#!/usr/bin/env python3
"""Measure frozen-payload quality and isolate its head-pruning approximation.

Teacher forcing uses identical tokens and reset boundaries for all three
circuits. Exact-with-mask changes only decay pruning; exported-poly evaluates
the payload's coefficients unchanged. This is a plaintext quality screen,
not encrypted inference or a guarantee on inputs outside these windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from manage_dgx_build import payload_sha256  # noqa: E402
from torch.nn import functional as F  # noqa: E402, N812

from fhemamba import __version__  # noqa: E402
from fhemamba.artifacts import current_git_commit  # noqa: E402
from fhemamba.m1_payload import _poly_ops_from_export  # noqa: E402
from fhemamba.ops import Exact  # noqa: E402
from fhemamba.reference import model_forward  # noqa: E402


class MaskedExactOps(Exact):
    def __init__(self, masks: list[list[float]], *, apply_mask: bool) -> None:
        self.masks = masks
        self.apply_mask = apply_mask
        self.decay_maxima: dict[int, torch.Tensor] = {}

    def exp(self, x, site):
        values = super().exp(x, site)
        layer = site[0]
        maximum = values.amax(dim=tuple(range(values.ndim - 1)))
        previous = self.decay_maxima.get(layer)
        self.decay_maxima[layer] = maximum if previous is None else maximum.maximum(previous)
        if self.apply_mask:
            values = values * values.new_tensor(self.masks[layer])
        return values


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--tokens-pt", type=Path, required=True)
    parser.add_argument("--data-description", required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--max-windows", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.windows) < 2 or args.max_windows < 1 or args.threads < 1 or args.offset < 0:
        parser.error("windows >= 2 and positive max-windows/threads are required")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)
    chain = json.loads((args.payload / "chain.json").read_text())
    if chain["n_layers"] != len(model.backbone.layers):
        raise ValueError("payload/model layer counts differ")
    masks = []
    for layer, block in zip(chain["layer_dirs"], model.backbone.layers, strict=True):
        path = args.payload / layer
        meta = json.loads((path / "meta.json").read_text())
        masks.append(meta["polys"]["decay_exp"].get("head_mask", [1.0] * block.mixer.num_heads))
        # Reject an accidentally selected checkpoint before spending time on
        # quality measurements. Hashes below identify all remaining weights.
        for name, weight in (
            ("in_proj_w", block.mixer.in_proj.weight),
            ("out_proj_w", block.mixer.out_proj.weight),
        ):
            payload_weight = np.fromfile(path / f"{name}.bin", dtype="<f4")
            if not np.array_equal(payload_weight, weight.cpu().numpy().reshape(-1)):
                raise ValueError(f"checkpoint projection differs from payload: {layer}/{name}")
    ids = torch.load(args.tokens_pt, weights_only=True, map_location="cpu")
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.int64:
        raise ValueError("token input must be int64 [1, length]")
    ids = ids[:, args.offset :]
    report = {
        "stage": "frozen-payload-quality",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": {
            str(path): _sha(path)
            for path in (
                Path(__file__),
                Path(__file__).resolve().parents[1] / "src/fhemamba/reference.py",
                Path(__file__).resolve().parents[1] / "src/fhemamba/ops.py",
                Path(__file__).resolve().parents[1] / "src/fhemamba/m1_payload.py",
                Path(__file__).resolve().parents[1] / "src/fhemamba/selective_gates.py",
                Path(__file__).resolve().parents[1] / "src/fhemamba/normalization.py",
            )
        },
        "input_payload_sha256": payload_sha256(args.payload),
        "checkpoint_sha256": {
            p.name: _sha(p)
            for p in sorted(args.checkpoint.iterdir())
            if p.suffix in {".safetensors", ".json"}
        },
        "token_file_sha256": _sha(args.tokens_pt),
        "data_description": args.data_description,
        "token_offset": args.offset,
        "tf32_enabled": False,
        "stabilized_gate_bundle_sha256": chain.get("stabilized_gate_bundle_sha256"),
        "normalization_bundle_sha256": chain.get("normalization_bundle_sha256"),
        "device": args.device,
        "torch_version": torch.__version__,
        "threads": args.threads,
        "pruned_heads": sum(mask.count(0.0) for mask in masks),
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "teacher_forcing": True,
            "window_resets_state": True,
            "full_dataset_quality_claimed": False,
            "claim": "Frozen coefficients and head-mask quality on the recorded token windows.",
        },
        "rows": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for length in args.windows:
        count = min(args.max_windows, ids.shape[1] // length)
        if count == 0:
            raise ValueError(f"need at least {length} input tokens")
        totals = {
            name: {
                "nll_sum": 0.0,
                "finite": True,
                "max_logit_error": 0.0,
                "top1_matches": 0,
                "seconds": 0.0,
            }
            for name in ("exact", "exact-with-mask", "exported-poly")
        }
        exact_ops = MaskedExactOps(masks, apply_mask=False)
        masked_ops = MaskedExactOps(masks, apply_mask=True)
        poly_ops = _poly_ops_from_export(args.payload, chain["n_layers"])
        for window in range(count):
            chunk = ids[:, window * length : (window + 1) * length].to(args.device)
            baseline = None
            for name, ops in (
                ("exact", exact_ops),
                ("exact-with-mask", masked_ops),
                ("exported-poly", poly_ops),
            ):
                start = time.monotonic()
                logits = model_forward(model, chunk, ops=ops, scan="chunked")["logits"]
                entry = totals[name]
                entry["seconds"] += time.monotonic() - start
                finite = bool(logits.isfinite().all())
                entry["finite"] = entry["finite"] and finite
                if baseline is None:
                    baseline = logits
                if finite:
                    entry["nll_sum"] += float(
                        F.cross_entropy(logits[0, :-1].float(), chunk[0, 1:], reduction="sum")
                    )
                    entry["max_logit_error"] = max(
                        entry["max_logit_error"], float((logits - baseline).abs().max())
                    )
                    entry["top1_matches"] += int(
                        (logits[0, :-1].argmax(-1) == baseline[0, :-1].argmax(-1)).sum()
                    )
                print(f"length={length} window={window} {name} finite={finite}", flush=True)
        tokens = count * (length - 1)
        for result in totals.values():
            nll_sum = result.pop("nll_sum")
            matches = result.pop("top1_matches")
            nll = nll_sum / tokens if result["finite"] else None
            result.update(
                nll_per_token=nll,
                ppl=math.exp(nll) if nll is not None and nll < 700 else None,
                top1_agreement=matches / tokens if result["finite"] else None,
            )
            if not result["finite"]:
                result["max_logit_error"] = None
        observed = [
            {
                "layer": layer,
                "head": head,
                "max_observed_exact_decay": float(exact_ops.decay_maxima[layer][head]),
            }
            for layer, mask in enumerate(masks)
            for head, active in enumerate(mask)
            if not active
        ]
        report["rows"].append(
            {
                "window": length,
                "windows": count,
                "predicted_tokens": tokens,
                "evaluated_token_ids_sha256": hashlib.sha256(
                    ids[:, : count * length].numpy().astype("<i8").tobytes()
                ).hexdigest(),
                "circuits": totals,
                "poly_domain_violations": poly_ops.violations,
                "pruned_head_observations_on_exact_path": observed,
            }
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["complete"] = True
    finite = all(
        circuit["finite"] for row in report["rows"] for circuit in row["circuits"].values()
    )
    report["status"] = "measured" if finite else "failed-non-finite"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)
    if not finite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
