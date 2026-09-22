#!/usr/bin/env python3
"""Clone a frozen payload with public state scales measured on independent text.

Only state-coordinate bounds change. Polynomial coefficients, weights, FIFO
and transient bounds, and evaluation references are copied unchanged. Both
exact and frozen-poly teacher-forced trajectories contribute to the scales.
Calibration maxima are observations, not worst-case bounds on arbitrary text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.artifacts import current_git_commit  # noqa: E402
from fhemamba.m1_payload import _poly_ops_from_export  # noqa: E402
from fhemamba.reference import init_states, model_forward  # noqa: E402
from fhemamba.state_coordinates import regularize_row_scales  # noqa: E402
from manage_dgx_build import payload_sha256  # noqa: E402

from fhemamba import __version__  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--calibration-text", type=Path)
    inputs.add_argument("--calibration-tokens-pt", type=Path)
    parser.add_argument("--offsets", type=int, nargs="+", default=[0])
    parser.add_argument("--calibration-description", required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--group-heads", type=int, default=4)
    parser.add_argument("--group-scale-floor", type=float, default=0.0)
    parser.add_argument("--output-chain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens, args.threads, args.group_heads) < 1:
        parser.error("positive tokens, threads, and group-heads are required")
    if not 0 <= args.group_scale_floor <= 1 or min(args.offsets) < 0:
        parser.error("floor in [0, 1] and nonnegative offsets are required")
    ordered = sorted(args.offsets)
    if any(right < left + args.tokens for left, right in pairwise(ordered)):
        parser.error("calibration windows must not overlap")
    if args.output_chain.exists():
        parser.error("output-chain must be a new directory")
    if args.output_chain.resolve().is_relative_to(args.payload.resolve()):
        parser.error("output-chain cannot be inside the source payload")
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval()
    source = args.calibration_text or args.calibration_tokens_pt
    if args.calibration_text:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        all_ids = tokenizer(source.read_text(), return_tensors="pt").input_ids
    else:
        all_ids = torch.load(source, weights_only=True, map_location="cpu")
    if all_ids.ndim != 2 or all_ids.shape[0] != 1 or all_ids.dtype != torch.int64:
        parser.error("calibration token input must be int64 [1, length]")
    if max(args.offsets) + args.tokens > all_ids.shape[1]:
        parser.error("calibration windows exceed token input")
    ids = torch.cat([all_ids[:, start : start + args.tokens] for start in args.offsets])
    chain = json.loads((args.payload / "chain.json").read_text())
    if chain["n_layers"] != len(model.backbone.layers):
        raise ValueError("payload/model layer counts differ")
    for directory, block in zip(chain["layer_dirs"], model.backbone.layers, strict=True):
        meta = json.loads((args.payload / directory / "meta.json").read_text())
        if meta["carried_bounds"].get("source") != "calibration_text":
            raise ValueError("parent payload must have independent carried-bound calibration")
        if block.mixer.num_heads % args.group_heads:
            raise ValueError("group-heads must divide the model's head count")
    states = init_states(model, batch_size=len(args.offsets))
    rows = [state.ssm.new_zeros(state.ssm.shape[1:3]) for state in states]
    per_circuit = {}
    for name, ops in (
        ("exact", None),
        ("exported-poly", _poly_ops_from_export(args.payload, chain["n_layers"])),
    ):
        states = init_states(model, batch_size=len(args.offsets))
        maxima = [state.ssm.new_zeros(state.ssm.shape[1:3]) for state in states]
        for token in range(ids.shape[1]):
            model_forward(
                model,
                ids[:, token : token + 1],
                ops=ops,
                scan="loop",
                states=states,
                output_logits=False,
            )
            for layer, state in enumerate(states):
                maxima[layer] = maxima[layer].maximum(state.ssm.abs().amax(dim=(0, 3)))
            if (token + 1) % 128 == 0:
                print(f"{name} calibration tokens={token + 1} per window", flush=True)
        if any(not bool(maximum.isfinite().all()) for maximum in maxima):
            report = {
                "stage": "state-coordinate-calibration",
                "version": __version__,
                "repo_commit": current_git_commit(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "status": "failed",
                "failure": f"non-finite {name} state on calibration data",
                "failed_circuit": name,
                "non_finite_layers": [
                    i for i, value in enumerate(maxima) if not bool(value.isfinite().all())
                ],
                "input_payload_sha256": payload_sha256(args.payload),
                "calibration_input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "calibration_token_ids_sha256": hashlib.sha256(
                    ids.numpy().astype("<i8").tobytes()
                ).hexdigest(),
                "calibration_description": args.calibration_description,
                "window_offsets": args.offsets,
                "tokens_per_window": ids.shape[1],
                "window_resets_state": True,
                "completed_circuit_layer_maxima": per_circuit,
                "output_payload_written": False,
                "measurement_scope": {
                    "artifact_level_report": True,
                    "full_model_correctness_claimed": False,
                    "encrypted_execution": False,
                    "claim": "Failed independent calibration; no new payload or scales accepted.",
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            raise SystemExit(f"{report['failure']}; no payload written")
        rows = [row.maximum(maximum) for row, maximum in zip(rows, maxima, strict=True)]
        per_circuit[name] = [float(maximum.max()) for maximum in maxima]
    provenance = {
        "source": "calibration_text",
        "description": args.calibration_description,
        "calibration_input_kind": "text" if args.calibration_text else "int64-token-file",
        "calibration_input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "calibration_token_ids_sha256": hashlib.sha256(
            ids.numpy().astype("<i8").tobytes()
        ).hexdigest(),
        "calibration_tokens": ids.numel(),
        "tokens_per_window": ids.shape[1],
        "window_offsets": args.offsets,
        "window_resets_state": True,
        "circuits": list(per_circuit),
        "layout": "flattened [head, channel]; maximum over time and state coordinate",
        "minimum_runtime_scale": 1e-6,
        "group_heads": args.group_heads,
        "group_scale_floor": args.group_scale_floor,
        "maximum_group_to_row_amplification": (
            1.0 / args.group_scale_floor if args.group_scale_floor > 0 else None
        ),
        "uniform_bound_claimed": False,
        "parent_payload_sha256": payload_sha256(args.payload),
        "checkpoint_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.checkpoint.iterdir())
            if path.suffix in {".safetensors", ".json"}
        },
    }
    scales = [
        torch.from_numpy(
            regularize_row_scales(row.numpy(), args.group_heads, args.group_scale_floor)
        )
        for row in rows
    ]
    shutil.copytree(args.payload, args.output_chain)
    for directory, row in zip(chain["layer_dirs"], scales, strict=True):
        path = args.output_chain / directory / "meta.json"
        meta = json.loads(path.read_text())
        bounds = meta["carried_bounds"]
        bounds["state_abs_max"] = float(row.max())
        bounds["state_head_abs_max"] = row.amax(dim=1).tolist()
        bounds["state_row_abs_max"] = row.flatten().tolist()
        bounds["state_coordinate_calibration"] = provenance
        # Legacy payload metadata represents an unbounded time_step_limit
        # with Infinity. Preserve that existing format; new scales were
        # checked finite above and the report below remains strict JSON.
        path.write_text(json.dumps(meta, indent=2) + "\n")
    chain["state_coordinate_calibration"] = provenance
    (args.output_chain / "chain.json").write_text(json.dumps(chain, indent=2) + "\n")
    summary = []
    for layer, row in enumerate(scales):
        # Report ciphertext grouping explicitly; it is separate from B/C groups.
        group = row.reshape(-1, args.group_heads * row.shape[1]).amax(dim=1)
        ratios = row.reshape(-1, args.group_heads * row.shape[1]) / group[:, None].clamp_min(1e-6)
        summary.append(
            {
                "layer": layer,
                "group_heads": args.group_heads,
                "group_maxima": group.tolist(),
                "mean_row_to_group_scale": float(ratios.mean()),
                "median_row_to_group_scale": float(ratios.median()),
            }
        )
    report = {
        "stage": "state-coordinate-calibration",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration": provenance,
        "output_payload_sha256": payload_sha256(args.output_chain),
        "per_circuit_layer_maxima": per_circuit,
        "rows": summary,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "uniform_bound_claimed": False,
            "claim": "State scales from independent exact and frozen-poly calibration runs.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
