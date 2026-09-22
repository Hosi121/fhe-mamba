#!/usr/bin/env python3
"""Isolate joint-gate error with every other model nonlinearity exact.

This ablation is not a fully polynomial/FHE circuit. It distinguishes the
new gate approximation from errors in convolution, activation and norm fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diagnose_payload_domain import DiagnosticPolyOps

from fhemamba import __version__
from fhemamba.artifacts import current_git_commit
from fhemamba.ops import PolyOps
from fhemamba.reference import model_forward
from fhemamba.selective_gates import DissipativeGate


class ExactExceptGates(DiagnosticPolyOps):
    def _apply(self, x, site, exact_fn):
        return exact_fn(x)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tokens-pt", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--offset", type=int, default=65536)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 2 or args.offset < 0:
        parser.error("at least two tokens and a nonnegative offset are required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)
    gates = {}
    with np.load(args.bundle, allow_pickle=False) as bundle:
        manifest = json.loads(str(bundle["manifest"]))
        if manifest["format"] != "dissipative-mamba2-gates-v1":
            parser.error("unsupported bundle format")
        if manifest["layers"] != list(range(len(model.backbone.layers))):
            parser.error("bundle must cover all checkpoint layers")
        for layer in manifest["layers"]:
            gates[layer] = (
                DissipativeGate(
                    *(bundle[f"layer_{layer}_{name}"] for name in ("lo", "hi", "p", "q"))
                ),
                bundle[f"layer_{layer}_rates"],
            )
    ops = ExactExceptGates(PolyOps({}, frozenset()), gates)
    ids = torch.load(args.tokens_pt, weights_only=True, map_location="cpu")
    ids = ids[:, args.offset : args.offset + args.tokens].to(args.device)
    if ids.shape[1] != args.tokens:
        parser.error("requested token window exceeds the input file")
    exact = model_forward(model, ids, scan="chunked")["logits"]
    exact_nll = torch.nn.functional.cross_entropy(exact[0, :-1], ids[0, 1:])
    exact = exact.cpu()
    logits = model_forward(model, ids, ops, scan="chunked")["logits"]
    if not bool(logits.isfinite().all()):
        raise FloatingPointError("non-finite gate-only ablation")
    nll = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
    logits = logits.cpu()
    quality = {
        "exact_ppl": float(exact_nll.exp()),
        "candidate_ppl": float(nll.exp()),
        "relative_ppl_increase": float((nll - exact_nll).expm1()),
        "max_logit_error": float((logits - exact).abs().max()),
        "top1_agreement": float(
            (logits[0, :-1].argmax(-1) == exact[0, :-1].argmax(-1)).float().mean()
        ),
        "candidate_finite": True,
        "predicted_tokens": args.tokens - 1,
    }
    sources = [Path(__file__), Path(__file__).with_name("diagnose_payload_domain.py")]
    sources.extend(
        Path(__file__).resolve().parents[1] / "src/fhemamba" / name
        for name in ("ops.py", "reference.py", "selective_gates.py")
    )
    report = {
        "stage": "dissipative-gate-only-ablation",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "input_payload_sha256": manifest["input_payload_sha256"],
        "token_ids_sha256": hashlib.sha256(ids.cpu().numpy().astype("<i8").tobytes()).hexdigest(),
        "tokens": args.tokens,
        "token_offset": args.offset,
        "device": args.device,
        "torch_version": torch.__version__,
        "tf32_enabled": False,
        "quality": quality,
        "events": ops.events,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "language_quality_measured": True,
            "fully_polynomial_model": False,
            "all_heads_retained": True,
            "claim": "Development ablation: only joint write/decay gates are approximated.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(quality, flush=True)


if __name__ == "__main__":
    main()
