#!/usr/bin/env python3
"""Check public evaluation-reference states against independent calibration.

This audits the declared bounds, not the bootstrap's mathematical domain or
encrypted noise. Evaluation references never change the calibration scales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba.artifacts import current_git_commit
from manage_dgx_build import payload_sha256

from fhemamba import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--group-heads", type=int, default=4)
    parser.add_argument("--margin", type=float, default=1.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.group_heads < 1 or not np.isfinite(args.margin) or args.margin < 1:
        parser.error("positive group-heads and finite margin >= 1 are required")
    rows = []
    for path in sorted(args.payload.glob("layer_*/meta.json")):
        meta = json.loads(path.read_text())
        bounds = meta["carried_bounds"]
        heads, width = meta["dims"]["num_heads"], meta["dims"]["head_dim"]
        if heads % args.group_heads:
            parser.error("group-heads must divide num_heads")
        row_bounds = np.maximum(np.asarray(bounds["state_row_abs_max"]).reshape(heads, width), 1e-6)
        group_bounds = np.asarray(bounds["state_head_abs_max"]).reshape(-1, args.group_heads).max(1)
        group_bounds = np.repeat(np.maximum(group_bounds, 1e-6), args.group_heads)[:, None]
        for tensor in ("test_state_output_poly", "autoregressive_poly_state_output"):
            if tensor not in meta["tensors"]:
                continue
            values = np.fromfile(path.parent / f"{tensor}.bin", dtype="<f4")
            values = np.abs(values.reshape(meta["tensors"][tensor])).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite reference: {path.parent.name}/{tensor}")
            for mode, scales in (("head-group", group_bounds), ("head-channel", row_bounds)):
                ratios = values / scales[None, :, :, None]
                rows.append(
                    {
                        "layer": meta["layer_index"],
                        "reference": tensor,
                        "scale_granularity": mode,
                        "max_reference_to_scale_ratio": float(ratios.max()),
                        "above_margin": int((ratios > args.margin).sum()),
                        "scalar_count": int(ratios.size),
                    }
                )
    if not rows:
        parser.error("no compatible state references found")
    report = {
        "stage": "state-scale-coverage-audit",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_payload_sha256": payload_sha256(args.payload),
        "group_heads": args.group_heads,
        "margin": args.margin,
        "rows": rows,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "calibration_modified": False,
            "claim": "Reference-state coverage of declared calibration scales, without refitting.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
