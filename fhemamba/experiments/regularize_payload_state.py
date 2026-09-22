#!/usr/bin/env python3
"""Clone a payload with a public floor on its existing row state scales.

No calibration/evaluation text is consumed and no model coefficient, reference,
FIFO or transient bound is changed. This isolates numerical conditioning from
the calibration data and the polynomial surrogate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba.artifacts import current_git_commit
from fhemamba.state_coordinates import regularize_row_scales
from manage_dgx_build import payload_sha256

from fhemamba import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--group-heads", type=int, default=4)
    parser.add_argument("--group-scale-floor", type=float, required=True)
    parser.add_argument("--output-chain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output_chain.exists() or args.output_chain.resolve().is_relative_to(
        args.payload.resolve()
    ):
        parser.error("output-chain must be a new directory outside the source payload")
    chain = json.loads((args.payload / "chain.json").read_text())
    parent_sha = payload_sha256(args.payload)
    metadata, summaries = [], []
    for directory in chain["layer_dirs"]:
        meta = json.loads((args.payload / directory / "meta.json").read_text())
        bounds = meta["carried_bounds"]
        if bounds.get("source") != "calibration_text" or not bounds.get(
            "state_coordinate_calibration"
        ):
            raise ValueError("independently calibrated row coordinates are required")
        dims = meta["dims"]
        old = np.asarray(bounds["state_row_abs_max"]).reshape(dims["num_heads"], dims["head_dim"])
        scales = regularize_row_scales(old, args.group_heads, args.group_scale_floor)
        # The raw observations and conditioning scales are separate fields.
        bounds["state_row_abs_max"] = scales.reshape(-1).tolist()
        bounds["state_row_observed_abs_max"] = old.reshape(-1).tolist()
        group = old.reshape(-1, args.group_heads * dims["head_dim"]).max(axis=1, keepdims=True)
        ratios = scales.reshape(group.shape[0], -1) / np.maximum(group, 1e-6)
        summaries.append(
            {
                "layer": meta["layer_index"],
                "raised_rows": int((scales > old).sum()),
                "mean_row_to_group_scale": float(ratios.mean()),
            }
        )
        metadata.append(meta)
    regularization = {
        "parent_payload_sha256": parent_sha,
        "group_heads": args.group_heads,
        "group_scale_floor": args.group_scale_floor,
        "maximum_group_to_row_amplification": 1 / args.group_scale_floor
        if args.group_scale_floor
        else None,
        "new_calibration_data": False,
        "evaluation_references_used_for_scales": False,
        "uniform_bound_claimed": False,
    }
    shutil.copytree(args.payload, args.output_chain)
    for directory, meta in zip(chain["layer_dirs"], metadata, strict=True):
        meta["carried_bounds"]["state_coordinate_regularization"] = regularization
        (args.output_chain / directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    chain["state_coordinate_regularization"] = regularization
    (args.output_chain / "chain.json").write_text(json.dumps(chain, indent=2) + "\n")
    report = {
        "stage": "state-coordinate-regularization",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).resolve().parents[1] / "src/fhemamba/state_coordinates.py",
            )
        },
        "regularization": regularization,
        "rows": summaries,
        "output_payload_sha256": payload_sha256(args.output_chain),
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "claim": (
                "Public conditioning of independently calibrated scales, "
                "without new data or reference fitting."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
