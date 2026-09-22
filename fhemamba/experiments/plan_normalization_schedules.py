#!/usr/bin/env python3
"""Build public variance-domain schedules; no text enters coefficient planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba.artifacts import current_git_commit
from fhemamba.m1_payload import _poly_ops_from_export
from fhemamba.normalization import certify_schedule, fixed_newton_cost, plan_invsqrt
from fhemamba.ops import (
    PolyInitNewton,
    SquaredPolyInitNewton,
    _poly_interval,
    positive_binomial_seed,
)
from manage_dgx_build import payload_sha256

from fhemamba import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-error", type=float, default=1e-7)
    parser.add_argument("--grid-size", type=int, default=8193)
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("grid-size must be at least two")
    if args.bundle.exists() or args.output.exists():
        parser.error("refusing to overwrite an existing frozen bundle/report")
    start = time.monotonic()
    chain = json.loads((args.payload / "chain.json").read_text())
    layers = chain["n_layers"]
    metadata = [
        json.loads((args.payload / d / "meta.json").read_text()) for d in chain["layer_dirs"]
    ]
    ops = _poly_ops_from_export(args.payload, layers)
    operators, measurements = [], []
    for (layer, name), old in sorted(ops.layer_polys.items()):
        if name not in ("rms_invsqrt", "gated_rms_invsqrt"):
            continue
        eps = (
            chain["final_norm_eps"]
            if layer == layers
            else metadata[layer]["eps"][
                "gated_norm" if name == "gated_rms_invsqrt" else "block_norm"
            ]
        )
        # Include the checkpoint epsilon rounded to float32 in the public domain.
        lo = min(float(eps), float(np.float32(eps)))
        hi = 4 * _poly_interval(old)[1]
        schedule = plan_invsqrt(lo, hi, tolerance=args.relative_error)
        operators.append({"layer": layer, "site": name, "recipe": schedule.recipe()})
        grid = torch.from_numpy(np.geomspace(lo, hi, args.grid_size))
        factor = schedule(grid) * grid.sqrt()
        squared = isinstance(old, SquaredPolyInitNewton)
        seed = positive_binomial_seed(hi, 63, power=0.25 if squared else 0.5)
        previous = SquaredPolyInitNewton(seed, 8, 0.85) if squared else PolyInitNewton(seed, 8)
        previous_factor = previous(grid) * grid.sqrt()
        measurements.append(
            {
                "layer": layer,
                "site": name,
                "interval": [lo, hi],
                "certificate": certify_schedule(schedule),
                "abstract_cost": schedule.abstract_cost(),
                "fixed_newton_same_accuracy": fixed_newton_cost(schedule),
                "sampled_float64_relative_error": float((factor - 1).abs().max()),
                "previous_binomial63_newton8_sampled_relative_error": float(
                    (previous_factor - 1).abs().max()
                ),
                "grid_size": args.grid_size,
            }
        )
    bundle = {
        "format": "fhemamba-normalization-schedules-v1",
        "input_payload_sha256": payload_sha256(args.payload),
        "domain_policy": "[min(eps64,eps32),4*old_hi]; upper endpoints remain conditional",
        "operators": operators,
    }
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    args.bundle.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    report = {
        "stage": "normalization-schedule-certificate",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "input_payload_sha256": bundle["input_payload_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/fhemamba/normalization.py").read_bytes()
        ).hexdigest(),
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "operator_count": len(operators),
        "all_certified": all(m["certificate"]["certified"] for m in measurements),
        "planning_seconds": time.monotonic() - start,
        "operators": measurements,
        "measurement_scope": {
            "artifact_level_report": True,
            "encrypted_execution": False,
            "full_model_correctness_claimed": False,
            "text_used_in_planning": False,
            "includes_ckks_error": False,
            "global_variance_domain_membership_proved": False,
            "claim": "Conditional real-arithmetic error certificates; symbolic ct-ct depth only.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"certified {len(operators)} schedules; bundle {report['bundle_sha256']}")


if __name__ == "__main__":
    main()
