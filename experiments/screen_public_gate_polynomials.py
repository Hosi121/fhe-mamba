#!/usr/bin/env python3
"""Screen SiLU fits on projection envelopes derived from public model weights.

The envelope follows from non-expansive RMS normalization; its applicability
requires the variance interval and numerical-error assumptions to hold.
Uniform-grid errors are a screen, not an approximation-error certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebinterpolate, chebval

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manage_dgx_build import payload_sha256

from fhemamba import __version__
from fhemamba.artifacts import current_git_commit


def silu(x):
    return x * np.exp(-np.logaddexp(0.0, -x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=[64, 128, 192, 256, 384])
    parser.add_argument("--margin", type=float, default=1.1)
    parser.add_argument("--grid-size", type=int, default=8193)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not np.isfinite(args.margin)
        or args.margin < 1.0
        or min(args.degrees) < 1
        or args.grid_size <= max(args.degrees)
    ):
        parser.error("margin >= 1, positive degrees and a larger validation grid are required")
    rows = []
    grid = np.linspace(-1.0, 1.0, args.grid_size)
    for path in sorted(args.payload.glob("layer_*/meta.json")):
        meta = json.loads(path.read_text())
        width, inner = meta["dims"]["d_model"], meta["dims"]["d_inner"]
        weight = np.fromfile(path.parent / "in_proj_w.bin", dtype="<f4").reshape(-1, width)
        gamma = np.fromfile(path.parent / "block_norm_w.bin", dtype="<f4")
        folded = weight[:inner].astype(float) * gamma.astype(float)[None, :]
        radius = float(args.margin * np.sqrt(width) * np.linalg.norm(folded, axis=1).max())
        points = radius * grid
        target = silu(points)
        old = meta["polys"]["gate_silu"]
        old_values = chebval(
            (2 * points - old["lo"] - old["hi"]) / (old["hi"] - old["lo"]), old["coeffs"]
        )
        for degree in args.degrees:
            coefficients = chebinterpolate(lambda t, radius=radius: silu(radius * t), degree)
            values = chebval(grid, coefficients)
            rows.append(
                {
                    "layer": meta["layer_index"],
                    "degree": degree,
                    "interval": [-radius, radius],
                    "sampled_max_abs_error": float(np.abs(values - target).max()),
                    "old_fit_sampled_max_abs_error_on_new_domain": float(
                        np.abs(old_values - target).max()
                    ),
                }
            )
    if not rows:
        parser.error("no layer payloads found")
    report = {
        "stage": "public-gate-polynomial-screen",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "status": "unpromoted",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_payload_sha256": payload_sha256(args.payload),
        "margin": args.margin,
        "grid_size": args.grid_size,
        "rows": rows,
        "summary": [
            {
                "degree": degree,
                "worst_sampled_error": max(
                    r["sampled_max_abs_error"] for r in rows if r["degree"] == degree
                ),
            }
            for degree in args.degrees
        ],
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "language_quality_measured": False,
            "uniform_approximation_error_certified": False,
            "claim": "Candidate gate fits on public-weight envelopes; no payload changed.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
