#!/usr/bin/env python3
"""Screen jointly bounded Mamba-2 write/decay gates on public weight domains.

Bernstein coefficients certify the recurrence invariant, but introduce a new
surrogate. No encrypted implementation, quality result, or converted-Chebyshev
certificate is implied by this probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manage_dgx_build import payload_sha256

from fhemamba import __version__
from fhemamba.artifacts import current_git_commit
from fhemamba.polynomial_certificate import certify_bernstein_state_invariant


def basis(degree, grid):
    k = np.arange(degree + 1)
    return (
        np.asarray([math.comb(degree, int(i)) for i in k], dtype=float)[None, :]
        * grid[:, None] ** k[None, :]
        * (1 - grid[:, None]) ** (degree - k[None, :])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--grid-size", type=int, default=1025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.degrees) < 1 or max(args.degrees) > 256 or args.grid_size <= max(args.degrees):
        parser.error("degrees in [1,256] and a larger validation grid are required")
    grid = np.linspace(0, 1, args.grid_size)
    bases = {degree: basis(degree, grid) for degree in args.degrees}
    rows = []
    for path in sorted(args.payload.glob("layer_*/meta.json")):
        meta = json.loads(path.read_text())
        dims = meta["dims"]
        width, heads = dims["d_model"], dims["num_heads"]
        w = np.fromfile(path.parent / "in_proj_w.bin", dtype="<f4").reshape(-1, width).astype(float)
        gamma = np.fromfile(path.parent / "block_norm_w.bin", dtype="<f4").astype(float)
        bias = np.fromfile(path.parent / "dt_bias.bin", dtype="<f4").astype(float)
        rates = np.exp(np.fromfile(path.parent / "a_log.bin", dtype="<f4").astype(float))
        # Non-expansive RMSNorm implies this projection envelope. The 10%
        # allowance is not a proof of floating-point or CKKS error coverage.
        radius = 1.1 * np.sqrt(width) * np.linalg.norm(w[-heads:] * gamma[None, :], axis=1)
        lo, hi = bias - radius, bias + radius
        targets = np.logaddexp(0, lo[None, :] + grid[:, None] * (hi - lo)[None, :])
        target_decay = np.exp(-targets * rates[None, :])
        for degree in args.degrees:
            knots = np.linspace(0, 1, degree + 1)
            write = np.logaddexp(0, lo[None, :] + knots[:, None] * (hi - lo)[None, :])
            decay = np.exp(-write * rates[None, :])
            gains = write.max(axis=0) + 1 / rates
            corrections = 0
            for head in range(heads):
                # Round the coefficient constraint inward using exact dyadic
                # rationals. In particular, rounded a_i=1 requires b_i=0.
                for knot in range(degree + 1):
                    cap = Fraction(float(gains[head])) * (1 - Fraction(float(decay[knot, head])))
                    if Fraction(float(write[knot, head])) > cap:
                        value = float(cap)
                        if Fraction(value) > cap:
                            value = float(np.nextafter(value, 0.0))
                        write[knot, head] = value
                        corrections += 1
                certificate = certify_bernstein_state_invariant(
                    decay[:, head].tolist(), write[:, head].tolist(), float(gains[head])
                )
                if not certificate["certified"]:
                    raise RuntimeError("constructed coefficients failed their invariant")
            predicted_write, predicted_decay = bases[degree] @ write, bases[degree] @ decay
            rows.append(
                {
                    "layer": meta["layer_index"],
                    "degree": degree,
                    "heads_certified": heads,
                    "input_domains": np.stack((lo, hi), axis=1).tolist(),
                    "state_gains": gains.tolist(),
                    "inward_coefficient_corrections": corrections,
                    "coefficients_sha256": hashlib.sha256(
                        decay.astype("<f8").tobytes() + write.astype("<f8").tobytes()
                    ).hexdigest(),
                    "sampled_max_write_error": float(np.abs(predicted_write - targets).max()),
                    "sampled_max_decay_error": float(np.abs(predicted_decay - target_decay).max()),
                    "sampled_max_memory_decay_bias": float((target_decay - predicted_decay).max()),
                }
            )
    if not rows:
        parser.error("no layer payloads found")
    report = {
        "stage": "bounded-selective-gate-probe",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "input_payload_sha256": payload_sha256(args.payload),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).resolve().parents[1] / "src/fhemamba/polynomial_certificate.py",
            )
        },
        "grid_size": args.grid_size,
        "rows": rows,
        "summary": [
            {
                "degree": d,
                "heads_certified": sum(r["heads_certified"] for r in rows if r["degree"] == d),
                "worst_sampled_write_error": max(
                    r["sampled_max_write_error"] for r in rows if r["degree"] == d
                ),
                "worst_sampled_decay_error": max(
                    r["sampled_max_decay_error"] for r in rows if r["degree"] == d
                ),
            }
            for d in args.degrees
        ],
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "language_quality_measured": False,
            "all_heads_retained": True,
            "input_domain_membership_proved": False,
            "chebyshev_conversion_certified": False,
            "ckks_rounding_error_certified": False,
            "claim": (
                "Exact-rational coefficientwise state invariants for a new "
                "joint Bernstein gate surrogate; sampled errors only."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
