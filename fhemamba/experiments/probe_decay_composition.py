#!/usr/bin/env python3
"""Screen direct exp(A*softplus(z)) polynomials on exported per-layer domains.

Independent uniform points check Chebyshev interpolants. This is neither a
formal interval certificate nor an FHE/PPL benchmark; report overshoots rather
than clipping them away. The existing decay head mask is reported separately.
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

from fhemamba.artifacts import current_git_commit

from fhemamba import __version__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--grid-size", type=int, default=4097)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.degrees or min(args.degrees) < 1 or args.grid_size <= max(args.degrees):
        parser.error("positive degrees and a larger validation grid are required")
    paths = sorted(args.payload.glob("layer_*/meta.json"))
    if not paths:
        parser.error("no layer payloads found")

    nodes = np.linspace(-1, 1, args.grid_size)
    rows, hashes, pruned_heads = [], {}, []
    for path in paths:
        meta = json.loads(path.read_text())
        spec = meta["polys"]["dt_softplus"]
        lo, hi = spec["lo"], spec["hi"]
        a_path = path.parent / "a_log.bin"
        for source in (path, a_path):
            hashes[str(source.relative_to(args.payload))] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
        a_values = -np.exp(np.fromfile(a_path, dtype="<f4").astype(float))
        active = np.asarray(meta["polys"]["decay_exp"].get("head_mask", np.ones_like(a_values)))
        threshold = float(meta["polys"]["decay_exp"].get("head_clip_threshold", 32.0))
        for head, a_value in enumerate(a_values):
            if not active[head]:
                # A<0: the largest exact decay is at the LOW input endpoint.
                maximum = float(np.exp(a_value * np.logaddexp(0, lo)))
                pruned_heads.append(
                    {
                        "layer": meta["layer_index"],
                        "head": head,
                        "input_interval": [lo, hi],
                        "max_exact_decay_on_interval": maximum,
                        "negligible_threshold": float(np.exp(-threshold)),
                        "uniformly_negligible_on_interval": bool(maximum <= np.exp(-threshold)),
                    }
                )
        z = (nodes + 1) * ((hi - lo) / 2) + lo
        for degree in args.degrees:
            for head, a_value in enumerate(a_values):
                coefficients = chebinterpolate(
                    lambda t, a_value=a_value, lo=lo, hi=hi: np.exp(
                        a_value * np.logaddexp(0, (t + 1) * ((hi - lo) / 2) + lo)
                    ),
                    degree,
                )
                values = chebval(nodes, coefficients)
                exact = np.exp(a_value * np.logaddexp(0, z))
                rows.append(
                    {
                        "layer": meta["layer_index"],
                        "head": head,
                        "degree": degree,
                        "existing_head_active": bool(active[head]),
                        "max_abs_error": float(np.max(np.abs(values - exact))),
                        "sampled_min": float(values.min()),
                        "sampled_max": float(values.max()),
                    }
                )
    summary = []
    for degree in args.degrees:
        for scope in ("all-heads", "existing-active-heads"):
            selected = [
                r
                for r in rows
                if r["degree"] == degree and (scope == "all-heads" or r["existing_head_active"])
            ]
            summary.append(
                {
                    "degree": degree,
                    "scope": scope,
                    "head_count": len(selected),
                    "max_abs_error": max(r["max_abs_error"] for r in selected),
                    "sampled_min": min(r["sampled_min"] for r in selected),
                    "sampled_max": max(r["sampled_max"] for r in selected),
                    "range_violating_heads": sum(
                        r["sampled_min"] < -1e-12 or r["sampled_max"] > 1 + 1e-12 for r in selected
                    ),
                }
            )
    report = {
        "stage": "decay-composition-screen",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "status": "unpromoted",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashes,
        "grid_size": args.grid_size,
        "summary": summary,
        "rows": rows,
        "head_pruning_audit": {
            "criterion": "For A<0, sup exp(A softplus(z)) occurs at the lower endpoint.",
            "pruned_head_count": len(pruned_heads),
            "uniformly_negligible_head_count": sum(
                bool(r["uniformly_negligible_on_interval"]) for r in pruned_heads
            ),
            "rows": pruned_heads,
            "scope": "Analytic extrema evaluated in float64 on the exported input interval; "
            "not a formal rounding certificate or evidence that actual tokens attain endpoints.",
        },
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "native_execution_measured": False,
            "claim": "Float64 sampled approximation to the exact composite, not equivalence "
            "to the existing polynomial circuit; no interval, PPL or FHE certificate.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
