#!/usr/bin/env python3
"""Certify Newton's non-expansive basin; compute public projection envelopes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manage_dgx_build import payload_sha256

from fhemamba import __version__
from fhemamba.artifacts import current_git_commit
from fhemamba.polynomial_certificate import certify_newton_initializer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.payload.glob("layer_*/meta.json")):
        meta = json.loads(path.read_text())
        certificates = {}
        for site in ("rms_invsqrt", "gated_rms_invsqrt"):
            certificates[site] = certify_newton_initializer(meta["polys"][site])
            print(path.parent.name, site, certificates[site]["certified"], flush=True)
        width, inner = meta["dims"]["d_model"], meta["dims"]["d_inner"]
        weight = np.fromfile(path.parent / "in_proj_w.bin", dtype="<f4").reshape(-1, width)
        gamma = np.fromfile(path.parent / "block_norm_w.bin", dtype="<f4")
        folded = weight.astype(np.float64) * gamma.astype(np.float64)[None, :]
        # Numerical estimates of an analytic Cauchy-Schwarz envelope. They
        # are deliberately separate from the exact-rational certificates.
        gate_bounds = np.sqrt(width) * np.linalg.norm(folded[:inner], axis=1)
        rows.append(
            {
                "layer": meta["layer_index"],
                "newton_certificates": certificates,
                "gate_envelope_max_estimate": float(gate_bounds.max()),
                "gate_envelope_median_estimate": float(np.median(gate_bounds)),
                "gate_envelope_per_channel_estimates": gate_bounds.tolist(),
                "existing_gate_interval": [
                    meta["polys"]["gate_silu"]["lo"],
                    meta["polys"]["gate_silu"]["hi"],
                ],
            }
        )
    if not rows:
        parser.error("no layer payloads found")
    source = Path(__file__).resolve().parents[1] / "src/fhemamba/polynomial_certificate.py"
    report = {
        "stage": "polynomial-range-certificate",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), source)
        },
        "input_payload_sha256": payload_sha256(args.payload),
        "all_newton_initializers_certified": all(
            cert["certified"] for row in rows for cert in row["newton_certificates"].values()
        ),
        "rows": rows,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "variance_domain_membership_proved": False,
            "ckks_rounding_error_certified": False,
            "inverse_sqrt_approximation_accuracy_certified": False,
            "claim": "Exact-rational Newton basin certificates on declared public intervals.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
