#!/usr/bin/env python3
"""Fit and certify shared-factor gates on public Mamba-2 weight domains.

The NPZ bundle is an experimental plaintext oracle, not a native payload.
No text, hidden activations or quality evaluation enter coefficient fitting.
"""

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

from manage_dgx_build import payload_sha256

from fhemamba import __version__
from fhemamba.artifacts import current_git_commit
from fhemamba.polynomial_certificate import certify_dissipative_gate
from fhemamba.selective_gates import fit_gate_roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--dissipation-degree", type=int, default=1024)
    parser.add_argument("--equilibrium-degree", type=int, default=512)
    parser.add_argument("--inward-margin", type=float, default=2e-6)
    parser.add_argument("--grid-size", type=int, default=8193)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.dissipation_degree, args.equilibrium_degree) < 1 or args.grid_size < 2:
        parser.error("positive degrees and a validation grid of at least two points required")
    if args.bundle.exists() or args.output.exists():
        parser.error("refusing to overwrite an existing frozen bundle or report")
    start = time.perf_counter()
    payload_hash = payload_sha256(args.payload)
    sources = [
        Path(__file__),
        Path(__file__).resolve().parents[1] / "src/fhemamba/selective_gates.py",
        Path(__file__).resolve().parents[1] / "src/fhemamba/polynomial_certificate.py",
    ]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    arrays, rows = {}, []
    for path in sorted(args.payload.glob("layer_*/meta.json")):
        meta = json.loads(path.read_text())
        layer = meta["layer_index"]
        width, heads = meta["dims"]["d_model"], meta["dims"]["num_heads"]
        w = np.fromfile(path.parent / "in_proj_w.bin", dtype="<f4").reshape(-1, width).astype(float)
        gamma = np.fromfile(path.parent / "block_norm_w.bin", dtype="<f4").astype(float)
        bias = np.fromfile(path.parent / "dt_bias.bin", dtype="<f4").astype(float)
        a_log = np.fromfile(path.parent / "a_log.bin", dtype="<f4")
        rates = torch.from_numpy(a_log).exp().numpy().astype(float)
        radius = 1.1 * np.sqrt(width) * np.linalg.norm(w[-heads:] * gamma[None, :], axis=1)
        lo, hi = bias - radius, bias + radius
        p, q = fit_gate_roots(
            lo,
            hi,
            rates,
            dissipation_degree=args.dissipation_degree,
            equilibrium_degree=args.equilibrium_degree,
            inward_margin=args.inward_margin,
        )
        certificates = []
        for head in range(heads):
            p_head, q_head = (
                np.trim_zeros(coefficients[:, head], trim="b").tolist() for coefficients in (p, q)
            )
            certificate = certify_dissipative_gate(p_head or [0.0], q_head or [0.0])
            certificates.append(
                dict(certificate, head=head, p_degree=len(p_head) - 1, q_degree=len(q_head) - 1)
            )
        grid = np.linspace(-1, 1, args.grid_size)
        z = bias[None, :] + grid[:, None] * radius[None, :]
        target_write = np.logaddexp(0, z)
        target_decay = np.exp(-target_write * rates)
        pv, qv = (np.polynomial.chebyshev.chebval(grid, coeff).T for coeff in (p, q))
        decay, write = 1 - pv * pv, pv * pv * qv * qv
        row = {
            "layer": layer,
            "input_domains": np.stack((lo, hi), axis=1).tolist(),
            "heads_certified": sum(c["certified"] for c in certificates),
            "certificates": certificates,
            "sampled_max_write_error": float(np.abs(write - target_write).max()),
            "sampled_max_decay_error": float(np.abs(decay - target_decay).max()),
            "sampled_min_decay": float(decay.min()),
            "sampled_max_decay": float(decay.max()),
            "coefficients_sha256": hashlib.sha256(
                p.astype("<f8").tobytes() + q.astype("<f8").tobytes()
            ).hexdigest(),
        }
        rows.append(row)
        for name, value in (("lo", lo), ("hi", hi), ("p", p), ("q", q), ("rates", rates)):
            arrays[f"layer_{layer}_{name}"] = value
        print(
            f"layer {layer}: certified {row['heads_certified']}/{heads}, "
            f"write error {row['sampled_max_write_error']:.3g}, "
            f"decay error {row['sampled_max_decay_error']:.3g}",
            flush=True,
        )
    if not rows:
        parser.error("no layer payloads found")
    certified = all(c["certified"] for r in rows for c in r["certificates"])
    policy = {
        "dissipation_degree": args.dissipation_degree,
        "equilibrium_degree": args.equilibrium_degree,
        "inward_margin": args.inward_margin,
        "trim_l1": 1e-11,
        "public_projection_margin": 1.1,
        "all_heads_retained": True,
        "rate_convention": "exp of checkpoint float32 A_log in torch float32",
    }
    bundle_hash = None
    if certified:
        manifest = {
            "format": "dissipative-mamba2-gates-v1",
            "input_payload_sha256": payload_hash,
            "source_sha256": source_hashes,
            "policy": policy,
            "layers": [r["layer"] for r in rows],
            "certified_heads": sum(r["heads_certified"] for r in rows),
        }
        arrays["manifest"] = np.array(json.dumps(manifest, sort_keys=True))
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        with args.bundle.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        bundle_hash = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    report = {
        "stage": "dissipative-selective-gate-fit",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "input_payload_sha256": payload_hash,
        "source_sha256": source_hashes,
        "policy": policy,
        "grid_size": args.grid_size,
        "all_certified": certified,
        "bundle_written": certified,
        "bundle_sha256": bundle_hash,
        "rows": rows,
        "summary": {
            "heads_certified": sum(r["heads_certified"] for r in rows),
            "worst_sampled_write_error": max(r["sampled_max_write_error"] for r in rows),
            "worst_sampled_decay_error": max(r["sampled_max_decay_error"] for r in rows),
            "max_state_gain": max(c["gain"] for r in rows for c in r["certificates"]),
        },
        "elapsed_seconds": time.perf_counter() - start,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "language_quality_measured": False,
            "input_domain_membership_proved": False,
            "approximation_error_certified": False,
            "ckks_rounding_error_certified": False,
            "claim": "Exact-rational state invariants for the actual shared-factor coefficients.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not certified:
        raise SystemExit("some heads failed certification; no bundle written")


if __name__ == "__main__":
    main()
