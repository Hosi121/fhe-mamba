"""Public-weight gate calibration and exact-rational basin certificates."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.chebyshev import chebinterpolate, chebval

from fhemamba.benchmarks.io import file_sha256, payload_sha256, sha256
from fhemamba.calibration.payload import folded_projection, layers, source_hashes
from fhemamba.calibration.payload import report as make_report
from fhemamba.polynomial_certificate import certify_dissipative_gate, certify_newton_initializer
from fhemamba.selective_gates import fit_gate_roots


def silu(x):
    return x * np.exp(-np.logaddexp(0.0, -x))


def certify_ranges(payload: Path) -> dict:
    sources = source_hashes(__file__, "polynomial_certificate.py")
    rows = []
    for directory, meta in layers(payload):
        certificates = {}
        for site in ("rms_invsqrt", "gated_rms_invsqrt"):
            certificates[site] = certify_newton_initializer(meta["polys"][site])
            print(directory.name, site, certificates[site]["certified"], flush=True)
        width, inner = (meta["dims"]["d_model"], meta["dims"]["d_inner"])
        folded = folded_projection(directory, meta, slice(None, inner))
        gate_bounds = np.sqrt(width) * np.linalg.norm(folded, axis=1)
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
        raise ValueError("no layer payloads found")
    report = make_report(
        "polynomial-range-certificate",
        sources,
        input_payload_sha256=payload_sha256(payload),
        all_newton_initializers_certified=all(
            cert["certified"] for row in rows for cert in row["newton_certificates"].values()
        ),
        rows=rows,
        measurement_scope={
            "variance_domain_membership_proved": False,
            "ckks_rounding_error_certified": False,
            "inverse_sqrt_approximation_accuracy_certified": False,
            "claim": "Exact-rational Newton basin certificates on declared public intervals.",
        },
    )
    return report


def screen_gates(
    payload: Path, *, degrees=(64, 128, 192, 256, 384), margin=1.1, grid_size=8193
) -> dict:
    sources = source_hashes(__file__)
    if (
        not np.isfinite(margin)
        or margin < 1.0
        or not degrees
        or min(degrees) < 1
        or (grid_size <= max(degrees))
    ):
        raise ValueError("margin >= 1, positive degrees and a larger validation grid are required")
    rows = []
    grid = np.linspace(-1.0, 1.0, grid_size)
    for directory, meta in layers(payload):
        width, inner = (meta["dims"]["d_model"], meta["dims"]["d_inner"])
        folded = folded_projection(directory, meta, slice(None, inner))
        radius = float(margin * np.sqrt(width) * np.linalg.norm(folded, axis=1).max())
        points = radius * grid
        target = silu(points)
        old = meta["polys"]["gate_silu"]
        old_values = chebval(
            (2 * points - old["lo"] - old["hi"]) / (old["hi"] - old["lo"]), old["coeffs"]
        )
        for degree in degrees:
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
        raise ValueError("no layer payloads found")
    report = make_report(
        "public-gate-polynomial-screen",
        sources,
        status="unpromoted",
        input_payload_sha256=payload_sha256(payload),
        margin=margin,
        grid_size=grid_size,
        rows=rows,
        summary=[
            {
                "degree": degree,
                "worst_sampled_error": max(
                    r["sampled_max_abs_error"] for r in rows if r["degree"] == degree
                ),
            }
            for degree in degrees
        ],
        measurement_scope={
            "language_quality_measured": False,
            "uniform_approximation_error_certified": False,
            "claim": "Candidate gate fits on public-weight envelopes; no payload changed.",
        },
    )
    print(report["summary"], flush=True)
    return report


def fit_gates(
    payload: Path,
    *,
    bundle: Path,
    dissipation_degree=1024,
    equilibrium_degree=512,
    inward_margin=2e-06,
    grid_size=8193,
) -> dict:
    sources = source_hashes(__file__, "selective_gates.py", "polynomial_certificate.py")
    if min(dissipation_degree, equilibrium_degree) < 1 or grid_size < 2:
        raise ValueError("positive degrees and a validation grid of at least two points required")
    if bundle.exists():
        raise ValueError("refusing to overwrite an existing frozen bundle or report")
    start = time.perf_counter()
    payload_hash = payload_sha256(payload)
    arrays, rows = ({}, [])
    for directory, meta in layers(payload):
        layer = meta["layer_index"]
        width, heads = (meta["dims"]["d_model"], meta["dims"]["num_heads"])
        bias = np.fromfile(directory / "dt_bias.bin", dtype="<f4").astype(float)
        a_log = np.fromfile(directory / "a_log.bin", dtype="<f4")
        rates = torch.from_numpy(a_log).exp().numpy().astype(float)
        radius = (
            1.1
            * np.sqrt(width)
            * np.linalg.norm(folded_projection(directory, meta, slice(-heads, None)), axis=1)
        )
        lo, hi = (bias - radius, bias + radius)
        p, q = fit_gate_roots(
            lo,
            hi,
            rates,
            dissipation_degree=dissipation_degree,
            equilibrium_degree=equilibrium_degree,
            inward_margin=inward_margin,
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
        grid = np.linspace(-1, 1, grid_size)
        z = bias[None, :] + grid[:, None] * radius[None, :]
        target_write = np.logaddexp(0, z)
        target_decay = np.exp(-target_write * rates)
        pv, qv = (np.polynomial.chebyshev.chebval(grid, coeff).T for coeff in (p, q))
        decay, write = (1 - pv * pv, pv * pv * qv * qv)
        row = {
            "layer": layer,
            "input_domains": np.stack((lo, hi), axis=1).tolist(),
            "heads_certified": sum(c["certified"] for c in certificates),
            "certificates": certificates,
            "sampled_max_write_error": float(np.abs(write - target_write).max()),
            "sampled_max_decay_error": float(np.abs(decay - target_decay).max()),
            "sampled_min_decay": float(decay.min()),
            "sampled_max_decay": float(decay.max()),
            "coefficients_sha256": sha256(p.astype("<f8").tobytes() + q.astype("<f8").tobytes()),
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
        raise ValueError("no layer payloads found")
    certified = all(c["certified"] for r in rows for c in r["certificates"])
    policy = {
        "dissipation_degree": dissipation_degree,
        "equilibrium_degree": equilibrium_degree,
        "inward_margin": inward_margin,
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
            "source_sha256": sources,
            "policy": policy,
            "layers": [r["layer"] for r in rows],
            "certified_heads": sum(r["heads_certified"] for r in rows),
        }
        arrays["manifest"] = np.array(json.dumps(manifest, sort_keys=True))
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        bundle_hash = file_sha256(bundle)
    report = make_report(
        "dissipative-selective-gate-fit",
        sources,
        input_payload_sha256=payload_hash,
        policy=policy,
        grid_size=grid_size,
        all_certified=certified,
        bundle_written=certified,
        bundle_sha256=bundle_hash,
        rows=rows,
        summary={
            "heads_certified": sum(r["heads_certified"] for r in rows),
            "worst_sampled_write_error": max(r["sampled_max_write_error"] for r in rows),
            "worst_sampled_decay_error": max(r["sampled_max_decay_error"] for r in rows),
            "max_state_gain": max(c["gain"] for r in rows for c in r["certificates"]),
        },
        elapsed_seconds=time.perf_counter() - start,
        measurement_scope={
            "language_quality_measured": False,
            "input_domain_membership_proved": False,
            "approximation_error_certified": False,
            "ckks_rounding_error_certified": False,
            "claim": "Exact-rational state invariants for the actual shared-factor coefficients.",
        },
    )
    return report
