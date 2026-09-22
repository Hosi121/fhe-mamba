"""Public-weight activation fits and validated shared-factor gate payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .polynomial_certificate import certify_dissipative_gate
from .selective_gates import DissipativeGate


def public_gate_domains(block):
    mixer = block.mixer
    weights = mixer.in_proj.weight.detach().cpu().double().numpy()
    gamma = block.norm.weight.detach().cpu().double().numpy()
    bias = mixer.dt_bias.detach().cpu().double().numpy()
    radius = (
        1.1
        * np.sqrt(weights.shape[1])
        * np.linalg.norm(weights[-mixer.num_heads :] * gamma[None, :], axis=1)
    )
    return bias - radius, bias + radius


def public_activation_specs(block, *, gate_degree=384, conv_degree=768):
    """Match the public-envelope plaintext candidate without fitting to text."""
    if min(gate_degree, conv_degree) < 1:
        raise ValueError("positive public activation degrees required")
    mixer = block.mixer
    weights = mixer.in_proj.weight.detach().cpu().double().numpy()
    gamma = block.norm.weight.detach().cpu().double().numpy()
    projection = np.sqrt(weights.shape[1]) * np.linalg.norm(weights * gamma[None, :], axis=1)
    inner, channels = mixer.intermediate_size, mixer.conv_dim
    convolution = mixer.conv1d.weight.detach().cpu().double().numpy().reshape(channels, -1)
    bias = (
        mixer.conv1d.bias.detach().cpu().double().numpy()
        if mixer.conv1d.bias is not None
        else np.zeros(channels)
    )
    radii = {
        "gate_silu": (
            float(
                1.1
                * np.sqrt(weights.shape[1])
                * np.linalg.norm(weights[:inner] * gamma[None, :], axis=1).max()
            ),
            gate_degree,
        ),
        "conv_silu": (
            float(
                1.1
                * (
                    projection[inner : inner + channels] * np.abs(convolution).sum(1) + np.abs(bias)
                ).max()
            ),
            conv_degree,
        ),
    }
    specs = {}
    for site, (radius, degree) in radii.items():
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("invalid public activation envelope")
        coefficients = np.polynomial.chebyshev.chebinterpolate(
            lambda t, r=radius: r * t * np.exp(-np.logaddexp(0, -r * t)), degree
        )
        specs[site] = {
            "kind": "cheb",
            "lo": -radius,
            "hi": radius,
            "coeffs": coefficients.tolist(),
            "coefficients_sha256": hashlib.sha256(coefficients.astype("<f8").tobytes()).hexdigest(),
        }
    return specs


def gate_from_spec(spec):
    if spec.get("kind") != "shared-dissipation-factor-v1":
        raise ValueError("unsupported joint gate kind")
    heads = len(spec["lo"])
    gate = DissipativeGate(
        spec["lo"],
        spec["hi"],
        np.asarray(spec["p"], dtype=float).reshape(-1, heads),
        np.asarray(spec["q"], dtype=float).reshape(-1, heads),
    )
    rates = np.asarray(spec["rates"], dtype=float)
    if rates.shape != (heads,) or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("joint gate requires one positive finite rate per head")
    return gate, rates


def stabilized_specs(model, bundle_path: str | Path):
    """Recheck coefficients and bind public domains/rates to the checkpoint.

    The original bundle's payload hash remains provenance, not the hash of
    this new payload with regenerated references. Domains and rates are
    independently recomputed from the actual checkpoint before any writes.
    """
    path = Path(bundle_path)
    specs = {}
    with np.load(path, allow_pickle=False) as bundle:
        manifest = json.loads(str(bundle["manifest"]))
        if manifest.get("format") != "dissipative-mamba2-gates-v1" or manifest.get(
            "layers"
        ) != list(range(len(model.backbone.layers))):
            raise ValueError("joint gate bundle must cover every checkpoint layer")
        for layer, block in enumerate(model.backbone.layers):
            if tuple(block.mixer.time_step_limit) != (0.0, float("inf")):
                raise ValueError("joint gates require unrestricted checkpoint time steps")
            lo, hi, p, q, rates = (
                bundle[f"layer_{layer}_{name}"] for name in ("lo", "hi", "p", "q", "rates")
            )
            gate = DissipativeGate(lo, hi, p, q)
            expected_lo, expected_hi = public_gate_domains(block)
            expected_rates = block.mixer.A_log.detach().cpu().float().exp().double().numpy()
            if (
                lo.shape != expected_lo.shape
                or not np.allclose(lo, expected_lo, rtol=1e-12, atol=1e-12)
                or not np.allclose(hi, expected_hi, rtol=1e-12, atol=1e-12)
                or rates.shape != expected_rates.shape
                or not np.allclose(rates, expected_rates, rtol=2e-7, atol=0)
            ):
                raise ValueError(f"joint gate domains/rates differ from checkpoint layer {layer}")
            for head in range(lo.size):
                columns = [
                    np.trim_zeros(c[:, head], trim="b").tolist() or [0.0]
                    for c in (gate.dissipation_root, gate.equilibrium_root)
                ]
                if not certify_dissipative_gate(*columns)["certified"]:
                    raise ValueError(f"uncertified joint gate layer {layer}, head {head}")
            specs[layer] = {
                "kind": "shared-dissipation-factor-v1",
                "lo": lo.tolist(),
                "hi": hi.tolist(),
                "p": p.reshape(-1).tolist(),
                "q": q.reshape(-1).tolist(),
                "rates": rates.tolist(),
                "coefficient_layout": "degree-major-head-minor",
                "all_heads_retained": True,
                "source_payload_sha256": manifest["input_payload_sha256"],
            }
    return specs, hashlib.sha256(path.read_bytes()).hexdigest()
