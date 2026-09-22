#!/usr/bin/env python3
"""Locate a frozen polynomial circuit's first non-finite operator/checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fhemamba.artifacts import current_git_commit  # noqa: E402
from fhemamba.m1_payload import _poly_ops_from_export  # noqa: E402
from fhemamba.normalization import ScheduledInvSqrt, certify_schedule  # noqa: E402
from fhemamba.ops import (  # noqa: E402
    ChebPoly,
    PolyInitNewton,
    PolyOps,
    SquaredPolyInitNewton,
    _poly_interval,
    positive_binomial_seed,
)
from fhemamba.reference import model_forward  # noqa: E402
from fhemamba.selective_gates import DissipativeGate  # noqa: E402
from manage_dgx_build import payload_sha256  # noqa: E402

from fhemamba import __version__  # noqa: E402


def finite_stats(x):
    finite = x.isfinite()
    values = x[finite]
    bad = (~finite).nonzero()
    return {
        "min": float(values.min()) if values.numel() else None,
        "max": float(values.max()) if values.numel() else None,
        "non_finite": int(bad.shape[0]),
        "first_non_finite_index": bad[0].tolist() if bad.numel() else None,
    }


class DiagnosticPolyOps(PolyOps):
    def __init__(self, base, joint_gates=None, exact_sites=(), local_errors=False):
        super().__init__(base.polys, base.enabled, base.layer_polys)
        self.events = []
        self.joint_gates = joint_gates or base.joint_gates
        self.exact_sites = frozenset(exact_sites)
        self.local_errors = local_errors

    def mamba2_gates(self, z, a_cont, layer, time_step_limit):
        if layer not in self.joint_gates:
            return super().mamba2_gates(z, a_cont, layer, time_step_limit)
        if tuple(time_step_limit) != (0.0, float("inf")):
            raise ValueError("joint-gate probe requires the checkpoint's unrestricted dt")
        gate, rates = self.joint_gates[layer]
        # CPU/CUDA exp implementations can differ by one float32 ulp.
        if not np.allclose(-a_cont.detach().cpu().numpy().astype(float), rates, rtol=2e-7, atol=0):
            raise ValueError("joint-gate rates differ from the current checkpoint")
        write, decay = gate(z)
        lo = torch.tensor(gate.lo, device=z.device, dtype=z.dtype)
        hi = torch.tensor(gate.hi, device=z.device, dtype=z.dtype)
        self.events.append(
            {
                "layer": layer,
                "site": "joint_selective_gates",
                "kind": "operator",
                "input": finite_stats(z),
                "input_domains": np.stack((gate.lo, gate.hi), axis=1).tolist(),
                "raw_input_outside_interval": int(((z < lo) | (z > hi)).sum()),
                "write": finite_stats(write),
                "decay": finite_stats(decay),
                "decay_outside_unit_interval": int(((decay < 0) | (decay > 1)).sum()),
            }
        )
        return self.checkpoint(write, (layer, "dt_out")), self.checkpoint(
            decay, (layer, "decay_output")
        )

    def _apply(self, x, site, exact_fn):
        poly = self.layer_polys.get(site) or self.polys[site[1]]
        lo, hi = _poly_interval(poly)
        exact_site = site in self.exact_sites
        value = exact_fn(x) if exact_site else super()._apply(x, site, exact_fn)
        event = {
            "layer": site[0],
            "site": site[1],
            "kind": "operator",
            "interval": [lo, hi],
            "input": finite_stats(x),
            "output": finite_stats(value),
            "raw_input_outside_interval": int(((x < lo) | (x > hi)).sum()),
            "exact_reference_ablation": exact_site,
        }
        if self.local_errors and bool(value.isfinite().all()):
            error = (value - exact_fn(x)).abs()
            event["local_error"] = {
                "max_abs": float(error.max()),
                "rmse": float(error.square().mean().sqrt()),
            }
            if "invsqrt" in site[1]:
                factor = value * x.sqrt()
                event["normalization_factor"] = finite_stats(factor)
                event["normalization_factor"]["min_flat_index"] = int(factor.argmin())
        self.events.append(event)
        if event["output"]["non_finite"]:
            raise FloatingPointError(f"first non-finite operator: {site}")
        return value

    def checkpoint(self, x, site):
        event = {"layer": site[0], "site": site[1], "kind": "checkpoint", "output": finite_stats(x)}
        self.events.append(event)
        if event["output"]["non_finite"]:
            raise FloatingPointError(f"first non-finite checkpoint: {site}")
        if site[1] == "layer_output":
            print(f"candidate layer {site[0]} finite", flush=True)
        return x


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--tokens-pt", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--quality-reference",
        action="store_true",
        help="also compute exact and candidate PPL on this token window",
    )
    parser.add_argument(
        "--public-gate-degree",
        type=int,
        help="unpromoted SiLU override on public-weight domains with 10%% margin",
    )
    parser.add_argument(
        "--public-conv-degree",
        type=int,
        help="unpromoted SiLU override on public convolution envelopes",
    )
    parser.add_argument(
        "--positive-newton-iterations",
        type=int,
        help="unpromoted degree-63 binomial seeds on [0, 4*old_hi]",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dissipative-gate-bundle", type=Path, help="frozen, certified joint-gate NPZ oracle"
    )
    parser.add_argument(
        "--norm-schedule-bundle",
        type=Path,
        help="frozen, certified polynomial normalization oracle",
    )
    parser.add_argument(
        "--exact-sites",
        nargs="*",
        default=[],
        help="diagnostic ablation only: sites like 0:gated_rms_invsqrt or rms_invsqrt",
    )
    parser.add_argument(
        "--local-errors",
        action="store_true",
        help="record exact-op errors without changing the candidate trajectory",
    )
    args = parser.parse_args()
    if min(args.tokens, args.threads) < 1 or args.offset < 0:
        parser.error("positive token and thread counts are required")
    if args.quality_reference and args.tokens < 2:
        parser.error("quality-reference requires at least two tokens")
    if args.public_gate_degree is not None and args.public_gate_degree < 1:
        parser.error("public-gate-degree must be positive")
    if args.public_conv_degree is not None and args.public_conv_degree < 1:
        parser.error("public-conv-degree must be positive")
    if args.positive_newton_iterations is not None and args.positive_newton_iterations < 1:
        parser.error("positive-newton-iterations must be positive")
    if args.norm_schedule_bundle is not None and args.positive_newton_iterations is not None:
        parser.error("choose one normalization override policy")
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval()
    base = _poly_ops_from_export(args.payload, len(model.backbone.layers))
    overrides = []
    if args.public_gate_degree is not None:
        for layer, block in enumerate(model.backbone.layers):
            inner = block.mixer.intermediate_size
            weights = block.mixer.in_proj.weight[:inner].double().numpy()
            gamma = block.norm.weight.double().numpy()
            radius = float(
                1.1
                * np.sqrt(weights.shape[1])
                * np.linalg.norm(weights * gamma[None, :], axis=1).max()
            )
            coefficients = np.polynomial.chebyshev.chebinterpolate(
                lambda t, radius=radius: radius * t * np.exp(-np.logaddexp(0, -radius * t)),
                args.public_gate_degree,
            )
            base.layer_polys[(layer, "gate_silu")] = ChebPoly(
                tuple(float(c) for c in coefficients), -radius, radius
            )
            overrides.append(
                {
                    "layer": layer,
                    "site": "gate_silu",
                    "lo": -radius,
                    "hi": radius,
                    "degree": args.public_gate_degree,
                    "coefficients_sha256": hashlib.sha256(
                        coefficients.astype("<f8").tobytes()
                    ).hexdigest(),
                }
            )
    if args.public_conv_degree is not None:
        for layer, block in enumerate(model.backbone.layers):
            inner, channels = block.mixer.intermediate_size, block.mixer.conv_dim
            weights = block.mixer.in_proj.weight[inner : inner + channels].double().numpy()
            gamma = block.norm.weight.double().numpy()
            projection = np.sqrt(weights.shape[1]) * np.linalg.norm(
                weights * gamma[None, :], axis=1
            )
            convolution = block.mixer.conv1d.weight.double().numpy().reshape(channels, -1)
            bias = (
                block.mixer.conv1d.bias.double().numpy()
                if block.mixer.conv1d.bias is not None
                else np.zeros(channels)
            )
            radius = float(1.1 * (projection * np.abs(convolution).sum(1) + np.abs(bias)).max())
            coefficients = np.polynomial.chebyshev.chebinterpolate(
                lambda t, radius=radius: radius * t * np.exp(-np.logaddexp(0, -radius * t)),
                args.public_conv_degree,
            )
            base.layer_polys[(layer, "conv_silu")] = ChebPoly(
                tuple(float(c) for c in coefficients), -radius, radius
            )
            overrides.append(
                {
                    "layer": layer,
                    "site": "conv_silu",
                    "lo": -radius,
                    "hi": radius,
                    "degree": args.public_conv_degree,
                    "coefficients_sha256": hashlib.sha256(
                        coefficients.astype("<f8").tobytes()
                    ).hexdigest(),
                }
            )
    if args.positive_newton_iterations is not None:
        for site, old in list(base.layer_polys.items()):
            if site[1] not in ("rms_invsqrt", "gated_rms_invsqrt"):
                continue
            squared = isinstance(old, SquaredPolyInitNewton)
            seed = positive_binomial_seed(
                _poly_interval(old)[1] * 4.0, 63, power=0.25 if squared else 0.5
            )
            base.layer_polys[site] = (
                SquaredPolyInitNewton(seed, args.positive_newton_iterations, 0.85)
                if squared
                else PolyInitNewton(seed, args.positive_newton_iterations)
            )
            overrides.append(
                {
                    "layer": site[0],
                    "site": site[1],
                    "kind": "positive-binomial-newton",
                    "power": 0.25 if squared else 0.5,
                    "lo": seed.lo,
                    "hi": seed.hi,
                    "degree": seed.degree,
                    "iterations": args.positive_newton_iterations,
                    "coefficients_sha256": hashlib.sha256(
                        np.asarray(seed.coeffs, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )
    norm_bundle_hash = None
    if args.norm_schedule_bundle is not None:
        norm_bundle_hash = hashlib.sha256(args.norm_schedule_bundle.read_bytes()).hexdigest()
        norm_bundle = json.loads(args.norm_schedule_bundle.read_text())
        if norm_bundle["format"] != "fhemamba-normalization-schedules-v1":
            raise ValueError("unsupported normalization bundle format")
        if norm_bundle["input_payload_sha256"] != payload_sha256(args.payload):
            raise ValueError("normalization bundle belongs to a different payload")
        expected_sites = {site for site in base.layer_polys if "invsqrt" in site[1]}
        loaded_sites = set()
        for operator in norm_bundle["operators"]:
            site = (operator["layer"], operator["site"])
            if site not in expected_sites or site in loaded_sites:
                raise ValueError("unexpected or duplicate normalization site")
            schedule = ScheduledInvSqrt.from_recipe(operator["recipe"])
            if not certify_schedule(schedule)["certified"]:
                raise ValueError(f"normalization recipe failed recertification: {site}")
            base.layer_polys[site] = schedule
            loaded_sites.add(site)
            overrides.append(
                {
                    "layer": site[0],
                    "site": site[1],
                    **schedule.recipe(),
                    "bundle_sha256": norm_bundle_hash,
                    "abstract_cost": schedule.abstract_cost(),
                }
            )
        if loaded_sites != expected_sites:
            raise ValueError("normalization bundle must cover every normalization site")
    joint_gates = {}
    bundle_hash = None
    if args.dissipative_gate_bundle is not None:
        bundle_hash = hashlib.sha256(args.dissipative_gate_bundle.read_bytes()).hexdigest()
        with np.load(args.dissipative_gate_bundle, allow_pickle=False) as bundle:
            manifest = json.loads(str(bundle["manifest"]))
            if manifest["format"] != "dissipative-mamba2-gates-v1":
                raise ValueError("unsupported joint-gate bundle format")
            if manifest["input_payload_sha256"] != payload_sha256(args.payload):
                raise ValueError("joint-gate bundle belongs to a different payload")
            if manifest["layers"] != list(range(len(model.backbone.layers))):
                raise ValueError("joint-gate bundle must cover every model layer")
            for layer in manifest["layers"]:
                gate = DissipativeGate(
                    *(bundle[f"layer_{layer}_{name}"] for name in ("lo", "hi", "p", "q"))
                )
                joint_gates[layer] = gate, bundle[f"layer_{layer}_rates"]
                overrides.append(
                    {
                        "layer": layer,
                        "site": "joint_selective_gates",
                        "kind": "shared-dissipation-factor",
                        "bundle_sha256": bundle_hash,
                        "all_heads_retained": True,
                        "coefficients_sha256": hashlib.sha256(
                            gate.dissipation_root.astype("<f8").tobytes()
                            + gate.equilibrium_root.astype("<f8").tobytes()
                        ).hexdigest(),
                    }
                )
    exact_sites = set()
    for requested in args.exact_sites:
        matched = {
            site for site in base.layer_polys if requested in (site[1], f"{site[0]}:{site[1]}")
        }
        if not matched:
            parser.error(f"no polynomial site matches {requested}")
        if any(site[1] in ("dt_softplus", "decay_exp") for site in matched) and joint_gates:
            parser.error("independent gate ablations cannot override a joint-gate bundle")
        exact_sites.update(matched)
    overrides.extend(
        {"layer": layer, "site": name, "kind": "exact-reference-ablation"}
        for layer, name in sorted(exact_sites)
    )
    ops = DiagnosticPolyOps(base, joint_gates, exact_sites, args.local_errors)
    model = model.to(args.device)
    ids = torch.load(args.tokens_pt, weights_only=True, map_location="cpu")
    ids = ids[:, args.offset : args.offset + args.tokens].to(args.device)
    if ids.shape[1] != args.tokens:
        parser.error("requested token window exceeds the input file")
    quality = None
    if args.quality_reference:
        exact_logits = model_forward(model, ids, scan="chunked")["logits"]
        exact_nll = torch.nn.functional.cross_entropy(exact_logits[0, :-1], ids[0, 1:])
        exact_logits = exact_logits.cpu()
        quality = {
            "exact_ppl": float(exact_nll.exp()),
            "candidate_ppl": None,
            "predicted_tokens": args.tokens - 1,
            "candidate_finite": False,
        }
        print(f"exact PPL {quality['exact_ppl']:.8f}", flush=True)
    failure = None
    try:
        result = model_forward(
            model, ids, ops, scan="chunked", output_logits=args.quality_reference
        )
        if args.quality_reference:
            logits = result["logits"]
            if not bool(logits.isfinite().all()):
                raise FloatingPointError("non-finite vocabulary projection")
            nll = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
            logits = logits.cpu()
            quality.update(
                candidate_ppl=float(nll.exp()),
                candidate_finite=True,
                relative_ppl_increase=float((nll - exact_nll).expm1()),
                max_logit_error=float((logits - exact_logits).abs().max()),
                top1_agreement=float(
                    (logits[0, :-1].argmax(-1) == exact_logits[0, :-1].argmax(-1)).float().mean()
                ),
            )
    except FloatingPointError as exc:
        failure = str(exc)
    report = {
        "stage": "payload-domain-diagnostic",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "src/fhemamba" / name).read_bytes()
            ).hexdigest()
            for name in ("ops.py", "reference.py", "selective_gates.py", "normalization.py")
        },
        "dissipative_gate_bundle_sha256": bundle_hash,
        "normalization_schedule_bundle_sha256": norm_bundle_hash,
        "device": args.device,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "tf32_enabled": False if args.device == "cuda" else None,
        "input_payload_sha256": payload_sha256(args.payload),
        "token_ids_sha256": hashlib.sha256(ids.cpu().numpy().astype("<i8").tobytes()).hexdigest(),
        "tokens": ids.shape[1],
        "token_offset": args.offset,
        "scan": "chunked",
        "first_failure": failure,
        "events": ops.events,
        "operator_overrides": overrides,
        "quality": quality,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            "identical_payload_circuit": not overrides,
            "language_quality_measured": args.quality_reference,
            "full_dataset_quality_claimed": False,
            "fully_polynomial_model": not exact_sites,
            "claim": "First non-finite stage, with all operator overrides recorded.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(failure or "all observed stages finite", flush=True)


if __name__ == "__main__":
    main()
