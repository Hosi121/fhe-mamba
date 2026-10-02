"""Mamba-3 SISO integration with the existing packed classical-128 recipe."""

from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import dataclass

from fhemamba.benchmarks.io import file_sha256
from fhemamba.inputs import generation_length, token_ids

from .contracts import PreparationProfile, capabilities, parse_options

# Settings from the recorded public-state-reuse B300 candidate. This selects
# parameters, not a qualification claim for a newly prepared input or binary.
_CKKS_OPTIONS = {
    "security": "128-classic",
    "security_digits": 4,
    "tolerance": 0.001,
    "planned_refresh": True,
    "bootstrap_passes": 2,
    "batch_refresh": True,
    "inplace_ops": True,
    "gpu_plaintext_ntt": True,
    "profile_evaluation": True,
    "naf_rotations": True,
    "reuse_dead_inputs": True,
    "direct_plaintext_upload": True,
    "compact_weights": True,
    "move_plaintext_coefficients": True,
    "borrow_plaintext_upload": True,
    "bsgs_routing_stages": True,
    "cache_plaintexts": True,
    "frontier_refresh": True,
    "s2c_first": True,
    "gpu_plaintext_rns": True,
    "hoist_rotations": True,
    "share_chebyshev": True,
    "prefetch_plaintexts": True,
    "prefetch_workers": 2,
    "gpu_addend_rns": True,
    "plaintext_cache_capacity": 2048,
    "merge_refresh_correction": True,
    "gpu_plaintext_fft": True,
    "frontier_live_limit": 256,
    "reuse_public_ciphertexts": True,
}


def validate_manifest(path, manifest):
    from fhemamba.benchmarks.packed import read_payload

    manifest = read_payload(path)
    if manifest.get("schema") != "fhemamba-mamba3-lm-v1":
        raise ValueError("a Mamba-3 language-model payload is required")
    if manifest.get("complete_backbone") is not True:
        raise ValueError("generation requires the complete backbone, not a layer probe")
    ids = token_ids(manifest["prompt_ids"])
    length = generation_length(manifest["generated_tokens"], input_length=len(ids))
    expected = token_ids(manifest["exact_token_ids"])
    if (
        len(expected) != length
        or expected != manifest["polynomial_token_ids"]
        or manifest["tokens"] != len(ids) + length - 1
        or len(manifest["output_names"]) != manifest["tokens"]
    ):
        raise ValueError("prepared references do not match the request")
    error = manifest["polynomial_hidden_max_abs_error"]
    if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 0.001:
        raise ValueError("prepared polynomial reference exceeds the 0.001 error gate")
    return manifest


def run_ckks(prepared, manifest, binary, output, timeout):
    from fhemamba.benchmarks.generation import validate_generation
    from fhemamba.benchmarks.packed import run
    from fhemamba.inference import GenerationResult

    record = run(binary, prepared.path, output, timeout=timeout, **_CKKS_OPTIONS)
    native_path = output / "native.json"
    native = {}
    error = None
    try:
        # Native failure records can contain non-finite measurements. Do not
        # copy those into our strict JSON result, or invent reference tokens.
        import json

        if native_path.exists():
            value = json.loads(native_path.read_text())
            if not isinstance(value, dict):
                raise ValueError("native result must be a JSON object")
            native = value
        validation = validate_generation(prepared.path, output, security="128-classic")
        if file_sha256(prepared.path / "manifest.json") != prepared.manifest_sha256:
            raise ValueError("prepared manifest changed during execution")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        validation = None
        error = str(exc)
    actual = native.get("generated_token_ids", [])
    if not isinstance(actual, list) or any(type(v) is not int or v < 0 for v in actual):
        actual = []
        error = "native output contains invalid generated token IDs"
        validation = None
    passed = record["passed"] is True and validation is not None
    result = GenerationResult(
        input_ids=list(manifest["prompt_ids"]),
        generated_ids=actual,
        backend="ckks",
        passed=passed,
        encrypted=native.get("encrypted") is True,
        stop_reason="length" if passed else "validation_failed",
        report={
            "profile": "classical-128",
            "manifest_sha256": prepared.manifest_sha256,
            "run_path": str(output),
            "timed_out": record["timed_out"],
            "validation": validation,
            "error": error,
            "client_server_process_separated": False,
        },
    )
    return result


def generate_cpu(model, ids, length, backend, prepared, manifest):
    import torch

    from fhemamba.inference import GenerationResult

    ops = None
    settings = {"factored": False, "rotary": "angle"}
    if backend == "polynomial":
        from fhemamba.ops import ChebPoly
        from fhemamba.packed_program import PolynomialTensorOps

        polynomials = {}
        for site, recipe in manifest["polynomials"].items():
            layer, name = site.split(":", 1)
            polynomials[int(layer), name] = ChebPoly(
                tuple(recipe["coefficients"]), recipe["lo"], recipe["hi"]
            )
        ops = PolynomialTensorOps(polynomials)
        settings = {
            "factored": manifest["state_representation"] == "factored",
            "rotary": manifest["rotary_representation"],
        }
    generated, trace = model.generate(ids, length, ops, **settings)
    if not all(bool(torch.isfinite(hidden).all()) for hidden in trace):
        raise ValueError("generation produced non-finite hidden states")
    checks = {"finite_hidden_states": True, "complete_generation": len(generated) == length}
    report = {
        "architecture": "mamba3",
        "checks": checks,
        "selection": "greedy",
        "stopping": "length",
    }
    if backend == "polynomial":
        import numpy as np

        with np.load(prepared.path / "fixture.npz", allow_pickle=False) as fixture:
            actual = torch.cat(trace).numpy()
            reference = fixture["exact_hidden"]
            if actual.shape != reference.shape:
                raise ValueError("prepared hidden references have the wrong shape")
            error = float(np.max(np.abs(actual - reference)))
        checks["matches_exact_tokens"] = generated == manifest["exact_token_ids"]
        checks["matches_polynomial_tokens"] = generated == manifest["polynomial_token_ids"]
        checks["hidden_error_within_0_001"] = math.isfinite(error) and error <= 0.001
        report["max_abs_error_vs_exact"] = error if math.isfinite(error) else None
        report["manifest_sha256"] = prepared.manifest_sha256
    passed = all(checks.values())
    return GenerationResult(
        ids,
        generated,
        backend,
        passed,
        False,
        "length" if passed else "validation_failed",
        report,
    )


def prepare_checkpoint(checkpoint, ids, length, output, tokenizer):
    generation_length(length, input_length=len(ids))
    from fhemamba.workloads.mamba3_export import export

    if tokenizer is None:
        raise ValueError("Mamba-3 preparation requires a local tokenizer for calibration")
    # The exporter owns model arithmetic, calibration gates and payload
    # writing. Preserve its legacy stdout while keeping this API's stdout clean.
    with contextlib.redirect_stdout(sys.stderr):
        export(
            checkpoint,
            tokenizer,
            output,
            input_ids=ids,
            new_tokens=length,
            state_representation="tiled",
            rotary="phasor",
            calibration_tokens=65,
            normalization_margin=1,
            positive_lower_fraction=0.25,
        )


@dataclass(frozen=True)
class Mamba3Preparation:
    """The frozen classical-128 recipe currently has no tunable model options."""


class Mamba3FHE:
    profiles = (
        PreparationProfile(
            "classical-128",
            "128-classic",
            (
                "Pinned Mamba-3 SISO 187M checkpoint and local tokenizer",
                "Source checkout for independent calibration; 1..64 new tokens, <=128 evaluations",
                "New inputs need domain/precision validation; recorded qualification: 16 tokens",
            ),
        ),
    )
    default_profile = "classical-128"
    validate_manifest = staticmethod(validate_manifest)
    run = staticmethod(run_ckks)

    def parse_options(self, value):
        return parse_options(Mamba3Preparation, value)

    def prepare(
        self, model, checkpoint, identity, ids, length, output, tokenizer, options, profile
    ):
        prepare_checkpoint(checkpoint, ids, length, output, tokenizer)


class Mamba3Adapter:
    architecture = "mamba3"
    fhe = Mamba3FHE()
    generate_cpu = staticmethod(generate_cpu)

    def capabilities(self, config):
        ssm = config.get("ssm_cfg") or {}
        unsupported = None
        if ssm.get("is_mimo", False):
            unsupported = "Mamba-3 MIMO is not implemented; the current adapter supports SISO"
        elif ssm.get("ngroups", 1) != 1 or config.get("attn_layer_idx"):
            unsupported = "grouped Mamba-3 SSMs and attention layers are unsupported"
        elif not config.get("rms_norm", True) or not config.get("tie_embeddings", True):
            unsupported = "Mamba-3 requires RMSNorm and tied embeddings"
        return capabilities(self.architecture, "state-spaces", self.fhe, unsupported=unsupported)

    def checkpoint_identity(self, checkpoint):
        return {
            name: file_sha256(checkpoint / name) for name in ("config.json", "pytorch_model.bin")
        }

    def load(self, checkpoint):
        from fhemamba.mamba3_lm import Mamba3LM

        return Mamba3LM.from_pretrained(checkpoint)

    def vocab_size(self, model):
        return model.backbone.embedding.num_embeddings


ADAPTER = Mamba3Adapter()
