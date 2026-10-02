"""Mamba-1 CPU reference and experimental classical-128 packed generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from fhemamba.checkpoints import hf_checkpoint_identity
from fhemamba.inputs import generation_length, token_ids

from .contracts import BackendCapability, PreparationProfile, capabilities, parse_options
from .packed import run_ckks

PROFILE = "mamba1-experimental"


@dataclass(frozen=True)
class Mamba1Preparation:
    """Independent calibration IDs, or the built-in prompts via a local tokenizer."""

    calibration_input_ids: tuple[tuple[int, ...], ...] | None = None
    calibration_new_tokens: int = 16

    def __post_init__(self):
        generation_length(self.calibration_new_tokens, input_length=1)
        if self.calibration_input_ids is not None:
            values = self.calibration_input_ids
            if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 16:
                raise ValueError("calibration_input_ids requires 1..16 independent sequences")
            values = tuple(tuple(token_ids(ids)) for ids in values)
            for ids in values:
                generation_length(self.calibration_new_tokens, input_length=len(ids))
            object.__setattr__(self, "calibration_input_ids", values)


def validate_manifest(path, manifest):
    from fhemamba.benchmarks.packed import read_payload

    manifest = read_payload(path)
    if (
        manifest.get("schema") != "fhemamba-mamba1-lm-v1"
        or manifest.get("architecture") != "mamba1"
        or manifest.get("profile") != PROFILE
        or manifest.get("security") != "128-classic"
        or manifest.get("client_protocol") != "packed-greedy-v1"
        or manifest.get("complete_backbone") is not True
        or manifest.get("state_representation") != "channel-tiled"
        or manifest["layers"] != manifest["checkpoint_layers"]
    ):
        raise ValueError("Mamba-1 requires a complete experimental packed generation request")
    vocab = manifest["vocab_size"]
    ids = token_ids(manifest["prompt_ids"], vocab_size=vocab)
    length = generation_length(manifest["generated_tokens"], input_length=len(ids))
    expected = token_ids(manifest["exact_token_ids"], vocab_size=vocab)
    if (
        len(expected) != length
        or expected != manifest["polynomial_token_ids"]
        or manifest["tokens"] != len(ids) + length - 1
        or len(manifest["output_names"]) != manifest["tokens"]
        or manifest["program"]["outputs"] != manifest["tokens"]
        or manifest["program"]["slots"] != 32768
        or manifest["calibration_excludes_evaluation_prompt"] is not True
        or ids in manifest["calibration_input_ids"]
        or manifest["calibration_generated_tokens"] < length
    ):
        raise ValueError("prepared Mamba-1 references do not match the request")
    error = manifest["polynomial_hidden_max_abs_error"]
    if type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 0.001:
        raise ValueError("prepared polynomial reference exceeds the 0.001 error gate")
    squarings = manifest["decay_squarings"]
    if len(squarings) != manifest["layers"] or any(
        type(count) is not int or not 0 <= count <= 32 for count in squarings
    ):
        raise ValueError("invalid Mamba-1 frozen decay squaring counts")
    return manifest


class Mamba1FHE:
    profiles = (
        PreparationProfile(
            PROFILE,
            "128-classic",
            (
                "SiLU, tied embeddings, independent calibration IDs or a local tokenizer",
                "Packed CKKS; 1..64 new tokens, <=128 evaluations; exact hidden error <=0.001",
                "Experimental Mamba-1 integration; no GPU-qualified checkpoint/request yet",
            ),
        ),
    )
    default_profile = None
    validate_manifest = staticmethod(validate_manifest)

    def parse_options(self, value):
        return parse_options(Mamba1Preparation, value)

    def prepare(
        self, model, checkpoint, identity, ids, length, output, tokenizer, options, profile
    ):
        from fhemamba.workloads.mamba1_export import export

        export(model, identity, ids, length, output, tokenizer, options, profile)

    def run(self, prepared, manifest, binary, output, timeout):
        return run_ckks(prepared, manifest, binary, output, timeout, profile=PROFILE)


def generate_polynomial(model, ids, length, prepared, manifest):
    import numpy as np
    import torch

    from fhemamba.inference import GenerationResult
    from fhemamba.mamba1 import Mamba1LM
    from fhemamba.ops import ChebPoly
    from fhemamba.packed_program import PolynomialTensorOps

    coefficients = {}
    for site, recipe in manifest["polynomials"].items():
        layer, name = site.split(":", 1)
        coefficients[int(layer), name] = ChebPoly(
            tuple(recipe["coefficients"]), recipe["lo"], recipe["hi"]
        )
    lowered = Mamba1LM(model, decay_squarings=manifest["decay_squarings"])
    generated, trace = lowered.generate(ids, length, PolynomialTensorOps(coefficients))
    actual = torch.cat(trace).numpy()
    errors = {}
    with np.load(prepared.path / "fixture.npz", allow_pickle=False) as fixture:
        for reference in ("exact", "polynomial"):
            expected = fixture[f"{reference}_hidden"]
            if actual.shape != expected.shape:
                raise ValueError("prepared hidden references have the wrong shape")
            error = float(np.max(np.abs(actual - expected)))
            errors[reference] = error if math.isfinite(error) else None
    checks = {
        "complete_generation": len(generated) == length,
        "matches_exact_tokens": generated == manifest["exact_token_ids"],
        "matches_polynomial_tokens": generated == manifest["polynomial_token_ids"],
        "within_operator_domains": True,  # PolynomialTensorOps raises on every domain violation.
        **{
            f"{name}_hidden_error_within_0_001": error is not None and error <= 0.001
            for name, error in errors.items()
        },
    }
    passed = all(checks.values())
    return GenerationResult(
        ids,
        generated,
        "polynomial",
        passed,
        False,
        "length" if passed else "validation_failed",
        {
            "architecture": "mamba1",
            "checks": checks,
            "selection": "greedy",
            "stopping": "length",
            "manifest_sha256": prepared.manifest_sha256,
            **{f"max_abs_error_vs_{name}": value for name, value in errors.items()},
        },
    )


class Mamba1Adapter:
    architecture = "mamba1"
    fhe = Mamba1FHE()
    checkpoint_identity = staticmethod(hf_checkpoint_identity)

    def capabilities(self, config):
        activation = config.get("hidden_act", "silu")
        unsupported = None
        if activation not in ("silu", "swish"):
            unsupported = (
                f"Mamba-1 reference requires SiLU; hidden_act={activation!r} is unsupported"
            )
        support = capabilities(self.architecture, "transformers", self.fhe, unsupported=unsupported)
        if unsupported:
            return support
        reason = None
        if not config.get("tie_word_embeddings", True):
            reason = "Mamba-1 packed feedback requires tied embeddings"
        hidden = config.get("hidden_size", 768)
        inner = config.get("intermediate_size", int(config.get("expand", 2) * hidden))
        state = config.get("state_size", 16)
        rank = config.get("time_step_rank", "auto")
        rank = math.ceil(hidden / 16) if rank == "auto" else rank
        if max(hidden, 2 * inner, rank + 2 * state) > 32768:
            reason = "Mamba-1 projections and one state row must fit in CKKS slots"
        if reason:
            return replace(
                support,
                backends={
                    **support.backends,
                    "polynomial": BackendCapability("unsupported", reason),
                    "ckks": BackendCapability("unsupported", reason),
                },
                profiles=(),
            )
        return support

    def load(self, checkpoint):
        from fhemamba._env import block_broken_torchvision

        block_broken_torchvision()
        try:
            from transformers import MambaForCausalLM
        except ImportError as exc:
            raise ValueError("Mamba-1 requires pip install 'fhemamba[experiments]'") from exc
        return (
            MambaForCausalLM.from_pretrained(checkpoint, local_files_only=True).float().cpu().eval()
        )

    def vocab_size(self, model):
        return model.get_input_embeddings().num_embeddings

    def generate_cpu(self, model, ids, length, backend, prepared, manifest):
        import torch

        from fhemamba.inference import GenerationResult
        from fhemamba.m1_payload import _trace_from_ids

        if backend == "polynomial":
            return generate_polynomial(model, ids, length, prepared, manifest)
        if backend != "exact":
            raise ValueError("Mamba-1 CPU generation requires exact or polynomial backend")
        trace = _trace_from_ids(
            model,
            torch.tensor([ids]),
            generate_tokens=length,
            record_layer_details=False,
        )
        checks = {
            "finite_hidden_states": True,  # The trace rejects non-finite hidden/logit values.
            "complete_generation": len(trace.generated_ids) == length,
        }
        passed = all(checks.values())
        return GenerationResult(
            ids,
            trace.generated_ids,
            backend,
            passed,
            False,
            "length" if passed else "validation_failed",
            {
                "architecture": self.architecture,
                "checks": checks,
                "selection": "greedy",
                "stopping": "length",
            },
        )


ADAPTER = Mamba1Adapter()
