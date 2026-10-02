"""Public generation API for local Mamba-2 and Mamba-3 SISO checkpoints.

Prepared requests bind one prompt and one fixed generation length. They are
research payloads containing plaintext references, not deployable server models.
Heavy model dependencies are imported only for CPU generation or preparation.
"""

from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fhemamba._version import __version__
from fhemamba.benchmarks.io import file_sha256, read_object, write_json
from fhemamba.checkpoints import architecture, checkpoint_identity, tokenizer_directory
from fhemamba.inputs import generation_length, token_ids

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


@dataclass(frozen=True)
class GenerationResult:
    """Actual generated IDs and their validation status, including failed runs."""

    input_ids: list[int]
    generated_ids: list[int]
    backend: str
    passed: bool
    encrypted: bool
    stop_reason: str
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "fhemamba-generation-v1", "version": __version__, **asdict(self)}


@dataclass(frozen=True)
class PreparedRequest:
    """A hash-checked, request-specific payload; no keys are stored here."""

    path: Path
    manifest_sha256: str

    @property
    def profile(self) -> str:
        return self.manifest().get("profile", "classical-128")

    def manifest(self) -> dict[str, Any]:
        from fhemamba.benchmarks.packed import read_payload

        if file_sha256(self.path / "manifest.json") != self.manifest_sha256:
            raise ValueError("prepared manifest changed; prepare a new request")
        manifest = read_object(self.path / "manifest.json")
        if manifest.get("schema") == "fhemamba-mamba2-generation-v1":
            from fhemamba.mamba2_inference import validate_manifest

            return validate_manifest(self.path, manifest)
        manifest = read_payload(self.path)
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

    def generate(
        self,
        *,
        binary: str | Path,
        output: str | Path,
        timeout: float = 2400,
    ) -> GenerationResult:
        """Execute the model-specific CKKS profile with its existing acceptance gates."""
        from fhemamba.benchmarks.generation import validate_generation
        from fhemamba.benchmarks.packed import run

        manifest = self.manifest()
        binary, output = Path(binary).resolve(), Path(output).resolve()
        if output.is_relative_to(self.path):
            raise ValueError("run output must be outside the prepared request")
        if manifest["architecture"] == "mamba2":
            from fhemamba.mamba2_inference import run as run_mamba2

            return run_mamba2(self, manifest, binary, output, timeout)
        record = run(binary, self.path, output, timeout=timeout, **_CKKS_OPTIONS)
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
            validation = validate_generation(self.path, output, security="128-classic")
            if file_sha256(self.path / "manifest.json") != self.manifest_sha256:
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
                "manifest_sha256": self.manifest_sha256,
                "run_path": str(output),
                "timed_out": record["timed_out"],
                "validation": validation,
                "error": error,
                "client_server_process_separated": False,
            },
        )
        write_json(output / "generation.json", result.to_dict())
        return result


def load_prepared(path: str | Path) -> PreparedRequest:
    """Load a new request or an existing full-model export without changing it."""
    path = Path(path).resolve()
    identity = file_sha256(path / "manifest.json")
    metadata = path / "request.json"
    if metadata.exists():
        request = read_object(metadata)
        if (
            request.get("schema") != "fhemamba-prepared-request-v1"
            or request.get("profile") not in ("classical-128", "mamba2-experimental")
            or request.get("manifest_sha256") != identity
        ):
            raise ValueError("prepared request identity or profile differs")
    prepared = PreparedRequest(path, identity)
    profile = prepared.profile
    if metadata.exists() and request["profile"] != profile:
        raise ValueError("prepared request profile differs from the model manifest")
    return prepared


class GenerationModel:
    """One local CPU model, with token-based generation and explicit FHE preparation."""

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint).resolve()
        self.architecture = architecture(self.checkpoint)
        self._identity = checkpoint_identity(self.checkpoint, self.architecture)
        if self.architecture == "mamba2":
            from fhemamba.mamba2_inference import load_checkpoint

            self._model = load_checkpoint(self.checkpoint)
            self._vocab_size = self._model.get_input_embeddings().num_embeddings
        else:
            from fhemamba.mamba3_lm import Mamba3LM

            self._model = Mamba3LM.from_pretrained(self.checkpoint)
            self._vocab_size = self._model.backbone.embedding.num_embeddings

    def generate(
        self,
        input_ids,
        *,
        max_new_tokens: int = 16,
        backend: str = "exact",
        prepared: PreparedRequest | None = None,
        binary: str | Path | None = None,
        output: str | Path | None = None,
        timeout: float = 2400,
    ) -> GenerationResult:
        """Greedy, fixed-length generation. EOS does not stop this research runner.

        Polynomial and CKKS modes require a matching prepared request. CPU
        polynomial mode evaluates the frozen coefficients, not cached token IDs.
        """
        import torch

        if backend not in ("exact", "polynomial", "ckks"):
            raise ValueError("backend must be exact, polynomial or ckks")
        ids = token_ids(input_ids, vocab_size=self._vocab_size)
        length = generation_length(max_new_tokens)
        manifest = None
        if prepared is not None:
            manifest = prepared.manifest()
            if manifest["architecture"] != self.architecture:
                raise ValueError("model architecture differs from the prepared request")
            if ids != manifest["prompt_ids"] or length != manifest["generated_tokens"]:
                raise ValueError("input IDs or generation length differ from the prepared request")
            if self._identity != manifest["checkpoint"]["files_sha256"]:
                raise ValueError("model checkpoint differs from the prepared request")
        if backend != "exact" and prepared is None:
            raise ValueError(f"{backend} requires model.prepare(...) or load_prepared(...)")
        if backend == "ckks":
            if binary is None or output is None:
                raise ValueError(
                    "CKKS requires an explicit native binary and fresh output directory"
                )
            return prepared.generate(binary=binary, output=output, timeout=timeout)
        if binary is not None or output is not None:
            raise ValueError("binary and output are CKKS execution options")
        if self.architecture == "mamba2":
            from fhemamba.mamba2_inference import generate

            return generate(self._model, ids, length, backend, prepared, manifest)
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
        generated, trace = self._model.generate(ids, length, ops, **settings)
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

    def prepare(
        self,
        input_ids,
        *,
        max_new_tokens: int = 16,
        output: str | Path,
        tokenizer: str | Path | None = None,
        profile: str | None = None,
        base_chain: str | Path | None = None,
    ) -> PreparedRequest:
        """Export one request using the architecture's existing preparation path.

        Mamba-2 requires a frozen base_chain and explicit mamba2-experimental
        profile. Mamba-3 uses the pinned checkpoint and independent calibration.
        Run preparation from a checkout; input_ids are never retokenized.
        """
        ids = token_ids(input_ids, vocab_size=self._vocab_size)
        generation_length(max_new_tokens)
        if checkpoint_identity(self.checkpoint, self.architecture) != self._identity:
            raise ValueError("checkpoint files changed after loading the model")
        output = Path(output).resolve()
        tokenizer = (
            Path(tokenizer) if tokenizer is not None else tokenizer_directory(self.checkpoint)
        )
        if self.architecture == "mamba2":
            from fhemamba.mamba2_inference import PROFILE, prepare

            if profile != PROFILE:
                raise ValueError(
                    "Mamba-2 requires explicit profile='mamba2-experimental' (security=not-set)"
                )
            prepare(
                self._model,
                self._identity,
                ids,
                max_new_tokens,
                output,
                base_chain,
                tokenizer,
            )
        else:
            profile = profile or "classical-128"
            if profile != "classical-128" or base_chain is not None:
                raise ValueError("Mamba-3 uses classical-128 preparation without base_chain")
            generation_length(max_new_tokens, input_length=len(ids))
            self._prepare_mamba3(ids, max_new_tokens, output, tokenizer)
        write_json(
            output / "request.json",
            {
                "schema": "fhemamba-prepared-request-v1",
                "profile": profile,
                "manifest_sha256": file_sha256(output / "manifest.json"),
                "selection": "greedy",
                "stopping": "length",
            },
        )
        return load_prepared(output)

    def _prepare_mamba3(self, ids, length, output, tokenizer):
        from fhemamba.workloads.mamba3_export import export

        if tokenizer is None:
            raise ValueError("Mamba-3 preparation requires a local tokenizer for calibration")
        # The exporter owns model arithmetic, calibration gates and payload
        # writing. Preserve its legacy stdout while keeping this API's stdout clean.
        with contextlib.redirect_stdout(sys.stderr):
            export(
                self.checkpoint,
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


def load_model(checkpoint: str | Path) -> GenerationModel:
    """Detect and load a local Mamba-2 or Mamba-3 SISO checkpoint on CPU."""
    return GenerationModel(checkpoint)
