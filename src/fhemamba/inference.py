"""Public request lifecycle for registered local model implementations.

Prepared requests bind one prompt and fixed generation length. Model arithmetic,
payload formats and FHE profiles belong to the selected integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fhemamba._version import __version__
from fhemamba.benchmarks.io import file_sha256, read_object, write_json
from fhemamba.checkpoints import read_config, tokenizer_directory
from fhemamba.inputs import generation_length, token_ids
from fhemamba.models.contracts import ModelCapabilities, preparation_profile
from fhemamba.models.registry import registry


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
        manifest = self.manifest()
        adapter = registry.for_manifest(manifest)
        return preparation_profile(adapter, manifest.get("profile")).name

    def manifest(self) -> dict[str, Any]:
        if file_sha256(self.path / "manifest.json") != self.manifest_sha256:
            raise ValueError("prepared manifest changed; prepare a new request")
        manifest = read_object(self.path / "manifest.json")
        adapter = registry.for_manifest(manifest)
        preparation_profile(adapter, manifest.get("profile"))
        return adapter.fhe.validate_manifest(self.path, manifest)

    def generate(
        self,
        *,
        binary: str | Path,
        output: str | Path,
        timeout: float = 2400,
    ) -> GenerationResult:
        """Execute the selected native adapter with its existing acceptance gates."""
        manifest = self.manifest()
        binary, output = Path(binary).resolve(), Path(output).resolve()
        if output.is_relative_to(self.path):
            raise ValueError("run output must be outside the prepared request")
        adapter = registry.for_manifest(manifest)
        result = adapter.fhe.run(self, manifest, binary, output, timeout)
        write_json(output / "generation.json", result.to_dict())
        return result


def load_prepared(path: str | Path) -> PreparedRequest:
    """Load a registered request format without changing it or loading a model."""
    path = Path(path).resolve()
    identity = file_sha256(path / "manifest.json")
    metadata = path / "request.json"
    if metadata.exists():
        request = read_object(metadata)
        if (
            request.get("schema") != "fhemamba-prepared-request-v1"
            or request.get("manifest_sha256") != identity
        ):
            raise ValueError("prepared request identity or profile differs")
    prepared = PreparedRequest(path, identity)
    profile = prepared.profile
    if metadata.exists() and request.get("profile") != profile:
        raise ValueError("prepared request profile differs from the model manifest")
    return prepared


def inspect_model(checkpoint: str | Path) -> ModelCapabilities:
    """Inspect configuration only: no weights, optional dependencies or GPU work."""
    config = read_config(Path(checkpoint).resolve())
    return registry.detect(config).capabilities(config)


def _preparation_options(adapter, profile, options, base_chain=None):
    selected = preparation_profile(adapter, profile)
    # Compatibility for the original keyword. New integrations own typed options;
    # they need no new parameters in the public API or parser.
    if base_chain is not None:
        if options is not None:
            raise ValueError("choose options or base_chain, not both")
        options = {"base_chain": base_chain}
    return selected, adapter.fhe.parse_options(options)


class GenerationModel:
    """One local CPU model, with token requests and explicit FHE preparation."""

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint).resolve()
        config = read_config(self.checkpoint)
        self._adapter = registry.detect(config)
        self.architecture = self._adapter.architecture
        self.capabilities = self._adapter.capabilities(config)
        self.capabilities.backends["exact"].require()
        self._identity = self._adapter.checkpoint_identity(self.checkpoint)
        self._model = self._adapter.load(self.checkpoint)
        self._vocab_size = self._adapter.vocab_size(self._model)

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
        if backend not in ("exact", "polynomial", "ckks"):
            raise ValueError("backend must be exact, polynomial or ckks")
        self.capabilities.backends[backend].require()
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
        return self._adapter.generate_cpu(self._model, ids, length, backend, prepared, manifest)

    def prepare(
        self,
        input_ids,
        *,
        max_new_tokens: int = 16,
        output: str | Path,
        tokenizer: str | Path | None = None,
        profile: str | None = None,
        options: Any = None,
        base_chain: str | Path | None = None,
    ) -> PreparedRequest:
        """Prepare with model-owned options and a registered, fixed FHE profile.

        options accepts the integration's dataclass or a strict mapping. base_chain
        remains a compatibility keyword for the original Mamba-2 API.
        """
        self.capabilities.backends["polynomial"].require()
        ids = token_ids(input_ids, vocab_size=self._vocab_size)
        generation_length(max_new_tokens)
        selected, options = _preparation_options(self._adapter, profile, options, base_chain)
        if self._adapter.checkpoint_identity(self.checkpoint) != self._identity:
            raise ValueError("checkpoint files changed after loading the model")
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        tokenizer = (
            Path(tokenizer) if tokenizer is not None else tokenizer_directory(self.checkpoint)
        )
        self._adapter.fhe.prepare(
            self._model,
            self.checkpoint,
            self._identity,
            ids,
            max_new_tokens,
            output,
            tokenizer,
            options,
            selected,
        )
        write_json(
            output / "request.json",
            {
                "schema": "fhemamba-prepared-request-v1",
                "profile": selected.name,
                "manifest_sha256": file_sha256(output / "manifest.json"),
                "selection": "greedy",
                "stopping": "length",
            },
        )
        return load_prepared(output)


def load_model(checkpoint: str | Path) -> GenerationModel:
    """Detect and load a registered local checkpoint on CPU."""
    return GenerationModel(checkpoint)
