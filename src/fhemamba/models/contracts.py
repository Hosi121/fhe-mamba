"""Lightweight contracts between the public API and model integrations.

An integration owns its arithmetic and prepared-payload format. The core owns
request identity, lifecycle and result persistence. No model libraries are
imported by these contracts or by registry discovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from fhemamba.inference import GenerationResult, PreparedRequest


@dataclass(frozen=True)
class PreparationProfile:
    name: str
    security: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class BackendCapability:
    status: str
    reason: str

    def require(self):
        if self.status == "unsupported":
            raise ValueError(self.reason)


@dataclass(frozen=True)
class ModelCapabilities:
    architecture: str
    checkpoint_format: str
    backends: dict[str, BackendCapability]
    profiles: tuple[PreparationProfile, ...] = ()

    def to_dict(self):
        return {
            "schema": "fhemamba-model-capabilities-v1",
            "inspection": "configuration_only",
            **asdict(self),
        }


class FHEImplementation(Protocol):
    """Model/backend-specific preparation, format validation and native adapter."""

    profiles: tuple[PreparationProfile, ...]
    default_profile: str | None

    def parse_options(self, value: Any) -> Any: ...

    def prepare(
        self,
        model,
        checkpoint: Path,
        identity: dict[str, str],
        ids: list[int],
        length: int,
        output: Path,
        tokenizer: Path | None,
        options: Any,
        profile: PreparationProfile,
    ) -> None: ...

    def validate_manifest(self, path: Path, manifest: dict) -> dict: ...

    def run(
        self,
        prepared: PreparedRequest,
        manifest: dict,
        binary: Path,
        output: Path,
        timeout: float,
    ) -> GenerationResult: ...


class ModelAdapter(Protocol):
    """The model owns weights, recurrent state and CPU reference execution."""

    architecture: str
    fhe: FHEImplementation | None

    def capabilities(self, config: dict) -> ModelCapabilities: ...

    def checkpoint_identity(self, checkpoint: Path) -> dict[str, str]: ...

    def load(self, checkpoint: Path) -> Any: ...

    def vocab_size(self, model) -> int: ...

    def generate_cpu(
        self,
        model,
        ids: list[int],
        length: int,
        backend: str,
        prepared: PreparedRequest | None,
        manifest: dict | None,
    ) -> GenerationResult: ...


def parse_options(options_type, value):
    """Only the selected integration interprets and validates its options."""
    if isinstance(value, options_type):
        return value
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"preparation options must be {options_type.__name__} or a JSON object")
    try:
        return options_type(**value)
    except TypeError as exc:
        raise ValueError(f"invalid {options_type.__name__}: {exc}") from exc


def capabilities(architecture, checkpoint_format, fhe, *, unsupported=None):
    if unsupported:
        backends = {
            name: BackendCapability("unsupported", unsupported)
            for name in ("exact", "polynomial", "ckks")
        }
    else:
        reason = f"{architecture} has no registered polynomial/CKKS implementation"
        backends = {
            "exact": BackendCapability(
                "implemented", "CPU reference; weights are not checked by inspection"
            ),
            "polynomial": BackendCapability(
                "requires_preparation" if fhe else "unsupported",
                "Requires a matching prepared request and frozen coefficients" if fhe else reason,
            ),
            "ckks": BackendCapability(
                "requires_preparation" if fhe else "unsupported",
                "Requires a prepared request, native GPU backend and per-request validation"
                if fhe
                else reason,
            ),
        }
    profiles = fhe.profiles if fhe and not unsupported else ()
    return ModelCapabilities(architecture, checkpoint_format, backends, profiles)


def preparation_profile(adapter: ModelAdapter, name: str | None) -> PreparationProfile:
    fhe = adapter.fhe
    if fhe is None:
        raise ValueError(f"{adapter.architecture} has no registered polynomial/CKKS implementation")
    name = name or fhe.default_profile
    if name is None:
        choices = ", ".join(f"{p.name} (security={p.security})" for p in fhe.profiles)
        raise ValueError(f"{adapter.architecture} requires explicit profile: {choices}")
    for profile in fhe.profiles:
        if profile.name == name:
            return profile
    raise ValueError(f"unsupported preparation profile {name!r} for {adapter.architecture}")
