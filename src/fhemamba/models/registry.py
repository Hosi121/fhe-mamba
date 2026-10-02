"""Explicit model registration; checkpoint metadata never supplies import paths."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

from .contracts import ModelAdapter


@dataclass(frozen=True)
class ModelRegistration:
    architecture: str
    implementation: str
    matches: Callable[[dict], bool]
    prepared_schemas: tuple[str, ...] = ()

    @cached_property
    def adapter(self) -> ModelAdapter:
        module, name = self.implementation.split(":", 1)
        adapter = getattr(importlib.import_module(module), name)
        if adapter.architecture != self.architecture:
            raise ValueError("registered architecture differs from its implementation")
        return adapter


class ModelRegistry:
    def __init__(self, registrations=()):
        self._registrations = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: ModelRegistration):
        if registration.architecture in self._registrations:
            raise ValueError(f"model already registered: {registration.architecture}")
        schemas = set(registration.prepared_schemas)
        if len(schemas) != len(registration.prepared_schemas) or any(
            schemas.intersection(item.prepared_schemas) for item in self._registrations.values()
        ):
            raise ValueError("prepared schema already registered")
        self._registrations[registration.architecture] = registration

    def get(self, architecture: str) -> ModelAdapter:
        try:
            return self._registrations[architecture].adapter
        except KeyError as exc:
            raise ValueError(f"unregistered model architecture: {architecture}") from exc

    def detect(self, config: dict) -> ModelAdapter:
        matches = [entry for entry in self._registrations.values() if entry.matches(config)]
        if not matches:
            raise ValueError("no registered model matches this checkpoint configuration")
        if len(matches) != 1:
            names = ", ".join(entry.architecture for entry in matches)
            raise ValueError(f"ambiguous checkpoint configuration: {names}")
        return matches[0].adapter

    def for_manifest(self, manifest: dict) -> ModelAdapter:
        for entry in self._registrations.values():
            if manifest.get("schema") in entry.prepared_schemas:
                if manifest.get("architecture") != entry.architecture:
                    raise ValueError("prepared architecture differs from its schema")
                adapter = entry.adapter
                if adapter.fhe is None:
                    raise ValueError("prepared schema has no FHE implementation")
                return adapter
        raise ValueError("no registered model supports this prepared schema")


registry = ModelRegistry(
    [
        ModelRegistration(
            "mamba2",
            "fhemamba.models.mamba2:ADAPTER",
            lambda config: config.get("model_type") == "mamba2",
            ("fhemamba-mamba2-generation-v1",),
        ),
        ModelRegistration(
            "mamba3",
            "fhemamba.models.mamba3:ADAPTER",
            lambda config: (config.get("ssm_cfg") or {}).get("layer") == "Mamba3",
            ("fhemamba-mamba3-lm-v1",),
        ),
    ]
)


def register_model(registration: ModelRegistration):
    """Register an installed integration explicitly before calling the public API."""
    registry.register(registration)
