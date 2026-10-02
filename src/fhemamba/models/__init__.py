"""Model integration contracts and explicit registration."""

from .contracts import (
    BackendCapability,
    FHEImplementation,
    ModelAdapter,
    ModelCapabilities,
    PreparationProfile,
)
from .registry import ModelRegistration, ModelRegistry, register_model

__all__ = [
    "BackendCapability",
    "FHEImplementation",
    "ModelAdapter",
    "ModelCapabilities",
    "ModelRegistration",
    "ModelRegistry",
    "PreparationProfile",
    "register_model",
]
