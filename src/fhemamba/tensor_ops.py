"""Tensor algebra boundary for plaintext references and packed CKKS lowering."""

from __future__ import annotations

import torch
from torch.nn import functional as F  # noqa: N812

from .ops import Exact


def nonlinear_function(kind: str, parameter: float = 0.0):
    if kind == "negative_a":
        return lambda x: -(x.clamp(min=0) + (1 - x.clamp(max=0)).reciprocal()).clamp(min=parameter)
    return {
        "silu": F.silu,
        "softplus": F.softplus,
        "exp": torch.exp,
        "inv_sqrt": torch.rsqrt,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "sin": torch.sin,
        "cos": torch.cos,
    }[kind]


class TensorOps:
    """Same arithmetic interface as PackedProgram, with injectable nonlinearities."""

    def __init__(self, nonlinearities=None):
        self.nonlinearities = nonlinearities if nonlinearities is not None else Exact()

    def linear(self, x, weight):
        return F.linear(x, weight)

    def nonlinear(self, x, kind, site, parameter=0.0):
        if kind in {"silu", "softplus", "exp", "inv_sqrt"}:
            return getattr(self.nonlinearities, kind)(x, site)
        return self.nonlinearities.unary(x, site, nonlinear_function(kind, parameter))

    def sum_last(self, x):
        return x.sum(-1, keepdim=True)

    def concatenate(self, values, axis=-1):
        return torch.cat(values, dim=axis)


def rms_normalize(x, weight, epsilon, ops, site):
    """Shared RMSNorm formula; all encrypted-dependent work goes through ops."""
    variance = ops.sum_last(x * x) * (1.0 / x.shape[-1]) + epsilon
    return (x * ops.nonlinear(variance, "inv_sqrt", site)) * weight


def ssm_readout(state, c, ops):
    return ops.sum_last(state * c[..., None, :])[..., 0]
