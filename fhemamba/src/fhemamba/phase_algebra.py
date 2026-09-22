"""Polynomial shear-phase research oracle; not a Mamba-3 substitution.

Three linear shears have determinant one in exact arithmetic. A fixed angle
preserves a modified quadratic energy, while changing angles can be unstable.
"""

from __future__ import annotations

import torch
from torch import Tensor


def shear_rotate(pair: Tensor, angle: Tensor) -> Tensor:
    """Apply three shears with slopes -angle/2, angle, -angle/2.

    ``pair`` ends in two real coordinates; ``angle`` broadcasts with the
    preceding dimensions. There are three sequential dependent products.
    This is an approximation to rotation, with a different phase frequency.
    Determinant one does not imply norm preservation or stability under
    changing angles, and excludes CKKS arithmetic/refresh error.
    """
    if pair.ndim < 1 or pair.shape[-1] != 2 or pair.is_complex() or angle.is_complex():
        raise ValueError("expected two real coordinates and a real angle")
    x, y = pair.unbind(-1)
    x = x - (angle / 2) * y
    y = y + angle * x
    x = x - (angle / 2) * y
    return torch.stack((x, y), dim=-1)
