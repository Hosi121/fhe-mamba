"""Public coordinate scales for an encrypted Mamba-2 recurrent state."""

from __future__ import annotations

import math

import numpy as np


def regularize_row_scales(
    observed: np.ndarray, group_heads: int, group_floor: float = 0.0
) -> np.ndarray:
    """Bound amplification relative to the enclosing ciphertext's scale.

    With floor f>0, max(row, f*group, 1e-6) retains at least the observed
    row maximum and limits group/row amplification to 1/f. This is a public
    conditioning rule, not a bound on future states or bootstrap error.
    """
    rows = np.asarray(observed, dtype=np.float64)
    if (
        rows.ndim != 2
        or min(rows.shape) == 0
        or group_heads < 1
        or rows.shape[0] % group_heads
        or not np.isfinite(rows).all()
        or (rows < 0).any()
        or not math.isfinite(group_floor)
        or not 0 <= group_floor <= 1
    ):
        raise ValueError(
            "finite nonnegative [head, channel] maxima and valid grouping/floor required"
        )
    grouped = rows.reshape(-1, group_heads * rows.shape[1])
    group = np.maximum(grouped.max(axis=1, keepdims=True), 1e-6)
    return np.maximum(np.maximum(grouped, 1e-6), group_floor * group).reshape(rows.shape)
