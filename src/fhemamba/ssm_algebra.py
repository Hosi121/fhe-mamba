"""Plaintext algebra oracles for proposed FHE schedules, not encrypted kernels.

These preserve the selective recurrence without inverses or encrypted branches.
See docs/research/2026-09-21-ssm-cryptographic-design.md for costs and limitations.
"""

from __future__ import annotations

import torch
from torch import Tensor


class DeferredMamba2State:
    """Keep a fixed checkpoint plus at most ``window`` rank-one updates.

    State shape is (heads, channels, state_size). Each step receives per-head
    scalar decay, u=dt*x, B and C. Readout contracts B with C before expanding
    the channel dimension. Only flush() materializes the new dense state.
    The schedule is public; no decision depends on secret tensor values.
    """

    def __init__(self, initial: Tensor, window: int):
        if initial.ndim != 3 or min(initial.shape) < 1:
            raise ValueError("initial must have nonempty (heads, channels, state_size) shape")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        self.base = initial.clone()
        self.window = window
        self.base_decay = initial.new_ones(initial.shape[0])
        self.pending: list[tuple[Tensor, Tensor, Tensor]] = []
        self.flushes = 0

    def step(self, decay: Tensor, update: Tensor, b: Tensor, c: Tensor) -> Tensor:
        heads, channels, state_size = self.base.shape
        if (
            decay.shape != (heads,)
            or update.shape != (heads, channels)
            or b.shape != (heads, state_size)
            or c.shape != (heads, state_size)
        ):
            raise ValueError("step inputs must match the checkpoint's head/channel/state shapes")
        self.base_decay = self.base_decay * decay
        self.pending = [(u, key, weight * decay) for u, key, weight in self.pending]
        self.pending.append((update.clone(), b.clone(), torch.ones_like(decay)))
        output = self.base_decay[:, None] * torch.einsum("hpn,hn->hp", self.base, c)
        for u, key, weight in self.pending:
            score = (key * c).sum(dim=-1)
            output = output + (weight * score)[:, None] * u
        if len(self.pending) == self.window:
            self.flush()
        return output

    def materialize(self) -> Tensor:
        """Inspection/flush oracle; calling this every step defeats the proposal."""
        state = self.base_decay[:, None, None] * self.base
        for u, key, weight in self.pending:
            state = state + (weight[:, None] * u)[:, :, None] * key[:, None, :]
        return state

    def flush(self) -> None:
        if not self.pending:
            return
        self.base = self.materialize()
        self.base_decay = torch.ones_like(self.base_decay)
        self.pending.clear()
        self.flushes += 1


def complex_trapezoid_scan(
    decay: Tensor,
    dt: Tensor,
    mixing: Tensor,
    phase: Tensor,
    x: Tensor,
    b: Tensor,
    c: Tensor,
) -> tuple[Tensor, Tensor]:
    """Direct complex form of the Mamba-3 SISO recurrence, zero initial state.

    decay/dt/mixing: (time, heads); x: (time, heads, channels);
    phase/b/c: (time, heads, complex_state_size). ``phase`` is exp(i*angle).
    b and c encode real coordinate pairs as real+i*imag; consequently readout
    uses conjugate(c). This explicitly states the sign convention.

    This is an algebra oracle only: no projections, learned checkpoint,
    BC normalization, polynomial phase approximation, or ciphertexts.
    """
    if (
        decay.ndim != 2
        or min(decay.shape) < 1
        or dt.shape != decay.shape
        or mixing.shape != decay.shape
        or x.ndim != 3
        or x.shape[:2] != decay.shape
        or b.ndim != 3
        or b.shape[:2] != decay.shape
        or c.shape != b.shape
        or phase.shape != b.shape
        or not b.is_complex()
        or not c.is_complex()
        or not phase.is_complex()
    ):
        raise ValueError("incompatible SISO sequence shapes or non-complex phase/B/C")
    _, heads, channels = x.shape
    state = b.new_zeros(heads, channels, b.shape[-1])
    previous = torch.zeros_like(state)
    outputs = []
    for time in range(decay.shape[0]):
        current = x[time, :, :, None] * b[time, :, None, :]
        alpha = decay[time, :, None, None] * phase[time, :, None, :]
        lag_weight = ((1 - mixing[time]) * dt[time])[:, None, None]
        write_weight = (mixing[time] * dt[time])[:, None, None]
        state = alpha * (state + lag_weight * previous) + write_weight * current
        outputs.append((state * c[time, :, None, :].conj()).sum(-1).real)
        previous = current
    return torch.stack(outputs), state


def cubic_phase(angle: Tensor) -> Tensor:
    """Third-order phase candidate, contractive only for |angle| <= sqrt(3).

    |p(angle)|^2 = 1 - angle^4/12 + angle^6/36. Contraction introduces memory
    loss; this function is not a replacement for exact rotary embeddings.
    """
    squared = angle * angle
    return (1 - squared / 2) + 1j * angle * (1 - squared / 6)


def rsqrt_residual_step(residual: Tensor) -> Tensor:
    """Exact residual identity q'=q^2(3+q)/4 for q=1-v*r^2.

    Assumes exact arithmetic and fixed positive v; it excludes CKKS errors.
    The positive-root convergence basin includes r>0 and 0<v*r^2<3.
    """
    return residual.square() * (3 + residual) / 4
