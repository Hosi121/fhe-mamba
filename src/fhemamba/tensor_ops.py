"""Tensor algebra boundary for plaintext references and packed CKKS lowering."""

from __future__ import annotations

from dataclasses import dataclass

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

    def checkpoint(self, value, site):
        """Observe a semantic boundary without adding arithmetic to the graph."""
        return value


def rms_normalize(x, weight, epsilon, ops, site):
    """Shared RMSNorm formula; all encrypted-dependent work goes through ops."""
    variance = ops.sum_last(x * x) * (1.0 / x.shape[-1]) + epsilon
    return (x * ops.nonlinear(variance, "inv_sqrt", site)) * weight


def ssm_readout(state, c, ops):
    return ops.sum_last(state * c[..., None, :])[..., 0]


@dataclass(frozen=True)
class HeadTiledState:
    """A recurrent (batch, head, channel, state) tensor split along its heads.

    Every block keeps whole heads, so updates and readout need no communication
    between blocks. This changes storage only; no state coordinates are dropped.
    """

    blocks: tuple

    def __post_init__(self):
        if not self.blocks or any(len(block.shape) != 4 for block in self.blocks):
            raise ValueError("state blocks must be nonempty rank-four tensors")
        batch, _, channels, states = self.blocks[0].shape
        if any(
            block.shape[0] != batch or block.shape[2:] != (channels, states) or min(block.shape) < 1
            for block in self.blocks
        ):
            raise ValueError("state blocks must have matching batch/channel/state dimensions")

    @property
    def shape(self):
        batch, _, channels, states = self.blocks[0].shape
        return batch, sum(block.shape[1] for block in self.blocks), channels, states


def ssm_update_readout(state, decay, writes, c, ops):
    """Apply decay and rank-one writes, then read a dense or head-tiled state.

    A write is (key, value, coefficient). Scaling its small value vector before
    the outer product avoids multiplying every state coordinate a second time.
    The same formula is used for Torch and encrypted packed blocks.
    """
    weighted = [(key, value * coefficient[..., None]) for key, value, coefficient in writes]

    def update(block, block_decay, block_writes, query):
        result = block * block_decay[..., None, None]
        for key, value in block_writes:
            result = result + value[..., None] * key[..., None, :]
        return result, ssm_readout(result, query, ops)

    if not isinstance(state, HeadTiledState):
        return update(state, decay, weighted, c)
    blocks, outputs, offset = [], [], 0
    for block in state.blocks:
        heads = slice(offset, offset + block.shape[1])
        next_block, output = update(
            block,
            decay[:, heads],
            [(key[:, heads], value[:, heads]) for key, value in weighted],
            c[:, heads],
        )
        blocks.append(next_block)
        outputs.append(output)
        offset += block.shape[1]
    return HeadTiledState(tuple(blocks)), ops.concatenate(outputs, axis=1)
