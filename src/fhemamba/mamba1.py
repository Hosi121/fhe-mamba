"""Mamba-1 token steps shared by polynomial evaluation and packed CKKS lowering.

Weights are public, while convolution history and every selective-SSM coordinate
remain encrypted in the packed program. Channel tiles bound ciphertext width;
they do not truncate or factor the state. The existing FP32 reference remains
the independent acceptance oracle for this FP64 lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .tensor_ops import TensorOps, rms_normalize


@dataclass(frozen=True)
class Mamba1State:
    conv: tuple
    ssm: tuple


def _weight(value):
    return None if value is None else value.detach().to(device="cpu", dtype=torch.float64)


def _linear(module):
    return SimpleNamespace(weight=_weight(module.weight), bias=_weight(module.bias))


def _norm(module):
    return SimpleNamespace(weight=_weight(module.weight), epsilon=module.variance_epsilon)


def _affine(x, module, ops):
    value = ops.linear(x, module.weight)
    return value if module.bias is None else value + module.bias


class Mamba1LM:
    """Read public HF weights without mutating the caller's CPU model or state."""

    def __init__(self, model, *, decay_squarings=None):
        self.embedding = _weight(model.get_input_embeddings().weight)
        self.head = (
            self.embedding
            if model.lm_head.weight is model.get_input_embeddings().weight
            else _weight(model.lm_head.weight)
        )
        self.head_bias = _weight(model.lm_head.bias)
        self.norm = _norm(model.backbone.norm_f)
        self.layers = []
        for block in model.backbone.layers:
            m = block.mixer
            self.layers.append(
                SimpleNamespace(
                    norm=_norm(block.norm),
                    inner=m.intermediate_size,
                    state_size=m.ssm_state_size,
                    rank=m.time_step_rank,
                    kernel=m.conv_kernel_size,
                    in_proj=_linear(m.in_proj),
                    x_proj=_linear(m.x_proj),
                    dt_proj=_linear(m.dt_proj),
                    out_proj=_linear(m.out_proj),
                    conv_weight=_weight(m.conv1d.weight[:, 0]),
                    conv_bias=_weight(m.conv1d.bias),
                    # Upstream/reference evaluate the public exponential in FP32.
                    a=(-torch.exp(m.A_log.detach().float())).double().cpu(),
                    d=_weight(m.D),
                )
            )
        self.decay_squarings = (
            (0,) * len(self.layers) if decay_squarings is None else tuple(decay_squarings)
        )
        if len(self.decay_squarings) != len(self.layers) or any(
            type(count) is not int or not 0 <= count <= 32 for count in self.decay_squarings
        ):
            raise ValueError("Mamba-1 requires one frozen decay squaring count (0..32) per layer")

    def initial_states(self, *, slots=32768, program=None):
        states = []
        for layer in self.layers:
            # sum_last pads a non-power-of-two state dimension before reduction.
            padded = 1 << (layer.state_size - 1).bit_length()
            channels = slots // padded
            if channels < 1 or max(2 * layer.inner, layer.rank + 2 * layer.state_size) > slots:
                raise ValueError("Mamba-1 projections and one state row must fit in CKKS slots")

            def zero(*shape):
                value = torch.zeros(shape, dtype=torch.float64)
                return value if program is None else program.constant(value)

            states.append(
                Mamba1State(
                    tuple(zero(1, layer.inner) for _ in range(layer.kernel - 1)),
                    tuple(
                        zero(1, min(channels, layer.inner - start), layer.state_size)
                        for start in range(0, layer.inner, channels)
                    ),
                )
            )
        return states

    def step(self, hidden, states, ops=None):
        ops = TensorOps() if ops is None else ops
        next_states = []
        for index, (layer, state) in enumerate(zip(self.layers, states, strict=True)):
            normed = rms_normalize(
                hidden, layer.norm.weight, layer.norm.epsilon, ops, (index, "rms_invsqrt")
            )
            projected = _affine(normed, layer.in_proj, ops)
            value, gate = projected[:, : layer.inner], projected[:, layer.inner :]
            window = (*state.conv, value)
            convolved = window[0] * layer.conv_weight[:, 0]
            for position in range(1, layer.kernel):
                convolved = convolved + window[position] * layer.conv_weight[:, position]
            if layer.conv_bias is not None:
                convolved = convolved + layer.conv_bias
            value = ops.nonlinear(convolved, "silu", (index, "conv_silu"))
            selected = _affine(value, layer.x_proj, ops)
            dt = ops.nonlinear(
                _affine(selected[:, : layer.rank], layer.dt_proj, ops),
                "softplus",
                (index, "dt_softplus"),
            )
            b = selected[:, layer.rank : layer.rank + layer.state_size]
            c = selected[:, layer.rank + layer.state_size :]
            blocks, readouts, offset = [], [], 0
            for block in state.ssm:
                channels = slice(offset, offset + block.shape[1])
                delta = dt[:, channels, None]
                exponent = delta * layer.a[channels]
                squarings = self.decay_squarings[index]
                if squarings:
                    exponent = exponent * (2.0**-squarings)
                decay = ops.nonlinear(exponent, "exp", (index, "decay_exp"))
                for _ in range(squarings):
                    decay = decay * decay
                updated = decay * block + (delta * b[:, None, :]) * value[:, channels, None]
                blocks.append(updated)
                readouts.append(ops.sum_last(updated * c[:, None, :])[:, :, 0])
                offset += block.shape[1]
            readout = readouts[0] if len(readouts) == 1 else ops.concatenate(readouts)
            mixed = (readout + value * layer.d) * ops.nonlinear(gate, "silu", (index, "gate_silu"))
            hidden = hidden + _affine(mixed, layer.out_proj, ops)
            next_states.append(Mamba1State(tuple(window[1:]), tuple(blocks)))
        return (
            rms_normalize(
                hidden, self.norm.weight, self.norm.epsilon, ops, (len(self.layers), "rms_invsqrt")
            ),
            next_states,
        )

    @torch.no_grad()
    def generate(self, ids, length, ops=None, *, slots=32768):
        states = self.initial_states(slots=slots)
        evaluated, generated, trace = list(ids), [], []
        for step in range(len(ids) + length - 1):
            hidden, states = self.step(self.embedding[evaluated[step]][None], states, ops)
            if not torch.isfinite(hidden).all():
                raise ValueError("Mamba-1 lowering produced non-finite hidden states")
            trace.append(hidden)
            if step >= len(ids) - 1:
                logits = hidden @ self.head.T
                if self.head_bias is not None:
                    logits = logits + self.head_bias
                if not torch.isfinite(logits).all():
                    raise ValueError("Mamba-1 lowering produced non-finite logits")
                token = int(logits.argmax())
                generated.append(token)
                evaluated.append(token)
        return generated, trace
