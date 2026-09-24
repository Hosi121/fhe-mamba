"""Mamba-3 SISO mixer: one formula for exact, polynomial and encrypted execution.

Matches state-spaces/mamba's input layout, B/C RMSNorm and biases,
tanh/dt rotary phase, exponential-trapezoidal recurrence and output gate.
Weights may come directly from an upstream Mamba3 module; no CUDA import is
needed to use the CPU module below. MIMO is rejected explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .tensor_ops import TensorOps, rms_normalize, ssm_readout


@dataclass
class Mamba3State:
    angle: object
    ssm: object
    k: object
    v: object


@dataclass
class FactoredMamba3State:
    """Exact finite-history representation, without a head/channel/state tensor.

    Each write is rank one. Keeping its key, value and scalar coefficient makes
    short prompt/decode sessions much cheaper in FHE; no rank truncation occurs.
    Storage and readout grow with sequence length, unlike the dense recurrence.
    """

    angle: object
    keys: tuple = ()
    values: tuple = ()
    weights: tuple = ()


def init_factored_mamba3_state(mixer, batch_size=1):
    validate_mamba3(mixer)
    return FactoredMamba3State(
        mixer.in_proj.weight.new_zeros(batch_size, mixer.nheads, mixer.num_rope_angles)
    )


def validate_mamba3(mixer):
    if mixer.is_mimo:
        raise NotImplementedError("Mamba-3 MIMO is not supported by this SISO backend")
    if mixer.num_bc_heads != 1:
        raise NotImplementedError("Mamba-3 SISO currently requires one B/C group")
    if mixer.d_inner != mixer.nheads * mixer.headdim:
        raise ValueError("inconsistent Mamba-3 head dimensions")
    if not 0 < 2 * mixer.num_rope_angles <= mixer.d_state:
        raise ValueError("invalid Mamba-3 rotary dimensions")


def init_mamba3_state(mixer, batch_size=1):
    validate_mamba3(mixer)
    w = mixer.in_proj.weight
    h, p, n = mixer.nheads, mixer.headdim, mixer.d_state
    return Mamba3State(
        w.new_zeros(batch_size, h, mixer.num_rope_angles),
        w.new_zeros(batch_size, h, p, n),
        w.new_zeros(batch_size, h, n),
        w.new_zeros(batch_size, h, p),
    )


def _rotate_bc(value, cosine, sine, angles, ops):
    # The SISO upstream kernel pairs adjacent coordinates. Unrotated channels
    # remain present when rope_fraction=0.5.
    even, odd = value[..., : 2 * angles : 2], value[..., 1 : 2 * angles : 2]
    pairs = ops.concatenate(
        ((even * cosine - odd * sine)[..., None], (even * sine + odd * cosine)[..., None]), axis=-1
    )
    rotated = pairs.reshape(*value.shape[:-1], 2 * angles)
    if 2 * angles < value.shape[-1]:
        rotated = ops.concatenate((rotated, value[..., 2 * angles :]), axis=-1)
    return rotated


def mamba3_step(mixer, x, state: Mamba3State, ops, layer_idx=0):
    """Evaluate one mixer step; `ops` owns all nonlinear/linear/reduction work.

    `x` has shape (batch, d_model). PackedProgram lowers the very same formula;
    its values implement only reshape, indexing and broadcast arithmetic.
    """
    validate_mamba3(mixer)
    h, p, n, a = mixer.nheads, mixer.headdim, mixer.d_state, mixer.num_rope_angles
    if len(x.shape) != 2 or x.shape[-1] != mixer.d_model:
        raise ValueError("Mamba-3 step input must be (batch, d_model)")
    batch = x.shape[0]
    factored = isinstance(state, FactoredMamba3State)
    if factored:
        valid = state.angle.shape == (batch, h, a)
        valid = valid and len(state.keys) == len(state.values) == len(state.weights)
        valid = valid and all(k.shape == (batch, h, n) for k in state.keys)
        valid = valid and all(v.shape == (batch, h, p) for v in state.values)
        valid = valid and all(w.shape == (batch, h) for w in state.weights)
    else:
        shapes = ((batch, h, a), (batch, h, p, n), (batch, h, n), (batch, h, p))
        valid = all(
            value.shape == shape
            for value, shape in zip((state.angle, state.ssm, state.k, state.v), shapes, strict=True)
        )
    if not valid:
        raise ValueError("Mamba-3 state does not match batch/head/state dimensions")
    projection = ops.linear(x, mixer.in_proj.weight)
    offset = 0
    parts = []
    for width in (mixer.d_inner, mixer.d_inner, n, n, h, h, h, a):
        parts.append(projection[..., offset : offset + width])
        offset += width
    z, v, b, c, dt, rate, mixing, phase = parts
    z, v = z.reshape(batch, h, p), v.reshape(batch, h, p)
    dt = ops.nonlinear(dt + mixer.dt_bias, "softplus", (layer_idx, "dt_softplus"))
    rate = ops.nonlinear(rate, "negative_a", (layer_idx, "m3_negative_a"), mixer.A_floor)
    decay = ops.nonlinear(rate * dt, "exp", (layer_idx, "decay_exp"))
    mixing = ops.nonlinear(mixing, "sigmoid", (layer_idx, "m3_mixing"))
    phase = ops.nonlinear(phase, "tanh", (layer_idx, "m3_angle_tanh"))
    angle = state.angle + (phase[:, None, :] * dt[..., None]) * math.pi
    cosine = ops.nonlinear(angle, "cos", (layer_idx, "m3_cos"))
    sine = ops.nonlinear(angle, "sin", (layer_idx, "m3_sin"))
    b = rms_normalize(b, mixer.B_norm.weight, mixer.B_norm.eps, ops, (layer_idx, "m3_b_rms"))
    c = rms_normalize(c, mixer.C_norm.weight, mixer.C_norm.eps, ops, (layer_idx, "m3_c_rms"))
    b = b[:, None, :] + mixer.B_bias[:, 0, :]
    c = c[:, None, :] + mixer.C_bias[:, 0, :]
    b, c = (_rotate_bc(value, cosine, sine, a, ops) for value in (b, c))
    lag = (1 - mixing) * dt * decay
    write = mixing * dt
    if factored:
        weights = [weight * decay for weight in state.weights]
        if weights:
            weights[-1] = weights[-1] + lag
        carry = FactoredMamba3State(angle, (*state.keys, b), (*state.values, v), (*weights, write))
        y = v * mixer.D[:, None]
        for key, value, weight in zip(carry.keys, carry.values, carry.weights, strict=True):
            score = ops.sum_last(key * c)[..., 0] * weight
            y = y + value * score[..., None]
    else:
        # k/v are the previous rotated B and x, including the lag write at t>0.
        ssm = state.ssm * decay[..., None, None]
        ssm = ssm + (state.v[..., None] * state.k[..., None, :]) * lag[..., None, None]
        ssm = ssm + (v[..., None] * b[..., None, :]) * write[..., None, None]
        y = ssm_readout(ssm, c, ops) + v * mixer.D[:, None]
        carry = Mamba3State(angle, ssm, b, v)
    if mixer.is_outproj_norm:
        weight = mixer.norm.weight.reshape(h, p)
        y = rms_normalize(y, weight, mixer.norm.eps, ops, (layer_idx, "m3_output_rms"))
    y = y * ops.nonlinear(z, "silu", (layer_idx, "gate_silu"))
    output = ops.linear(y.reshape(batch, mixer.d_inner), mixer.out_proj.weight)
    return output, carry


def mixer3_forward(mixer, input_states, ops=None, layer_idx=0, scan="loop", state=None):
    if scan != "loop":
        raise NotImplementedError("Mamba-3 reference currently supports scan='loop' only")
    validate_mamba3(mixer)
    if input_states.ndim != 3 or input_states.shape[-1] != mixer.d_model:
        raise ValueError("expected (batch, tokens, d_model) Mamba-3 input")
    if input_states.shape[1] == 0:
        raise ValueError("Mamba-3 input must contain at least one token")
    algebra = ops if isinstance(ops, TensorOps) else TensorOps(ops)
    carry = state if state is not None else init_mamba3_state(mixer, input_states.shape[0])
    outputs = []
    for token in range(input_states.shape[1]):
        y, carry = mamba3_step(mixer, input_states[:, token], carry, algebra, layer_idx)
        outputs.append(y)
    if state is not None:
        state.angle, state.ssm, state.k, state.v = carry.angle, carry.ssm, carry.k, carry.v
    return torch.stack(outputs, dim=1)


class _RMSNorm(nn.Module):
    """Upstream-compatible parameters, also usable with the minimum Torch 2.2."""

    def __init__(self, width, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width, **kwargs))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba3Mixer(nn.Module):
    """CPU-friendly SISO module with upstream-compatible parameter names/shapes."""

    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        headdim=64,
        rope_fraction=0.5,
        is_outproj_norm=False,
        A_floor=1e-4,  # noqa: N803 - upstream parameter name
        device=None,
        dtype=None,
    ):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in (d_model, d_state, expand, headdim)):
            raise ValueError(
                "model, state, expansion and head dimensions must be positive integers"
            )
        if rope_fraction not in (0.5, 1.0) or not math.isfinite(A_floor) or A_floor <= 0:
            raise ValueError("invalid rotary fraction or A_floor")
        self.d_model, self.d_state, self.d_inner = d_model, d_state, expand * d_model
        if self.d_inner % headdim:
            raise ValueError("expanded model width must be divisible by headdim")
        self.headdim, self.nheads = headdim, self.d_inner // headdim
        self.num_rope_angles = int(d_state * rope_fraction) // 2
        self.num_bc_heads, self.is_mimo, self.mimo_rank = 1, False, 1
        self.A_floor, self.is_outproj_norm = A_floor, is_outproj_norm
        kw = {"device": device, "dtype": dtype}
        width = 2 * self.d_inner + 2 * d_state + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, width, bias=False, **kw)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **kw)
        # Same initialization range as upstream, with the stable inverse softplus.
        dt = torch.empty(self.nheads, **kw).uniform_(math.log(0.001), math.log(0.1)).exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state, **kw))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state, **kw))
        self.D = nn.Parameter(torch.ones(self.nheads, **kw))
        self.B_norm = _RMSNorm(d_state, eps=1e-5, **kw)
        self.C_norm = _RMSNorm(d_state, eps=1e-5, **kw)
        if is_outproj_norm:
            self.norm = _RMSNorm(self.d_inner, eps=1e-5, **kw)
        validate_mamba3(self)

    def forward(self, x, state=None, ops=None):
        return mixer3_forward(self, x, ops, state=state)
