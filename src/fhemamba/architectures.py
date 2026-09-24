"""Architecture dispatch and state allocation shared by reference callers.

An architecture owns its recurrent state; callers never infer a convolution
FIFO from the existence of an SSM. Native support is a separate capability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LayerState:
    conv: Tensor
    ssm: Tensor


def mixer_architecture(mixer) -> str:
    name = type(mixer).__name__
    if name in {"Mamba3", "Mamba3Mixer"}:
        return "mamba3"
    if name == "Mamba2Mixer":
        return "mamba2"
    if name == "MambaMixer":
        return "mamba1"
    raise ValueError(f"unsupported mixer architecture: {name}")


def init_mixer_state(mixer, batch_size: int = 1):
    architecture = mixer_architecture(mixer)
    if architecture == "mamba3":
        from .mamba3 import init_mamba3_state

        return init_mamba3_state(mixer, batch_size)
    weight = mixer.conv1d.weight
    conv = weight.new_zeros(batch_size, weight.shape[0], mixer.conv_kernel_size - 1)
    shape = (
        (batch_size, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size)
        if architecture == "mamba2"
        else (batch_size, mixer.intermediate_size, mixer.ssm_state_size)
    )
    return LayerState(conv, torch.zeros(shape, device=weight.device, dtype=weight.dtype))


def mixer_forward_dispatch(mixer):
    from .mamba3 import mixer3_forward
    from .reference import mixer2_forward, mixer_forward

    return {"mamba1": mixer_forward, "mamba2": mixer2_forward, "mamba3": mixer3_forward}[
        mixer_architecture(mixer)
    ]
