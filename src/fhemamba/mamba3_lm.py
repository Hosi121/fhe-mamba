"""Checkpoint-compatible Mamba-3 SISO language model, including gated MLPs.

The CPU oracle and encrypted lowering share the same residual/norm/MLP graph.
Embedding lookup and the tied vocabulary head belong to the generation client,
as in the existing Mamba-2 protocol. The entire backbone stays encrypted.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .mamba3 import (
    Mamba3Mixer,
    _RMSNorm,
    init_factored_mamba3_state,
    init_mamba3_state,
    mamba3_step,
)
from .tensor_ops import TensorOps, rms_normalize


class Mamba3LM(nn.Module):
    def __init__(self, config, *, dtype=torch.float64, device=None):
        super().__init__()
        cfg = dict(config)
        ssm = dict(cfg["ssm_cfg"])
        if ssm.pop("layer") != "Mamba3" or ssm.pop("is_mimo", False):
            raise ValueError("a Mamba-3 SISO checkpoint is required")
        if ssm.pop("ngroups", 1) != 1 or cfg.get("attn_layer_idx"):
            raise ValueError("grouped SSMs and attention layers are unsupported")
        if not cfg.get("rms_norm", True) or not cfg.get("tie_embeddings", True):
            raise ValueError("RMSNorm and tied embeddings are required")
        # Initialization and GPU scan options do not change checkpoint inference.
        for key in ("dt_min", "dt_max", "dt_init_floor", "chunk_size"):
            ssm.pop(key, None)
        self.config = SimpleNamespace(**cfg)
        kw = {"dtype": dtype, "device": device}
        width = cfg["d_model"]
        intermediate = cfg["d_intermediate"]
        if intermediate <= 0 or intermediate % 128:
            raise ValueError("checkpoint MLP width must be a positive multiple of 128")
        eps = cfg.get("norm_epsilon", 1e-5)
        vocab = cfg["vocab_size"]
        multiple = cfg.get("pad_vocab_size_multiple", 16)
        vocab = (vocab + multiple - 1) // multiple * multiple
        layers = []
        for _ in range(cfg["n_layer"]):
            layer = nn.Module()
            layer.norm = _RMSNorm(width, eps, **kw)
            layer.mixer = Mamba3Mixer(width, **ssm, **kw)
            layer.norm2 = _RMSNorm(width, eps, **kw)
            layer.mlp = nn.Module()
            layer.mlp.fc1 = nn.Linear(width, 2 * intermediate, bias=False, **kw)
            layer.mlp.fc2 = nn.Linear(intermediate, width, bias=False, **kw)
            layers.append(layer)
        self.backbone = nn.Module()
        self.backbone.embedding = nn.Embedding(vocab, width, **kw)
        self.backbone.layers = nn.ModuleList(layers)
        self.backbone.norm_f = _RMSNorm(width, eps, **kw)
        self.lm_head = nn.Linear(width, vocab, bias=False, **kw)
        self.lm_head.weight = self.backbone.embedding.weight

    @classmethod
    def from_pretrained(cls, directory, *, dtype=torch.float64, device="cpu"):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        model = cls(config, dtype=dtype, device="meta")
        weights = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
        # assign=True avoids allocating/randomizing a second full checkpoint.
        model.load_state_dict(weights, strict=True, assign=True)
        if not torch.equal(model.lm_head.weight, model.backbone.embedding.weight):
            raise ValueError("checkpoint claims tied embeddings but the tensors differ")
        model.lm_head.weight = model.backbone.embedding.weight
        return model.to(device=device, dtype=dtype).eval()

    def initial_states(self, *, factored=True):
        initialize = init_factored_mamba3_state if factored else init_mamba3_state
        return [initialize(layer.mixer) for layer in self.backbone.layers]

    def step(self, embedding, states, ops=None):
        ops = TensorOps() if ops is None else ops
        if len(states) != len(self.backbone.layers):
            raise ValueError("one carried state per layer is required")
        hidden, carry = embedding, []
        for index, (layer, state) in enumerate(zip(self.backbone.layers, states, strict=True)):
            x = rms_normalize(hidden, layer.norm.weight, layer.norm.eps, ops, (index, "block_rms"))
            mixed, state = mamba3_step(layer.mixer, x, state, ops, index)
            hidden = hidden + mixed
            x = rms_normalize(hidden, layer.norm2.weight, layer.norm2.eps, ops, (index, "mlp_rms"))
            projection = ops.linear(x, layer.mlp.fc1.weight)
            width = layer.mlp.fc2.in_features
            gated = projection[..., :width] * ops.nonlinear(
                projection[..., width:], "silu", (index, "mlp_silu")
            )
            hidden = hidden + ops.linear(gated, layer.mlp.fc2.weight)
            carry.append(state)
        norm = self.backbone.norm_f
        return rms_normalize(hidden, norm.weight, norm.eps, ops, (-1, "final_rms")), carry

    @torch.no_grad()
    def generate(self, prompt_ids, new_tokens, ops=None):
        if not prompt_ids or new_tokens < 1:
            raise ValueError("nonempty prompt and positive generation length are required")
        states = self.initial_states()
        generated, trace = [], []
        token_ids = list(prompt_ids)
        for step in range(len(prompt_ids) + new_tokens - 1):
            token = token_ids[step]
            hidden, states = self.step(self.backbone.embedding.weight[token][None], states, ops)
            trace.append(hidden)
            if step >= len(prompt_ids) - 1:
                token = int(self.lm_head(hidden).argmax(-1).item())
                token_ids.append(token)
                generated.append(token)
        return generated, trace
