#!/usr/bin/env python3
"""Verify a trained complete backbone against original upstream CPU functions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from check_mamba3_upstream import reference
from einops import rearrange
from torch.nn import functional as F  # noqa: N812

from fhemamba.mamba3 import Mamba3State, init_mamba3_state
from fhemamba.mamba3_lm import Mamba3LM


class Norm:
    def __init__(self, norm):
        self.weight, self.eps = norm.weight, norm.eps

    def __call__(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


@torch.no_grad()
def check(source, checkpoint):
    paths = {
        name: source / path
        for name, path in {
            "module": "mamba_ssm/modules/mamba3.py",
            "rotary": "mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py",
            "step": "mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py",
            "block": "mamba_ssm/modules/block.py",
            "mlp": "mamba_ssm/modules/mlp.py",
        }.items()
    }
    namespace = {
        "torch": torch,
        "math": math,
        "F": F,
        "rearrange": rearrange,
        "Optional": Optional,
        "Tensor": torch.Tensor,
    }
    reference(paths["module"], "heavy_tail_activation", namespace)
    preprocess = reference(paths["module"], "_preprocess", namespace, owner="Mamba3")
    rotary = reference(paths["rotary"], "apply_rotary_qk_inference_reference", namespace)
    recurrence = reference(paths["step"], "selective_state_update_fused_ref_v2", namespace)
    block = reference(paths["block"], "forward", namespace, owner="Block")
    mlp = reference(paths["mlp"], "forward", namespace, owner="GatedMLP")
    model = Mamba3LM.from_pretrained(checkpoint)
    ours = model.initial_states()
    theirs = [init_mamba3_state(layer.mixer) for layer in model.backbone.layers]

    def mixer_step(index, mixer, input_states):
        state = theirs[index]
        h, p, n, a = mixer.nheads, mixer.headdim, mixer.d_state, mixer.num_rope_angles
        z, x, b, c, dt, rate, trap, angle = mixer.in_proj(input_states).split(
            [h * p, h * p, n, n, h, h, h, a], dim=-1
        )
        dt, b, c, x, z, trap, rate, angle = preprocess(mixer, rate, dt, b, c, x, z, trap, angle)
        c, b, next_angle = rotary(
            c, b, state.angle, angle, dt, mixer.C_bias.transpose(0, 1), mixer.B_bias.transpose(0, 1)
        )
        ones = torch.ones(1, h, p, dtype=torch.float64)
        y, ssm = recurrence(
            state.ssm,
            rate,
            b,
            c,
            ones,
            x,
            ones,
            z,
            dt,
            state.k[:, None],
            state.v,
            trap,
            mixer.D,
            ones,
            compute_dtype=torch.float64,
        )
        theirs[index] = Mamba3State(next_angle, ssm, b[:, 0], x)
        return mixer.out_proj(y.reshape(1, h * p))

    tokens = [791, 6864]  # Complete prompt: "The capital", no BOS.
    output_error, state_error = 0.0, 0.0
    actual_tokens, upstream_tokens = [], []
    for step in range(5):
        embedding = model.backbone.embedding.weight[tokens[step]][None]
        actual, ours = model.step(embedding, ours)
        hidden, residual = embedding, None
        for index, layer in enumerate(model.backbone.layers):
            # Original Block.forward and GatedMLP.forward determine the order;
            # only the fused GPU mixer is substituted with upstream's oracle.
            wrapper = SimpleNamespace(
                fused_add_norm=False,
                residual_in_fp32=False,
                norm=Norm(layer.norm),
                norm2=Norm(layer.norm2),
                mixer=lambda x, inference_params=None, i=index, m=layer.mixer: mixer_step(i, m, x),
                mlp=lambda x, m=layer.mlp: mlp(
                    SimpleNamespace(fc1=m.fc1, fc2=m.fc2, activation=F.silu), x
                ),
            )
            hidden, residual = block(wrapper, hidden, residual)
        expected = Norm(model.backbone.norm_f)(hidden + residual)
        output_error = max(output_error, float((actual - expected).abs().max()))
        for factored, dense in zip(ours, theirs, strict=True):
            restored = sum(
                v[..., None] * k[..., None, :] * w[..., None, None]
                for k, v, w in zip(factored.keys, factored.values, factored.weights, strict=True)
            )
            state_error = max(state_error, float((restored - dense.ssm).abs().max()))
        if step >= 1:
            a_token, b_token = (
                int(model.lm_head(actual).argmax()),
                int(model.lm_head(expected).argmax()),
            )
            actual_tokens.append(a_token)
            upstream_tokens.append(b_token)
            tokens.append(b_token)
    return {
        "passed": output_error < 1e-4 and state_error < 1e-4 and actual_tokens == upstream_tokens,
        "hidden_max_abs_error": output_error,
        "state_max_abs_error": state_error,
        "actual_tokens": actual_tokens,
        "upstream_tokens": upstream_tokens,
        "layers": len(model.backbone.layers),
        "evaluations": 5,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths.values()
        },
        "scope": (
            "original upstream CPU preprocessing, rotary, recurrence, residual block and MLP; "
            "no fused CUDA comparison"
        ),
        "precision": (
            "FP64 oracle with upstream FP32 A preprocessing; "
            "residual FP32 rounding disabled for both oracles"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mamba3-siso-187m"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = check(args.source, args.checkpoint)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
