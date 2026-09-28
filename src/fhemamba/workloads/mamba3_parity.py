#!/usr/bin/env python3
"""Compare with upstream's CPU reference functions from a pinned source checkout.

Only named reference function definitions are loaded, so CuteDSL, Triton and
CUDA are not required. Source hashes and the checkout revision are recorded.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F  # noqa: N812

from fhemamba.benchmarks.io import file_sha256, write_json
from fhemamba.mamba3 import Mamba3Mixer, Mamba3State, init_mamba3_state, mamba3_step
from fhemamba.tensor_ops import TensorOps


def reference(path, name, namespace, *, owner=None):
    tree = ast.parse(path.read_text())
    definitions = (
        tree.body
        if owner is None
        else next(
            node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner
        )
    )
    definition = next(
        node for node in definitions if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class UpstreamMamba3:
    """Original upstream CPU functions, shared by mixer and backbone checks.

    The oracle loads upstream arithmetic, never the local model implementation.
    Only explicitly requested definitions execute; importing GPU packages is unnecessary.
    """

    def __init__(self, source: Path):
        from einops import rearrange

        self.source, self.files = source, set()
        self.namespace = {
            "torch": torch,
            "math": math,
            "F": F,
            "rearrange": rearrange,
            "Optional": Optional,
            "Tensor": torch.Tensor,
        }
        module = "mamba_ssm/modules/mamba3.py"
        self.function(module, "heavy_tail_activation")
        self.preprocess = self.function(module, "_preprocess", owner="Mamba3")
        self.rotary = self.function(
            "mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py",
            "apply_rotary_qk_inference_reference",
        )
        self.recurrence = self.function(
            "mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py", "selective_state_update_fused_ref_v2"
        )

    def function(self, relative_path, name, *, owner=None):
        path = self.source / relative_path
        self.files.add(path)
        return reference(path, name, self.namespace, owner=owner)

    def identity(self):
        return {
            "upstream_commit": subprocess.check_output(
                ["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_sha256": {
                str(path.relative_to(self.source)): file_sha256(path) for path in sorted(self.files)
            },
        }

    def step(self, mixer, input_states, state):
        h, p, n, a = mixer.nheads, mixer.headdim, mixer.d_state, mixer.num_rope_angles
        z, x, b, c, dt, rate, trap, angle = mixer.in_proj(input_states).split(
            [h * p, h * p, n, n, h, h, h, a], dim=-1
        )
        dt, b, c, x, z, trap, rate, angle = self.preprocess(
            mixer, rate, dt, b, c, x, z, trap, angle
        )
        c, b, next_angle = self.rotary(
            c, b, state.angle, angle, dt, mixer.C_bias.transpose(0, 1), mixer.B_bias.transpose(0, 1)
        )
        ones = torch.ones(1, h, p, dtype=torch.float64)
        out_norm = mixer.is_outproj_norm
        y, ssm = self.recurrence(
            state.ssm,
            rate,
            b,
            c,
            ones,
            x,
            ones,
            None if out_norm else z,
            dt,
            state.k[:, None],
            state.v,
            trap,
            mixer.D,
            None if out_norm else ones,
            compute_dtype=torch.float64,
        )
        if out_norm:
            y = F.rms_norm(y[:, 0], (p,), None, mixer.norm.eps)
            y = y * mixer.norm.weight.reshape(h, p) * F.silu(z)
        output = mixer.out_proj(y.reshape(input_states.shape[0], h * p))
        return output, Mamba3State(next_angle, ssm, b[:, 0], x)


def check(source: Path):
    upstream = UpstreamMamba3(source)
    results = []
    with torch.no_grad():
        for fraction in (0.5, 1.0):
            for out_norm in (False, True):
                torch.manual_seed(77)
                mixer = Mamba3Mixer(
                    8,
                    d_state=8,
                    headdim=8,
                    rope_fraction=fraction,
                    is_outproj_norm=out_norm,
                    dtype=torch.float64,
                )
                mixer.B_bias.normal_()
                mixer.C_bias.normal_()
                data = torch.randn(2, 7, 8, dtype=torch.float64)
                ours, theirs = init_mamba3_state(mixer, 2), init_mamba3_state(mixer, 2)
                output_error, state_error = 0.0, 0.0
                for t in range(data.shape[1]):
                    actual, ours = mamba3_step(mixer, data[:, t], ours, TensorOps())
                    expected, theirs = upstream.step(mixer, data[:, t], theirs)
                    output_error = max(output_error, float((actual - expected).abs().max()))
                    for name in ("angle", "ssm", "k", "v"):
                        state_error = max(
                            state_error,
                            float((getattr(ours, name) - getattr(theirs, name)).abs().max()),
                        )
                results.append(
                    {
                        "rope_fraction": fraction,
                        "outproj_norm": out_norm,
                        "output_max_abs_error": output_error,
                        "state_max_abs_error": state_error,
                    }
                )
    return {
        "passed": all(
            max(r["output_max_abs_error"], r["state_max_abs_error"]) < 1e-7 for r in results
        ),
        **upstream.identity(),
        "tolerance": 1e-7,
        "precision_note": "upstream preprocessing casts A to float32; our oracle keeps float64",
        "cases": results,
        "scope": "upstream CPU reference parity; no fused CUDA kernel validation",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fhemamba workload check-mixer", description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = check(args.source)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
