#!/usr/bin/env python3
"""Compare with upstream's CPU reference functions from a pinned source checkout.

Only named reference function definitions are loaded, so CuteDSL, Triton and
CUDA are not required. Source hashes and the checkout revision are recorded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Optional

import torch
from einops import rearrange
from torch.nn import functional as F  # noqa: N812

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


def check(source: Path):
    files = {
        "module": source / "mamba_ssm/modules/mamba3.py",
        "rotary": source / "mamba_ssm/ops/triton/mamba3/mamba3_mimo_rotary_step.py",
        "step": source / "mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py",
    }
    namespace = {"torch": torch, "math": math, "F": F, "rearrange": rearrange, "Optional": Optional}
    reference(files["module"], "heavy_tail_activation", namespace)
    preprocess = reference(files["module"], "_preprocess", namespace, owner="Mamba3")
    rotary = reference(files["rotary"], "apply_rotary_qk_inference_reference", namespace)
    step = reference(files["step"], "selective_state_update_fused_ref_v2", namespace)
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
                    projected = mixer.in_proj(data[:, t])
                    z, x, b, c, dt, rate, trap, angle = projected.split(
                        [16, 16, 8, 8, 2, 2, 2, mixer.num_rope_angles], dim=-1
                    )
                    dt, b, c, x, z, trap, rate, angle = preprocess(
                        mixer, rate, dt, b, c, x, z, trap, angle
                    )
                    c, b, next_angle = rotary(
                        c,
                        b,
                        theirs.angle,
                        angle,
                        dt,
                        mixer.C_bias.transpose(0, 1),
                        mixer.B_bias.transpose(0, 1),
                    )
                    ones = torch.ones(1, 2, 8, dtype=torch.float64)
                    y, ssm = step(
                        theirs.ssm,
                        rate,
                        b,
                        c,
                        ones,
                        x,
                        ones,
                        None if out_norm else z,
                        dt,
                        theirs.k[:, None],
                        theirs.v,
                        trap,
                        mixer.D,
                        None if out_norm else ones,
                        compute_dtype=torch.float64,
                    )
                    if out_norm:
                        y = F.rms_norm(y[:, 0], (8,), None, mixer.norm.eps)
                        y = y * mixer.norm.weight.reshape(2, 8) * F.silu(z)
                    expected = mixer.out_proj(y.reshape(2, 16))
                    theirs = Mamba3State(next_angle, ssm, b[:, 0], x)
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
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "passed": all(
            max(r["output_max_abs_error"], r["state_max_abs_error"]) < 1e-7 for r in results
        ),
        "upstream_commit": revision,
        "tolerance": 1e-7,
        "precision_note": "upstream preprocessing casts A to float32; our oracle keeps float64",
        "source_sha256": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files.values()
        },
        "cases": results,
        "scope": "upstream CPU reference parity; no fused CUDA kernel validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.source)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
