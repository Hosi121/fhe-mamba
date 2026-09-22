#!/usr/bin/env python3
"""Reproduce algebra gates on real Mamba-2 factors and synthetic phase dynamics.

This is a CPU/float64 reassociation probe, not an encrypted benchmark or a
Mamba-3 checkpoint evaluation. The existing reference supplies real factors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()

import torch  # noqa: E402
from fhemamba.artifacts import current_git_commit  # noqa: E402
from fhemamba.ops import Exact  # noqa: E402
from fhemamba.reference import model_forward  # noqa: E402
from fhemamba.ssm_algebra import DeferredMamba2State, cubic_phase  # noqa: E402

from fhemamba import __version__  # noqa: E402

TEXT = (
    "A private language model must keep its recurrent state encrypted while processing "
    "a long sequence. Reassociating an outer product and a dot product gives the same "
    "answer in exact arithmetic, but approximate encryption changes the error budget. "
    "We compare the two schedules before designing a ciphertext layout. "
) * 4


class CaptureFactors(Exact):
    def __init__(self):
        self.values = {}

    def checkpoint(self, x, site):
        if site[1] in {"conv_silu_out", "dt_out", "decay_output"}:
            self.values[site] = x.detach().double().clone()
        return x


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mamba2-130m-hf"))
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or any(window < 1 for window in args.windows):
        parser.error("tokens and windows must be positive")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True).float().eval()
    )
    ids = tokenizer(TEXT, return_tensors="pt").input_ids[:, : args.tokens]
    capture = CaptureFactors()
    model_forward(model, ids, capture, scan="chunked", output_logits=False)
    rows = []
    passed = True
    for layer, block in enumerate(model.backbone.layers):
        mixer = block.mixer
        heads, channels, size = mixer.num_heads, mixer.head_dim, mixer.ssm_state_size
        packed = capture.values[layer, "conv_silu_out"][0]
        x, b, c = packed.split(
            [mixer.intermediate_size, mixer.n_groups * size, mixer.n_groups * size], dim=-1
        )
        length = x.shape[0]
        u = capture.values[layer, "dt_out"][0, :, :, None] * x.reshape(length, heads, channels)
        b = b.reshape(length, mixer.n_groups, size).repeat_interleave(heads // mixer.n_groups, 1)
        c = c.reshape(length, mixer.n_groups, size).repeat_interleave(heads // mixer.n_groups, 1)
        decay = capture.values[layer, "decay_output"][0]
        dense = u.new_zeros(heads, channels, size)
        outputs = []
        for t in range(length):
            dense = decay[t, :, None, None] * dense + u[t, :, :, None] * b[t, :, None, :]
            outputs.append(torch.einsum("hpn,hn->hp", dense, c[t]))
        expected = torch.stack(outputs)
        for window in args.windows:
            deferred = DeferredMamba2State(torch.zeros_like(dense), window)
            actual = torch.stack([deferred.step(decay[t], u[t], b[t], c[t]) for t in range(length)])
            output_error = float((actual - expected).abs().max())
            state_error = float((deferred.materialize() - dense).abs().max())
            row_passed = bool(
                torch.allclose(actual, expected, atol=1e-9, rtol=1e-11)
                and torch.allclose(deferred.materialize(), dense, atol=1e-9, rtol=1e-11)
            )
            passed &= row_passed
            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "tokens": length,
                    "output_max_abs_error": output_error,
                    "state_max_abs_error": state_error,
                    "dense_state_elements": dense.numel(),
                    "extra_factor_elements_upper_bound": heads
                    * (1 + window * (channels + size + 1)),
                    "flushes": deferred.flushes,
                    "pending_updates": len(deferred.pending),
                    "passed": row_passed,
                }
            )

    angles = [0.1, 0.25, 0.5, 1.0, math.sqrt(3), 2.0]
    phase_rows = []
    for angle in angles:
        phase = complex(cubic_phase(torch.tensor(angle, dtype=torch.float64)))
        phase_rows.append(
            {
                "angle": angle,
                "modulus": abs(phase),
                "phase_error_radians": abs(math.atan2(phase.imag, phase.real) - angle),
                "retained_magnitude_after_1024_steps": abs(phase) ** 1024,
                "contractive_interval": angle <= math.sqrt(3),
            }
        )
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__),
        root / "fhemamba/src/fhemamba/ssm_algebra.py",
        root / "fhemamba/src/fhemamba/reference.py",
    ]
    weights = sorted(args.checkpoint.glob("*.safetensors")) + sorted(args.checkpoint.glob("*.bin"))
    report = {
        "stage": "ssm-algebra-report",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in source_paths},
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": {p.name: sha256(p) for p in weights},
        "input_token_ids": ids[0].tolist(),
        "dtype": "float64 factors from float32 reference",
        "rows": rows,
        "cubic_phase_negative_control": phase_rows,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "native_execution_measured": False,
            "claim": "Exact recurrence reassociation on one real factor stream per layer; "
            "no FHE latency/precision or Mamba-3 model-quality claim.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "rows": len(rows),
                "tokens": ids.shape[1],
                "max_output_error": max(row["output_max_abs_error"] for row in rows),
                "max_state_error": max(row["state_max_abs_error"] for row in rows),
            }
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
