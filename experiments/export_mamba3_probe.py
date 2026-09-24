#!/usr/bin/env python3
"""Export a complete synthetic Mamba-3 SISO mixer to shared packed arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from fhemamba.mamba3 import Mamba3Mixer, Mamba3State, init_mamba3_state, mamba3_step
from fhemamba.packed_program import Calibration, PackedProgram
from fhemamba.tensor_ops import TensorOps


def export_probe(output: Path, *, tokens=4, seed=19, d_model=32, d_state=16, headdim=32):
    if tokens < 2 or tokens > 16:
        raise ValueError("the small probe requires between 2 and 16 recurrent steps")
    output.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        mixer = Mamba3Mixer(d_model, d_state=d_state, headdim=headdim, dtype=torch.float64).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    calibration = torch.rand(1, 128, d_model, generator=generator, dtype=torch.float64) - 0.5
    inputs = torch.rand(1, tokens, d_model, generator=generator, dtype=torch.float64) - 0.5
    recorder = Calibration()
    with torch.no_grad():
        mixer(calibration, ops=recorder)
        polynomial_ops, recipes = recorder.fit(tolerance=1e-6, max_degree=511)
        # Validate on calibration data without modifying the frozen domains.
        mixer(calibration, ops=polynomial_ops)
        exact_state, poly_state = init_mamba3_state(mixer), init_mamba3_state(mixer)
        slots = 1 << max(10, (mixer.d_inner * d_state - 1).bit_length())
        program = PackedProgram(polynomial_ops.polynomials, slots=slots, bound=64)
        encrypted_state = Mamba3State(
            *(program.constant(getattr(poly_state, f.name)) for f in fields(poly_state))
        )
        errors = []
        for token in range(tokens):
            exact, exact_state = mamba3_step(mixer, inputs[:, token], exact_state, TensorOps())
            polynomial, poly_state = mamba3_step(
                mixer, inputs[:, token], poly_state, polynomial_ops
            )
            encrypted, encrypted_state = mamba3_step(
                mixer, program.input(inputs[:, token]), encrypted_state, program
            )
            np.testing.assert_allclose(encrypted.value, polynomial.numpy(), rtol=1e-10, atol=1e-10)
            program.output(encrypted, exact)
            output_names = [f"t{token}.output"]
            for f in fields(exact_state):
                program.output(getattr(encrypted_state, f.name), getattr(exact_state, f.name))
                output_names.append(f"t{token}.{f.name}")
            errors.append(float((polynomial - exact).abs().max()))
            if token == 0:
                names = output_names
            else:
                names.extend(output_names)
    program_path = output / "program.txt"
    program.write(program_path)
    weights = {name: value.detach().numpy() for name, value in mixer.state_dict().items()}
    np.savez(
        output / "fixture.npz", **weights, calibration=calibration.numpy(), inputs=inputs.numpy()
    )
    report = {
        "schema": "fhemamba-mamba3-probe-v1",
        "architecture": "mamba3",
        "variant": "siso",
        "weights": "synthetic-random",
        "seed": seed,
        "d_model": d_model,
        "d_state": d_state,
        "headdim": headdim,
        "tokens": tokens,
        "calibration_tokens": 128,
        "calibration_and_evaluation_inputs_disjoint": True,
        "program": program.summary(),
        "polynomials": recipes,
        "output_names": names,
        "polynomial_output_errors_vs_exact": errors,
        "files_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("program.txt", "fixture.npz")
        },
        "scope": "one complete SISO mixer; no language-model quality or security claim",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    report = export_probe(args.output, tokens=args.tokens, seed=args.seed)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("architecture", "tokens", "program", "polynomial_output_errors_vs_exact")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
