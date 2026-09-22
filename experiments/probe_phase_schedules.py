#!/usr/bin/env python3
"""Record constant-angle and switching counterexamples for polynomial phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba import __version__
from fhemamba.artifacts import current_git_commit
from fhemamba.phase_algebra import shear_rotate
from fhemamba.ssm_algebra import cubic_phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    angle = torch.tensor(0.5, dtype=torch.float64)
    pair = torch.tensor([1.0, 0.0], dtype=torch.float64)
    norms = []
    energy_errors = []
    for _ in range(1024):
        pair = shear_rotate(pair, angle)
        norms.append(float(pair.norm()))
        energy_errors.append(
            float(abs(pair[0].square() + (1 - angle.square() / 4) * pair[1].square() - 1))
        )
    identity = torch.eye(2, dtype=torch.float64)
    a = shear_rotate(identity, angle).T
    b = shear_rotate(identity, torch.tensor(0.1, dtype=torch.float64)).T
    period = torch.linalg.matrix_power(b, 16) @ torch.linalg.matrix_power(a, 3)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "src/fhemamba/phase_algebra.py",
        root / "src/fhemamba/ssm_algebra.py",
    ]
    report = {
        "stage": "phase-schedule-report",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "status": "unpromoted",
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "constant_angle": {
            "angle": 0.5,
            "steps": 1024,
            "cubic_retained_magnitude": abs(complex(cubic_phase(angle))) ** 1024,
            "shear_norm_min": min(norms),
            "shear_norm_max": max(norms),
            "shear_norm_final": norms[-1],
            "shear_modified_energy_max_error": max(energy_errors),
            "shear_accumulated_frequency_error_radians": (2 * math.asin(0.25) - 0.5) * 1024,
        },
        "switching_counterexample": {
            "period": [{"angle": 0.5, "steps": 3}, {"angle": 0.1, "steps": 16}],
            "period_matrix": period.tolist(),
            "determinant": float(torch.linalg.det(period)),
            "trace": float(torch.trace(period)),
            "spectral_radius": float(torch.linalg.eigvals(period).abs().max()),
        },
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "native_execution_measured": False,
            "claim": "Float64 phase-algebra screen: constant-angle amplitude preservation "
            "does not imply correct phase or stability under switching. "
            "No checkpoint, CKKS precision or performance claim.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "constant_angle": report["constant_angle"],
                "switching_counterexample": report["switching_counterexample"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
