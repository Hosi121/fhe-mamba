#!/usr/bin/env python3
"""Exercise automatic refresh with a value-preserving, level-consuming circuit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fhemamba.packed_program import PackedProgram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = np.linspace(-0.25, 0.25, 128)
    program = PackedProgram({}, slots=1024, bound=64)
    value = program.input(inputs)
    for _ in range(45):
        value = value * np.ones(128)
    program.output(value, inputs)
    program.write(args.output / "program.txt")
    np.savez(args.output / "fixture.npz", inputs=inputs)
    manifest = {
        "schema": "fhemamba-packed-refresh-probe-v1",
        "architecture": "arithmetic-refresh-probe",
        "tokens": 0,
        "output_names": ["identity_after_refresh"],
        "program": program.summary(),
        "files_sha256": {
            name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
            for name in ("program.txt", "fixture.npz")
        },
        "scope": "automatic refresh arithmetic; not a Mamba sequence result",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
