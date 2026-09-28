#!/usr/bin/env python3
"""Export selected node references from a frozen Mamba-3 generation circuit.

No polynomial is refitted. The original graph is rebuilt using its frozen
coefficients, and its canonical node-prefix hash must be checked against the
original program before these diagnostic references are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

from fhemamba.benchmarks.io import file_sha256 as digest
from fhemamba.mamba3 import Mamba3State, RotaryPhase
from fhemamba.mamba3_lm import Mamba3LM
from fhemamba.ops import ChebPoly
from fhemamba.packed_program import PackedProgram

MAGIC = b"FHEMDG01"


class ReferenceProgram(PackedProgram):
    def __init__(self, polynomials, output, steps, all_nodes=False, node_ranges=(), original=None):
        super().__init__(polynomials, slots=32768, bound=1e8, extended=True)
        self.step, self.steps, self.all_nodes = -1, set(steps), all_nodes
        self.node_ranges = tuple(node_ranges)
        self.original = original.open() if original else None
        if self.original:
            next(self.original)
        self.bound_rebindings, self.max_relative_bound_drift = 0, 0.0
        self.references = {}
        self.node_hash = hashlib.sha256()
        self.stream = output.open("wb")
        self.stream.write(MAGIC + struct.pack("<Q", 0))

    def record(self, value, name):
        if self.step not in self.steps:
            return value
        if self.node_ranges and not any(a <= value.index < b for a, b in self.node_ranges):
            return value
        if value.index not in self.references:
            self.references[value.index] = {
                "step": self.step,
                "names": [],
                "size": value.value.size,
            }
            self.stream.write(struct.pack("<II", value.index, value.value.size))
            self.stream.write(np.asarray(value.value, dtype="<f8").tobytes())
        self.references[value.index]["names"].append(name)
        return value

    def node(self, operation, parents, data, value):
        out = super().node(operation, parents, data, value)
        op, ps, size, constants = self.nodes[out.index]
        fields = [
            op,
            str(size),
            format(self.node_bounds[out.index], ".17g"),
            str(len(ps)),
            *map(str, ps),
            str(len(constants)),
            *(format(x, ".17g") for x in constants),
        ]
        if self.original:
            expected = next(self.original).split()
            if (
                len(expected) != len(fields)
                or fields[:2] + fields[3:] != expected[:2] + expected[3:]
            ):
                raise ValueError(f"frozen arithmetic/constant mismatch at node {out.index}")
            bound, frozen_bound = float(fields[2]), float(expected[2])
            if not math.isclose(bound, frozen_bound, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"reference refresh-bound drift at node {out.index}")
            if fields[2] != expected[2]:
                self.bound_rebindings += 1
                self.max_relative_bound_drift = max(
                    self.max_relative_bound_drift, abs(bound - frozen_bound) / frozen_bound
                )
            # Floating-point reference reductions may differ by a few ulps.
            # The executed circuit retains its original public bound verbatim;
            # constants, operators, widths and parents must match exactly.
            fields[2] = expected[2]
            self.node_bounds[out.index] = frozen_bound
        self.node_hash.update((" ".join(fields) + "\n").encode())
        if self.all_nodes:
            self.record(out, operation)
        return out

    def nonlinear(self, x, kind, site, parameter=0.0):
        self.record(x, f"{site[0]}:{site[1]}.input")
        return self.record(
            super().nonlinear(x, kind, site, parameter), f"{site[0]}:{site[1]}.output"
        )

    def checkpoint(self, value, site):
        return self.record(value, f"{site[0]}:{site[1]}")

    def close(self):
        self.stream.seek(8)
        self.stream.write(struct.pack("<Q", len(self.references)))
        self.stream.close()
        if self.original:
            self.original.close()


@torch.no_grad()
def export(
    checkpoint,
    manifest_path,
    fixture_path,
    output,
    steps,
    *,
    all_nodes=False,
    node_ranges=(),
    original=None,
):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "fhemamba-mamba3-lm-v1" or (
        manifest["state_representation"],
        manifest["rotary_representation"],
    ) != ("tiled", "phasor"):
        raise ValueError("diagnostic exporter requires a frozen tiled/phasor LM payload")
    if not steps or min(steps) < 0 or max(steps) >= manifest["tokens"]:
        raise ValueError("steps must be zero-based server evaluations within the payload")
    if any(a < 0 or b <= a for a, b in node_ranges):
        raise ValueError("node ranges must be nonempty half-open intervals")
    for name, expected in manifest["checkpoint"]["files_sha256"].items():
        if digest(checkpoint / name) != expected:
            raise ValueError(f"checkpoint digest mismatch: {name}")
    if digest(fixture_path) != manifest["files_sha256"]["fixture.npz"]:
        raise ValueError("fixture digest mismatch")
    if original and digest(original) != manifest["files_sha256"]["program.txt"]:
        raise ValueError("original program digest mismatch")
    output.mkdir(parents=True, exist_ok=False)
    polys = {
        (int(k.split(":", 1)[0]), k.split(":", 1)[1]): ChebPoly(
            tuple(p["coefficients"]), p["lo"], p["hi"]
        )
        for k, p in manifest["polynomials"].items()
    }
    model = Mamba3LM.from_pretrained(checkpoint)
    if manifest["layers"] != len(model.backbone.layers):
        raise ValueError("diagnostic exporter requires the complete backbone")
    program = ReferenceProgram(
        polys, output / "references.bin", steps, all_nodes, node_ranges, original
    )
    states = [
        Mamba3State(
            RotaryPhase(program.constant(s.angle.cosine), program.constant(s.angle.sine)),
            program.recurrent_state(s.ssm),
            program.constant(s.k),
            program.constant(s.v),
        )
        for s in model.initial_states(factored=False, rotary="phasor")
    ]
    ids = manifest["prompt_ids"] + manifest["polynomial_token_ids"]
    endpoints, hidden = [], None
    try:
        with np.load(fixture_path, allow_pickle=False) as fixture:
            for step in range(max(steps) + 1):
                program.step = step
                embedding = model.backbone.embedding.weight[ids[step]][None]
                x = (
                    program.input(embedding)
                    if step < len(manifest["prompt_ids"])
                    else program.client_input(hidden, embedding)
                )
                hidden, states = model.step(x, states, program)
                np.testing.assert_allclose(
                    hidden.value.ravel(), fixture["polynomial_hidden"][step], atol=1e-9, rtol=1e-9
                )
                program.record(hidden, "final_hidden")
                for layer, state in enumerate(states):
                    for name, value in [
                        ("phase_cos", state.angle.cosine),
                        ("phase_sin", state.angle.sine),
                        ("previous_key", state.k),
                        ("previous_value", state.v),
                    ]:
                        program.record(value, f"{layer}:carry.{name}")
                    for tile, value in enumerate(state.ssm.blocks):
                        program.record(value, f"{layer}:carry.ssm.{tile}")
                endpoints.append(hidden.index)
                print(
                    f"reference_step={step} nodes={len(program.nodes)} "
                    f"references={len(program.references)}",
                    flush=True,
                )
    finally:
        program.close()
    metadata = {
        "schema": "fhemamba-packed-diagnostic-v1",
        "diagnostic_only": True,
        "manifest_sha256": digest(manifest_path),
        "program_sha256": manifest["files_sha256"]["program.txt"],
        "references_sha256": digest(output / "references.bin"),
        "node_prefix_sha256": program.node_hash.hexdigest(),
        "node_count": len(program.nodes),
        "steps": sorted(set(steps)),
        "all_nodes": all_nodes,
        "node_ranges": list(node_ranges),
        "bound_rebindings": program.bound_rebindings,
        "max_relative_bound_drift": program.max_relative_bound_drift,
        "output_nodes": endpoints,
        "references": program.references,
    }
    (output / "references.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(prog="fhemamba diagnose export", description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument(
        "--program", type=Path, help="bind the original public bounds and verify every node"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
        help="zero-based server evaluations; with a two-token prompt selection k uses step k",
    )
    p.add_argument("--all-nodes", action="store_true")
    p.add_argument(
        "--node-range",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("FIRST", "STOP"),
        help="restrict observations to a half-open original node-ID interval; repeatable",
    )
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    if a.threads < 1:
        p.error("threads must be positive")
    torch.set_num_threads(a.threads)
    export(
        a.checkpoint,
        a.manifest,
        a.fixture,
        a.output,
        a.steps,
        all_nodes=a.all_nodes,
        node_ranges=a.node_range,
        original=a.program,
    )


if __name__ == "__main__":
    main()
