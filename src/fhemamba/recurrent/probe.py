#!/usr/bin/env python3
"""Capture a trained recurrence and compare bounded and exact-history storage.

The FHE payload starts at the state-update boundary: projections, nonlinearities,
rotary phase and token selection are performed by the plaintext trace producer.
This measures recurrent arithmetic, not end-to-end encrypted generation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fhemamba import mamba3
from fhemamba.benchmarks.io import file_identity, repository_root, write_json
from fhemamba.mamba3_lm import Mamba3LM
from fhemamba.packed_program import PackedProgram
from fhemamba.tensor_ops import TensorOps

INPUTS = ("b", "c", "v", "decay", "lag", "write")


@torch.no_grad()
def capture(checkpoint, tokenizer_path, output, *, steps=64, layer=0, prompt="The capital"):
    """Record one layer during actual greedy *plaintext* model generation."""
    from transformers import AutoTokenizer

    if not 1 <= steps <= 1024:
        raise ValueError("capture requires 1..1024 steps")
    pins = json.loads((repository_root() / "config/mamba3-reproduction.json").read_text())
    for directory, section in ((checkpoint, "checkpoint"), (tokenizer_path, "tokenizer")):
        for name, expected in pins[section]["files_sha256"].items():
            if file_identity(directory / name)["sha256"] != expected:
                raise ValueError(f"{section} differs from pinned file: {name}")
    model = Mamba3LM.from_pretrained(checkpoint)
    if not 0 <= layer < len(model.backbone.layers):
        raise ValueError("layer must identify an existing model layer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids or len(prompt_ids) > steps:
        raise ValueError("prompt must contain 1..steps tokens")
    states = model.initial_states(factored=False)
    original = mamba3.mamba3_state_step
    rows = {name: [] for name in (*INPUTS, "reference_y")}
    calls = 0

    def record(state, b, c, v, decay, lag, write, skip, ops):
        nonlocal calls
        selected = calls % len(states) == layer
        calls += 1
        result, carry = original(state, b, c, v, decay, lag, write, skip, ops)
        if selected:
            for name, value in zip(INPUTS, (b, c, v, decay, lag, write), strict=True):
                rows[name].append(value.detach().cpu().numpy().copy())
            rows["reference_y"].append(result.detach().cpu().numpy().copy())
        return result, carry

    token_ids = list(prompt_ids)
    with patch.object(mamba3, "mamba3_state_step", record):
        for step in range(steps):
            embedding = model.backbone.embedding.weight[token_ids[step]][None]
            hidden, states = model.step(embedding, states)
            if step >= len(prompt_ids) - 1:
                token_ids.append(int(model.lm_head(hidden).argmax(-1).item()))
    if calls != steps * len(states) or any(len(value) != steps for value in rows.values()):
        raise RuntimeError("unexpected recurrence capture count")
    output.mkdir(parents=True, exist_ok=False)
    mixer = model.backbone.layers[layer].mixer
    np.savez_compressed(
        output / "trace.npz",
        **{name: np.stack(value) for name, value in rows.items()},
        skip_weight=mixer.D.detach().cpu().numpy(),
    )
    manifest = {
        "schema": "fhemamba-recurrence-trace-v1",
        "checkpoint": pins["checkpoint"],
        "tokenizer": pins["tokenizer"],
        "layer": layer,
        "steps": steps,
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "consumed_token_ids": token_ids[:steps],
        "plaintext_generated_token_ids": token_ids[len(prompt_ids) :],
        "heads": mixer.nheads,
        "channels": mixer.headdim,
        "state_size": mixer.d_state,
        "trace_sha256": file_identity(output / "trace.npz")["sha256"],
        "scope": "plaintext autoregressive trace; exact nonlinearities; one recurrence boundary",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def export(trace, output, *, steps, representation, slots=32768):
    """Export the same encrypted boundary inputs for either state representation."""
    manifest = json.loads((trace / "manifest.json").read_text())
    if manifest.get("schema") != "fhemamba-recurrence-trace-v1":
        raise ValueError("unsupported recurrence trace")
    if file_identity(trace / "trace.npz")["sha256"] != manifest["trace_sha256"]:
        raise ValueError("recurrence trace digest differs")
    if not 1 <= steps <= manifest["steps"] or representation not in ("tiled", "factored"):
        raise ValueError("invalid step count or state representation")
    with np.load(trace / "trace.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in (*INPUTS, "reference_y", "skip_weight")}
    heads, channels, states = (manifest[name] for name in ("heads", "channels", "state_size"))
    program = PackedProgram({}, slots=slots, bound=1e8, extended=True)
    angle = program.constant(np.zeros((1, heads, 1)))
    if representation == "tiled":
        carry = mamba3.Mamba3State(
            angle,
            program.recurrent_state(np.zeros((1, heads, channels, states))),
            program.constant(np.zeros((1, heads, states))),
            program.constant(np.zeros((1, heads, channels))),
        )
    else:
        carry = mamba3.FactoredMamba3State(angle)
    node_ranges, names, oracle_errors = [], [], []
    reference_state = np.zeros((1, heads, channels, states))
    previous_key = np.zeros((1, heads, states))
    previous_value = np.zeros((1, heads, channels))
    for step in range(steps):
        start = len(program.nodes)
        values = {name: program.input(arrays[name][step]) for name in INPUTS}
        skip = values["v"] * arrays["skip_weight"][:, None]
        result, carry = mamba3.mamba3_state_step(carry, **values, skip=skip, ops=program)
        # Independent dense NumPy oracle retains the original outer-product order.
        b, c, v, decay, lag, write = (arrays[name][step] for name in INPUTS)
        reference_state = reference_state * decay[..., None, None]
        reference_state += (
            previous_value[..., None] * previous_key[..., None, :] * lag[..., None, None]
        )
        reference_state += v[..., None] * b[..., None, :] * write[..., None, None]
        reference_y = (reference_state * c[..., None, :]).sum(-1) + v * arrays["skip_weight"][
            :, None
        ]
        np.testing.assert_allclose(reference_y, arrays["reference_y"][step], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(result.value, reference_y, rtol=1e-10, atol=1e-9)
        oracle_errors.append(float(np.max(np.abs(result.value - reference_y))))
        previous_key, previous_value = b, v
        # Boundary inputs are offline, so a readout has no future consumer.
        # Retain *every* readout or the native liveness pass would remove it.
        program.output(result, reference_y)
        names.append({"name": f"step{step + 1}.readout", "step": step + 1, "node": result.index})
        state_values = (
            (*carry.ssm.blocks, carry.k, carry.v)
            if representation == "tiled"
            else (*carry.keys, *carry.values, *carry.weights)
        )
        node_ranges.append(
            {
                "step": step + 1,
                "start": start,
                "stop": len(program.nodes),
                "operations": dict(Counter(node[0] for node in program.nodes[start:])),
                "carried_ciphertexts": len(state_values),
                "carried_coordinates": sum(value.value.size for value in state_values),
            }
        )
    if representation == "tiled":
        offset = 0
        for index, block in enumerate(carry.ssm.blocks):
            block_heads = block.shape[1]
            program.output(block, reference_state[:, offset : offset + block_heads])
            names.append({"name": f"final.state{index}", "step": steps, "node": block.index})
            offset += block_heads
    output.mkdir(parents=True, exist_ok=False)
    program.write(output / "program.txt")
    np.savez_compressed(output / "fixture.npz", reference_y=arrays["reference_y"][:steps])
    report = {
        "schema": "fhemamba-recurrence-probe-v1",
        "architecture": "mamba3",
        "representation": representation,
        "tokens": steps,
        "layer": manifest["layer"],
        "geometry": {"heads": heads, "channels": channels, "state_size": states},
        "trace_sha256": manifest["trace_sha256"],
        "trace_manifest": manifest,
        "program": program.summary(),
        "output_names": names,
        "steps": node_ranges,
        "max_plaintext_error_vs_dense": max(oracle_errors),
        "refresh_bounds": "trace maximum times eight, minimum one; not certified",
        "scope": "recurrence only; precomputed boundary inputs; no encrypted token feedback",
        "memory_scope": "carried state excludes validation outputs, keys and the unrolled DAG",
        "files_sha256": {
            name: file_identity(output / name)["sha256"] for name in ("program.txt", "fixture.npz")
        },
    }
    write_json(output / "manifest.json", report)
    return report


@torch.no_grad()
def audit_domains(checkpoint, trace, polynomial_manifest):
    """Replay the captured token sequence exactly and audit existing fit domains.

    This is a necessary domain screen, not polynomial/FHE error qualification.
    No approximation interval or coefficient is changed by the audit.
    """
    captured = json.loads((trace / "manifest.json").read_text())
    frozen = json.loads(polynomial_manifest.read_text())
    if captured.get("schema") != "fhemamba-recurrence-trace-v1":
        raise ValueError("unsupported recurrence trace")
    if any(
        captured["checkpoint"][key] != frozen["checkpoint"][key] for key in ("repo_id", "revision")
    ) or any(
        captured["checkpoint"]["files_sha256"][name] != frozen["checkpoint"]["files_sha256"][name]
        for name in ("config.json", "pytorch_model.bin")
    ):
        raise ValueError("polynomial manifest and trace use different checkpoints")
    for name, expected in captured["checkpoint"]["files_sha256"].items():
        if file_identity(checkpoint / name)["sha256"] != expected:
            raise ValueError(f"checkpoint differs from captured model: {name}")
    model = Mamba3LM.from_pretrained(checkpoint)
    sites = {}

    class Audit(TensorOps):
        step = 0

        def nonlinear(self, x, kind, site, parameter=0.0):
            name = f"{site[0]}:{site[1]}"
            recipe = frozen["polynomials"][name]
            if recipe["kind"] != kind or recipe["parameter"] != parameter:
                raise ValueError(f"polynomial operator differs at {name}")
            low, high = float(x.min()), float(x.max())
            if not torch.isfinite(x).all():
                raise ValueError(f"non-finite plaintext input at {name}")
            row = sites.setdefault(
                name,
                {
                    "kind": kind,
                    "degree": recipe["degree"],
                    "frozen_lo": recipe["lo"],
                    "frozen_hi": recipe["hi"],
                    "observed_lo": low,
                    "observed_hi": high,
                    "first_outside_step": None,
                    "outside_steps": 0,
                },
            )
            row["observed_lo"] = min(row["observed_lo"], low)
            row["observed_hi"] = max(row["observed_hi"], high)
            if low < recipe["lo"] or high > recipe["hi"]:
                if row["first_outside_step"] is None:
                    row["first_outside_step"] = self.step
                row["outside_steps"] += 1
            return super().nonlinear(x, kind, site, parameter)

    ops, states = Audit(), model.initial_states(factored=False)
    for step, token in enumerate(captured["consumed_token_ids"], start=1):
        ops.step = step
        _, states = model.step(model.backbone.embedding.weight[token][None], states, ops)
    outside = [row["first_outside_step"] for row in sites.values() if row["outside_steps"]]
    return {
        "schema": "fhemamba-recurrence-domain-audit-v1",
        "steps": len(captured["consumed_token_ids"]),
        "trace_manifest_sha256": file_identity(trace / "manifest.json")["sha256"],
        "polynomial_manifest_sha256": file_identity(polynomial_manifest)["sha256"],
        "sites": sites,
        "sites_outside": len(outside),
        "first_outside_step": min(outside) if outside else None,
        "all_observed_inputs_within_domains": not outside,
        "scope": "exact plaintext replay; necessary domain screen; not an FHE accuracy result",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fhemamba recurrent probe", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("capture")
    record.add_argument("--checkpoint", type=Path, required=True)
    record.add_argument("--tokenizer", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--steps", type=int, default=64)
    record.add_argument("--layer", type=int, default=0)
    record.add_argument("--prompt", default="The capital")
    compile_ = commands.add_parser("export")
    compile_.add_argument("--trace", type=Path, required=True)
    compile_.add_argument("--output", type=Path, required=True)
    compile_.add_argument("--steps", type=int, required=True)
    compile_.add_argument("--representation", choices=("tiled", "factored"), required=True)
    compile_.add_argument("--slots", type=int, default=32768)
    audit = commands.add_parser("audit-domains")
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--trace", type=Path, required=True)
    audit.add_argument("--polynomial-manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.command == "capture":
        result = capture(
            args.checkpoint,
            args.tokenizer,
            args.output,
            steps=args.steps,
            layer=args.layer,
            prompt=args.prompt,
        )
        print(json.dumps({key: result[key] for key in ("steps", "layer", "trace_sha256")}))
    elif args.command == "export":
        result = export(
            args.trace,
            args.output,
            steps=args.steps,
            representation=args.representation,
            slots=args.slots,
        )
        print(json.dumps({key: result[key] for key in ("tokens", "representation", "program")}))
    else:
        result = audit_domains(args.checkpoint, args.trace, args.polynomial_manifest)
        write_json(args.output, result)
        print(
            json.dumps(
                {key: result[key] for key in ("steps", "sites_outside", "first_outside_step")}
            )
        )


if __name__ == "__main__":
    main()
