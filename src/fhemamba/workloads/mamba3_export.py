#!/usr/bin/env python3
"""Compile a trained Mamba-3 backbone and real client feedback to packed CKKS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fhemamba.benchmarks.io import file_sha256 as digest
from fhemamba.benchmarks.io import repository_root
from fhemamba.inputs import generation_length, token_ids
from fhemamba.mamba3 import FactoredMamba3State, Mamba3State, RotaryPhase
from fhemamba.mamba3_lm import Mamba3LM
from fhemamba.packed_program import Calibration, PackedProgram
from fhemamba.workloads.client_head import share_client_head, write_client_head

MODEL_ID = "state-spaces/mamba3-siso-187m"
MODEL_REVISION = "6792c27c00f3bb41506db1066dcd1c51bb0f4b02"
CALIBRATION_PROMPTS = (
    "A country",
    "The city",
    "In Europe",
    "This country",
    "Once upon",
    "Paris is",
)


def prefix_payload(source, output, new_tokens):
    """Derive an exact prefix without refitting coefficients or recomputing bounds."""
    from collections import Counter

    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema") != "fhemamba-mamba3-lm-v1":
        raise ValueError("prefix source must be a language-model payload")
    if not 1 <= new_tokens <= manifest["generated_tokens"]:
        raise ValueError("prefix token count must be within the source generation")
    for name, expected in manifest["files_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"prefix source digest differs: {name}")
    evaluations = len(manifest["prompt_ids"]) + new_tokens - 1
    with (source / "program.txt").open() as stream:
        header = stream.readline().split()
        for _ in range(int(header[3])):
            stream.readline()
        outputs = [stream.readline() for _ in range(evaluations)]
    stop = int(outputs[-1].split(maxsplit=1)[0]) + 1
    output.mkdir(parents=True, exist_ok=False)
    operations = Counter()
    with (source / "program.txt").open() as src, (output / "program.txt").open("w") as dest:
        src.readline()
        dest.write(" ".join([*header[:3], str(stop), str(evaluations)]) + "\n")
        for _ in range(stop):
            line = src.readline()
            operations[line.split(maxsplit=1)[0]] += 1
            dest.write(line)
        dest.writelines(outputs)
    share_client_head(source / "client_head.f32", output / "client_head.f32")
    with np.load(source / "fixture.npz", allow_pickle=False) as fixture:
        arrays = {
            key: fixture[key][: evaluations if key.endswith("_hidden") else new_tokens]
            for key in fixture.files
            if key != "prompt_ids"
        }
        np.savez(output / "fixture.npz", prompt_ids=fixture["prompt_ids"], **arrays)
    manifest.update(
        generated_tokens=new_tokens,
        tokens=evaluations,
        exact_token_ids=manifest["exact_token_ids"][:new_tokens],
        polynomial_token_ids=manifest["polynomial_token_ids"][:new_tokens],
        output_names=manifest["output_names"][:evaluations],
        prefix_source_manifest_sha256=digest(source / "manifest.json"),
        polynomial_hidden_max_abs_error=float(
            np.max(np.abs(arrays["exact_hidden"] - arrays["polynomial_hidden"]))
        ),
    )
    # Text for the entire source would falsely describe a shorter generation.
    manifest.pop("reference_text", None)
    manifest["program"].update(nodes=stop, outputs=evaluations, operations=dict(operations))
    manifest["files_sha256"] = {
        name: digest(output / name) for name in ("program.txt", "fixture.npz", "client_head.f32")
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@torch.no_grad()
def export(
    checkpoint,
    tokenizer_path,
    output,
    *,
    prompt=None,
    input_ids=None,
    new_tokens=4,
    probe_layers=None,
    state_representation="factored",
    rotary="angle",
    calibration_tokens=None,
    calibration_margin=0.2,
    positive_lower_fraction=0.5,
    normalization_margin=None,
):
    from transformers import AutoTokenizer

    generation_length(new_tokens, input_length=1)
    if input_ids is not None and prompt is not None:
        raise ValueError("choose prompt or input_ids, not both")
    if input_ids is not None:
        input_ids = token_ids(input_ids)
        generation_length(new_tokens, input_length=len(input_ids))
    elif prompt is None:
        prompt = "The capital"
    if state_representation not in ("factored", "tiled") or rotary not in ("angle", "phasor"):
        raise ValueError("invalid state or rotary representation")
    calibration_tokens = new_tokens + 1 if calibration_tokens is None else calibration_tokens
    if not new_tokens <= calibration_tokens <= 128:
        raise ValueError("calibration length must cover generation and be at most 128 tokens")
    pins = json.loads((repository_root() / "config/mamba3-reproduction.json").read_text())[
        "checkpoint"
    ]
    if pins["repo_id"] != MODEL_ID or pins["revision"] != MODEL_REVISION:
        raise ValueError("exporter and checkpoint manifest disagree")
    for name in ("config.json", "pytorch_model.bin"):
        if digest(checkpoint / name) != pins["files_sha256"][name]:
            raise ValueError(f"checkpoint differs from the pinned trained model: {name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = Mamba3LM.from_pretrained(checkpoint)
    original_layers = len(model.backbone.layers)
    if probe_layers is not None:
        if not 1 <= probe_layers <= original_layers:
            raise ValueError("invalid probe layer count")
        model.backbone.layers = model.backbone.layers[:probe_layers]
    prompt_ids = token_ids(
        tokenizer.encode(prompt, add_special_tokens=False) if input_ids is None else input_ids,
        vocab_size=model.backbone.embedding.num_embeddings,
    )
    generation_length(new_tokens, input_length=len(prompt_ids))
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    recorder = Calibration()
    for text in CALIBRATION_PROMPTS:
        model.generate(
            tokenizer.encode(text, add_special_tokens=False),
            calibration_tokens,
            recorder,
            factored=False,
            rotary=rotary,
        )
    poly, recipes = recorder.fit(
        tolerance=1e-6,
        max_degree=1023,
        margin=calibration_margin,
        quadrature=True,
        kind_tolerances={"negative_a": 1e-4},
        positive_lower_fraction=positive_lower_fraction,
        kind_margins=None if normalization_margin is None else {"inv_sqrt": normalization_margin},
    )
    exact_ids, exact_trace = model.generate(prompt_ids, new_tokens, factored=False)
    poly_ids, poly_trace = model.generate(
        prompt_ids, new_tokens, poly, factored=state_representation == "factored", rotary=rotary
    )
    if exact_ids != poly_ids:
        raise ValueError("polynomial approximation changes greedy tokens on the chosen prompt")
    hidden_error = max(
        float((a - b).abs().max()) for a, b in zip(exact_trace, poly_trace, strict=True)
    )
    if not np.isfinite(hidden_error) or hidden_error > 0.001:
        raise ValueError(f"polynomial hidden error exceeds 0.001: {hidden_error}")
    output.mkdir(parents=True, exist_ok=False)
    program = PackedProgram(poly.polynomials, slots=32768, bound=1e8, extended=True)
    states = []
    for state in model.initial_states(factored=state_representation == "factored", rotary=rotary):
        angle = (
            RotaryPhase(program.constant(state.angle.cosine), program.constant(state.angle.sine))
            if isinstance(state.angle, RotaryPhase)
            else program.constant(state.angle)
        )
        states.append(
            FactoredMamba3State(angle)
            if state_representation == "factored"
            else Mamba3State(
                angle,
                program.recurrent_state(state.ssm),
                program.constant(state.k),
                program.constant(state.v),
            )
        )
    all_ids = prompt_ids + poly_ids
    names = []
    hidden = None
    for step in range(len(exact_trace)):
        embedding = model.backbone.embedding.weight[all_ids[step]][None]
        x = (
            program.input(embedding)
            if step < len(prompt_ids)
            else program.client_input(hidden, embedding)
        )
        hidden, states = model.step(x, states, program)
        np.testing.assert_allclose(hidden.value, poly_trace[step].numpy(), atol=1e-9, rtol=1e-9)
        program.output(hidden, exact_trace[step])
        names.append(f"step{step}.final_hidden")
        print(f"compiled_step={step + 1}/{len(exact_trace)} nodes={len(program.nodes)}", flush=True)
    program.write(output / "program.txt")
    write_client_head(output / "client_head.f32", model.lm_head.weight.detach().numpy())
    np.savez(
        output / "fixture.npz",
        prompt_ids=prompt_ids,
        exact_token_ids=exact_ids,
        polynomial_token_ids=poly_ids,
        exact_hidden=torch.cat(exact_trace).numpy(),
        polynomial_hidden=torch.cat(poly_trace).numpy(),
    )
    report = {
        "schema": "fhemamba-mamba3-lm-v1",
        "architecture": "mamba3",
        "variant": "siso",
        "checkpoint": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "files_sha256": {
                name: digest(checkpoint / name) for name in ("config.json", "pytorch_model.bin")
            },
        },
        "tokenizer_files_sha256": {
            p.name: digest(p) for p in sorted(tokenizer_path.glob("*.json"))
        },
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "add_special_tokens": False,
        "generated_tokens": new_tokens,
        "tokens": len(exact_trace),
        "layers": len(model.backbone.layers),
        "checkpoint_layers": original_layers,
        "complete_backbone": len(model.backbone.layers) == original_layers,
        "exact_token_ids": exact_ids,
        "polynomial_token_ids": poly_ids,
        "reference_text": tokenizer.decode(prompt_ids + exact_ids),
        "calibration_prompts": list(CALIBRATION_PROMPTS),
        "calibration_excludes_evaluation_prompt": (
            prompt not in CALIBRATION_PROMPTS
            if input_ids is None
            else all(
                prompt_ids != tokenizer.encode(text, add_special_tokens=False)
                for text in CALIBRATION_PROMPTS
            )
        ),
        "calibration_generated_tokens": calibration_tokens,
        "calibration_margin": calibration_margin,
        "calibration_positive_lower_fraction": positive_lower_fraction,
        "calibration_normalization_margin": normalization_margin,
        "polynomial_hidden_max_abs_error": hidden_error,
        "polynomials": recipes,
        "program": program.summary(),
        "output_names": names,
        "client": (
            "tied vocabulary head and greedy selection from actual decrypted final hidden; "
            "fresh encrypted embedding feedback"
        ),
        "state": (
            "exact finite-history factorization; no rank truncation"
            if state_representation == "factored"
            else "complete fixed-size state tiled by heads; no rank truncation"
        ),
        "state_representation": state_representation,
        "rotary_representation": rotary,
        "refresh_bounds": (
            "per-node bounds from compiler proxy, 8x headroom with minimum 1; "
            "empirical, not interval-certified"
        ),
        "files_sha256": {
            name: digest(output / name)
            for name in ("program.txt", "fixture.npz", "client_head.f32")
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "reference_text",
                    "layers",
                    "tokens",
                    "polynomial_hidden_max_abs_error",
                    "program",
                )
            },
            indent=2,
        )
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fhemamba workload export-mamba3", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mamba3-siso-187m"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/mamba3-siso-187m/tokenizer")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital")
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--probe-layers", type=int)
    parser.add_argument("--state-representation", choices=("factored", "tiled"), default="factored")
    parser.add_argument("--rotary", choices=("angle", "phasor"), default="angle")
    parser.add_argument("--calibration-tokens", type=int)
    parser.add_argument("--calibration-margin", type=float, default=0.2)
    parser.add_argument("--positive-lower-fraction", type=float, default=0.5)
    parser.add_argument("--normalization-margin", type=float)
    parser.add_argument("--prefix-source", type=Path, help="derive a prefix without any refitting")
    args = parser.parse_args(argv)
    if args.prefix_source is not None:
        prefix_payload(args.prefix_source, args.output, args.new_tokens)
        return
    torch.set_num_threads(4)
    export(
        args.checkpoint,
        args.tokenizer,
        args.output,
        prompt=args.prompt,
        new_tokens=args.new_tokens,
        probe_layers=args.probe_layers,
        state_representation=args.state_representation,
        rotary=args.rotary,
        calibration_tokens=args.calibration_tokens,
        calibration_margin=args.calibration_margin,
        positive_lower_fraction=args.positive_lower_fraction,
        normalization_margin=args.normalization_margin,
    )


if __name__ == "__main__":
    main()
