#!/usr/bin/env python3
"""Compile a trained Mamba-3 backbone and real client feedback to packed CKKS."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

from fhemamba.mamba3 import FactoredMamba3State
from fhemamba.mamba3_lm import Mamba3LM
from fhemamba.packed_program import Calibration, PackedProgram

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


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


@torch.no_grad()
def export(
    checkpoint, tokenizer_path, output, *, prompt="The capital", new_tokens=4, probe_layers=None
):
    from transformers import AutoTokenizer

    if not 1 <= new_tokens <= 16:
        raise ValueError("short-session export supports 1 to 16 generated tokens")
    pins = json.loads(
        (Path(__file__).resolve().parents[1] / "config/mamba3-reproduction.json").read_text()
    )["checkpoint"]
    if pins["repo_id"] != MODEL_ID or pins["revision"] != MODEL_REVISION:
        raise ValueError("exporter and checkpoint manifest disagree")
    for name in ("config.json", "pytorch_model.bin"):
        if digest(checkpoint / name) != pins["files_sha256"][name]:
            raise ValueError(f"checkpoint differs from the pinned trained model: {name}")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = Mamba3LM.from_pretrained(checkpoint)
    original_layers = len(model.backbone.layers)
    if probe_layers is not None:
        if not 1 <= probe_layers <= original_layers:
            raise ValueError("invalid probe layer count")
        model.backbone.layers = model.backbone.layers[:probe_layers]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids or len(prompt_ids) + new_tokens - 1 > 32:
        raise ValueError("prompt plus generation must fit the 32-step short-session limit")
    recorder = Calibration()
    for text in CALIBRATION_PROMPTS:
        model.generate(tokenizer.encode(text, add_special_tokens=False), new_tokens + 1, recorder)
    poly, recipes = recorder.fit(
        tolerance=1e-6,
        max_degree=1023,
        margin=0.2,
        quadrature=True,
        kind_tolerances={"negative_a": 1e-4},
    )
    exact_ids, exact_trace = model.generate(prompt_ids, new_tokens)
    poly_ids, poly_trace = model.generate(prompt_ids, new_tokens, poly)
    if exact_ids != poly_ids:
        raise ValueError("polynomial approximation changes greedy tokens on the chosen prompt")
    program = PackedProgram(poly.polynomials, slots=32768, bound=1e8, extended=True)
    states = [
        FactoredMamba3State(program.constant(state.angle)) for state in model.initial_states()
    ]
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
    with (output / "client_head.f32").open("wb") as stream:
        weight = model.lm_head.weight.detach().numpy().astype("<f4")
        stream.write(struct.pack("<II", *weight.shape))
        weight.tofile(stream)
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
        "calibration_excludes_evaluation_prompt": prompt not in CALIBRATION_PROMPTS,
        "polynomial_hidden_max_abs_error": max(
            float((a - b).abs().max()) for a, b in zip(exact_trace, poly_trace, strict=True)
        ),
        "polynomials": recipes,
        "program": program.summary(),
        "output_names": names,
        "client": (
            "tied vocabulary head and greedy selection from actual decrypted final hidden; "
            "fresh encrypted embedding feedback"
        ),
        "state": "exact finite-history factorization; no rank truncation",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mamba3-siso-187m"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/mamba3-siso-187m/tokenizer")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital")
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--probe-layers", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(
        args.checkpoint,
        args.tokenizer,
        args.output,
        prompt=args.prompt,
        new_tokens=args.new_tokens,
        probe_layers=args.probe_layers,
    )


if __name__ == "__main__":
    main()
