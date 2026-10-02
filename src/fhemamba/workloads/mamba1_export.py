"""Prepare an independently calibrated Mamba-1 packed generation request."""

from __future__ import annotations

import math

import numpy as np
import torch

from fhemamba.benchmarks.io import file_sha256, write_json
from fhemamba.checkpoints import tokenizer_identity
from fhemamba.inputs import generation_length, token_ids
from fhemamba.m1_payload import _trace_from_ids
from fhemamba.mamba1 import Mamba1LM
from fhemamba.packed_program import Calibration, PackedProgram
from fhemamba.workloads.client_head import write_client_head

CALIBRATION_PROMPTS = (
    "A country",
    "The city",
    "In Europe",
    "This country",
    "Once upon",
    "Paris is",
)


def calibrate(model, calibration_ids, new_tokens):
    """Freeze coefficients using only independent calibration sequences."""
    recorder = Calibration()
    model.decay_squarings = (0,) * len(model.layers)
    for values in calibration_ids:
        model.generate(values, new_tokens, recorder)
    squarings = []
    for layer in range(len(model.layers)):
        site = (layer, "decay_exp")
        lo, hi = recorder.ranges[site]
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("non-finite Mamba-1 decay calibration range")
        count = max(0, math.ceil(math.log2(max(1.0, abs(lo), abs(hi)))))
        if count > 32:
            raise ValueError("Mamba-1 decay range exceeds the 32-squaring limit")
        # exp(x) = exp(x / 2**count)**(2**count). Avoid fitting a huge negative
        # interval or padding it into overflow; no clipping changes the model.
        recorder.ranges[site] = lo * 2.0**-count, hi * 2.0**-count
        squarings.append(count)
    model.decay_squarings = tuple(squarings)
    return recorder.fit(
        tolerance=1e-6,
        max_degree=1023,
        quadrature=True,
        margin=0.5,
        positive_lower_fraction=0.25,
        kind_margins={"inv_sqrt": 1.0},
        kind_tolerances={"exp": 1e-12},
    )


@torch.no_grad()
def export(model, identity, ids, length, output, tokenizer, options, profile):
    generation_length(length, input_length=len(ids))
    if options.calibration_new_tokens < length:
        raise ValueError("calibration_new_tokens must cover the requested generation length")
    embedding = model.get_input_embeddings().weight
    if model.lm_head.bias is not None or not torch.equal(model.lm_head.weight, embedding):
        raise ValueError("Mamba-1 packed feedback requires tied embeddings and a bias-free head")
    calibration_ids = options.calibration_input_ids
    if calibration_ids is None:
        if tokenizer is None:
            raise ValueError("Mamba-1 preparation requires a tokenizer or calibration_input_ids")
        from transformers import AutoTokenizer

        encoder = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        calibration_ids = [
            encoder.encode(text, add_special_tokens=False) for text in CALIBRATION_PROMPTS
        ]
    calibration_ids = [
        token_ids(values, vocab_size=embedding.shape[0]) for values in calibration_ids
    ]
    for values in calibration_ids:
        generation_length(options.calibration_new_tokens, input_length=len(values))
        if values == ids:
            raise ValueError("calibration_input_ids must exclude the evaluation prompt")

    lowered = Mamba1LM(model)
    polynomial, recipes = calibrate(lowered, calibration_ids, options.calibration_new_tokens)
    exact = _trace_from_ids(
        model, torch.tensor([ids]), generate_tokens=length, record_layer_details=False
    )
    poly_ids, poly_trace = lowered.generate(ids, length, polynomial)
    poly_hidden = torch.cat(poly_trace).numpy()
    exact_hidden = exact.expected_final.numpy()
    error = float(np.max(np.abs(poly_hidden - exact_hidden)))
    if poly_ids != exact.generated_ids:
        raise ValueError("Mamba-1 polynomial approximation changes greedy tokens")
    if not np.isfinite(error) or error > 0.001:
        raise ValueError(f"Mamba-1 polynomial hidden error exceeds 0.001: {error}")

    program = PackedProgram(polynomial.polynomials, slots=32768, bound=1e8, extended=True)
    states = lowered.initial_states(program=program)
    evaluated = ids + poly_ids
    hidden = None
    for step in range(len(poly_trace)):
        embedding = lowered.embedding[evaluated[step]][None]
        x = program.input(embedding) if step < len(ids) else program.client_input(hidden, embedding)
        hidden, states = lowered.step(x, states, program)
        np.testing.assert_allclose(hidden.value, poly_hidden[step : step + 1], atol=1e-9, rtol=1e-9)
        program.output(hidden, exact_hidden[step])
    if len(program.nodes) > 1_000_000:
        raise ValueError("Mamba-1 request exceeds the native packed program node limit")

    # No output is created until calibration, domains and reference gates pass.
    output.mkdir(parents=True, exist_ok=False)
    program.write(output / "program.txt")
    write_client_head(output / "client_head.f32", lowered.embedding.numpy())
    np.savez(
        output / "fixture.npz",
        prompt_ids=ids,
        exact_token_ids=exact.generated_ids,
        polynomial_token_ids=poly_ids,
        exact_hidden=exact_hidden,
        polynomial_hidden=poly_hidden,
    )
    write_json(
        output / "manifest.json",
        {
            "schema": "fhemamba-mamba1-lm-v1",
            "architecture": "mamba1",
            "profile": profile.name,
            "security": profile.security,
            "checkpoint": {"files_sha256": identity},
            "tokenizer_files_sha256": tokenizer_identity(tokenizer),
            "prompt": None,
            "prompt_ids": ids,
            "generated_tokens": length,
            "tokens": len(poly_trace),
            "layers": len(lowered.layers),
            "checkpoint_layers": len(model.backbone.layers),
            "complete_backbone": True,
            "vocab_size": lowered.embedding.shape[0],
            "hidden_size": lowered.embedding.shape[1],
            "exact_token_ids": exact.generated_ids,
            "polynomial_token_ids": poly_ids,
            "polynomial_hidden_max_abs_error": error,
            "polynomials": recipes,
            "decay_squarings": list(lowered.decay_squarings),
            "calibration_input_ids": calibration_ids,
            "calibration_generated_tokens": options.calibration_new_tokens,
            "calibration_excludes_evaluation_prompt": True,
            "program": program.summary(),
            "output_names": [f"step{step}.final_hidden" for step in range(len(poly_trace))],
            "client_protocol": "packed-greedy-v1",
            "client": (
                "tied vocabulary head; actual greedy selection and encrypted embedding feedback"
            ),
            "state_representation": "channel-tiled",
            "state": "complete convolution history and selective SSM; no state truncation",
            "refresh_bounds": (
                "per-node proxy bounds with 8x headroom, minimum 1; empirical, not certified"
            ),
            "files_sha256": {
                name: file_sha256(output / name)
                for name in ("program.txt", "fixture.npz", "client_head.f32")
            },
        },
    )
