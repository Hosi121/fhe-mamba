"""Independent state observations and public conditioning of frozen row scales."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from fhemamba._env import block_broken_torchvision
from fhemamba.benchmarks.io import file_sha256, payload_sha256, sha256
from fhemamba.calibration.payload import (
    derivative_destination,
    layers,
    source_hashes,
    write_derivative,
)
from fhemamba.calibration.payload import report as make_report
from fhemamba.m1_payload import _poly_ops_from_export
from fhemamba.reference import init_states, model_forward
from fhemamba.state_coordinates import regularize_row_scales


def regularize_state(
    payload: Path, *, output_chain: Path, group_scale_floor: float, group_heads=4
) -> dict:
    sources = source_hashes(__file__, "state_coordinates.py")
    derivative_destination(payload, output_chain)
    chain = json.loads((payload / "chain.json").read_text())
    parent_sha = payload_sha256(payload)
    metadata, summaries = ([], [])
    for _, meta in layers(payload, chain):
        bounds = meta["carried_bounds"]
        if bounds.get("source") != "calibration_text" or not bounds.get(
            "state_coordinate_calibration"
        ):
            raise ValueError("independently calibrated row coordinates are required")
        dims = meta["dims"]
        old = np.asarray(bounds["state_row_abs_max"]).reshape(dims["num_heads"], dims["head_dim"])
        scales = regularize_row_scales(old, group_heads, group_scale_floor)
        bounds["state_row_abs_max"] = scales.reshape(-1).tolist()
        bounds["state_row_observed_abs_max"] = old.reshape(-1).tolist()
        group = old.reshape(-1, group_heads * dims["head_dim"]).max(axis=1, keepdims=True)
        ratios = scales.reshape(group.shape[0], -1) / np.maximum(group, 1e-06)
        summaries.append(
            {
                "layer": meta["layer_index"],
                "raised_rows": int((scales > old).sum()),
                "mean_row_to_group_scale": float(ratios.mean()),
            }
        )
        metadata.append(meta)
    regularization = {
        "parent_payload_sha256": parent_sha,
        "group_heads": group_heads,
        "group_scale_floor": group_scale_floor,
        "maximum_group_to_row_amplification": 1 / group_scale_floor if group_scale_floor else None,
        "new_calibration_data": False,
        "evaluation_references_used_for_scales": False,
        "uniform_bound_claimed": False,
    }
    write_derivative(
        payload, output_chain, chain, metadata, "state_coordinate_regularization", regularization
    )
    report = make_report(
        "state-coordinate-regularization",
        sources,
        regularization=regularization,
        rows=summaries,
        output_payload_sha256=payload_sha256(output_chain),
        measurement_scope={
            "claim": (
                "Public conditioning of independently calibrated scales, "
                "without new data or reference fitting."
            )
        },
    )
    return report


@torch.no_grad()
def calibrate_state(
    payload: Path,
    *,
    checkpoint: Path,
    output_chain: Path,
    calibration_description: str,
    calibration_text: Path | None = None,
    calibration_tokens_pt: Path | None = None,
    offsets=(0,),
    tokens=512,
    threads=4,
    group_heads=4,
    group_scale_floor=0.0,
) -> dict:
    sources = source_hashes(__file__, "state_coordinates.py", "m1_payload.py")
    if (calibration_text is None) == (calibration_tokens_pt is None):
        raise ValueError("supply exactly one calibration text or token file")
    if min(tokens, threads, group_heads) < 1:
        raise ValueError("positive tokens, threads, and group-heads are required")
    if not 0 <= group_scale_floor <= 1 or not offsets or min(offsets) < 0:
        raise ValueError("floor in [0, 1] and nonnegative offsets are required")
    ordered = sorted(offsets)
    if any((right < left + tokens for left, right in pairwise(ordered))):
        raise ValueError("calibration windows must not overlap")
    derivative_destination(payload, output_chain)
    torch.set_num_threads(threads)
    block_broken_torchvision()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(checkpoint).float().eval()
    source = calibration_text or calibration_tokens_pt
    if calibration_text:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        all_ids = tokenizer(source.read_text(), return_tensors="pt").input_ids
    else:
        all_ids = torch.load(source, weights_only=True, map_location="cpu")
    if all_ids.ndim != 2 or all_ids.shape[0] != 1 or all_ids.dtype != torch.int64:
        raise ValueError("calibration token input must be int64 [1, length]")
    if max(offsets) + tokens > all_ids.shape[1]:
        raise ValueError("calibration windows exceed token input")
    ids = torch.cat([all_ids[:, start : start + tokens] for start in offsets])
    chain = json.loads((payload / "chain.json").read_text())
    if chain["n_layers"] != len(model.backbone.layers):
        raise ValueError("payload/model layer counts differ")
    metadata = [meta for _, meta in layers(payload, chain)]
    for meta, block in zip(metadata, model.backbone.layers, strict=True):
        if meta["carried_bounds"].get("source") != "calibration_text":
            raise ValueError("parent payload must have independent carried-bound calibration")
        if block.mixer.num_heads % group_heads:
            raise ValueError("group-heads must divide the model's head count")
    states = init_states(model, batch_size=len(offsets))
    rows = [state.ssm.new_zeros(state.ssm.shape[1:3]) for state in states]
    per_circuit = {}
    for name, ops in (
        ("exact", None),
        ("exported-poly", _poly_ops_from_export(payload, chain["n_layers"])),
    ):
        states = init_states(model, batch_size=len(offsets))
        maxima = [state.ssm.new_zeros(state.ssm.shape[1:3]) for state in states]
        for token in range(ids.shape[1]):
            model_forward(
                model,
                ids[:, token : token + 1],
                ops=ops,
                scan="loop",
                states=states,
                output_logits=False,
            )
            for layer, state in enumerate(states):
                maxima[layer] = maxima[layer].maximum(state.ssm.abs().amax(dim=(0, 3)))
            if (token + 1) % 128 == 0:
                print(f"{name} calibration tokens={token + 1} per window", flush=True)
        if any(not bool(maximum.isfinite().all()) for maximum in maxima):
            report = make_report(
                "state-coordinate-calibration",
                sources,
                status="failed",
                failure=f"non-finite {name} state on calibration data",
                failed_circuit=name,
                non_finite_layers=[
                    i for i, value in enumerate(maxima) if not bool(value.isfinite().all())
                ],
                input_payload_sha256=payload_sha256(payload),
                calibration_input_sha256=file_sha256(source),
                calibration_token_ids_sha256=sha256(ids.numpy().astype("<i8").tobytes()),
                calibration_description=calibration_description,
                window_offsets=offsets,
                tokens_per_window=ids.shape[1],
                window_resets_state=True,
                completed_circuit_layer_maxima=per_circuit,
                output_payload_written=False,
                measurement_scope={
                    "claim": "Failed independent calibration; no new payload or scales accepted."
                },
            )
            return report
        rows = [row.maximum(maximum) for row, maximum in zip(rows, maxima, strict=True)]
        per_circuit[name] = [float(maximum.max()) for maximum in maxima]
    provenance = {
        "source": "calibration_text",
        "description": calibration_description,
        "calibration_input_kind": "text" if calibration_text else "int64-token-file",
        "calibration_input_sha256": file_sha256(source),
        "calibration_token_ids_sha256": sha256(ids.numpy().astype("<i8").tobytes()),
        "calibration_tokens": ids.numel(),
        "tokens_per_window": ids.shape[1],
        "window_offsets": offsets,
        "window_resets_state": True,
        "circuits": list(per_circuit),
        "layout": "flattened [head, channel]; maximum over time and state coordinate",
        "minimum_runtime_scale": 1e-06,
        "group_heads": group_heads,
        "group_scale_floor": group_scale_floor,
        "maximum_group_to_row_amplification": 1.0 / group_scale_floor
        if group_scale_floor > 0
        else None,
        "uniform_bound_claimed": False,
        "parent_payload_sha256": payload_sha256(payload),
        "checkpoint_sha256": {
            path.name: file_sha256(path)
            for path in sorted(checkpoint.iterdir())
            if path.suffix in {".safetensors", ".json"}
        },
    }
    scales = [
        torch.from_numpy(regularize_row_scales(row.numpy(), group_heads, group_scale_floor))
        for row in rows
    ]
    for meta, row in zip(metadata, scales, strict=True):
        bounds = meta["carried_bounds"]
        bounds["state_abs_max"] = float(row.max())
        bounds["state_head_abs_max"] = row.amax(dim=1).tolist()
        bounds["state_row_abs_max"] = row.flatten().tolist()
    write_derivative(
        payload, output_chain, chain, metadata, "state_coordinate_calibration", provenance
    )
    summary = []
    for layer, row in enumerate(scales):
        group = row.reshape(-1, group_heads * row.shape[1]).amax(dim=1)
        ratios = row.reshape(-1, group_heads * row.shape[1]) / group[:, None].clamp_min(1e-06)
        summary.append(
            {
                "layer": layer,
                "group_heads": group_heads,
                "group_maxima": group.tolist(),
                "mean_row_to_group_scale": float(ratios.mean()),
                "median_row_to_group_scale": float(ratios.median()),
            }
        )
    report = make_report(
        "state-coordinate-calibration",
        sources,
        calibration=provenance,
        output_payload_sha256=payload_sha256(output_chain),
        per_circuit_layer_maxima=per_circuit,
        rows=summary,
        measurement_scope={
            "uniform_bound_claimed": False,
            "claim": "State scales from independent exact and frozen-poly calibration runs.",
        },
    )
    return report
