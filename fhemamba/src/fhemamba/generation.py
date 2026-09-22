"""Prompt preparation and honest text reporting for the native client loop."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from fhemamba import __version__


def prepare_generation(model, tokenizer, source, destination, *, prompt, generate_tokens):
    """Copy a frozen payload, then add references for the entire input prompt."""
    import numpy as np

    from fhemamba.m1_payload import export_autoregressive_client_payload

    if not prompt.strip() or generate_tokens < 1:
        raise ValueError("a nonempty prompt and positive generation length are required")
    source, destination = Path(source), Path(destination)
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("generation destination must be outside the source payload")
    chain = json.loads((source / "chain.json").read_text())
    if chain.get("format") != "fhemamba-m2-chain-joint-v1":
        raise ValueError("generation requires the stabilized joint-gate payload")
    if chain["n_layers"] != len(model.backbone.layers):
        raise ValueError("checkpoint and payload layer counts differ")

    def check_weight(directory, name, weight):
        exported = np.fromfile(directory / f"{name}.bin", dtype="<f4")
        actual = weight.detach().cpu().numpy().reshape(-1)
        if not np.array_equal(exported, actual):
            raise ValueError(f"checkpoint differs from payload: {directory.name}/{name}")

    check_weight(source, "final_norm_w", model.backbone.norm_f.weight)
    if chain["final_norm_eps"] != model.backbone.norm_f.variance_epsilon:
        raise ValueError("checkpoint differs from payload: final normalization epsilon")
    for directory, block in zip(chain["layer_dirs"], model.backbone.layers, strict=True):
        mixer = block.mixer
        meta = json.loads((source / directory / "meta.json").read_text())
        if meta["eps"] != {
            "block_norm": block.norm.variance_epsilon,
            "gated_norm": mixer.norm.variance_epsilon,
        } or meta["time_step_limit"] != list(mixer.time_step_limit):
            raise ValueError(f"checkpoint differs from payload: {directory}/configuration")
        for name, weight in (
            ("in_proj_w", mixer.in_proj.weight),
            ("conv_w", mixer.conv1d.weight),
            ("dt_bias", mixer.dt_bias),
            ("a_log", mixer.A_log),
            ("d_skip", mixer.D),
            ("block_norm_w", block.norm.weight),
            ("gated_norm_w", mixer.norm.weight),
            ("out_proj_w", mixer.out_proj.weight),
        ):
            check_weight(source / directory, name, weight)
        bias = np.fromfile(source / directory / "conv_b.bin", dtype="<f4")
        if mixer.conv1d.bias is None:
            if np.any(bias):
                raise ValueError(f"checkpoint differs from payload: {directory}/conv_b")
        else:
            check_weight(source / directory, "conv_b", mixer.conv1d.bias)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    if not prompt_ids:
        raise ValueError("prompt tokenization is empty")
    # Never modify the measured source or share writable hard links with it.
    shutil.copytree(source, destination)
    export_autoregressive_client_payload(
        model,
        tokenizer,
        destination,
        prompt=prompt,
        prompt_tokens=len(prompt_ids),
        generate_tokens=generate_tokens,
    )
    prepared = json.loads((destination / "chain.json").read_text())
    if prepared["autoregressive"]["prompt_ids"] != prompt_ids:
        raise ValueError("exported prompt differs from the full requested prompt")
    if prepared.get("final_norm_poly") != chain.get("final_norm_poly"):
        raise ValueError("generation preparation changed final normalization")
    for directory in chain["layer_dirs"]:
        before = json.loads((source / directory / "meta.json").read_text())
        after = json.loads((destination / directory / "meta.json").read_text())
        for key in ("polys", "joint_gates", "carried_bounds"):
            if before[key] != after[key]:
                raise ValueError(f"generation preparation changed {directory}/{key}")
    return {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "prompt_tokens": len(prompt_ids),
        "generate_tokens": generate_tokens,
        "server_evaluations": len(prompt_ids) + generate_tokens - 1,
        "n_layers": chain["n_layers"],
        "frozen_operators_and_bounds_unchanged": True,
    }


def generation_report(native, chain, request, tokenizer, *, payload_sha256):
    """Decode measured IDs, never replace them with the expected completion."""
    if native.get("input_payload_sha256") != payload_sha256:
        raise ValueError("native result belongs to a different payload")
    parameters = native["parameters"]
    expected = {
        "autoregressive_client_loop": True,
        "autoregressive_prompt_tokens": request["prompt_tokens"],
        "autoregressive_generate_tokens": request["generate_tokens"],
        "n_layers_loaded": request["n_layers"],
        "tokens": request["server_evaluations"],
        "final_norm_applied": True,
    }
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise ValueError("native result does not cover the requested generation")
    ar = chain["autoregressive"]
    if ar["prompt_ids"] != request["prompt_ids"]:
        raise ValueError("payload prompt differs from the request")
    measurements = native["measurements"]
    selected = measurements["autoregressive_selected_ids"]
    vocab_size = chain["tensors"]["client_embedding_w"][0]
    if any(type(value) is not int or not 0 <= value < vocab_size for value in selected):
        raise ValueError("native result contains invalid token IDs")
    errors = measurements["per_token_max_abs_error"]
    decrypt_ok = measurements["per_token_decrypt_ok"]
    scope = native["measurement_scope"]
    checks = {
        "native_passed": native.get("passed") is True,
        "encrypted_full_chain": native.get("encrypted") is True
        and all(
            scope.get(key) is True
            for key in (
                "fideslib_encrypted_execution",
                "full_layer_chain",
                "multi_token_ciphertext_state_carry",
                "ciphertext_conv_fifo",
                "per_token_fresh_embedding_encryption",
                "client_output_decrypts_are_protocol_boundary",
            )
        ),
        "complete_generation": len(selected) == request["generate_tokens"],
        "matches_polynomial_tokens": selected == ar["poly_generated_ids"],
        "all_outputs_decrypt": len(decrypt_ok) == request["server_evaluations"] and all(decrypt_ok),
        "all_errors_within_0_05": len(errors) == request["server_evaluations"]
        and all(math.isfinite(value) and 0 <= value <= 0.05 for value in errors),
        "no_intermediate_diagnostic_decrypts": scope.get("zero_intermediate_decrypts") is True,
        "client_output_decrypts": scope.get("autoregressive_client_output_decrypt_count")
        == request["generate_tokens"],
    }

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    passed = all(checks.values())
    return {
        "version": __version__,
        "stage": "fhemamba-client-generation",
        "repo_commit": native["repo_commit"],
        "binary_sha256": native["binary_sha256"],
        "input_payload_sha256": payload_sha256,
        "backend": native["backend"],
        "config": native["config"],
        "encrypted": native.get("encrypted") is True,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "parameters": {**request, "security": parameters["security"], "selection": "greedy"},
        "checks": checks,
        "text": {
            "prompt": decode(request["prompt_ids"]),
            "generated": decode(selected),
            "full": decode(request["prompt_ids"] + selected),
            "polynomial_reference": decode(ar["poly_generated_ids"]),
            "exact_reference": decode(ar["exact_generated_ids"]),
        },
        "measurements": {
            "generated_ids": selected,
            "per_token_max_abs_error": errors,
            "per_token_max_abs_error_vs_exact": measurements["per_token_max_abs_error_vs_exact"],
            "matches_exact_tokens": selected == ar["exact_generated_ids"],
            "peak_rss_gib": measurements["peak_rss_gib"],
            "operator_domain_violations": ar["operator_domain_violations"],
        },
        "operation_counts": native["operation_counts"],
        "timing": native["timing"],
        "measurement_scope": {
            "full_model_correctness_claimed": False,
            "encrypted_recurrent_state_and_fifo": checks["encrypted_full_chain"],
            "client_selects_from_full_vocabulary": True,
            "zero_intermediate_decrypts": checks["no_intermediate_diagnostic_decrypts"],
            "client_server_process_separated": False,
            "fully_encrypted_token_selection": False,
            "claim": "Single-process interactive generation with client output decryption, "
            "full-vocabulary greedy selection and encrypted embedding feedback. "
            "This does not establish server key separation or 128-bit full-chain security.",
        },
    }
