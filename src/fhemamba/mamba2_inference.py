"""Mamba-2 adapters for the public API; retain the existing frozen-chain protocol."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from fhemamba._version import __version__
from fhemamba.benchmarks.io import (
    file_sha256,
    payload_sha256,
    read_object,
    repository_root,
    write_json,
)
from fhemamba.checkpoints import tokenizer_identity
from fhemamba.inputs import generation_length, token_ids

PROFILE = "mamba2-experimental"


def load_checkpoint(path):
    from fhemamba._env import block_broken_torchvision

    block_broken_torchvision()
    try:
        from transformers import Mamba2ForCausalLM
    except ImportError as exc:
        raise ValueError("Mamba-2 requires pip install 'fhemamba[experiments]'") from exc
    return Mamba2ForCausalLM.from_pretrained(path, local_files_only=True).float().cpu().eval()


def execution_recipe(layers, evaluations):
    """Freeze the existing shell argument builder and campaign recipe once.

    Preparation needs a checkout; the saved argv and acceptance criteria let
    execution run from the installed wheel without importing experiment scripts.
    No ambient experimental environment overrides are consumed here.
    """
    root = repository_root()
    recipe = root / "experiments/manifests/dgx_spark_stabilized_generation.json"
    helper = root / "experiments/dgx_mamba2_common.sh"
    campaign = read_object(recipe)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1"; init_dgx_mamba2_defaults; '
            'build_dgx_mamba2_args "$2" "$3"; printf "%s\\0" "${DGX_MAMBA2_ARGS[@]}"',
            "fhemamba",
            str(helper),
            str(layers),
            str(evaluations),
        ],
        env={
            "PATH": os.defpath,
            "FHEMAMBA_REMOTE_ROOT": ".",
            "ARTIFACT_VERSION": __version__,
            **campaign["defaults"],
            "INPUT_CHAIN": "payload",
        },
        check=True,
        capture_output=True,
    )
    return {
        "arguments": completed.stdout.decode().split("\0")[:-1],
        "acceptance": {**campaign["acceptance"], "layers": layers, "tokens": evaluations},
        "sources_sha256": {p.name: file_sha256(p) for p in (recipe, helper)},
    }


def prepare(model, identity, ids, length, output, base_chain, tokenizer):
    from fhemamba.generation import prepare_generation

    if base_chain is None:
        raise ValueError(
            "Mamba-2 preparation requires base_chain with frozen joint-gate coefficients"
        )
    source = Path(base_chain).resolve()
    if output.is_relative_to(source):
        raise ValueError("prepared request must be outside the base chain")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    recipe = execution_recipe(len(model.backbone.layers), len(ids) + length - 1)
    before = payload_sha256(source)
    request = prepare_generation(
        model,
        None,
        source,
        output / "payload",
        input_ids=ids,
        generate_tokens=length,
    )
    if payload_sha256(source) != before:
        raise ValueError("base chain changed during preparation")
    chain = json.loads((output / "payload/chain.json").read_text())
    ar = chain["autoregressive"]
    manifest = {
        "schema": "fhemamba-mamba2-generation-v1",
        "architecture": "mamba2",
        "profile": PROFILE,
        "checkpoint": {"files_sha256": identity},
        "tokenizer_files_sha256": tokenizer_identity(tokenizer),
        "prompt_ids": ids,
        "generated_tokens": length,
        "tokens": request["server_evaluations"],
        "layers": request["n_layers"],
        "complete_backbone": True,
        "exact_token_ids": ar["exact_generated_ids"],
        "polynomial_token_ids": ar["poly_generated_ids"],
        "source_payload_sha256": before,
        "payload_sha256": payload_sha256(output / "payload"),
        "execution": recipe,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_manifest(path, manifest):
    if manifest.get("profile") != PROFILE or manifest.get("complete_backbone") is not True:
        raise ValueError("Mamba-2 requires a complete experimental joint-gate request")
    payload = path / "payload"
    if payload_sha256(payload) != manifest["payload_sha256"]:
        raise ValueError("prepared Mamba-2 payload digest differs")
    chain = json.loads((payload / "chain.json").read_text())
    if chain.get("format") != "fhemamba-m2-chain-joint-v1":
        raise ValueError("Mamba-2 requires a stabilized joint-gate chain")
    ids = token_ids(manifest["prompt_ids"], vocab_size=chain["tensors"]["client_embedding_w"][0])
    length = generation_length(manifest["generated_tokens"])
    ar = chain["autoregressive"]
    if (
        ar["prompt_ids"] != ids
        or ar["generate_tokens"] != length
        or ar["server_evaluations"] != len(ids) + length - 1
        or manifest["tokens"] != ar["server_evaluations"]
        or manifest["layers"] != chain["n_layers"]
        or ar["poly_generated_ids"] != manifest["polynomial_token_ids"]
        or ar["exact_generated_ids"] != manifest["exact_token_ids"]
        or len(ar["poly_generated_ids"]) != length
        or len(ar["exact_generated_ids"]) != length
        or any(count[0] for count in ar["operator_domain_violations"].values())
    ):
        raise ValueError("prepared Mamba-2 references do not match the request")
    return manifest


def generate(model, ids, length, backend, prepared, manifest):
    import numpy as np
    import torch

    from fhemamba.inference import GenerationResult
    from fhemamba.m1_payload import _poly_ops_from_export, _trace_from_ids

    ops = (
        _poly_ops_from_export(prepared.path / "payload", manifest["layers"])
        if manifest and backend == "polynomial"
        else None
    )
    trace = _trace_from_ids(
        model,
        torch.tensor([ids]),
        ops,
        generate_tokens=length,
        record_layer_details=False,
    )
    checks = {
        "finite_hidden_states": True,
        "complete_generation": len(trace.generated_ids) == length,
    }
    report = {
        "architecture": "mamba2",
        "checks": checks,
        "selection": "greedy",
        "stopping": "length",
    }
    if backend == "polynomial":
        chain = json.loads((prepared.path / "payload/chain.json").read_text())
        actual = trace.expected_final.numpy()
        errors = {}
        for reference in ("poly", "exact"):
            name = f"autoregressive_{reference}_expected_final"
            values = np.fromfile(prepared.path / "payload" / f"{name}.bin", dtype="<f4")
            expected = values.reshape(chain["tensors"][name])
            if actual.shape != expected.shape:
                raise ValueError("prepared hidden references have the wrong shape")
            error = float(np.max(np.abs(actual - expected)))
            errors[reference] = error if math.isfinite(error) else None
        checks.update(
            matches_polynomial_tokens=trace.generated_ids == manifest["polynomial_token_ids"],
            within_polynomial_error_gate=errors["poly"] is not None and errors["poly"] <= 0.05,
            within_operator_domains=not any(value[0] for value in ops.violations.values()),
        )
        report.update(
            matches_exact_tokens=trace.generated_ids == manifest["exact_token_ids"],
            max_abs_error_vs_exact=errors["exact"],
            max_abs_error_vs_polynomial=errors["poly"],
            manifest_sha256=prepared.manifest_sha256,
        )
    passed = all(checks.values())
    return GenerationResult(
        ids,
        trace.generated_ids,
        backend,
        passed,
        False,
        "length" if passed else "validation_failed",
        report,
    )


def run(prepared, manifest, binary, output, timeout):
    from fhemamba.benchmarks.acceptance import evaluate_acceptance
    from fhemamba.benchmarks.jobs import run_job
    from fhemamba.generation import generation_report
    from fhemamba.inference import GenerationResult

    payload = prepared.path / "payload"
    identity = file_sha256(binary)
    command = [
        str(binary),
        *manifest["execution"]["arguments"],
        "--artifact-version",
        __version__,
        "--repo-commit",
        "working-tree",
        "--binary-sha256",
        identity,
        "--output-json",
        "${output}/native.json",
    ]
    spec = {
        "schema_version": 1,
        "command": command,
        "cwd": str(prepared.path),
        "env": {"CUDA_LAUNCH_BLOCKING": "1", "FIDESLIB_SYNC_PROFILE": "full"},
        "inputs": {"binary": str(binary), "manifest": str(prepared.path / "manifest.json")},
        "timeout_seconds": timeout,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=output.parent) as stream:
        json.dump(spec, stream)
        stream.flush()
        record = run_job(Path(stream.name), output)
    native, checks, error = {}, {}, None
    try:
        value = json.loads((output / "native.json").read_text())
        if not isinstance(value, dict):
            raise ValueError("native result must be a JSON object")
        native = value
        prepared.manifest()  # Verify every payload byte again before accepting a result.
        for section in ("parameters", "measurements", "measurement_scope"):
            if not isinstance(native.get(section), dict):
                raise ValueError(f"native result must contain a {section} object")
        if native.get("binary_sha256") != identity:
            raise ValueError("native result belongs to a different binary")
        if native.get("parameters", {}).get("security") != "not-set":
            raise ValueError("native result differs from the Mamba-2 experimental security profile")
        # The native M2 binary does not emit a payload hash. Bind the derived
        # report to the checked inputs without rewriting its original bytes.
        bound = {"input_payload_sha256": manifest["payload_sha256"], **native}
        request = {
            "prompt_ids": manifest["prompt_ids"],
            "prompt_tokens": len(manifest["prompt_ids"]),
            "generate_tokens": manifest["generated_tokens"],
            "n_layers": manifest["layers"],
            "server_evaluations": manifest["tokens"],
        }
        validation = generation_report(
            bound,
            json.loads((payload / "chain.json").read_text()),
            request,
            None,
            payload_sha256=manifest["payload_sha256"],
        )
        checks = validation["checks"]
        acceptance = evaluate_acceptance(
            manifest["execution"]["acceptance"],
            [{"infrastructure_ok": record["state"] == "completed", "artifacts": [native]}],
            complete=True,
            dry_run=False,
        )
        checks["campaign_acceptance"] = acceptance["passed"] is True
        if not all(checks.values()):
            error = "; ".join(acceptance["issues"] or [k for k, v in checks.items() if not v])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error = str(exc)
    measurements = native.get("measurements", {})
    actual = (
        measurements.get("autoregressive_selected_ids", [])
        if isinstance(measurements, dict)
        else []
    )
    if not isinstance(actual, list) or any(type(v) is not int or v < 0 for v in actual):
        actual, error = [], "native result contains invalid generated token IDs"
    result = GenerationResult(
        manifest["prompt_ids"],
        actual,
        "ckks",
        bool(checks) and all(checks.values()) and error is None,
        native.get("encrypted") is True,
        "length" if error is None else "validation_failed",
        {
            "architecture": "mamba2",
            "profile": PROFILE,
            "security": "not-set",
            "checks": checks,
            "error": error,
            "run_path": str(output),
            "timed_out": record["state"] == "timed_out",
            "manifest_sha256": prepared.manifest_sha256,
            "native_sha256": file_sha256(output / "native.json")
            if (output / "native.json").exists()
            else None,
            "client_server_process_separated": False,
        },
    )
    write_json(output / "generation.json", result.to_dict())
    return result
