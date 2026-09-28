#!/usr/bin/env python3
"""Decode actual FHE client tokens after verifying the run and payload bindings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from fhemamba.benchmarks.io import file_sha256 as digest


def validate_generation(payload, run, *, security=None):
    """Validate either runner's evidence before decoding or starting a longer run.

    Kept free of Torch/Transformers so GPU hosts can use it as a completion hook.
    A successful process alone is insufficient (some CUDA failures exit zero).
    """
    manifest = json.loads((payload / "manifest.json").read_text())
    record = json.loads((run / "run.json").read_text())
    native = json.loads((run / "native.json").read_text())
    if manifest.get("schema") != "fhemamba-mamba3-lm-v1":
        raise ValueError("not a language-model payload")
    if record.get("schema_version") == 1:
        if record.get("state") != "completed" or record.get("exit_code") != 0:
            raise ValueError("generation process did not complete successfully")
        expected = {"manifest": digest(payload / "manifest.json")}
        expected.update(
            (key, manifest["files_sha256"][name])
            for key, name in (
                ("program", "program.txt"),
                ("fixture", "fixture.npz"),
                ("client_head", "client_head.f32"),
            )
        )
        if any(record["inputs"][key]["sha256"] != value for key, value in expected.items()):
            raise ValueError("run belongs to a different payload")
    else:
        if not record.get("passed"):
            raise ValueError("generation runner did not pass")
        if record.get("manifest_sha256") != digest(payload / "manifest.json"):
            raise ValueError("run belongs to a different payload")
        if record.get("native_sha256") != digest(run / "native.json"):
            raise ValueError("native output was changed after the run")
    tokens = native.get("generated_token_ids", [])
    if (
        native.get("schema") != "fhemamba-packed-result-v1"
        or native.get("passed") is not True
        or native.get("encrypted") is not True
        or native.get("non_finite") != 0
        or tokens != manifest["exact_token_ids"]
        or tokens != manifest["polynomial_token_ids"]
        or len(tokens) != manifest["generated_tokens"]
        or native.get("client_output_decrypt_count") != len(tokens)
        or native.get("evaluation_decryptions") != 0
        or native.get("diagnostic_only", False) is not False
        or len(native.get("per_output_errors", [])) != manifest["tokens"]
        or manifest["tokens"] != len(manifest["prompt_ids"]) + len(tokens) - 1
    ):
        raise ValueError("encrypted generation did not pass its reference/protocol gates")
    for name in ("max_abs_error_vs_polynomial", "max_abs_error_vs_exact"):
        value = native.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 0.001:
            raise ValueError(f"generation exceeds 0.001 error gate: {name}")
    for errors in native["per_output_errors"]:
        for name in ("polynomial", "exact"):
            value = errors[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 0.001
            ):
                raise ValueError("a generation output exceeds the 0.001 error gate")
    if security is not None and native.get("security") != security:
        raise ValueError("generation security profile differs")
    if native.get("security") == "128-classic":
        audit = native.get("security_audit", {})
        expected = {
            "ring_dimension": 131072,
            "q_bits": 2656,
            "p_bits": 720,
            "qp_bits": 3376,
            "hybrid_digits": 4,
            "guideline_max_qp_bits": 3523,
            "uniform_ternary": True,
            "hybrid": True,
            "library_classical128": True,
            "passed": True,
        }
        if any(type(audit.get(k)) is not type(v) or audit[k] != v for k, v in expected.items()):
            raise ValueError("generation differs from the qualified classical-128 parameters")
        if (
            native.get("gpu_dual_ring") is not False
            or native.get("bootstrap_passes") != 2
            or native.get("ring_dimension") != 131072
            or native.get("refresh_ring_dimension") != 131072
            or native.get("exact_tolerance") != 0.001
            or native.get("polynomial_tolerance") != 0.001
            or native.get("depth") != 44
            or native.get("scale_bits") != 59
            or type(audit.get("error_sigma")) not in (int, float)
            or not math.isfinite(audit["error_sigma"])
            or audit["error_sigma"] < 3.19
        ):
            raise ValueError("generation changed the ring, refresh passes or error tolerance")
    seconds = native.get("eval_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("generation timing must be finite and positive")
    return {
        "passed": True,
        "complete_backbone": manifest["complete_backbone"],
        "layers": manifest["layers"],
        "prompt": manifest["prompt"],
        "generated_token_ids": tokens,
        "generated_tokens": len(tokens),
        "server_evaluations": manifest["tokens"],
        "eval_seconds": seconds,
        "seconds_per_generated_token": seconds / len(tokens),
        "max_abs_error_vs_polynomial": native["max_abs_error_vs_polynomial"],
        "max_abs_error_vs_exact": native["max_abs_error_vs_exact"],
        "security": native["security"],
        "client_protocol": manifest["client"],
        "manifest_sha256": digest(payload / "manifest.json"),
        "native_sha256": digest(run / "native.json"),
        "timing_scope": (
            "evaluation including prompt and inline feedback; excludes setup, final client "
            "token selection, output validation, teardown and external transport"
        ),
    }


def report_generation(payload, run, tokenizer_path, *, security=None):
    from transformers import AutoTokenizer

    result = validate_generation(payload, run, security=security)
    manifest = json.loads((payload / "manifest.json").read_text())
    for name, expected in manifest["tokenizer_files_sha256"].items():
        if digest(tokenizer_path / name) != expected:
            raise ValueError("tokenizer differs from the export")
    tokens = result["generated_token_ids"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return {
        **result,
        "generated_text": tokenizer.decode(tokens),
        "text": tokenizer.decode(manifest["prompt_ids"] + tokens),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fhemamba benchmark generation-report", description=__doc__
    )
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="no tokenizer dependencies")
    parser.add_argument("--security", choices=("128-classic", "not-set"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.validate_only and args.tokenizer is None:
        parser.error("--tokenizer is required unless --validate-only is used")
    report = (
        validate_generation(args.payload, args.run, security=args.security)
        if args.validate_only
        else report_generation(args.payload, args.run, args.tokenizer, security=args.security)
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
