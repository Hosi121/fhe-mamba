#!/usr/bin/env python3
"""Write, validate, and attach immutable B300 build provenance."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from fhemamba.benchmarks.io import file_sha256 as _sha256
from fhemamba.benchmarks.io import read_object as _read_object
from fhemamba.benchmarks.io import write_json as _write_object


def _require_sha(name: str, value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _write_metadata(args: argparse.Namespace) -> None:
    _require_sha("platform config SHA-256", args.platform_config_sha256)
    _require_sha("patch-set SHA-256", args.patch_set_sha256)
    if re.fullmatch(r"[0-9a-f]{40}", args.fideslib_commit) is None:
        raise ValueError("FIDESlib commit must be a full lowercase commit SHA")
    binary_sha256 = _sha256(args.binary)
    payload = {
        "schema_version": 1,
        "platform_config_version": args.platform_config_version,
        "platform_config_sha256": args.platform_config_sha256,
        "image": {"reference": args.image, "id": args.image_id},
        "cuda": {"configured_version": args.cuda_version, "nvcc": args.nvcc},
        "fideslib": {
            "repository": args.fideslib_repository,
            "commit": args.fideslib_commit,
            "arch": args.fideslib_arch,
            "sm": args.fideslib_sm,
            "sync_profile": args.sync_profile,
            "variant": args.variant,
            "patch_set_sha256": args.patch_set_sha256,
        },
        "toolchain": {
            "cxx_path": args.cxx_path,
            "cxx_version": args.cxx_version,
            "cmake_version": args.cmake_version,
        },
        "binary": {
            "relative_path": args.binary_relative_path,
            "sha256": binary_sha256,
        },
    }
    _write_object(args.output, payload)


def _metadata_issues(args: argparse.Namespace, payload: dict[str, Any]) -> list[str]:
    image_value = payload.get("image")
    cuda_value = payload.get("cuda")
    fideslib_value = payload.get("fideslib")
    binary_value = payload.get("binary")
    toolchain_value = payload.get("toolchain")
    image = image_value if isinstance(image_value, dict) else {}
    cuda = cuda_value if isinstance(cuda_value, dict) else {}
    fideslib = fideslib_value if isinstance(fideslib_value, dict) else {}
    binary = binary_value if isinstance(binary_value, dict) else {}
    toolchain = toolchain_value if isinstance(toolchain_value, dict) else {}
    expected = {
        "schema_version": (payload.get("schema_version"), 1),
        "platform_config_version": (
            payload.get("platform_config_version"),
            args.platform_config_version,
        ),
        "platform_config_sha256": (
            payload.get("platform_config_sha256"),
            args.platform_config_sha256,
        ),
        "image.reference": (image.get("reference"), args.image),
        "image.id": (image.get("id"), args.image_id),
        "cuda.configured_version": (cuda.get("configured_version"), args.cuda_version),
        "fideslib.commit": (fideslib.get("commit"), args.fideslib_commit),
        "fideslib.repository": (
            fideslib.get("repository"),
            args.fideslib_repository,
        ),
        "fideslib.arch": (fideslib.get("arch"), args.fideslib_arch),
        "fideslib.sm": (fideslib.get("sm"), args.fideslib_sm),
        "fideslib.sync_profile": (fideslib.get("sync_profile"), args.sync_profile),
        "fideslib.variant": (fideslib.get("variant"), args.variant),
        "fideslib.patch_set_sha256": (
            fideslib.get("patch_set_sha256"),
            args.patch_set_sha256,
        ),
        "binary.relative_path": (
            binary.get("relative_path"),
            args.binary_relative_path,
        ),
        "binary.sha256": (binary.get("sha256"), _sha256(args.binary)),
    }
    issues = [
        f"{name} mismatch: expected {wanted!r}, got {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    required_strings = {
        "cuda.nvcc": cuda.get("nvcc"),
        "toolchain.cxx_path": toolchain.get("cxx_path"),
        "toolchain.cxx_version": toolchain.get("cxx_version"),
        "toolchain.cmake_version": toolchain.get("cmake_version"),
    }
    issues.extend(
        f"{name} must be a non-empty string"
        for name, value in required_strings.items()
        if not isinstance(value, str) or not value
    )
    return issues


def _validate_metadata(args: argparse.Namespace) -> dict[str, Any]:
    _require_sha("platform config SHA-256", args.platform_config_sha256)
    _require_sha("patch-set SHA-256", args.patch_set_sha256)
    payload = _read_object(args.metadata)
    issues = _metadata_issues(args, payload)
    if issues:
        raise ValueError("invalid B300 build metadata: " + "; ".join(issues))
    return payload


def _attach_metadata(args: argparse.Namespace) -> None:
    metadata = _validate_metadata(args)
    artifact = _read_object(args.artifact)
    artifact_sha256 = artifact.get("binary_sha256")
    metadata_sha256 = metadata["binary"]["sha256"]
    if artifact_sha256 != metadata_sha256:
        raise ValueError("artifact binary_sha256 does not match validated B300 build metadata")
    artifact["build_provenance"] = metadata
    _write_object(args.artifact, artifact)


def _add_expectation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--platform-config-version", required=True)
    parser.add_argument("--platform-config-sha256", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--fideslib-repository", required=True)
    parser.add_argument("--fideslib-commit", required=True)
    parser.add_argument("--fideslib-arch", required=True)
    parser.add_argument("--fideslib-sm", required=True)
    parser.add_argument("--sync-profile", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--patch-set-sha256", required=True)
    parser.add_argument("--binary-relative-path", required=True)
    parser.add_argument("--binary", type=Path, required=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fhemamba benchmark b300-build")
    subparsers = parser.add_subparsers(dest="command", required=True)

    writer = subparsers.add_parser("write")
    _add_expectation_args(writer)
    writer.add_argument("--output", type=Path, required=True)
    writer.add_argument("--nvcc", required=True)
    writer.add_argument("--cxx-path", required=True)
    writer.add_argument("--cxx-version", required=True)
    writer.add_argument("--cmake-version", required=True)

    validator = subparsers.add_parser("validate")
    _add_expectation_args(validator)
    validator.add_argument("--metadata", type=Path, required=True)

    attacher = subparsers.add_parser("attach")
    _add_expectation_args(attacher)
    attacher.add_argument("--metadata", type=Path, required=True)
    attacher.add_argument("--artifact", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "write":
            _write_metadata(args)
        elif args.command == "validate":
            _validate_metadata(args)
        else:
            _attach_metadata(args)
    except (OSError, ValueError) as exc:
        print(f"fhemamba benchmark b300-build: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
