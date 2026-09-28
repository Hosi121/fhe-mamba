#!/usr/bin/env python3
"""Write, validate, and attach immutable B300 build provenance."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from fhemamba.benchmarks.io import field
from fhemamba.benchmarks.io import file_sha256 as _sha256
from fhemamba.benchmarks.io import read_object as _read_object
from fhemamba.benchmarks.io import write_json as _write_object

# One mapping defines the CLI, serialized identity and validation contract.
_IDENTITY_FIELDS = {
    "platform_config_version": "platform_config_version",
    "platform_config_sha256": "platform_config_sha256",
    "image.reference": "image",
    "image.id": "image_id",
    "cuda.configured_version": "cuda_version",
    "fideslib.repository": "fideslib_repository",
    "fideslib.commit": "fideslib_commit",
    "fideslib.arch": "fideslib_arch",
    "fideslib.sm": "fideslib_sm",
    "fideslib.sync_profile": "sync_profile",
    "fideslib.variant": "variant",
    "fideslib.patch_set_sha256": "patch_set_sha256",
    "binary.relative_path": "binary_relative_path",
}
_DETAIL_FIELDS = {
    "cuda.nvcc": "nvcc",
    "toolchain.cxx_path": "cxx_path",
    "toolchain.cxx_version": "cxx_version",
    "toolchain.cmake_version": "cmake_version",
}


def _expected(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        **{name: getattr(args, attribute) for name, attribute in _IDENTITY_FIELDS.items()},
        "binary.sha256": _sha256(args.binary),
    }


def _require_sha(name: str, value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _write_metadata(args: argparse.Namespace) -> None:
    _require_sha("platform config SHA-256", args.platform_config_sha256)
    _require_sha("patch-set SHA-256", args.patch_set_sha256)
    if re.fullmatch(r"[0-9a-f]{40}", args.fideslib_commit) is None:
        raise ValueError("FIDESlib commit must be a full lowercase commit SHA")
    fields = {
        **_expected(args),
        **{name: getattr(args, attribute) for name, attribute in _DETAIL_FIELDS.items()},
    }
    payload = {}
    for name, value in fields.items():
        if "." in name:
            group, key = name.split(".")
            payload.setdefault(group, {})[key] = value
        else:
            payload[name] = value
    _write_object(args.output, payload)


def _metadata_issues(args: argparse.Namespace, payload: dict[str, Any]) -> list[str]:
    issues = [
        f"{name} mismatch: expected {wanted!r}, got {actual!r}"
        for name, wanted in _expected(args).items()
        if (actual := field(payload, name, None)) != wanted
    ]
    issues.extend(
        f"{name} must be a non-empty string"
        for name in _DETAIL_FIELDS
        if not isinstance(value := field(payload, name, None), str) or not value
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
    for attribute in _IDENTITY_FIELDS.values():
        parser.add_argument("--" + attribute.replace("_", "-"), required=True)
    parser.add_argument("--binary", type=Path, required=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fhemamba benchmark b300-build")
    subparsers = parser.add_subparsers(dest="command", required=True)

    writer = subparsers.add_parser("write")
    _add_expectation_args(writer)
    writer.add_argument("--output", type=Path, required=True)
    for attribute in _DETAIL_FIELDS.values():
        writer.add_argument("--" + attribute.replace("_", "-"), required=True)

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
