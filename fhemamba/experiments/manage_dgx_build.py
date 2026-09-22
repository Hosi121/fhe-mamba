#!/usr/bin/env python3
"""Native Spark build identity, including the actual shared-library contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    files = [REPO / "config/dgx-spark.env", REPO / "fhemamba/src/fhemamba/_version.py"]
    native = REPO / "native/fideslib_stage0"
    files += [p for p in native.rglob("*") if p.is_file()]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(f"{path.relative_to(REPO)}\0{sha256(path)}\n".encode())
    return digest.hexdigest()


def payload_sha256(directory: Path) -> str:
    """Bind both file names and bytes, including references/calibration assets."""
    files = sorted(p for p in directory.rglob("*") if p.suffix in {".json", ".bin"} and p.is_file())
    if not (directory / "chain.json").is_file():
        raise ValueError(f"chain payload is missing chain.json: {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(directory)}\0{sha256(path)}\n".encode())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate(root: Path) -> dict:
    binary = root / "spark/kernel/stage1_mamba2_decode_fideslib"
    metadata = json.loads(binary.with_suffix(".build.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("platform") != "dgx-spark":
        raise ValueError("unrecognized Spark build metadata")
    if metadata.get("source_sha256") != source_sha256():
        raise ValueError("native sources/configuration changed; rebuild the Spark binary")
    if metadata.get("binary_sha256") != sha256(binary):
        raise ValueError("Spark binary hash mismatch")
    libraries = metadata.get("shared_libraries")
    if not isinstance(libraries, dict) or not libraries:
        raise ValueError("missing shared-library provenance")
    for path, expected in libraries.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Spark shared library changed: {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["write", "validate", "attach", "library-path", "payload-hash"]
    )
    parser.add_argument("--root", type=Path, default=Path.home() / "fhemamba")
    parser.add_argument("--fideslib-prefix", type=Path)
    parser.add_argument("--openfhe-prefix", type=Path)
    parser.add_argument("--patch-set-sha256")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--payload-sha256")
    args = parser.parse_args()
    binary = args.root / "spark/kernel/stage1_mamba2_decode_fideslib"
    if args.action == "payload-hash":
        if not args.payload:
            parser.error("payload-hash requires --payload")
        print(payload_sha256(args.payload))
    elif args.action == "write":
        if not args.fideslib_prefix or not args.openfhe_prefix or not args.patch_set_sha256:
            parser.error("write requires dependency prefixes and patch-set hash")
        config = dict(
            line.split("=", 1)
            for line in (REPO / "config/dgx-spark.env").read_text().splitlines()
            if line and not line.startswith("#")
        )
        libraries = sorted(
            {
                p.resolve()
                for prefix in (args.fideslib_prefix, args.openfhe_prefix)
                for p in prefix.rglob("*.so*")
                if p.is_file()
            }
        )
        metadata = {
            "schema_version": 1,
            "platform": "dgx-spark",
            "source_sha256": source_sha256(),
            "binary_sha256": sha256(binary),
            "cuda": {"configured_version": config["SPARK_CUDA_VERSION"]},
            "fideslib": {
                "commit": config["SPARK_FIDESLIB_COMMIT"],
                "arch": config["SPARK_FIDESLIB_ARCH"],
                "sync_profile": config["SPARK_SYNC_PROFILE"],
                "patch_set_sha256": args.patch_set_sha256,
            },
            "toolchain": {
                "cxx": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
            },
            "library_path": ":".join(str(p) for p in sorted({p.parent for p in libraries})),
            "shared_libraries": {str(p): sha256(p) for p in libraries},
        }
        write_json(binary.with_suffix(".build.json"), metadata)
    else:
        metadata = validate(args.root)
        if args.action == "library-path":
            print(metadata["library_path"])
        elif args.action == "attach":
            if not args.artifact:
                parser.error("attach requires --artifact")
            artifact = json.loads(args.artifact.read_text())
            if artifact.get("binary_sha256") != metadata["binary_sha256"]:
                raise ValueError("artifact binary hash differs from Spark build")
            artifact["build_provenance"] = metadata
            if args.payload_sha256:
                artifact["input_payload_sha256"] = args.payload_sha256
            write_json(args.artifact, artifact)


if __name__ == "__main__":
    main()
