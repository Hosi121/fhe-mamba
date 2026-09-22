#!/usr/bin/env python3
"""Download or verify the public checkpoint pinned in config/reproduction.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mamba2-130m-hf"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/reproduction.json").read_text())
    checkpoint = config["checkpoint"]
    for name, expected in config["public_coefficients"].items():
        if digest(ROOT / name) != expected:
            parser.error(f"public coefficient hash mismatch: {name}")
    missing = []
    for name, expected in checkpoint["files_sha256"].items():
        path = args.output / name
        if not path.is_file():
            missing.append(name)
        elif digest(path) != expected:
            parser.error(f"existing checkpoint file differs: {path}; choose a new output directory")
    if missing:
        if args.verify_only:
            parser.error("missing checkpoint files: " + ", ".join(missing))
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=checkpoint["repo_id"],
            revision=checkpoint["revision"],
            local_dir=args.output,
            allow_patterns=missing,
        )
    for name, expected in checkpoint["files_sha256"].items():
        if digest(args.output / name) != expected:
            parser.error(f"downloaded checkpoint hash mismatch: {name}")
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "revision": checkpoint["revision"],
                "files_verified": len(checkpoint["files_sha256"]),
                "public_coefficient_bundles_verified": len(config["public_coefficients"]),
            }
        )
    )


if __name__ == "__main__":
    main()
