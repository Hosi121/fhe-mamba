#!/usr/bin/env python3
"""Download the pinned Mamba-3 checkpoint and compatible public tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mamba3-siso-187m"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/mamba3-reproduction.json").read_text())
    for kind, destination in (
        ("checkpoint", args.output),
        ("tokenizer", args.output / "tokenizer"),
    ):
        entry = config[kind]
        missing = []
        for name, expected in entry["files_sha256"].items():
            path = destination / name
            if not path.exists():
                missing.append(name)
            elif digest(path) != expected:
                parser.error(f"existing {path} differs; choose another output directory")
        if missing:
            if args.verify_only:
                parser.error(f"missing {kind} files: {missing}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                entry["repo_id"],
                revision=entry["revision"],
                local_dir=destination,
                allow_patterns=missing,
            )
        for name, expected in entry["files_sha256"].items():
            if digest(destination / name) != expected:
                parser.error(f"download hash mismatch: {destination / name}")
        print(f"verified {kind}: {entry['repo_id']}@{entry['revision']}")


if __name__ == "__main__":
    main()
