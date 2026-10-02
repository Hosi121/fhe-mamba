"""Local checkpoint discovery without loading weights or optional model libraries."""

from __future__ import annotations

import json
from pathlib import Path

from fhemamba.benchmarks.io import file_sha256, read_object


def read_config(path: Path) -> dict:
    # HF Mamba-2 configs can contain an unbounded time_step_limit (Infinity).
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError("checkpoint config.json must be an object")
    if config.get("ssm_cfg") is not None and not isinstance(config["ssm_cfg"], dict):
        raise ValueError("checkpoint ssm_cfg must be an object")
    return config


def architecture(path: Path) -> str:
    from fhemamba.models.registry import registry

    return registry.detect(read_config(path)).architecture


def checkpoint_identity(path: Path, kind: str) -> dict[str, str]:
    from fhemamba.models.registry import registry

    return registry.get(kind).checkpoint_identity(path)


def hf_checkpoint_identity(path: Path) -> dict[str, str]:
    """Hash the single or sharded weight files selected by Transformers."""
    names = ["config.json"]
    for filename in (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        if not (path / filename).is_file():
            continue
        names.append(filename)
        if filename.endswith(".index.json"):
            shards = set(read_object(path / filename)["weight_map"].values())
            if any(not isinstance(s, str) or Path(s).name != s for s in shards):
                raise ValueError("checkpoint shards must be files in the checkpoint directory")
            names.extend(sorted(shards))
        break
    else:
        raise ValueError("checkpoint has no local safetensors or PyTorch weights")
    return {name: file_sha256(path / name) for name in names}


def tokenizer_directory(checkpoint: Path) -> Path | None:
    if (checkpoint / "tokenizer").is_dir():
        return checkpoint / "tokenizer"
    if (checkpoint / "tokenizer_config.json").is_file():
        return checkpoint
    return None


def tokenizer_identity(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {
        p.name: file_sha256(p)
        for p in sorted(path.iterdir())
        if p.is_file()
        and (
            p.name.startswith(("tokenizer", "special_tokens", "added_tokens"))
            or p.name in ("vocab.json", "vocab.txt", "merges.txt", "spiece.model")
        )
    }
