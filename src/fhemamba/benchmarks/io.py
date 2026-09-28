"""Small shared contracts for experiment records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def file_sha256(path: Path) -> str:
    """Stream large payloads rather than allocating a second in-memory copy."""
    return file_identity(path)["sha256"]


def payload_sha256(directory: Path) -> str:
    """Stable chain identity, including names and reference/calibration bytes."""
    if not (directory / "chain.json").is_file():
        raise ValueError(f"chain payload is missing chain.json: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.suffix in {".json", ".bin"} and p.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(f"{path.relative_to(directory)}\0{file_sha256(path)}\n".encode())
    return digest.hexdigest()


def repository_root(start: Path | None = None) -> Path:
    """Find a source checkout from the caller's working directory, not site-packages."""
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file() and (path / "native/fideslib_stage0").is_dir():
            return path
    raise ValueError("this command needs a source checkout; run from the repository or pass --repo")


def write_json(path: Path, value: Any) -> None:
    """Replace a record atomically; readers never observe half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value
