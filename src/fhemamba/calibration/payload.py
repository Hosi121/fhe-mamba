"""Shared payload I/O and provenance for calibration operations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fhemamba import __version__
from fhemamba.artifacts import current_git_commit
from fhemamba.benchmarks.io import file_sha256


def layers(payload: Path, chain: dict | None = None):
    """Keep chain order when provided, otherwise the original sorted layer order."""
    paths = (
        (payload / name / "meta.json" for name in chain["layer_dirs"])
        if chain is not None
        else sorted(payload.glob("layer_*/meta.json"))
    )
    for path in paths:
        yield path.parent, json.loads(path.read_text())


def folded_projection(directory: Path, meta: dict, rows: slice):
    """Fold public RMS gamma into selected projection rows in float64."""
    import numpy as np

    width = meta["dims"]["d_model"]
    weight = np.fromfile(directory / "in_proj_w.bin", dtype="<f4").reshape(-1, width)
    gamma = np.fromfile(directory / "block_norm_w.bin", dtype="<f4")
    return weight[rows].astype(float) * gamma.astype(float)[None, :]


def derivative_destination(payload: Path, output: Path) -> None:
    if output.exists() or output.resolve().is_relative_to(payload.resolve()):
        raise ValueError("output-chain must be a new directory outside the source payload")


def write_derivative(
    payload: Path, output: Path, chain: dict, metadata: list, key: str, provenance: dict
) -> None:
    """Copy unchanged weights/references; preserve legacy Infinity in metadata."""
    derivative_destination(payload, output)
    shutil.copytree(payload, output)
    for directory, meta in zip(chain["layer_dirs"], metadata, strict=True):
        meta["carried_bounds"][key] = provenance
        (output / directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "chain.json").write_text(json.dumps({**chain, key: provenance}, indent=2) + "\n")


def source_hashes(source: str, *dependencies: str) -> dict:
    package = Path(__file__).parents[1]
    paths = [Path(source), Path(__file__), *(package / name for name in dependencies)]
    return {str(path): file_sha256(path) for path in paths}


def report(stage: str, sources: dict, *, measurement_scope: dict, **values) -> dict:
    return {
        "stage": stage,
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": sources,
        **values,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "encrypted_execution": False,
            **measurement_scope,
        },
    }
