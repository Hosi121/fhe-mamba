"""Command-line interface for FHE Mamba and its reusable research tools."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from fhemamba import __version__
from fhemamba._commands import dispatch
from fhemamba.artifacts import validate_artifact_file

GROUPS = {
    "generate": (
        "fhemamba.inference_cli",
        "generate_main",
        "generate from text or token IDs",
    ),
    "prepare": (
        "fhemamba.inference_cli",
        "prepare_main",
        "prepare one Mamba-2/3 request for polynomial/CKKS generation",
    ),
    "calibrate": (
        "fhemamba.calibration.__main__",
        "main",
        "frozen payload calibration and certificates",
    ),
    "benchmark": ("fhemamba.benchmarks.__main__", "main", "jobs, qualification and evidence"),
    "diagnose": ("fhemamba.diagnostics.__main__", "main", "frozen-circuit error diagnosis"),
    "profile": ("fhemamba.profiling.__main__", "main", "Nsight CSV/SQLite analysis"),
    "recurrent": ("fhemamba.recurrent.__main__", "main", "recurrent-state component studies"),
    "workload": ("fhemamba.workloads.__main__", "main", "Mamba-3 export and upstream parity"),
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in GROUPS:
        return dispatch("fhemamba", GROUPS, argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _validate_artifacts(args: argparse.Namespace) -> int:
    results = [
        {
            "path": str(path),
            **validate_artifact_file(path, require_commit=args.require_commit).to_json_dict(),
        }
        for path in args.artifacts
    ]
    payload = {
        "version": __version__,
        "stage": "artifact-validation",
        "artifact_count": len(results),
        "valid": all(result["valid"] for result in results),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fhemamba")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, _, description) in GROUPS.items():
        subparsers.add_parser(name, help=description)

    validate = subparsers.add_parser(
        "validate-artifacts",
        help="validate benchmark/probe JSON artifacts",
    )
    validate.add_argument("artifacts", nargs="+", type=Path)
    validate.add_argument("--require-commit", action="store_true")
    validate.set_defaults(handler=_validate_artifacts)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
