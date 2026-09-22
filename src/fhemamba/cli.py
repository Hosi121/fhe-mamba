"""Command-line interface for the active FHE Mamba-2 package."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fhemamba import __version__
from fhemamba.artifacts import validate_artifact_file


def main(argv: Sequence[str] | None = None) -> int:
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
