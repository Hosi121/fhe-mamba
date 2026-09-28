"""Command line for portable benchmark jobs and curated evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fhemamba._commands import dispatch

from .comparison import compare
from .evidence import Redactor, extract_provenance, publish, verify
from .io import read_object
from .jobs import run_job

TOOLS = {
    "packed": ("fhemamba.benchmarks.packed", "main", "run and qualify packed inference"),
    "generation-report": ("fhemamba.benchmarks.generation", "main", "qualify generated tokens"),
    "spark-build": ("fhemamba.benchmarks.builds", "main", "validate Spark build identity"),
    "b300-build": ("fhemamba.benchmarks.b300_build", "main", "legacy B300 build identity"),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in TOOLS:
        return dispatch("fhemamba benchmark", TOOLS, argv)
    parser = argparse.ArgumentParser(prog="fhemamba benchmark", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, _, description) in TOOLS.items():
        commands.add_parser(name, help=description)
    run = commands.add_parser("run", help="execute one job and write a completion event")
    run.add_argument("spec", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--settings", type=Path, help="local JSON variables, not shell code")
    run.add_argument("--lock", type=Path, help="shared advisory lock for jobs using one device")
    run.add_argument("--events", type=Path, help="append completion events for a watcher")
    export = commands.add_parser("publish", help="create a separate public evidence tree")
    export.add_argument("source", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--policy", type=Path, help="local JSON with literal replacements")
    check = commands.add_parser("verify", help="verify public files and archive members")
    check.add_argument("roots", type=Path, nargs="+")
    comparison = commands.add_parser(
        "compare", help="check conditions and compare measured samples"
    )
    comparison.add_argument("--baseline", type=Path, nargs="+", required=True)
    comparison.add_argument("--candidate", type=Path, nargs="+", required=True)
    comparison.add_argument("--contract", type=Path, required=True)
    extract = commands.add_parser("extract", help="inspect archived provenance in a new directory")
    extract.add_argument("root", type=Path)
    extract.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_job(
                args.spec,
                args.output,
                settings=read_object(args.settings) if args.settings else None,
                lock_path=args.lock,
                events_path=args.events,
            )
            print(json.dumps({key: result[key] for key in ("state", "exit_code", "wall_seconds")}))
            return 0 if result["state"] == "completed" else 1
        if args.command == "publish":
            policy = read_object(args.policy) if args.policy else {}
            if set(policy) - {"replacements", "binary_suffixes"}:
                raise ValueError("policy supports only replacements and binary_suffixes")
            redactor = Redactor(
                policy.get("replacements"), binary_suffixes=policy.get("binary_suffixes")
            )
            result = publish(args.source, args.destination, redactor)
            print(json.dumps({"files": len(result["files"]), "output": str(args.destination)}))
        elif args.command == "compare":
            result = compare(args.baseline, args.candidate, read_object(args.contract))
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["passed"] else 1
        elif args.command == "verify":
            print(json.dumps([{"path": str(root), **verify(root)} for root in args.roots]))
        else:
            extract_provenance(args.root, args.destination)
            print(json.dumps({"output": str(args.destination)}))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
