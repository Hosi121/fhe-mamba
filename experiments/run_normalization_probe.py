#!/usr/bin/env python3
"""Run an isolated native normalization probe with build/input provenance.

Standard-library only so the DGX does not need a Python ML environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--mode", choices=("balanced", "coupled", "weighted"), required=True)
    parser.add_argument(
        "--input-mode", choices=("variance", "normalize", "vector"), default="variance"
    )
    parser.add_argument(
        "--fixture", type=Path, help="certified vector fixture; requires vector input mode"
    )
    parser.add_argument("--gamma-placement", choices=("before", "after"), default="before")
    parser.add_argument("--refresh", choices=("none", "output", "output-meta"), default="none")
    parser.add_argument("--meta-alpha", type=int, default=12)
    parser.add_argument("--meta-iterations", type=int, choices=(2, 3), default=2)
    parser.add_argument("--refresh-coordinates", choices=("global", "channel"), default="global")
    parser.add_argument("--internal-refresh", choices=("none", "meta"), default="none")
    parser.add_argument("--inverse-coordinates", choices=("global", "stage"), default="global")
    parser.add_argument(
        "--input-level",
        type=int,
        default=0,
        help="consume real identity/rescale levels before the RMS probe",
    )
    parser.add_argument(
        "--refresh-trigger",
        type=int,
        help="consumed inverse level triggering internal refresh; default depth-6",
    )
    parser.add_argument("--depth", type=int, default=44)
    parser.add_argument("--scale", type=int, default=59)
    parser.add_argument("--security", choices=("128-classic", "not-set"), default="128-classic")
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.refresh_trigger is None:
        args.refresh_trigger = args.depth - 6
    if (args.fixture is not None) != (args.input_mode == "vector"):
        parser.error("vector input mode and --fixture must be supplied together")
    if args.input_mode == "vector" and (args.mode == "coupled" or args.scale < 54):
        parser.error("vector probe requires balanced/weighted mode and scale >=54")
    if args.input_mode != "vector" and (args.refresh != "none" or args.gamma_placement != "before"):
        parser.error("refresh and gamma placement apply only to vector mode")
    if args.refresh_coordinates != "global" and (
        args.input_mode != "vector" or args.refresh == "none"
    ):
        parser.error("channel refresh coordinates require a vector refresh")
    if (args.internal_refresh != "none" or args.input_level != 0) and args.input_mode != "vector":
        parser.error("internal refresh and input-level pressure require vector mode")
    if args.internal_refresh != "none" and (
        args.mode != "balanced" or not 24 <= args.refresh_trigger <= args.depth - 4
    ):
        parser.error("internal refresh requires balanced mode and a trigger in [24,depth-4]")
    if args.inverse_coordinates != "global" and args.internal_refresh == "none":
        parser.error("stage inverse coordinates require internal refresh")
    if not 0 <= args.input_level <= max(0, args.depth - 8):
        parser.error("input-level pressure exceeds the normalization seed budget")
    if not 0 <= args.meta_alpha <= 20:
        parser.error("meta-alpha must be in [0,20]")
    if not 4 <= args.depth <= 44 or not 30 <= args.scale <= 59:
        parser.error("pinned FIDESlib HYBRID-3 probe requires depth 4..44 and scale 30..59")
    raw, log = args.output.with_suffix(".raw.json"), args.output.with_suffix(".log")
    if any(path.exists() for path in (args.output, raw, log)):
        parser.error("refusing to overwrite an existing probe artifact")
    build_manifest = json.loads(args.binary.with_suffix(".build.json").read_text())
    if sha256(args.binary) != build_manifest["binary_sha256"]:
        parser.error("binary differs from its build manifest")
    for name, expected in build_manifest["library_sha256"].items():
        if sha256(Path(name)) != expected:
            parser.error(f"runtime library differs from build manifest: {name}")
    manifest = json.loads((args.recipe.parent / "manifest.json").read_text())
    record = next((r for r in manifest["operators"] if r["file"] == args.recipe.name), None)
    if record is None or record["sha256"] != sha256(args.recipe):
        parser.error("recipe differs from its certified export manifest")
    if args.mode == "weighted" and not record.get("weighted_certificate", {}).get("certified"):
        parser.error("weighted mode requires a rounded-ratio certificate")
    fixture_record = None
    if args.fixture is not None:
        fixture_manifest = json.loads((args.fixture.parent / "manifest.json").read_text())
        fixture_record = next(
            (r for r in fixture_manifest["operators"] if r["file"] == args.fixture.name), None
        )
        if fixture_record is None or fixture_record["sha256"] != sha256(args.fixture):
            parser.error("fixture differs from its export manifest")
        if fixture_record["recipe_sha256"] != record["sha256"]:
            parser.error("fixture belongs to a different normalization recipe")
    command = [
        str(args.binary.resolve()),
        str(raw.resolve()),
        str(args.recipe.resolve()),
        args.mode,
        str(args.depth),
        str(args.scale),
        args.security,
        manifest["repo_commit"],
        build_manifest["binary_sha256"],
        record["sha256"],
        str(args.relative_tolerance),
        args.input_mode,
    ]
    if args.fixture is not None:
        command.extend([str(args.fixture.resolve()), args.gamma_placement, args.refresh])
        command.extend(
            [
                str(args.meta_alpha),
                args.refresh_coordinates,
                args.internal_refresh,
                str(args.input_level),
                str(args.refresh_trigger),
                args.inverse_coordinates,
                str(args.meta_iterations),
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    failure = None
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1", OMP_NUM_THREADS="8")
    with log.open("w") as stream:
        try:
            result = subprocess.run(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
            failure = "probe exceeded timeout"
    if raw.exists():
        report = json.loads(raw.read_text())
    else:
        report = {
            "stage": "normalization-ckks-probe",
            "version": manifest["version"],
            "repo_commit": manifest["repo_commit"],
            "passed": False,
            "status": "failed",
            "mode": args.mode,
            "input_mode": args.input_mode,
            "parameters": {"depth": args.depth, "scale": args.scale, "security": args.security},
            "failure": failure or "native probe failed before producing measurements",
            "measurement_scope": {
                "artifact_level_report": True,
                "full_model_correctness_claimed": False,
                "claim": "Failed isolated encrypted normalization probe; see preserved log.",
            },
        }
    if returncode != 0:
        report.update(passed=False, status="failed")
    report.update(
        build_provenance=build_manifest,
        input_provenance=record,
        normalization_bundle_sha256=manifest["bundle_sha256"],
        runner_source_sha256=sha256(Path(__file__)),
        process_returncode=returncode,
        process_seconds=time.monotonic() - start,
        log_sha256=sha256(log),
        execution_environment={"CUDA_LAUNCH_BLOCKING": "1", "OMP_NUM_THREADS": "8"},
    )
    if fixture_record is not None:
        report.update(
            fixture_provenance=fixture_record,
            fixture_manifest_sha256=sha256(args.fixture.parent / "manifest.json"),
            gamma_placement=args.gamma_placement,
            refresh=args.refresh,
            refresh_coordinates=args.refresh_coordinates,
            internal_refresh=args.internal_refresh,
            input_level=args.input_level,
            refresh_trigger=args.refresh_trigger,
            inverse_coordinates=args.inverse_coordinates,
            meta_iterations=args.meta_iterations,
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "measurements": report.get("measurements")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
