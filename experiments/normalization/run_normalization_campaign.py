#!/usr/bin/env python3
"""Run frozen normalization recipes sequentially with a fresh key per probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modes", nargs="+", choices=("balanced", "weighted"), default=["weighted"]
    )
    parser.add_argument("--sites", nargs="*", help="recipe filenames; default: every exported site")
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--fixtures", type=Path, help="optional packed vector RMS fixtures")
    parser.add_argument("--gamma-placement", choices=("before", "after"), default="before")
    parser.add_argument("--refresh", choices=("none", "output", "output-meta"), default="none")
    parser.add_argument("--meta-alpha", type=int, default=12)
    parser.add_argument("--refresh-coordinates", choices=("global", "channel"), default="global")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refusing to overwrite a campaign directory")
    manifest = json.loads((args.recipes / "manifest.json").read_text())
    selected = [r for r in manifest["operators"] if not args.sites or r["file"] in args.sites]
    if not selected or (args.sites and {r["file"] for r in selected} != set(args.sites)):
        parser.error("no recipes match, or a requested site is absent")
    fixture_records = {}
    if args.fixtures is not None:
        fixture_manifest = json.loads((args.fixtures / "manifest.json").read_text())
        fixture_records = {r["recipe_file"]: r for r in fixture_manifest["operators"]}
        if any(r["file"] not in fixture_records for r in selected):
            parser.error("selected recipe has no vector fixture")
    elif (
        args.refresh != "none"
        or args.gamma_placement != "before"
        or args.refresh_coordinates != "global"
    ):
        parser.error("refresh and gamma placement require vector fixtures")
    if args.refresh_coordinates == "channel" and args.refresh == "none":
        parser.error("channel coordinates require a refresh")
    if not 0 <= args.meta_alpha <= 20:
        parser.error("meta-alpha must be in [0,20]")
    args.output_dir.mkdir(parents=True)
    report = {
        "stage": "vector-rms-ckks-campaign" if args.fixtures else "normalization-ckks-campaign",
        "version": manifest["version"],
        "repo_commit": manifest["repo_commit"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization_bundle_sha256": manifest["bundle_sha256"],
        "input_manifest_sha256": hashlib.sha256(
            (args.recipes / "manifest.json").read_bytes()
        ).hexdigest(),
        "requested_modes": args.modes,
        "requested_sites": [r["file"] for r in selected],
        "expected_runs": len(selected) * len(args.modes),
        "passed": False,
        "status": "running",
        "runs": [],
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Packed vector RMS with learned gamma; fresh keys and explicit refresh policy."
                if args.fixtures
                else "Scalar RMS normalization with fresh keys; no refresh or vector reduction."
            ),
        },
    }
    if args.fixtures:
        report.update(
            fixture_manifest_sha256=hashlib.sha256(
                (args.fixtures / "manifest.json").read_bytes()
            ).hexdigest(),
            gamma_placement=args.gamma_placement,
            refresh=args.refresh,
            meta_alpha=args.meta_alpha,
            refresh_coordinates=args.refresh_coordinates,
        )
    runner = Path(__file__).with_name("run_normalization_probe.py")
    summary = args.output_dir / "campaign.json"
    for record in selected:
        for iteration, mode in enumerate(args.modes, start=1):
            output = args.output_dir / f"{Path(record['file']).stem}-{mode}-{iteration}.json"
            extra = []
            if args.fixtures:
                extra = [
                    "--fixture",
                    str(args.fixtures / fixture_records[record["file"]]["file"]),
                    "--gamma-placement",
                    args.gamma_placement,
                    "--refresh",
                    args.refresh,
                    "--meta-alpha",
                    str(args.meta_alpha),
                    "--refresh-coordinates",
                    args.refresh_coordinates,
                ]
            process = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--binary",
                    str(args.binary),
                    "--recipe",
                    str(args.recipes / record["file"]),
                    "--mode",
                    mode,
                    "--input-mode",
                    "vector" if args.fixtures else "normalize",
                    "--depth",
                    str(args.depth),
                    "--output",
                    str(output),
                    *extra,
                ],
                capture_output=True,
                text=True,
            )
            if output.exists():
                result = json.loads(output.read_text())
                entry = {
                    "file": output.name,
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "passed": result["passed"],
                    "measurements": result.get("measurements"),
                }
            else:
                entry = {"file": output.name, "passed": False, "failure": process.stderr[-4000:]}
            report["runs"].append(entry)
            summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"{output.name}: {'passed' if entry['passed'] else 'FAILED'}", flush=True)
    report["passed"] = all(r["passed"] for r in report["runs"])
    report["status"] = "passed" if report["passed"] else "failed"
    summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
