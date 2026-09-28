"""Isolated native normalization probes and campaigns with common preflight checks.

Standard-library only; each run launches a new native process with fresh keys.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

from fhemamba.benchmarks.io import file_sha256 as sha256
from fhemamba.benchmarks.io import write_json
from fhemamba.benchmarks.process import run_process


@dataclass
class Probe:
    binary: Path
    recipe: Path
    output: Path
    mode: str
    input_mode: str = "variance"
    fixture: Path | None = None
    gamma_placement: str = "before"
    refresh: str = "none"
    meta_alpha: int = 12
    meta_iterations: int = 2
    refresh_coordinates: str = "global"
    internal_refresh: str = "none"
    inverse_coordinates: str = "global"
    input_level: int = 0
    refresh_trigger: int | None = None
    depth: int = 44
    scale: int = 59
    security: str = "128-classic"
    relative_tolerance: float = 1e-4
    timeout: int = 600


CHOICES = {
    "mode": ("balanced", "coupled", "weighted"),
    "input_mode": ("variance", "normalize", "vector"),
    "gamma_placement": ("before", "after"),
    "refresh": ("none", "output", "output-meta"),
    "meta_iterations": (2, 3),
    "refresh_coordinates": ("global", "channel"),
    "internal_refresh": ("none", "meta"),
    "inverse_coordinates": ("global", "stage"),
    "security": ("128-classic", "not-set"),
}


def run(args: Probe) -> dict:
    """Validate inputs, run one native process, retain log/raw output and return its report."""
    for name, choices in CHOICES.items():
        if getattr(args, name) not in choices:
            raise ValueError(f"invalid {name}: {getattr(args, name)!r}")
    if args.refresh_trigger is None:
        args.refresh_trigger = args.depth - 6
    if (args.fixture is not None) != (args.input_mode == "vector"):
        raise ValueError("vector input mode and --fixture must be supplied together")
    if args.input_mode == "vector" and (args.mode == "coupled" or args.scale < 54):
        raise ValueError("vector probe requires balanced/weighted mode and scale >=54")
    if args.input_mode != "vector" and (args.refresh != "none" or args.gamma_placement != "before"):
        raise ValueError("refresh and gamma placement apply only to vector mode")
    if args.refresh_coordinates != "global" and (
        args.input_mode != "vector" or args.refresh == "none"
    ):
        raise ValueError("channel refresh coordinates require a vector refresh")
    if (args.internal_refresh != "none" or args.input_level != 0) and args.input_mode != "vector":
        raise ValueError("internal refresh and input-level pressure require vector mode")
    if args.internal_refresh != "none" and (
        args.mode != "balanced" or not 24 <= args.refresh_trigger <= args.depth - 4
    ):
        raise ValueError("internal refresh requires balanced mode and a trigger in [24,depth-4]")
    if args.inverse_coordinates != "global" and args.internal_refresh == "none":
        raise ValueError("stage inverse coordinates require internal refresh")
    if not 0 <= args.input_level <= max(0, args.depth - 8):
        raise ValueError("input-level pressure exceeds the normalization seed budget")
    if not 0 <= args.meta_alpha <= 20:
        raise ValueError("meta-alpha must be in [0,20]")
    if not 4 <= args.depth <= 44 or not 30 <= args.scale <= 59:
        raise ValueError("pinned FIDESlib HYBRID-3 probe requires depth 4..44 and scale 30..59")
    raw, log = args.output.with_suffix(".raw.json"), args.output.with_suffix(".log")
    if any(path.exists() for path in (args.output, raw, log)):
        raise ValueError("refusing to overwrite an existing probe artifact")
    build_manifest = json.loads(args.binary.with_suffix(".build.json").read_text())
    if sha256(args.binary) != build_manifest["binary_sha256"]:
        raise ValueError("binary differs from its build manifest")
    for name, expected in build_manifest["library_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"runtime library differs from build manifest: {name}")
    manifest = json.loads((args.recipe.parent / "manifest.json").read_text())
    record = next((r for r in manifest["operators"] if r["file"] == args.recipe.name), None)
    if record is None or record["sha256"] != sha256(args.recipe):
        raise ValueError("recipe differs from its certified export manifest")
    if args.mode == "weighted" and not record.get("weighted_certificate", {}).get("certified"):
        raise ValueError("weighted mode requires a rounded-ratio certificate")
    fixture_record = None
    if args.fixture is not None:
        fixture_manifest = json.loads((args.fixture.parent / "manifest.json").read_text())
        fixture_record = next(
            (r for r in fixture_manifest["operators"] if r["file"] == args.fixture.name), None
        )
        if fixture_record is None or fixture_record["sha256"] != sha256(args.fixture):
            raise ValueError("fixture differs from its export manifest")
        if fixture_record["recipe_sha256"] != record["sha256"]:
            raise ValueError("fixture belongs to a different normalization recipe")
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
        returncode, timed_out = run_process(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout
        )
        if timed_out:
            returncode, failure = -1, "probe exceeded timeout"
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
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "measurements": report.get("measurements")}))
    return report


@dataclass
class Campaign:
    binary: Path
    recipes: Path
    output_dir: Path
    modes: tuple[str, ...] = ("weighted",)
    sites: tuple[str, ...] = ()
    depth: int = 40
    fixtures: Path | None = None
    gamma_placement: str = "before"
    refresh: str = "none"
    meta_alpha: int = 12
    refresh_coordinates: str = "global"


def campaign(args: Campaign) -> dict:
    """Run selected recipes sequentially through the same isolated probe API."""
    if not args.modes or set(args.modes) - {"balanced", "weighted"}:
        raise ValueError("campaign modes must be balanced or weighted")
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite a campaign directory")
    manifest = json.loads((args.recipes / "manifest.json").read_text())
    selected = [r for r in manifest["operators"] if not args.sites or r["file"] in args.sites]
    if not selected or (args.sites and {r["file"] for r in selected} != set(args.sites)):
        raise ValueError("no recipes match, or a requested site is absent")
    fixture_records = {}
    if args.fixtures is not None:
        fixture_manifest = json.loads((args.fixtures / "manifest.json").read_text())
        fixture_records = {r["recipe_file"]: r for r in fixture_manifest["operators"]}
        if any(r["file"] not in fixture_records for r in selected):
            raise ValueError("selected recipe has no vector fixture")
    elif (
        args.refresh != "none"
        or args.gamma_placement != "before"
        or args.refresh_coordinates != "global"
    ):
        raise ValueError("refresh and gamma placement require vector fixtures")
    if args.refresh_coordinates == "channel" and args.refresh == "none":
        raise ValueError("channel coordinates require a refresh")
    if not 0 <= args.meta_alpha <= 20:
        raise ValueError("meta-alpha must be in [0,20]")
    args.output_dir.mkdir(parents=True)
    report = {
        "stage": "vector-rms-ckks-campaign" if args.fixtures else "normalization-ckks-campaign",
        "version": manifest["version"],
        "repo_commit": manifest["repo_commit"],
        "source_sha256": sha256(Path(__file__)),
        "normalization_bundle_sha256": manifest["bundle_sha256"],
        "input_manifest_sha256": sha256(args.recipes / "manifest.json"),
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
            fixture_manifest_sha256=sha256(args.fixtures / "manifest.json"),
            gamma_placement=args.gamma_placement,
            refresh=args.refresh,
            meta_alpha=args.meta_alpha,
            refresh_coordinates=args.refresh_coordinates,
        )
    summary = args.output_dir / "campaign.json"
    for record in selected:
        for iteration, mode in enumerate(args.modes, start=1):
            output = args.output_dir / f"{Path(record['file']).stem}-{mode}-{iteration}.json"
            try:
                result = run(
                    Probe(
                        binary=args.binary,
                        recipe=args.recipes / record["file"],
                        output=output,
                        mode=mode,
                        depth=args.depth,
                        input_mode="vector" if args.fixtures else "normalize",
                        fixture=(
                            args.fixtures / fixture_records[record["file"]]["file"]
                            if args.fixtures
                            else None
                        ),
                        gamma_placement=args.gamma_placement,
                        refresh=args.refresh,
                        meta_alpha=args.meta_alpha,
                        refresh_coordinates=args.refresh_coordinates,
                    )
                )
                entry = {
                    "file": output.name,
                    "sha256": sha256(output),
                    "passed": result["passed"],
                    "measurements": result.get("measurements"),
                }
            except (OSError, ValueError, KeyError) as error:
                entry = {"file": output.name, "passed": False, "failure": str(error)[-4000:]}
            report["runs"].append(entry)
            write_json(summary, report)
            print(f"{output.name}: {'passed' if entry['passed'] else 'FAILED'}", flush=True)
    report["passed"] = all(r["passed"] for r in report["runs"])
    report["status"] = "passed" if report["passed"] else "failed"
    write_json(summary, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, config in (("probe", Probe), ("campaign", Campaign)):
        child = commands.add_parser(name)
        for option in fields(config):
            key = option.name
            settings = {}
            if key in {"modes", "sites"}:
                settings["nargs"] = "+" if key == "modes" else "*"
            if key == "modes":
                settings["choices"] = ("balanced", "weighted")
            elif key in CHOICES:
                settings["choices"] = CHOICES[key]
            kind = {"Path": Path, "int": int, "float": float}.get(option.type.split(" | ")[0], str)
            if option.default is MISSING:
                settings["required"] = True
            else:
                settings["default"] = option.default
            child.add_argument("--" + key.replace("_", "-"), type=kind, **settings)
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    try:
        report = run(Probe(**options)) if command == "probe" else campaign(Campaign(**options))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
