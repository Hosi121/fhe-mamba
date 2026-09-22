#!/usr/bin/env python3
"""Certify and export frozen normalization coefficients to the native micro-probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba import __version__
from fhemamba.artifacts import current_git_commit
from fhemamba.normalization import (
    ScheduledInvSqrt,
    certify_schedule,
    certify_weighted_schedule,
    fixed_newton_cost,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refusing to overwrite a probe input directory")
    bundle = json.loads(args.bundle.read_text())
    if bundle["format"] != "fhemamba-normalization-schedules-v1":
        parser.error("unsupported bundle format")
    records, files = [], {}
    for entry in bundle["operators"]:
        layer, site = entry["layer"], entry["site"]
        if (
            site not in ("rms_invsqrt", "gated_rms_invsqrt")
            or not isinstance(layer, int)
            or layer < 0
        ):
            parser.error("invalid normalization site")
        schedule = ScheduledInvSqrt.from_recipe(entry["recipe"])
        certificate = certify_schedule(schedule)
        weighted_certificate = certify_weighted_schedule(schedule)
        if not certificate["certified"] or not weighted_certificate["certified"]:
            parser.error(f"uncertified recipe at {layer}:{site}")
        name = f"l{layer:02d}_{site}.txt"
        if name in files:
            parser.error("duplicate normalization site")
        lines = [
            "fhemamba-invsqrt-v1",
            f"{schedule.lo:.17g} {schedule.hi:.17g} {schedule.seed:.17g} "
            f"{len(schedule.coefficients)}",
            *(f"{a:.17g} {b:.17g}" for a, b in schedule.coefficients),
        ]
        content = "\n".join(lines) + "\n"
        files[name] = content
        records.append(
            {
                "layer": layer,
                "site": site,
                "file": name,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "interval": schedule.interval,
                "certified_relative_error": certificate["relative_error_bound"],
                "weighted_certificate": weighted_certificate,
                "coupled_cost": schedule.abstract_cost(),
                "balanced_cost": schedule.balanced_cost(),
                "fixed_newton_same_accuracy": fixed_newton_cost(schedule),
            }
        )
    if not records:
        parser.error("empty normalization bundle")
    report = {
        "stage": "normalization-probe-inputs",
        "version": __version__,
        "repo_commit": current_git_commit(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/fhemamba/normalization.py").read_bytes()
        ).hexdigest(),
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "input_payload_sha256": bundle["input_payload_sha256"],
        "operators": records,
        "measurement_scope": {
            "artifact_level_report": True,
            "encrypted_execution": False,
            "full_model_correctness_claimed": False,
            "claim": "Certified normalization recipes and the rounded-ratio weighted variant.",
        },
    }
    args.output_dir.mkdir(parents=True)
    for name, content in files.items():
        (args.output_dir / name).write_text(content)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(f"exported {len(records)} certified recipes")


if __name__ == "__main__":
    main()
