"""Conditional normalization schedules and certified native probe inputs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from fhemamba.benchmarks.io import file_sha256, payload_sha256, sha256, write_json
from fhemamba.calibration.payload import layers, source_hashes
from fhemamba.calibration.payload import report as make_report
from fhemamba.m1_payload import _poly_ops_from_export
from fhemamba.normalization import (
    ScheduledInvSqrt,
    certify_schedule,
    certify_weighted_schedule,
    fixed_newton_cost,
    plan_invsqrt,
)
from fhemamba.ops import (
    PolyInitNewton,
    SquaredPolyInitNewton,
    _poly_interval,
    positive_binomial_seed,
)


def plan_normalization(
    payload: Path, *, bundle: Path, relative_error=1e-07, grid_size=8193
) -> dict:
    sources = source_hashes(__file__, "normalization.py")
    if grid_size < 2:
        raise ValueError("grid-size must be at least two")
    if bundle.exists():
        raise ValueError("refusing to overwrite an existing frozen bundle/report")
    start = time.monotonic()
    chain = json.loads((payload / "chain.json").read_text())
    layer_count = chain["n_layers"]
    metadata = [meta for _, meta in layers(payload, chain)]
    ops = _poly_ops_from_export(payload, layer_count)
    operators, measurements = ([], [])
    for (layer, name), old in sorted(ops.layer_polys.items()):
        if name not in ("rms_invsqrt", "gated_rms_invsqrt"):
            continue
        eps = (
            chain["final_norm_eps"]
            if layer == layer_count
            else metadata[layer]["eps"][
                "gated_norm" if name == "gated_rms_invsqrt" else "block_norm"
            ]
        )
        lo = min(float(eps), float(np.float32(eps)))
        hi = 4 * _poly_interval(old)[1]
        schedule = plan_invsqrt(lo, hi, tolerance=relative_error)
        operators.append({"layer": layer, "site": name, "recipe": schedule.recipe()})
        grid = torch.from_numpy(np.geomspace(lo, hi, grid_size))
        factor = schedule(grid) * grid.sqrt()
        squared = isinstance(old, SquaredPolyInitNewton)
        seed = positive_binomial_seed(hi, 63, power=0.25 if squared else 0.5)
        previous = SquaredPolyInitNewton(seed, 8, 0.85) if squared else PolyInitNewton(seed, 8)
        previous_factor = previous(grid) * grid.sqrt()
        measurements.append(
            {
                "layer": layer,
                "site": name,
                "interval": [lo, hi],
                "certificate": certify_schedule(schedule),
                "abstract_cost": schedule.abstract_cost(),
                "fixed_newton_same_accuracy": fixed_newton_cost(schedule),
                "sampled_float64_relative_error": float((factor - 1).abs().max()),
                "previous_binomial63_newton8_sampled_relative_error": float(
                    (previous_factor - 1).abs().max()
                ),
                "grid_size": grid_size,
            }
        )
    bundle_data = {
        "format": "fhemamba-normalization-schedules-v1",
        "input_payload_sha256": payload_sha256(payload),
        "domain_policy": "[min(eps64,eps32),4*old_hi]; upper endpoints remain conditional",
        "operators": operators,
    }
    write_json(bundle, bundle_data)
    report = make_report(
        "normalization-schedule-certificate",
        sources,
        input_payload_sha256=bundle_data["input_payload_sha256"],
        normalization_source_sha256=file_sha256(Path(__file__).parents[1] / "normalization.py"),
        bundle_sha256=file_sha256(bundle),
        operator_count=len(operators),
        all_certified=all(m["certificate"]["certified"] for m in measurements),
        planning_seconds=time.monotonic() - start,
        operators=measurements,
        measurement_scope={
            "text_used_in_planning": False,
            "includes_ckks_error": False,
            "global_variance_domain_membership_proved": False,
            "claim": "Conditional real-arithmetic error certificates; symbolic ct-ct depth only.",
        },
    )
    print(f"certified {len(operators)} schedules; bundle {report['bundle_sha256']}")
    return report


def export_normalization(bundle: Path, *, output_dir: Path) -> dict:
    sources = source_hashes(__file__, "normalization.py")
    if output_dir.exists():
        raise ValueError("refusing to overwrite a probe input directory")
    bundle_path = bundle
    bundle = json.loads(bundle_path.read_text())
    if bundle["format"] != "fhemamba-normalization-schedules-v1":
        raise ValueError("unsupported bundle format")
    records, files = [], {}
    for entry in bundle["operators"]:
        layer, site = entry["layer"], entry["site"]
        if (
            site not in ("rms_invsqrt", "gated_rms_invsqrt")
            or not isinstance(layer, int)
            or layer < 0
        ):
            raise ValueError("invalid normalization site")
        schedule = ScheduledInvSqrt.from_recipe(entry["recipe"])
        certificate = certify_schedule(schedule)
        weighted_certificate = certify_weighted_schedule(schedule)
        if not certificate["certified"] or not weighted_certificate["certified"]:
            raise ValueError(f"uncertified recipe at {layer}:{site}")
        name = f"l{layer:02d}_{site}.txt"
        if name in files:
            raise ValueError("duplicate normalization site")
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
                "sha256": sha256(content.encode()),
                "interval": schedule.interval,
                "certified_relative_error": certificate["relative_error_bound"],
                "weighted_certificate": weighted_certificate,
                "coupled_cost": schedule.abstract_cost(),
                "balanced_cost": schedule.balanced_cost(),
                "fixed_newton_same_accuracy": fixed_newton_cost(schedule),
            }
        )
    if not records:
        raise ValueError("empty normalization bundle")
    report = make_report(
        "normalization-probe-inputs",
        sources,
        normalization_source_sha256=file_sha256(Path(__file__).parents[1] / "normalization.py"),
        bundle_sha256=file_sha256(bundle_path),
        input_payload_sha256=bundle["input_payload_sha256"],
        operators=records,
        measurement_scope={
            "claim": "Certified normalization recipes and the rounded-ratio weighted variant."
        },
    )
    output_dir.mkdir(parents=True)
    for name, content in files.items():
        (output_dir / name).write_text(content)
    write_json(output_dir / "manifest.json", report)
    return report
