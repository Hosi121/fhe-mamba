#!/usr/bin/env python3
"""Select hardware counters from an Nsight Compute --page raw --csv export."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METRICS = {
    "duration": "gpu__time_duration.sum",
    "sm_throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_bandwidth": "dram__bytes.sum.per_second",
    "achieved_occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "registers_per_thread": "launch__registers_per_thread",
    "shared_memory_per_block": "launch__shared_mem_per_block",
}


def summarize(stream, additional_metrics=()):
    requested = {**METRICS, **{name: name for name in additional_metrics}}
    rows = list(csv.DictReader(stream))
    if not rows or rows[0].get("ID") != "" or not set(requested.values()) <= rows[0].keys():
        raise ValueError("expected a raw CSV export with its units row and required metrics")
    units, kernels, seen = rows[0], [], set()
    for row in rows[1:]:
        identifier = int(row["ID"])
        if identifier in seen:
            raise ValueError("duplicate kernel ID")
        seen.add(identifier)
        metrics = {}
        for name, column in requested.items():
            value = float(row[column].replace(",", ""))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid hardware counter: {column}")
            metrics[name] = {"value": value, "unit": units[column], "counter": column}
        kernels.append(
            {
                "id": identifier,
                "name": row["Kernel Name"],
                "device": row["Device"],
                "block_size": row["Block Size"],
                "grid_size": row["Grid Size"],
                "metrics": metrics,
            }
        )
    if not kernels:
        raise ValueError("no profiled kernels")
    return {
        "schema_version": 1,
        "kernels": kernels,
        "interpretation": (
            "Selected launches under profiler replay and its cache policy; these are not "
            "full-model averages or unprofiled kernel durations. Keep units and launch filters."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fhemamba profile ncu", description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric", action="append", default=[], help="extra raw counter column; repeatable"
    )
    args = parser.parse_args(argv)
    with args.csv.open(newline="") as stream:
        result = summarize(stream, args.metric)
    result["input"] = {
        "sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "bytes": args.csv.stat().st_size,
    }
    result["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
