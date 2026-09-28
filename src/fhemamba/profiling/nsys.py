#!/usr/bin/env python3
"""Summarize an Nsight Systems SQLite export without counting overlap twice."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from fhemamba.benchmarks.io import file_sha256 as _digest

_GPU_TABLES = {
    "kernel": "CUPTI_ACTIVITY_KIND_KERNEL",
    "copy": "CUPTI_ACTIVITY_KIND_MEMCPY",
    "memset": "CUPTI_ACTIVITY_KIND_MEMSET",
    "graph": "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
}


def merged_intervals(intervals):
    """Merge start-sorted half-open intervals, including nested streams/graphs."""
    left = right = None
    for start, end in intervals:
        if end < start:
            raise ValueError("negative activity duration")
        if left is None:
            left, right = start, end
        elif start > right:
            yield left, right
            left, right = start, end
        else:
            right = max(right, end)
    if left is not None:
        yield left, right


def overlap_duration(first, second):
    """Intersection of two ordered, disjoint interval streams."""
    first, second = iter(first), iter(second)
    a, b = next(first, None), next(second, None)
    total = 0
    while a is not None and b is not None:
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] < b[1]:
            a = next(first, None)
        else:
            b = next(second, None)
    return total


def summarize(connection, *, device=None, top=20, bin_seconds=10, start_seconds=0):
    if top < 1 or bin_seconds < 1:
        raise ValueError("top and bin_seconds must be positive")
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and nonnegative")
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    activity_tables = {kind: table for kind, table in _GPU_TABLES.items() if table in tables}
    if not activity_tables or "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
        raise ValueError("trace needs GPU activities and CUDA runtime events")
    devices = set()
    for table in activity_tables.values():
        devices.update(
            row[0] for row in connection.execute(f"SELECT DISTINCT deviceId FROM {table}")
        )
    if device is None and len(devices) != 1:
        raise ValueError("select --device for a trace containing zero or multiple devices")
    device = next(iter(devices)) if device is None else device
    if device not in devices:
        raise ValueError("selected device has no activities")
    # Runtime APIs cannot all be assigned to a device; do not silently mix them
    # into a supposedly per-device utilization or CPU-wait report.
    if len(devices) != 1:
        raise ValueError("use an isolated single-device capture for CPU/GPU attribution")
    unions = " UNION ALL ".join(
        f"SELECT start,end FROM {table} WHERE deviceId=?" for table in activity_tables.values()
    )
    parameters = [device] * len(activity_tables)
    gpu_start, gpu_end = connection.execute(
        f"SELECT MIN(start),MAX(end) FROM ({unions})", parameters
    ).fetchone()
    api_start, api_end = connection.execute(
        "SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()
    if api_start is None:
        raise ValueError("trace has no CUDA runtime events")
    start, end = min(gpu_start, api_start), max(gpu_end, api_end)
    start += round(start_seconds * 1e9)
    span = end - start
    if span <= 0:
        raise ValueError("trace has no positive time span")
    width = bin_seconds * 1_000_000_000
    bins = [0] * ((span + width - 1) // width)
    covered = longest_gap = 0
    gap_count = gap_total = 0
    previous = start
    for left, right in merged_intervals(
        connection.execute(
            f"SELECT MAX(start,?),MIN(end,?) FROM ({unions}) "
            "WHERE end>? AND start<? ORDER BY start,end",
            [start, end, *parameters, start, end],
        )
    ):
        gap = left - previous
        longest_gap = max(longest_gap, gap)
        if gap >= 1_000_000:
            gap_count += 1
            gap_total += gap
        covered += right - left
        if right > left:
            for i in range((left - start) // width, (right - 1 - start) // width + 1):
                bins[i] += min(right, start + (i + 1) * width) - max(left, start + i * width)
        previous = right
    tail = end - previous
    longest_gap = max(longest_gap, tail)
    if tail >= 1_000_000:
        gap_count += 1
        gap_total += tail
    gpu_query = (
        f"SELECT MAX(start,?),MIN(end,?) FROM ({unions}) WHERE end>? AND start<? ORDER BY start,end"
    )
    api_query = (
        "SELECT MAX(start,?),MIN(end,?) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE end>? AND start<? ORDER BY start,end"
    )
    api_parameters = [start, end, start, end]
    api_covered = sum(
        right - left
        for left, right in merged_intervals(connection.execute(api_query, api_parameters))
    )
    api_gpu_overlap = overlap_duration(
        merged_intervals(connection.execute(gpu_query, [start, end, *parameters, start, end])),
        merged_intervals(connection.execute(api_query, api_parameters)),
    )
    totals = {}
    for kind, table in activity_tables.items():
        count, duration = connection.execute(
            f"SELECT COUNT(*),COALESCE(SUM(MIN(end,?)-MAX(start,?)),0) FROM {table} "
            "WHERE deviceId=? AND end>? AND start<?",
            (end, start, device, start, end),
        ).fetchone()
        totals[kind] = {"instances": count, "summed_seconds": duration / 1e9}

    def timing_rows(table, name_column):
        return [
            {
                "name": name,
                "instances": count,
                "summed_seconds": duration / 1e9,
                "mean_microseconds": duration / count / 1e3,
            }
            for name, count, duration in connection.execute(
                f"SELECT s.value,t.n,t.duration FROM (SELECT {name_column} AS name,"
                f"COUNT(*) AS n,SUM(MIN(end,?)-MAX(start,?)) AS duration FROM {table} "
                f"WHERE end>? AND start<? GROUP BY {name_column} "
                "ORDER BY duration DESC LIMIT ?) t JOIN StringIds s ON s.id=t.name",
                (end, start, start, end, top),
            )
        ]

    result = {
        "schema_version": 1,
        "device": device,
        "window_start_seconds": start_seconds,
        "trace_window_seconds": span / 1e9,
        "gpu_activity_union_seconds": covered / 1e9,
        "gpu_activity_coverage": covered / span,
        "cuda_api_union_seconds": api_covered / 1e9,
        "gpu_and_cuda_api_overlap_seconds": api_gpu_overlap / 1e9,
        "gpu_idle_inside_cuda_api_seconds": (api_covered - api_gpu_overlap) / 1e9,
        "gpu_idle_outside_cuda_api_seconds": (span - covered - api_covered + api_gpu_overlap) / 1e9,
        "longest_gpu_gap_seconds": longest_gap / 1e9,
        "gpu_gaps_at_least_1ms": {"count": gap_count, "seconds": gap_total / 1e9},
        "activities": totals,
        "cuda_api": timing_rows("CUPTI_ACTIVITY_KIND_RUNTIME", "nameId"),
        "kernels": timing_rows(_GPU_TABLES["kernel"], "demangledName")
        if _GPU_TABLES["kernel"] in tables
        else [],
        "time_bins": [
            {
                "offset_seconds": start_seconds + i * bin_seconds,
                "duration_seconds": min(width, span - i * width) / 1e9,
                "gpu_activity_coverage": value / min(width, span - i * width),
            }
            for i, value in enumerate(bins)
        ],
        "interpretation": [
            "The window spans traced CUDA APIs and GPU activities, not process setup.",
            "Activity coverage is a timeline union, not SM occupancy or HBM utilization.",
            "Graph envelopes include internal gaps when graph-level tracing is enabled.",
            "Summed API and kernel durations overlap; they are not a wall-time partition.",
            "GPU-idle/API overlap locates gaps; it does not prove their cause or avoidable cost.",
            "Profiling perturbs execution; use independent unprofiled runs for speed claims.",
        ],
    }
    if "OSRT_API" in tables:
        result["os_runtime"] = timing_rows("OSRT_API", "nameId")
    if {"COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS"} <= tables:
        result["cpu_leaf_samples"] = [
            {
                "thread": thread,
                "symbol": symbol,
                "module": module,
                "unresolved": bool(unresolved),
                "samples": count,
            }
            for thread, symbol, module, unresolved, count in connection.execute(
                "SELECT e.globalTid,s.value,m.value,c.unresolved,COUNT(*) AS n "
                "FROM COMPOSITE_EVENTS e "
                "JOIN SAMPLING_CALLCHAINS c ON c.id=e.id AND c.stackDepth=0 "
                "JOIN StringIds s ON s.id=c.symbol LEFT JOIN StringIds m ON m.id=c.module "
                "WHERE e.start>=? AND e.start<? "
                "GROUP BY e.globalTid,c.symbol,c.module,c.unresolved "
                "ORDER BY n DESC LIMIT ?",
                (start, end, top),
            )
        ]
        result["interpretation"].append("CPU leaf sample counts are not measured wall seconds.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fhemamba profile nsys", description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--bin-seconds", type=int, default=10)
    parser.add_argument(
        "--start-seconds", type=float, default=0, help="clip away a startup interval"
    )
    args = parser.parse_args(argv)
    with closing(sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        result = summarize(
            conn,
            device=args.device,
            top=args.top,
            bin_seconds=args.bin_seconds,
            start_seconds=args.start_seconds,
        )
    result["input"] = {"sha256": _digest(args.sqlite), "bytes": args.sqlite.stat().st_size}
    result["analyzer_sha256"] = _digest(Path(__file__))
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
