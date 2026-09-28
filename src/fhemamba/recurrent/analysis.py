#!/usr/bin/env python3
"""Validate matched recurrence runs and report measured crossover brackets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from fhemamba.benchmarks.comparison import compare, validate_samples
from fhemamba.benchmarks.io import file_identity, read_object, write_json

CONTRACT = {
    "schema_version": 1,
    "metric": "eval_seconds",
    "equal": [
        "security_audit",
        "slots",
        "payload_slots",
        "ring_dimension",
        "depth",
        "scale_bits",
        "refresh_ring_dimension",
        "bootstrap_passes",
        "refresh_ceiling",
        "refreshed_level",
        "plaintext_cache_capacity",
        "prefetch_workers",
        "exact_tolerance",
        "polynomial_tolerance",
        "rotation_decomposition",
        "refresh_policy",
        "routing_method",
    ],
    "require": {
        "passed": True,
        "encrypted": True,
        "security": "128-classic",
        "security_audit.passed": True,
        "security_audit.uniform_ternary": True,
        "security_audit.library_classical128": True,
        "gpu_dual_ring": False,
        "evaluation_decryptions": 0,
        "client_output_decrypt_count": 0,
        "non_finite": 0,
        "bootstrap_passes": 2,
        "s2c_first": True,
        "batch_refresh": True,
        "frontier_refresh": True,
        "merge_refresh_correction": True,
        "gpu_plaintext_fft": True,
        "gpu_plaintext_ntt": True,
        "gpu_plaintext_rns": True,
        "gpu_addend_rns": True,
        "hoist_rotations": True,
        "share_chebyshev": True,
        "reuse_dead_inputs": True,
        "compact_weights": True,
        "profile_evaluation": True,
        "inplace_ops": True,
        "cache_plaintexts": True,
        "fast_plaintext_upload": True,
        "direct_plaintext_upload": True,
        "move_plaintext_coefficients": True,
        "borrow_plaintext_upload": True,
        "bsgs_routing_stages": True,
        "prefetch_plaintexts": True,
        "exact_tolerance": 0.001,
        "polynomial_tolerance": 0.001,
    },
    "maximum": {"max_abs_error_vs_exact": 0.001, "max_abs_error_vs_polynomial": 0.001},
    "minimum_reduction": 0,
}


IDENTITY_FIELDS = ("binary", "fideslib", "openfhe_core", "source", "native_source", "trace")


def load_measurement(manifest_path, directory, mode, steps):
    """Bind a run to all retained readouts and preserve unsuccessful processes."""
    case = directory.name
    manifest = read_object(manifest_path)
    record = read_object(directory / "run.json")
    if manifest["representation"] != mode or manifest["tokens"] != steps:
        raise ValueError(f"{case}: mismatched representation or step count")
    readouts = [item["step"] for item in manifest["output_names"] if "readout" in item["name"]]
    if readouts != list(range(1, steps + 1)):
        raise ValueError(f"{case}: every step must retain its readout")
    for name in ("program", "fixture"):
        suffix = "txt" if name == "program" else "npz"
        if record["inputs"][name]["sha256"] != manifest["files_sha256"][f"{name}.{suffix}"]:
            raise ValueError(f"{case}: {name} digest differs")
    if record["inputs"]["manifest"]["sha256"] != file_identity(manifest_path)["sha256"]:
        raise ValueError(f"{case}: manifest digest differs")
    if record["inputs"]["trace"]["sha256"] != manifest["trace_sha256"]:
        raise ValueError(f"{case}: trace digest differs")
    native_path = directory / "native.json"
    if record["state"] != "completed" or record["exit_code"] != 0 or not native_path.exists():
        log_path = directory / "run.log"
        log = log_path.read_text() if log_path.exists() else ""
        failure = {
            "case": case,
            "process_state": record["state"],
            "process_exit_code": record["exit_code"],
            "process_wall_seconds": record["wall_seconds"],
            "native_result_exists": native_path.exists(),
            "cuda_out_of_memory": "out of memory" in log.lower(),
            "reason": "missing native result or unsuccessful process; no timing comparison",
        }
        return manifest, record, None, failure
    native = read_object(native_path)
    if len(native["per_output_errors"]) != len(manifest["output_names"]):
        raise ValueError(f"{case}: missing output validation")
    if native["operation_stats"]["sum"]["nodes"] != manifest["program"]["operations"]["sum"]:
        raise ValueError(f"{case}: readout operations were pruned")
    return manifest, record, native, None


def analyze(payload, results, lengths):
    pairs, identities, failures = [], [], []
    valid_paths = {"factored": [], "tiled": []}
    for steps in lengths:
        manifests, measurements, records = {}, {}, {}
        for mode in ("factored", "tiled"):
            case = f"{mode}-{steps}"
            manifest_path = payload / case / "manifest.json"
            manifest, record, native, failure = load_measurement(
                manifest_path, results / case, mode, steps
            )
            identities.append({name: record["inputs"][name]["sha256"] for name in IDENTITY_FIELDS})
            manifests[mode], records[mode] = manifest, record
            if failure:
                failures.append(failure)
                measurements[mode] = None
                continue
            measurements[mode] = native
            valid_paths[mode].append(results / case / "native.json")
        for field in ("geometry", "tokens", "layer", "trace_sha256"):
            if manifests["factored"][field] != manifests["tiled"][field]:
                raise ValueError(f"matched arms differ in {field}")
        row = {"steps": steps, "complete_pair": all(measurements.values()), "arms": {}}
        for mode, native in measurements.items():
            if native is None:
                row["arms"][mode] = {"qualified": False}
                continue
            row["arms"][mode] = {
                key: native[key]
                for key in (
                    "eval_seconds",
                    "setup_seconds",
                    "bootstrap_seconds",
                    "bootstraps",
                    "ct_ct_mul",
                    "ct_pt_mul",
                    "rotations",
                    "max_abs_error_vs_exact",
                    "max_abs_error_vs_polynomial",
                    "per_output_errors",
                    "peak_rss_gib",
                )
            }
            arithmetic = sum(
                stats["seconds"]
                for op, stats in native["operation_stats"].items()
                if op not in ("input", "public")
            )
            materialization = sum(
                stats["seconds"]
                for op, stats in native["operation_stats"].items()
                if op in ("input", "public")
            )
            row["arms"][mode].update(
                qualified=True,
                evaluation_seconds_per_step=native["eval_seconds"] / steps,
                state_arithmetic_seconds=arithmetic,
                state_arithmetic_seconds_per_step=arithmetic / steps,
                input_materialization_seconds=materialization,
                process_wall_seconds=records[mode]["wall_seconds"],
                carried_ciphertexts=manifests[mode]["steps"][-1]["carried_ciphertexts"],
                carried_coordinates=manifests[mode]["steps"][-1]["carried_coordinates"],
            )
        if row["complete_pair"]:
            baseline = measurements["factored"]["eval_seconds"]
            candidate = measurements["tiled"]["eval_seconds"]
            row["tiled_evaluation_faster"] = candidate <= baseline
            row["reduction_fraction"] = 1 - candidate / baseline
            row["speedup"] = baseline / candidate
            state_baseline = row["arms"]["factored"]["state_arithmetic_seconds"]
            state_candidate = row["arms"]["tiled"]["state_arithmetic_seconds"]
            row["state_arithmetic_speedup"] = state_baseline / state_candidate
        pairs.append(row)
    if not identities or any(identity != identities[0] for identity in identities):
        raise ValueError("runs do not share a binary, libraries, source and input trace")
    # Validate every completed sample against the same crypto/runtime contract,
    # including successful arms whose partner failed. Discard the cross-length
    # aggregate timing: speed comparisons above only use equal-length pairs.
    # compare raises on any crypto/accuracy mismatch; a slower arm is valid data.
    compare(valid_paths["factored"], valid_paths["tiled"], CONTRACT)
    return {
        "schema": "fhemamba-recurrence-comparison-v1",
        "passed": not failures,
        "all_tiled_runs_qualified": all(row["arms"]["tiled"]["qualified"] for row in pairs),
        "failures": failures,
        "identities": identities[0],
        "pairs": pairs,
        "first_measured_tiled_evaluation_win": next(
            (row["steps"] for row in pairs if row.get("speedup", 0) > 1), None
        ),
        "first_measured_tiled_arithmetic_win": next(
            (row["steps"] for row in pairs if row.get("state_arithmetic_speedup", 0) > 1), None
        ),
        "scope": "one recurrence layer; offline encrypted inputs; setup excluded from eval",
        "limitations": [
            "one process per arm and length; no statistical confidence interval",
            "no claim about full-model latency, encrypted autoregressive feedback or text quality",
            "bounded carried state; executor DAG and retained validation outputs still grow",
            "finite trace accuracy test, not a long-horizon numerical stability proof",
        ],
    }


def progress_samples(path):
    """Read completed-operation samples, including logs without native JSON."""
    samples = []
    for line in path.read_text().splitlines():
        match = re.match(r"node=(\d+)/(\d+) op=\S+ seconds=([\d.e+-]+) ", line)
        if match is None:
            continue
        row = {
            "completed_nodes": int(match[1]),
            "total_nodes": int(match[2]),
            "seconds": float(match[3]),
        }
        for key in (
            "bootstraps",
            "live_values",
            "peak_live_values",
            "cache_entries",
            "device_used_bytes",
            "sampled_peak_device_bytes",
        ):
            value = re.search(rf"\b{key}=(\d+)\b", line)
            if value:
                row[key] = int(value[1])
        samples.append(row)
    return samples


def progress_summary(path):
    samples = progress_samples(path)
    return {"sample_count": len(samples), "last": samples[-1] if samples else None}


def _schedule_command(command):
    """Remove only output/container identities and the tested scheduling option."""
    normalized = list(command)
    for flag in ("--name", "--frontier-live-limit"):
        if flag == "--name" and flag not in normalized:
            continue
        if normalized.count(flag) != 1:
            raise ValueError(f"expected exactly one {flag}")
        normalized[normalized.index(flag) + 1] = f"<{flag}>"
    outputs = [i for i, item in enumerate(normalized) if item.endswith("/native.json")]
    if len(outputs) != 1:
        raise ValueError("expected exactly one native output path")
    normalized[outputs[0]] = "<native-output>"
    return normalized


def analyze_schedule(payload, results, study):
    """Qualify a memory admission study without treating OOM as a speed baseline."""
    if study.get("schema_version") != 1 or not study.get("cases"):
        raise ValueError("schedule study requires schema_version 1 and cases")
    records, manifests, natives, rows, paths = {}, {}, {}, {}, []
    identity = None
    for case in study["cases"]:
        name, mode, steps, limit = (case[key] for key in ("name", "mode", "steps", "limit"))
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in records:
            raise ValueError("case names must be unique directory names")
        if mode not in ("factored", "tiled") or type(steps) is not int or steps <= 0:
            raise ValueError("invalid representation or step count")
        if type(limit) is not int or not 0 <= limit <= 1000000:
            raise ValueError("invalid frontier live limit")
        manifest, record, native, failure = load_measurement(
            payload / f"{mode}-{steps}" / "manifest.json", results / name, mode, steps
        )
        command = record["spec"]["command"]
        _schedule_command(command)
        if command[command.index("--frontier-live-limit") + 1] != str(limit):
            raise ValueError(f"{name}: command limit differs from study")
        current = {key: record["inputs"][key]["sha256"] for key in IDENTITY_FIELDS}
        if identity is not None and current != identity:
            raise ValueError("runs do not share a binary, libraries, source and input trace")
        identity = current
        records[name], manifests[name], natives[name] = record, manifest, native
        row = dict(
            case,
            qualified=native is not None,
            job_wall_seconds=record["wall_seconds"],
            lock_wait_seconds=record.get("lock_wait_seconds", 0),
            process_wall_seconds=record["wall_seconds"] - record.get("lock_wait_seconds", 0),
            progress=progress_summary(results / name / "run.log"),
        )
        if failure:
            row["failure"] = failure
        else:
            if native["frontier_live_limit"] != limit:
                raise ValueError(f"{name}: measured limit differs from study")
            for key in (
                "frontier_limit_selections",
                "maximum_live_dag_values",
                "sampled_peak_device_bytes",
                "device_total_bytes",
            ):
                if type(native[key]) is not int or native[key] < 0:
                    raise ValueError(f"{name}: invalid {key}")
            if not 0 < native["sampled_peak_device_bytes"] <= native["device_total_bytes"]:
                raise ValueError(f"{name}: invalid sampled device memory")
            if not limit and native["frontier_limit_selections"]:
                raise ValueError(f"{name}: unlimited control applied an admission limit")
            paths.append(results / name / "native.json")
            row["measurement"] = {
                key: native[key]
                for key in (
                    "eval_seconds",
                    "setup_seconds",
                    "bootstrap_seconds",
                    "bootstraps",
                    "frontier_limit_selections",
                    "maximum_live_dag_values",
                    "sampled_peak_device_bytes",
                    "device_total_bytes",
                    "plaintext_cache_entries",
                    "max_abs_error_vs_exact",
                    "max_abs_error_vs_polynomial",
                    "peak_rss_gib",
                )
            }
            row["measurement"]["state_arithmetic_seconds"] = sum(
                stats["seconds"]
                for op, stats in native["operation_stats"].items()
                if op not in ("input", "public")
            )
        rows[name] = row
    # A candidate must pass the original cryptographic and numerical gate even
    # if its baseline ran out of memory, or it is the only completed sample.
    if paths:
        validate_samples(paths, CONTRACT)
    comparisons = []
    for pair in study.get("pairs", []):
        if type(pair.get("compare_timing", False)) is not bool:
            raise ValueError("compare_timing must be a boolean")
        base, candidate = pair["baseline"], pair["candidate"]
        if base == candidate or base not in records or candidate not in records:
            raise ValueError("comparison needs two distinct known cases")
        if manifests[base] != manifests[candidate]:
            raise ValueError("schedule comparison changed the payload")
        if rows[base]["limit"] != 0 or rows[candidate]["limit"] <= 0:
            raise ValueError("comparison must test a positive limit against the unlimited control")
        a, b = records[base]["spec"], records[candidate]["spec"]
        if (
            a.get("cwd") != b.get("cwd")
            or a.get("env") != b.get("env")
            or _schedule_command(a["command"]) != _schedule_command(b["command"])
        ):
            raise ValueError("schedule comparison changed command, device or CPU placement")
        complete = natives[base] is not None and natives[candidate] is not None
        result = dict(pair, complete_pair=complete)
        if complete and pair.get("compare_timing", False):
            result["evaluation"] = compare(
                [results / base / "native.json"], [results / candidate / "native.json"], CONTRACT
            )
            result["state_arithmetic_reduction"] = 1 - (
                rows[candidate]["measurement"]["state_arithmetic_seconds"]
                / rows[base]["measurement"]["state_arithmetic_seconds"]
            )
        comparisons.append(result)
    return {
        "schema": "fhemamba-recurrence-schedule-v1",
        "all_runs_qualified": all(row["qualified"] for row in rows.values()),
        "identities": identity,
        "cases": rows,
        "pairs": comparisons,
        "scope": "one recurrence layer; offline inputs; same payload within each pair",
        "limitations": [
            "live-value admission is a soft threshold, not a byte or allocation cap",
            "device memory is sampled after completed operations, not inside kernels",
            "samples include all device allocations, not an attribution to ciphertexts",
            "timing comparisons are descriptive; missing results have no speedup",
            "not a full-model or encrypted autoregressive token-latency measurement",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fhemamba recurrent analyze", description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument(
        "--schedule-study", type=Path, help="case/limit/pair manifest for a matched scheduler study"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.lengths != sorted(set(args.lengths)):
        parser.error("lengths must be unique and increasing")
    result = (
        analyze_schedule(args.payload, args.results, read_object(args.schedule_study))
        if args.schedule_study
        else analyze(args.payload, args.results, args.lengths)
    )
    write_json(args.output, result)


if __name__ == "__main__":
    main()
