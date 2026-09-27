#!/usr/bin/env python3
"""Run a packed FHE probe with a hard timeout and bind its input/binary hashes.

This runner uses only the Python standard library and can run on the GPU host.
It never labels a timeout, missing result or non-finite error as a passing run.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _run(
    binary,
    payload,
    output,
    *,
    timeout=600,
    tolerance=0.001,
    legacy_routing=False,
    trace_levels=False,
    planned_refresh=False,
    bootstrap_passes=2,
    batch_refresh=False,
    merge_refresh_correction=False,
    profile_evaluation=False,
    inplace_ops=False,
    cache_plaintexts=False,
    plaintext_cache_capacity=None,
    fast_plaintext_upload=False,
    gpu_plaintext_ntt=False,
    direct_plaintext_upload=False,
    move_plaintext_coefficients=False,
    borrow_plaintext_upload=False,
    bsgs_routing_stages=False,
    naf_rotations=False,
    reuse_dead_inputs=False,
    compact_weights=False,
    frontier_refresh=False,
    s2c_first=False,
    gpu_plaintext_rns=False,
    gpu_addend_rns=False,
    gpu_plaintext_fft=False,
    batch_plaintext_rns=False,
    fuse_plaintext_rns_ntt=False,
    prefetch_plaintexts=False,
    prefetch_workers=1,
    hoist_rotations=False,
    share_chebyshev=False,
    gpu_dual_ring=False,
    security="not-set",
    security_digits=None,
):
    binary, payload, output = binary.resolve(), payload.resolve(), output.resolve()
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("timeout and tolerance must be finite and positive")
    if bootstrap_passes not in (1, 2):
        raise ValueError("bootstrap_passes must be 1 or 2")
    if planned_refresh and legacy_routing:
        raise ValueError("planned refresh requires radix8 routing")
    if batch_refresh and (not planned_refresh or bootstrap_passes != 2):
        raise ValueError("batch refresh requires planned refresh and two bootstrap passes")
    if merge_refresh_correction and (not planned_refresh or bootstrap_passes != 2):
        raise ValueError("refresh correction merging requires planned two-pass refresh")
    if s2c_first and not batch_refresh:
        raise ValueError("S2C-first requires planned two-pass batch refresh")
    if plaintext_cache_capacity is not None:
        if type(plaintext_cache_capacity) is not int or not 0 <= plaintext_cache_capacity <= 4096:
            raise ValueError("plaintext cache capacity must be 0..4096")
        if not cache_plaintexts:
            raise ValueError("plaintext cache capacity requires cache_plaintexts")
    if gpu_addend_rns and not (gpu_plaintext_rns or batch_plaintext_rns or fuse_plaintext_rns_ntt):
        raise ValueError("GPU addend RNS requires gpu_plaintext_rns")
    if gpu_plaintext_fft and (
        not gpu_plaintext_rns or batch_plaintext_rns or fuse_plaintext_rns_ntt
    ):
        raise ValueError("GPU FFT requires compact per-limb GPU RNS encoding")
    if security not in ("not-set", "128-classic"):
        raise ValueError("security must be not-set or 128-classic")
    if security == "128-classic" and gpu_dual_ring:
        raise ValueError("classical-128 does not support the experimental dual-ring bridge")
    if security_digits is not None and (
        type(security_digits) is not int
        or not 1 <= security_digits <= 16
        or security != "128-classic"
    ):
        raise ValueError("security digits must be 1..16 and require 128-classic")
    if gpu_dual_ring and not s2c_first:
        raise ValueError("GPU dual ring requires S2C-first planned two-pass refresh")
    if batch_plaintext_rns and fuse_plaintext_rns_ntt:
        raise ValueError("choose one compact RNS strategy")
    if type(prefetch_workers) is not int or prefetch_workers not in (1, 2):
        raise ValueError("prefetch workers must be 1 or 2")
    if prefetch_workers != 1 and not prefetch_plaintexts:
        raise ValueError("parallel preparation requires plaintext prefetch")
    if prefetch_plaintexts and not (
        gpu_plaintext_ntt
        or move_plaintext_coefficients
        or gpu_plaintext_rns
        or batch_plaintext_rns
        or fuse_plaintext_rns_ntt
    ):
        raise ValueError("plaintext prefetch requires coefficient encoding")
    if frontier_refresh and not batch_refresh:
        raise ValueError("frontier refresh requires batch refresh")
    manifest = json.loads((payload / "manifest.json").read_text())
    client = manifest.get("schema") == "fhemamba-mamba3-lm-v1"
    names = (
        ("program.txt", "fixture.npz", "client_head.f32")
        if client
        else ("program.txt", "fixture.npz")
    )
    for name in names:
        if digest(payload / name) != manifest["files_sha256"][name]:
            raise ValueError(f"payload digest differs: {name}")
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(binary),
        str(payload / "program.txt"),
        str(output / "native.json"),
        str(tolerance),
        str(tolerance),
    ]
    if client:
        command.extend(["--client-head", str(payload / "client_head.f32")])
    if legacy_routing:
        command.append("--legacy-routing")
    if trace_levels:
        command.append("--trace-levels")
    if planned_refresh:
        command.append("--planned-refresh")
    if bootstrap_passes != 2:
        command.extend(["--bootstrap-passes", str(bootstrap_passes)])
    if batch_refresh:
        command.append("--batch-refresh")
    if profile_evaluation:
        command.append("--profile-evaluation")
    if inplace_ops:
        command.append("--inplace-ops")
    if cache_plaintexts:
        command.append("--cache-plaintexts")
    if fast_plaintext_upload:
        command.append("--fast-plaintext-upload")
    if gpu_plaintext_ntt:
        command.append("--gpu-plaintext-ntt")
    for enabled, flag in (
        (merge_refresh_correction, "--merge-refresh-correction"),
        (direct_plaintext_upload, "--direct-plaintext-upload"),
        (move_plaintext_coefficients, "--move-plaintext-coefficients"),
        (borrow_plaintext_upload, "--borrow-plaintext-upload"),
        (bsgs_routing_stages, "--bsgs-routing-stages"),
        (naf_rotations, "--naf-rotations"),
        (reuse_dead_inputs, "--reuse-dead-inputs"),
        (compact_weights, "--compact-weights"),
        (frontier_refresh, "--frontier-refresh"),
        (s2c_first, "--s2c-first"),
        (gpu_plaintext_rns, "--gpu-plaintext-rns"),
        (gpu_addend_rns, "--gpu-addend-rns"),
        (gpu_plaintext_fft, "--gpu-plaintext-fft"),
        (batch_plaintext_rns, "--batch-plaintext-rns"),
        (fuse_plaintext_rns_ntt, "--fuse-plaintext-rns-ntt"),
        (prefetch_plaintexts, "--prefetch-plaintexts"),
        (hoist_rotations, "--hoist-rotations"),
        (share_chebyshev, "--share-chebyshev"),
        (gpu_dual_ring, "--gpu-dual-ring"),
    ):
        if enabled:
            command.append(flag)
    if prefetch_workers != 1:
        command.extend(["--prefetch-workers", str(prefetch_workers)])
    if security != "not-set":
        command.extend(["--security", security])
    if security_digits is not None:
        command.extend(["--security-digits", str(security_digits)])
    if plaintext_cache_capacity is not None:
        command.extend(["--plaintext-cache-capacity", str(plaintext_cache_capacity)])
    record = {
        "schema": "fhemamba-packed-run-v1",
        "requested_security": security,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command,
        "binary_sha256": digest(binary),
        "manifest_sha256": digest(payload / "manifest.json"),
        "program_sha256": digest(payload / "program.txt"),
        "timeout_seconds": timeout,
        "architecture": manifest["architecture"],
        "tokens": manifest["tokens"],
    }
    started = time.monotonic()
    with (output / "native.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=timeout)
            record["timed_out"] = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode, record["timed_out"] = process.returncode, True
    record.update(returncode=returncode, wall_seconds=time.monotonic() - started, passed=False)
    native = output / "native.json"
    if native.exists():
        result = json.loads(native.read_text())
        errors = [result["max_abs_error_vs_polynomial"], result["max_abs_error_vs_exact"]]
        record["native_sha256"] = digest(native)
        record["passed"] = (
            returncode == 0
            and not record["timed_out"]
            and result.get("passed") is True
            and result.get("schema") == "fhemamba-packed-result-v1"
            and result.get("encrypted") is True
            and len(result.get("per_output_errors", [])) == len(manifest["output_names"])
            and result.get("non_finite") == 0
            and all(math.isfinite(error) and 0 <= error <= tolerance for error in errors)
        )
        if merge_refresh_correction:
            counts = [
                result.get(key)
                for key in ("merged_refreshes", "merge_refresh_fallbacks", "bootstraps")
            ]
            record["passed"] = record["passed"] and (
                result.get("merge_refresh_correction") is True
                and all(type(value) is int and value >= 0 for value in counts)
                and 2 * (counts[0] + counts[1]) == counts[2]
            )
        if gpu_plaintext_fft:
            counts = [
                result.get(key) for key in ("gpu_fft_encodes", "gpu_fft_fallbacks", "host_encodes")
            ]
            record["passed"] = record["passed"] and (
                result.get("gpu_plaintext_fft") is True
                and all(type(value) is int and value >= 0 for value in counts)
                and counts[0] > 0
                and counts[0] + counts[1] <= counts[2]
            )
        if security == "128-classic":
            audit = result.get("security_audit") or {}
            bound = {
                1024: 26,
                2048: 53,
                4096: 106,
                8192: 214,
                16384: 430,
                32768: 868,
                65536: 1747,
                131072: 3523,
            }.get(audit.get("ring_dimension"), 0)
            qp_bits, sigma = audit.get("qp_bits"), audit.get("error_sigma")
            record["passed"] = record["passed"] and (
                result.get("security") == security
                and result.get("gpu_dual_ring") is False
                and audit.get("passed") is True
                and audit.get("library_classical128") is True
                and audit.get("uniform_ternary") is True
                and audit.get("hybrid") is True
                and bound > 0
                and audit.get("guideline_max_qp_bits") == bound
                and type(qp_bits) is int
                and 0 < qp_bits <= bound
                and type(sigma) in (int, float)
                and math.isfinite(sigma)
                and sigma >= 3.19
                and result.get("ring_dimension") == audit.get("ring_dimension")
                and result.get("refresh_ring_dimension") == audit.get("ring_dimension")
                and audit.get("hybrid_digits") == (security_digits or 4)
                and result.get("evaluation_decryptions") == 0
            )
        if client:
            record["passed"] = record["passed"] and (
                result.get("generated_token_ids") == manifest["exact_token_ids"]
                and result.get("client_output_decrypt_count") == manifest["generated_tokens"]
                and result.get("evaluation_decryptions") == 0
            )
            record["complete_backbone"] = manifest["complete_backbone"]
            record["expected_text"] = manifest["reference_text"]
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run(
    binary,
    payload,
    output,
    *,
    timeout=600,
    tolerance=0.001,
    legacy_routing=False,
    trace_levels=False,
    planned_refresh=False,
    bootstrap_passes=2,
    batch_refresh=False,
    merge_refresh_correction=False,
    profile_evaluation=False,
    inplace_ops=False,
    cache_plaintexts=False,
    plaintext_cache_capacity=None,
    fast_plaintext_upload=False,
    gpu_plaintext_ntt=False,
    direct_plaintext_upload=False,
    move_plaintext_coefficients=False,
    borrow_plaintext_upload=False,
    bsgs_routing_stages=False,
    naf_rotations=False,
    reuse_dead_inputs=False,
    compact_weights=False,
    frontier_refresh=False,
    s2c_first=False,
    gpu_plaintext_rns=False,
    gpu_addend_rns=False,
    gpu_plaintext_fft=False,
    batch_plaintext_rns=False,
    fuse_plaintext_rns_ntt=False,
    prefetch_plaintexts=False,
    prefetch_workers=1,
    hoist_rotations=False,
    share_chebyshev=False,
    gpu_dual_ring=False,
    security="not-set",
    security_digits=None,
    budget_file=None,
    budget_seconds=None,
):
    """Charge total runner wall time conservatively against a shared GPU budget."""
    if budget_file is None:
        if budget_seconds is not None:
            raise ValueError("budget_seconds requires a budget_file")
        return _run(
            binary,
            payload,
            output,
            timeout=timeout,
            tolerance=tolerance,
            legacy_routing=legacy_routing,
            trace_levels=trace_levels,
            planned_refresh=planned_refresh,
            bootstrap_passes=bootstrap_passes,
            batch_refresh=batch_refresh,
            merge_refresh_correction=merge_refresh_correction,
            profile_evaluation=profile_evaluation,
            inplace_ops=inplace_ops,
            cache_plaintexts=cache_plaintexts,
            plaintext_cache_capacity=plaintext_cache_capacity,
            fast_plaintext_upload=fast_plaintext_upload,
            gpu_plaintext_ntt=gpu_plaintext_ntt,
            direct_plaintext_upload=direct_plaintext_upload,
            move_plaintext_coefficients=move_plaintext_coefficients,
            borrow_plaintext_upload=borrow_plaintext_upload,
            bsgs_routing_stages=bsgs_routing_stages,
            naf_rotations=naf_rotations,
            reuse_dead_inputs=reuse_dead_inputs,
            compact_weights=compact_weights,
            frontier_refresh=frontier_refresh,
            s2c_first=s2c_first,
            gpu_plaintext_rns=gpu_plaintext_rns,
            gpu_addend_rns=gpu_addend_rns,
            gpu_plaintext_fft=gpu_plaintext_fft,
            batch_plaintext_rns=batch_plaintext_rns,
            fuse_plaintext_rns_ntt=fuse_plaintext_rns_ntt,
            prefetch_plaintexts=prefetch_plaintexts,
            prefetch_workers=prefetch_workers,
            hoist_rotations=hoist_rotations,
            share_chebyshev=share_chebyshev,
            gpu_dual_ring=gpu_dual_ring,
            security=security,
            security_digits=security_digits,
        )
    import fcntl

    if budget_seconds is None or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("a positive explicit budget is required")
    budget_file = Path(budget_file)
    budget_file.parent.mkdir(parents=True, exist_ok=True)
    with budget_file.open("a+") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        ledger.seek(0)
        contents = ledger.read()
        budget = (
            json.loads(contents)
            if contents
            else {
                "limit_seconds": budget_seconds,
                "used_seconds": 0.0,
                "runs": [],
            }
        )
        if budget["limit_seconds"] != budget_seconds:
            raise ValueError("cannot change an existing campaign budget")
        remaining = budget_seconds - budget["used_seconds"]
        # Leave room for process-group termination if the hard timeout fires.
        if remaining <= 15:
            raise ValueError("GPU campaign budget exhausted")
        started = time.monotonic()
        try:
            record = _run(
                binary,
                payload,
                output,
                timeout=min(timeout, remaining - 15),
                tolerance=tolerance,
                legacy_routing=legacy_routing,
                trace_levels=trace_levels,
                planned_refresh=planned_refresh,
                bootstrap_passes=bootstrap_passes,
                batch_refresh=batch_refresh,
                merge_refresh_correction=merge_refresh_correction,
                profile_evaluation=profile_evaluation,
                inplace_ops=inplace_ops,
                cache_plaintexts=cache_plaintexts,
                plaintext_cache_capacity=plaintext_cache_capacity,
                fast_plaintext_upload=fast_plaintext_upload,
                gpu_plaintext_ntt=gpu_plaintext_ntt,
                direct_plaintext_upload=direct_plaintext_upload,
                move_plaintext_coefficients=move_plaintext_coefficients,
                borrow_plaintext_upload=borrow_plaintext_upload,
                bsgs_routing_stages=bsgs_routing_stages,
                naf_rotations=naf_rotations,
                reuse_dead_inputs=reuse_dead_inputs,
                compact_weights=compact_weights,
                frontier_refresh=frontier_refresh,
                s2c_first=s2c_first,
                gpu_plaintext_rns=gpu_plaintext_rns,
                gpu_addend_rns=gpu_addend_rns,
                gpu_plaintext_fft=gpu_plaintext_fft,
                batch_plaintext_rns=batch_plaintext_rns,
                fuse_plaintext_rns_ntt=fuse_plaintext_rns_ntt,
                prefetch_plaintexts=prefetch_plaintexts,
                prefetch_workers=prefetch_workers,
                hoist_rotations=hoist_rotations,
                share_chebyshev=share_chebyshev,
                gpu_dual_ring=gpu_dual_ring,
                security=security,
                security_digits=security_digits,
            )
        finally:
            charged = time.monotonic() - started
            budget["used_seconds"] += charged
            budget["runs"].append({"output": str(output), "charged_seconds": charged})
            ledger.seek(0)
            ledger.truncate()
            json.dump(budget, ledger, indent=2)
            ledger.flush()
            os.fsync(ledger.fileno())
        record["campaign_budget_seconds"] = budget_seconds
        record["campaign_used_seconds"] = budget["used_seconds"]
        (Path(output) / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--legacy-routing", action="store_true")
    parser.add_argument("--trace-levels", action="store_true")
    parser.add_argument("--planned-refresh", action="store_true")
    parser.add_argument("--bootstrap-passes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--batch-refresh", action="store_true")
    parser.add_argument("--merge-refresh-correction", action="store_true")
    parser.add_argument("--profile-evaluation", action="store_true")
    parser.add_argument("--inplace-ops", action="store_true")
    parser.add_argument("--cache-plaintexts", action="store_true")
    parser.add_argument("--plaintext-cache-capacity", type=int)
    parser.add_argument("--fast-plaintext-upload", action="store_true")
    parser.add_argument("--gpu-plaintext-ntt", action="store_true")
    parser.add_argument("--direct-plaintext-upload", action="store_true")
    parser.add_argument("--move-plaintext-coefficients", action="store_true")
    parser.add_argument("--borrow-plaintext-upload", action="store_true")
    parser.add_argument("--bsgs-routing-stages", action="store_true")
    parser.add_argument("--naf-rotations", action="store_true")
    parser.add_argument("--reuse-dead-inputs", action="store_true")
    parser.add_argument("--compact-weights", action="store_true")
    parser.add_argument("--frontier-refresh", action="store_true")
    parser.add_argument("--s2c-first", action="store_true")
    parser.add_argument("--gpu-plaintext-rns", action="store_true")
    parser.add_argument("--gpu-addend-rns", action="store_true")
    parser.add_argument("--gpu-plaintext-fft", action="store_true")
    rns = parser.add_mutually_exclusive_group()
    rns.add_argument("--batch-plaintext-rns", action="store_true")
    rns.add_argument("--fuse-plaintext-rns-ntt", action="store_true")
    parser.add_argument("--prefetch-plaintexts", action="store_true")
    parser.add_argument("--prefetch-workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--hoist-rotations", action="store_true")
    parser.add_argument("--share-chebyshev", action="store_true")
    parser.add_argument("--gpu-dual-ring", action="store_true")
    parser.add_argument("--security", choices=("not-set", "128-classic"), default="not-set")
    parser.add_argument("--security-digits", type=int)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--budget-seconds", type=float)
    args = parser.parse_args()
    record = run(
        args.binary,
        args.payload,
        args.output,
        timeout=args.timeout,
        tolerance=args.tolerance,
        legacy_routing=args.legacy_routing,
        trace_levels=args.trace_levels,
        planned_refresh=args.planned_refresh,
        bootstrap_passes=args.bootstrap_passes,
        batch_refresh=args.batch_refresh,
        merge_refresh_correction=args.merge_refresh_correction,
        profile_evaluation=args.profile_evaluation,
        inplace_ops=args.inplace_ops,
        cache_plaintexts=args.cache_plaintexts,
        plaintext_cache_capacity=args.plaintext_cache_capacity,
        fast_plaintext_upload=args.fast_plaintext_upload,
        gpu_plaintext_ntt=args.gpu_plaintext_ntt,
        direct_plaintext_upload=args.direct_plaintext_upload,
        move_plaintext_coefficients=args.move_plaintext_coefficients,
        borrow_plaintext_upload=args.borrow_plaintext_upload,
        bsgs_routing_stages=args.bsgs_routing_stages,
        naf_rotations=args.naf_rotations,
        reuse_dead_inputs=args.reuse_dead_inputs,
        compact_weights=args.compact_weights,
        frontier_refresh=args.frontier_refresh,
        s2c_first=args.s2c_first,
        gpu_plaintext_rns=args.gpu_plaintext_rns,
        gpu_addend_rns=args.gpu_addend_rns,
        gpu_plaintext_fft=args.gpu_plaintext_fft,
        batch_plaintext_rns=args.batch_plaintext_rns,
        fuse_plaintext_rns_ntt=args.fuse_plaintext_rns_ntt,
        prefetch_plaintexts=args.prefetch_plaintexts,
        prefetch_workers=args.prefetch_workers,
        hoist_rotations=args.hoist_rotations,
        share_chebyshev=args.share_chebyshev,
        gpu_dual_ring=args.gpu_dual_ring,
        security=args.security,
        security_digits=args.security_digits,
        budget_file=args.budget_file,
        budget_seconds=args.budget_seconds,
    )
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
