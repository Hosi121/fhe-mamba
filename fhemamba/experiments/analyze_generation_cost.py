#!/usr/bin/env python3
"""Separate arithmetic bounds, representation cost, and calibrated timing ceilings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from manage_dgx_build import payload_sha256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(artifact_path: Path, payload: Path, bandwidth_gbs: float) -> dict:
    native = json.loads(artifact_path.read_text())
    chain_path = payload / "chain.json"
    chain = json.loads(chain_path.read_text())
    if payload_sha256(payload) != native["input_payload_sha256"]:
        raise ValueError("the supplied payload does not match the measured artifact")
    sources = {str(artifact_path): sha256(artifact_path), str(chain_path): sha256(chain_path)}
    gates = []
    for index, directory in enumerate(chain["layer_dirs"]):
        meta_path = payload / directory / "meta.json"
        spec = json.loads(meta_path.read_text())["joint_gates"]
        sources[str(meta_path)] = sha256(meta_path)
        heads = len(spec["lo"])
        degrees = {
            name: [
                max(
                    degree
                    for degree in range(len(spec[name]) // heads)
                    if spec[name][degree * heads + head] != 0
                )
                for head in range(heads)
            ]
            for name in ("p", "q")
        }
        # Leading coefficients of p^2*q^2 cannot cancel within one head.
        write_degrees = [2 * (p + q) for p, q in zip(degrees["p"], degrees["q"], strict=True)]
        degree = max(write_degrees)
        gates.append(
            {
                "layer": index,
                "p_degree_by_head": degrees["p"],
                "q_degree_by_head": degrees["q"],
                "max_write_degree": degree,
                "minimum_multiplication_depth": (degree - 1).bit_length(),
                "dense_coefficient_rows_per_evaluation": sum(
                    len(spec[name]) // heads for name in ("p", "q")
                ),
            }
        )

    evaluations = len(native["timing"]["token_step_seconds"])
    rows = evaluations * sum(g["dense_coefficient_rows_per_evaluation"] for g in gates)
    total = native["timing"]["eval_seconds"]
    gate_seconds = native["phase_timings"]["joint_selective_gates"]
    bootstrap_seconds = native["timing"]["bootstrap_eval_seconds"]
    # Other phase timers overlap bootstrap. Only these two disjoint buckets
    # can be subtracted from total without reconstructing the entire timeline.
    assert native["phase_operation_counts"]["joint_selective_gates"]["bootstraps"] == 0
    other_seconds = total - gate_seconds - bootstrap_seconds
    assert other_seconds >= 0
    scenarios = []
    for gate_factor, bootstrap_factor in (
        (2, 1),
        (None, 1),
        (1, 2),
        (1, None),
        (2, 2),
        (None, None),
    ):
        seconds = (
            other_seconds
            + (gate_seconds / gate_factor if gate_factor else 0)
            + (bootstrap_seconds / bootstrap_factor if bootstrap_factor else 0)
        )
        scenarios.append(
            {
                "gate_speed_factor": gate_factor or "infinite",
                "bootstrap_speed_factor": bootstrap_factor or "infinite",
                "conditional_eval_seconds": seconds,
                "speed_factor": total / seconds,
            }
        )

    n = native["parameters"]["ring_dimension"]
    period = native["parameters"]["joint_gate_schedule"]["coefficient_packing_slots"]
    small_n = 2 * period
    assert n > 0
    assert n & (n - 1) == 0
    assert n % small_n == 0
    before = n * (n.bit_length() - 1) // 2
    after = small_n * (small_n.bit_length() - 1) // 2
    representation = []
    for level in (0, 21, 26, 34):
        # This frozen FLEXIBLEAUTO context has depth+1 RNS primes.
        towers = native["parameters"]["multiplicative_depth"] + 1 - level
        full_bytes = n * towers * 8
        compact_bytes = small_n * towers * 8
        representation.append(
            {
                "consumption_level": level,
                "rns_towers": towers,
                "full_plaintext_bytes": full_bytes,
                "compact_ntt_bytes": compact_bytes,
                "full_ntt_butterflies": towers * before,
                "small_ntt_butterflies": towers * after,
                "one_full_write_ideal_seconds": full_bytes / (bandwidth_gbs * 1e9),
                "all_rows_full_write_scenario_seconds": rows * full_bytes / (bandwidth_gbs * 1e9),
            }
        )

    bootstrap_groups = defaultdict(lambda: {"events": 0, "physical_calls": 0, "seconds": 0.0})
    for event in native["measurements"]["bootstrap_events"]:
        name = event["checkpoint"].split(".", 2)[-1]
        if ".scheduled_norm." in name:
            name = name.split(".scheduled_norm.")[0] + ".scheduled_norm"
        if event["carried"]:
            name = "carried"
        group = bootstrap_groups[name]
        group["events"] += 1
        group["physical_calls"] += event["physical_bootstraps"]
        group["seconds"] += event["seconds"]
    calls = sum(group["physical_calls"] for group in bootstrap_groups.values())
    assert calls == native["operation_counts"]["bootstraps"]

    return {
        "analysis_kind": "conditional-cost-model-not-a-benchmark",
        "source_sha256": sources,
        "script_sha256": sha256(Path(__file__)),
        "payload_sha256_from_native": native["input_payload_sha256"],
        "evaluations": evaluations,
        "measured_disjoint_seconds": {
            "total": total,
            "joint_gates": gate_seconds,
            "bootstrap": bootstrap_seconds,
            "other": other_seconds,
        },
        "amdahl_scenarios": scenarios,
        "gate_algebra": {
            "assumption": (
                "Exact fixed per-head polynomials; ordinary binary multiplication/addition circuit."
            ),
            "proof": "At multiplication depth d, polynomial degree is at most 2^d; write=p^2*q^2.",
            "max_write_degree": max(g["max_write_degree"] for g in gates),
            "max_minimum_multiplication_depth": max(
                g["minimum_multiplication_depth"] for g in gates
            ),
            "layers": gates,
        },
        "coefficient_representation": {
            "ring_dimension": n,
            "period": period,
            "dense_coefficient_rows": rows,
            "row_count_scope": (
                "Degree-based count before exactly-zero row skipping; not a measured call count."
            ),
            "butterfly_reduction_factor": before / after,
            "expanded_to_compact_bytes_factor": n / small_n,
            "complexity_before": "O(L*N*log2(N)) modular work per coefficient row",
            "complexity_after": "O(L*2s*log2(2s)) modular work plus O(L*N) materialization",
            "scenarios": representation,
        },
        "bootstrap": {
            "physical_calls": calls,
            "mean_measured_seconds_per_call": bootstrap_seconds / calls,
            "groups": dict(bootstrap_groups),
        },
        "hardware": {
            "peak_memory_bandwidth_gbs": bandwidth_gbs,
            "primary_source": "https://docs.nvidia.com/dgx/dgx-spark/hardware.html",
        },
        "limits": [
            "Amdahl scenarios hold all unmodified time fixed; "
            "they are not fundamental FHE lower bounds.",
            "Butterfly counts exclude slot FFT, allocation, expansion, "
            "transfers and ciphertext arithmetic.",
            "Bandwidth scenarios assume those bytes cross main memory once at peak bandwidth; "
            "cache residency and actual DRAM traffic are not measured.",
            "Hardware FP4 tensor FLOPS do not specify 59-bit modular arithmetic throughput.",
            "No absolute ideal model latency is claimed without modular throughput, "
            "mandatory traffic and a dependency DAG.",
            "Changing polynomial approximation or bootstrap schedule "
            "needs a new error/depth certificate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bandwidth-gbs", type=float, default=273.0)
    args = parser.parse_args()
    if not math.isfinite(args.bandwidth_gbs) or args.bandwidth_gbs <= 0:
        parser.error("bandwidth-gbs must be finite and positive")
    report = analyze(args.artifact, args.payload, args.bandwidth_gbs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("measured_disjoint_seconds", "amdahl_scenarios")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
