"""Recompute static opportunities and conditional Amdahl scenarios (stdlib only)."""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
inventory = json.loads((HERE / "inventory.json").read_text())
profile = json.loads(
    (ROOT / "results/dgx/2026-09-24/owned-arithmetic/m3-full-candidate/native.json").read_text()
)


def naf(offset, slots=32768):
    offset %= slots
    if offset > slots // 2:
        offset -= slots
    steps, power = [], 1
    while offset:
        if offset & 1:
            digit = 2 - (offset & 3)
            steps.append(digit * power)
            offset -= digit
        offset //= 2
        power *= 2
    return steps


rot = inventory["rotations"]
assert rot["baseline"] == profile["rotations"] - profile["refresh_rotations"]
prefix_saved = hoisting_saved = 0
for group in rot["groups"]:
    prefixes, sources = set(), set()
    baseline = 0
    for offset in group["offsets"]:
        path = ()
        steps = naf(offset)
        assert sum(steps) % profile["slots"] == offset % profile["slots"]
        for step in steps:
            baseline += 1
            sources.add(path)
            path = (*path, step)
            prefixes.add(path)
    prefix_saved += (baseline - len(prefixes)) * group["calls"]
    hoisting_saved += (baseline - len(sources)) * group["calls"]
assert rot["baseline"] - prefix_saved == rot["prefix_edges"]
assert rot["baseline"] - hoisting_saved == rot["hoisted_sources"]

poly = inventory["polynomials"]
ctct = poly["basis_multiplies"] + poly["recombines"] + poly["dag_multiplies"]
assert ctct == profile["ct_ct_mul"]
assert poly["calls"] == profile["operation_stats"]["cheb"]["nodes"]
squares = poly["basis_squares"] + poly["dag_squares"]
shared_saving = sum(g["basis_calls"] - g["shared_basis_calls"] for g in poly["shared_input_groups"])

# A graph antichain: each cheb increments the maximum ancestor cheb rank.
# Feedback epochs prevent grouping across the sequential client boundary.
# Equal degree makes the existing recursive split/baby size identical.
# This is NOT a level-aware executable packing or refresh schedule.
groups = defaultdict(list)
for node in inventory["nonlinear_frontiers"]:
    assert node["ctct"] == len(node["basis"]) - 1 + node["recombines"]
    groups[node["epoch"], node["frontier"], node["degree"]].append(node)
groups_report = []
for key, nodes in groups.items():
    epoch, frontier, degree = key
    basis = set().union(*(set(n["basis"]) for n in nodes))
    recombines = {n["recombines"] for n in nodes}
    assert len(recombines) == 1
    occupied = sum(n["size"] for n in nodes)
    assert occupied <= profile["slots"]
    groups_report.append(
        {
            "epoch": epoch,
            "frontier": frontier,
            "degree": degree,
            "nodes": [n["id"] for n in nodes],
            "occupied_slots": occupied,
            "separate_ctct": sum(n["ctct"] for n in nodes),
            "joint_polynomial_ctct": len(basis) - 1 + next(iter(recombines)),
        }
    )
separate = sum(g["separate_ctct"] for g in groups_report)
joint = sum(g["joint_polynomial_ctct"] for g in groups_report)
assert separate == poly["basis_multiplies"] + poly["recombines"]

stats = profile["operation_stats"]
exclusive = {op: v["seconds"] - v["bootstrap_seconds"] for op, v in stats.items()}
categories = {
    "refresh": profile["bootstrap_seconds"],
    "routing_nodes_outside_refresh": sum(
        exclusive[op] for op in ["gather", "scatter", "repeat", "sum"]
    ),
    "linear_nodes_outside_refresh": exclusive["linear"] + exclusive["linear_ref"],
    "polynomial_nodes_outside_refresh": exclusive["cheb"],
}
evaluation = profile["eval_seconds"]
summary = {
    "scope": (
        "Static counts and conditional scenarios, "
        "not measured speedup or an executable packing plan"
    ),
    "baseline_eval_seconds": evaluation,
    "matched_profile_counts": True,
    "squares": {"calls": squares, "all_ctct": ctct, "fraction": squares / ctct},
    "shared_same_input_basis": {
        "groups": len(poly["shared_input_groups"]),
        "ctct_saved": shared_saving,
    },
    "rotation_prefix": {
        "baseline": rot["baseline"],
        "candidate": rot["prefix_edges"],
        "saved": prefix_saved,
    },
    "rotation_modup_with_prefix_and_sibling_hoisting": {
        "baseline": rot["baseline"],
        "candidate": rot["hoisted_sources"],
        "saved": hoisting_saved,
    },
    "nonlinear_frontier_packing": {
        "nodes": poly["calls"],
        "groups": len(groups),
        "multi_groups": sum(len(g) > 1 for g in groups.values()),
        "maximum_group_nodes": max(map(len, groups.values())),
        "maximum_group_slots": max(g["occupied_slots"] for g in groups_report),
        "separate_polynomial_ctct": separate,
        "joint_polynomial_ctct": joint,
        "polynomial_ctct_saved": separate - joint,
        "excludes": [
            "level/scale compatibility",
            "packing/unpacking cost",
            "plaintext-vector preparation",
            "changed refresh placement",
            "error and live-state validation",
        ],
    },
    "category_seconds": categories,
    "one_category_twice_as_fast_others_fixed": {
        name: {
            "eval_seconds": evaluation - seconds / 2,
            "time_reduction_fraction": seconds / (2 * evaluation),
        }
        for name, seconds in categories.items()
    },
    "nonrefresh_eliminated_refresh_fixed_max_speedup": evaluation / profile["bootstrap_seconds"],
    "host_encoding_nested_seconds_do_not_add": profile["host_encoding_seconds"],
}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(HERE / "frontier-groups.json").write_text(json.dumps(groups_report, indent=2) + "\n")
print(json.dumps(summary, indent=2))
