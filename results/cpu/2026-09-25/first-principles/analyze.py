"""Bounded static/algebraic research; no encrypted execution or production edits.

Run from the repository root with .venv/bin/python and the unchanged full payload.
The reader hashes the payload and skips parsing the large public weight tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PROFILE = ROOT / "results/dgx/2026-09-25/packed-frontiers/final-candidate-full/native.json"
MANIFEST = ROOT / "results/dgx/2026-09-25/packed-frontiers/payload-manifests/lm-full/manifest.json"
INVENTORY = ROOT / "results/cpu/2026-09-25/math-kernel-audit/inventory.json"


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_graph(path):
    nodes, outputs = [], []
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        line = stream.readline()
        hasher.update(line)
        magic, slots, _, count, output_count = line.split()
        assert magic == b"fhemamba-packed-v2"
        for _ in range(int(count)):
            line = stream.readline()
            hasher.update(line)
            prefix = line.split(b" ", 4)
            operation, size, bound, parent_count = prefix[:4]
            parent_count = int(parent_count)
            tail = prefix[4].split(b" ", parent_count + 1)
            parents = list(map(int, tail[:parent_count]))
            data_count = int(tail[parent_count])
            data = None
            if operation != b"linear":
                data = np.fromstring(tail[-1].decode(), sep=" ") if data_count else np.array([])
                assert len(data) == data_count
            nodes.append(
                {
                    "op": operation.decode(),
                    "size": int(size),
                    "bound": float(bound),
                    "parents": parents,
                    "data": data,
                }
            )
        for _ in range(int(output_count)):
            line = stream.readline()
            hasher.update(line)
            outputs.append(int(line.split(b" ", 1)[0]))
        assert not stream.read()
    live = set(outputs)
    for index in range(len(nodes) - 1, -1, -1):
        if index in live:
            live.update(nodes[index]["parents"])
    return nodes, live, int(slots), hasher.hexdigest()


def polynomial_key(data):
    return tuple(map(float, data))


def algebra_checks(polynomials):
    """Check identities and expose where exact trigonometry changes the surrogate."""
    rng = np.random.default_rng(20260925)
    rotary = []
    rate = []
    gram_error, complex_error = 0.0, 0.0
    for layer in range(12):
        cosine = polynomials[f"{layer}:m3_cos"]
        sine = polynomials[f"{layer}:m3_sin"]
        assert (cosine["lo"], cosine["hi"]) == (sine["lo"], sine["hi"])
        # Includes endpoints and is independent of the exporter's 8,193-point grid.
        u = np.linspace(-1, 1, 65537)
        c = np.polynomial.chebyshev.chebval(u, cosine["coefficients"])
        s = np.polynomial.chebyshev.chebval(u, sine["coefficients"])
        q = c * c + s * s
        b, v = rng.normal(size=(2, len(u), 2))
        rb = np.column_stack((c * b[:, 0] - s * b[:, 1], s * b[:, 0] + c * b[:, 1]))
        rv = np.column_stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]))
        lhs = np.sum(rb * rv, axis=1)
        rhs = q * np.sum(b * v, axis=1)
        gram_error = max(gram_error, float(np.max(np.abs(lhs - rhs))))
        z = (c + 1j * s) * (b[:, 0] + 1j * b[:, 1])
        complex_error = max(
            complex_error,
            float(np.max(np.abs(z.real - rb[:, 0]))),
            float(np.max(np.abs(z.imag - rb[:, 1]))),
        )
        rotary.append({"layer": layer, "sampled_unitarity_defect": float(np.max(np.abs(q - 1)))})
        p = polynomials[f"{layer}:m3_negative_a"]
        lo, hi, floor = p["lo"], p["hi"], p["parameter"]
        assert lo < 0 < hi
        x = np.linspace(lo, hi, 65537)
        exact = np.maximum(np.maximum(x, 0) + 1 / (1 - np.minimum(x, 0)), floor)
        piecewise = np.where(x >= 0, 1 + x, 1 / (1 - np.minimum(x, 0)))
        minimum = 1 / (1 - lo)
        rate.append(
            {
                "layer": layer,
                "degree": p["degree"],
                "domain": [lo, hi],
                "floor": floor,
                "analytic_minimum_unfloored": minimum,
                "floor_inactive_on_whole_domain": minimum > floor,
                "piecewise_identity_max_error": float(np.max(np.abs(exact - piecewise))),
            }
        )
    assert gram_error < 1e-12
    assert complex_error < 1e-12
    assert all(row["floor_inactive_on_whole_domain"] for row in rate)
    assert all(row["piecewise_identity_max_error"] < 1e-12 for row in rate)
    return {
        "scope": "Float64 identity checks; sampled defects are not interval/error certificates",
        "grid_points_per_site": 65537,
        "seed": 20260925,
        "same_angle_gram_identity_max_error": gram_error,
        "frozen_polynomial_complex_multiply_max_error": complex_error,
        "rotary_sites": rotary,
        "negative_a_sites": rate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--program", type=Path, default=ROOT / "runs/mamba3-lm-full-20260924/program.txt"
    )
    parser.add_argument("--output", type=Path, default=HERE / "summary.json")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    profile = json.loads(PROFILE.read_text())
    inventory = json.loads(INVENTORY.read_text())
    nodes, live, slots, program_hash = read_graph(args.program)
    assert program_hash == manifest["files_sha256"]["program.txt"]
    assert len(live) == profile["evaluated_nodes"]
    lookup = {
        polynomial_key([p["lo"], p["hi"], *p["coefficients"]]): (site, p["kind"])
        for site, p in manifest["polynomials"].items()
    }
    by_kind = defaultdict(Counter)
    by_size = defaultdict(Counter)
    site_rows = []
    inventory_nodes = {row["id"]: row for row in inventory["nonlinear_frontiers"]}
    children = defaultdict(list)
    for index in sorted(live):
        for parent in nodes[index]["parents"]:
            children[parent].append(index)
    norm_repeats = []
    for index in sorted(live):
        node = nodes[index]
        if node["op"] != "cheb":
            continue
        site, kind = lookup[polynomial_key(node["data"])]
        old = inventory_nodes[index]
        assert old["size"] == node["size"]
        assert old["degree"] == len(node["data"]) - 3
        for bucket in (by_kind[kind], by_size[node["size"]]):
            bucket["nodes"] += 1
            bucket["ctct_core_calls"] += old["ctct"]
            bucket["logical_values"] += node["size"]
        site_rows.append(
            {
                "id": index,
                "site": site,
                "kind": kind,
                "size": node["size"],
                "degree": old["degree"],
                "ctct_core_calls": old["ctct"],
            }
        )
        if kind == "inv_sqrt":
            for child in children[index]:
                if nodes[child]["op"] == "repeat":
                    norm_repeats.append(
                        {"polynomial": index, "repeat": child, "target_size": nodes[child]["size"]}
                    )
    total_poly = sum(row["ctct_core_calls"] for row in site_rows)
    small = sum(row["ctct_core_calls"] for row in site_rows if row["size"] <= 32)
    dag_mul = sum(nodes[i]["op"] == "mul" for i in live)
    assert total_poly + dag_mul == profile["ct_ct_mul"]
    assert len(site_rows) == profile["operation_stats"]["cheb"]["nodes"]
    categories = {}
    groups = {
        "routing": ["gather", "scatter", "repeat", "sum"],
        "linear": ["linear", "linear_ref"],
        "polynomial": ["cheb"],
    }
    for name, operations in groups.items():
        categories[name] = sum(
            profile["operation_stats"][op]["seconds"]
            - profile["operation_stats"][op]["bootstrap_seconds"]
            for op in operations
        )
    total = profile["eval_seconds"]
    refresh = profile["bootstrap_seconds"]
    outside = total - refresh
    categories["refresh_including_wrapper"] = refresh
    categories["remaining_including_timer_gaps"] = total - sum(categories.values())
    scenarios = {}
    for name, seconds in {**categories, "all_nonrefresh": outside}.items():
        if name == "remaining_including_timer_gaps":
            continue
        scenarios[name] = {
            "seconds_if_category_halved_and_others_fixed": total - seconds / 2,
            "total_reduction_fraction": seconds / (2 * total),
        }
    # Ideal 64 -> 32-bit storage at equal total modulus; this is not a traffic/timing model.
    representation = [
        {
            "limbs64": count,
            "bytes64_two_components": 2 * 65536 * count * 8,
            "limbs32_min_for_same_bits": math.ceil(count * 59 / 30),
            "bytes32_two_components": 2 * 65536 * math.ceil(count * 59 / 30) * 4,
        }
        for count in (20, 40)
    ]
    result = {
        "scope": (
            "Static attribution, algebra checks and conditional scenarios; no new GPU speed result"
        ),
        "base_commit": "f2452997f603c6dfd4556a73f8023ae621469253",
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in (MANIFEST, PROFILE, INVENTORY)},
        "program_sha256": program_hash,
        "graph": {
            "nodes": len(nodes),
            "live_nodes": len(live),
            "slots": slots,
            "maximum_logical_node_size": max(nodes[i]["size"] for i in live),
            "warning": "Logical sizes exclude replicated BSGS scratch and packed refresh groups",
            "live_size_histogram": dict(sorted(Counter(nodes[i]["size"] for i in live).items())),
        },
        "polynomials": {
            "by_kind": dict(by_kind),
            "by_size": dict(by_size),
            "total_core_ctct_calls": total_poly,
            "dag_ctct_calls": dag_mul,
            "small_nodes": sum(row["size"] <= 32 for row in site_rows),
            "small_core_ctct_calls": small,
            "small_core_fraction": small / total_poly,
            "scalar_rms_repeat_candidates": norm_repeats,
            "sites": site_rows,
        },
        "profile": {
            "eval_seconds": total,
            "category_seconds": categories,
            "host_encoding_nested_seconds_do_not_add": profile["host_encoding_seconds"],
            "physical_two_pass_refresh_events": profile["bootstraps"] // 2,
        },
        "conditional_scenarios_not_forecasts": scenarios,
        "dual_ring_break_even": {
            "assumption": (
                "All nonrefresh time halves, old per-event refresh cost, no conversion cost"
            ),
            "maximum_refresh_count_multiplier_for_any_gain": 1 + outside / (2 * refresh),
            "seconds_if_refresh_count_increases_50_percent": outside / 2 + 1.5 * refresh,
        },
        "ideal_rns_storage_not_traffic_model": representation,
        "algebra": algebra_checks(manifest["polynomials"]),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "written": str(args.output),
                "small_polynomial_ctct_fraction": small / total_poly,
                "polynomial_ctct": total_poly,
                "reconciled_ctct": total_poly + dag_mul,
                "scalar_norm_repeat_candidates": len(norm_repeats),
                "max_logical_size": result["graph"]["maximum_logical_node_size"],
            }
        )
    )


if __name__ == "__main__":
    main()
