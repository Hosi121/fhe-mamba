"""Attribute packed polynomial work to frozen calibration sites, without decryption."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from analyze import program_metadata


def summarize(program, manifest, native=None):
    if program["sha256"] != manifest["files_sha256"]["program.txt"]:
        raise ValueError("manifest and program identity differ")
    recipes = defaultdict(list)
    for site, recipe in manifest["polynomials"].items():
        key = (recipe["lo"], recipe["hi"], *recipe["coefficients"])
        recipes[key].append((site, recipe["kind"]))
    nodes = program["nodes"]
    live, pending = set(), list(program["outputs"])
    while pending:
        index = pending.pop()
        if index not in live:
            live.add(index)
            pending.extend(nodes[index]["parents"])
    groups = {}
    labels = {}
    for index in sorted(live):
        node = nodes[index]
        if node["op"] != "cheb":
            continue
        matches = recipes.get(tuple(node["data"]), [])
        kinds = {kind for _, kind in matches}
        if len(kinds) != 1:
            raise ValueError(f"polynomial node {index} has missing or ambiguous calibration")
        kind = kinds.pop()
        labels[index] = kind
        group = groups.setdefault(kind, {"nodes": 0, "degrees": Counter(), "sites": {}})
        group["nodes"] += 1
        group["degrees"][len(node["data"]) - 3] += 1
        for site, _ in matches:
            recipe = manifest["polynomials"][site]
            group["sites"][site] = {
                key: recipe[key] for key in ("lo", "hi", "degree", "grid_max_abs_error")
            }
            if kind == "inv_sqrt":
                group["sites"][site]["interval_ratio"] = recipe["hi"] / recipe["lo"]
    profiled = native is not None
    if profiled:
        if native.get("passed") is not True or not native.get("profile_evaluation"):
            raise ValueError("a successful profiled native result is required")
        if native["nodes"] != len(nodes) or native["evaluated_nodes"] != len(live):
            raise ValueError("native workload size differs")
        seen = set()
        for sample in native["polynomial_stats"]:
            index = sample["node"]
            if index in seen or index not in labels:
                raise ValueError("unexpected or duplicate polynomial timing")
            seen.add(index)
            for field in ("seconds", "bootstrap_seconds", "bootstraps"):
                value = sample[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid polynomial timing/counter")
                if field == "bootstraps" and type(value) is not int:
                    raise ValueError("bootstrap count must be integral")
                group = groups[labels[index]]
                group[field] = group.get(field, 0) + value
        if seen != set(labels):
            raise ValueError("polynomial timings are incomplete")
        total = native["operation_stats"]["cheb"]
        if total["nodes"] != len(seen):
            raise ValueError("polynomial node totals differ")
        for field in ("seconds", "bootstrap_seconds", "bootstraps"):
            actual = sum(group[field] for group in groups.values())
            if not math.isclose(actual, total[field], rel_tol=1e-8, abs_tol=1e-7):
                raise ValueError("polynomial timing totals differ")
        for group in groups.values():
            group["evaluation_fraction"] = group["seconds"] / native["eval_seconds"]
    for group in groups.values():
        group["degrees"] = dict(sorted(group["degrees"].items()))
    return {
        "schema_version": 1,
        "program_sha256": program["sha256"],
        "live_polynomial_nodes": len(labels),
        "profiled": profiled,
        "groups": groups,
        "scope": (
            "Polynomial-node work only; excludes surrounding reductions, masks and products. "
            "Refresh batches are charged to their initiating node, not exclusively caused by it. "
            "Calibration intervals and grid errors are not input-distribution certificates. "
            "Timing attribution is not a predicted saving."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--program", type=Path)
    inputs.add_argument("--metadata", type=Path, help="program_metadata() output from analyze.py")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native", type=Path, help="profile from the same frozen program")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    program = (
        program_metadata(args.program) if args.program else json.loads(args.metadata.read_text())
    )
    result = summarize(
        program,
        json.loads(args.manifest.read_text()),
        json.loads(args.native.read_text()) if args.native else None,
    )
    result["input_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("manifest", args.manifest), ("native", args.native))
        if path is not None
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
