"""Estimate modulus slack from a public packed program and an observed schedule.

This is a static cost screen, not a latency or encrypted-accuracy claim. The
trace contains only operation/level metadata, never decrypted intermediate data.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def program_metadata(path):
    digest = hashlib.sha256()
    nodes, outputs = [], []
    with path.open("rb") as stream:
        header = next(stream)
        digest.update(header)
        magic, slots, bound, count, output_count = header.split()
        if magic != b"fhemamba-packed-v2":
            raise ValueError("this screen requires a v2 packed program")
        for index, line in enumerate(stream):
            digest.update(line)
            if index >= int(count):
                outputs.append(int(line.split(maxsplit=1)[0]))
                continue
            op, size, node_bound, count_parents, rest = line.split(maxsplit=4)
            parts = rest.split(maxsplit=int(count_parents) + 1)
            parents = list(map(int, parts[: int(count_parents)]))
            data_count = int(parts[int(count_parents)])
            data = []
            if op != b"linear" and data_count:
                data = list(map(float, parts[int(count_parents) + 1].split()))
            nodes.append(
                {
                    "id": index,
                    "op": op.decode(),
                    "size": int(size),
                    "bound": float(node_bound),
                    "parents": parents,
                    "data": data,
                    "data_count": data_count,
                }
            )
    if len(nodes) != int(count) or len(outputs) != int(output_count):
        raise ValueError("program record count mismatch")
    return {
        "sha256": digest.hexdigest(),
        "slots": int(slots),
        "bound": float(bound),
        "nodes": nodes,
        "outputs": outputs,
    }


def analyze(program, log, refreshed=18, ceiling=35, towers=45):
    versions, actions, current = [], [], {}
    pending = None
    refresh_calls = 0

    def version(node, level, degree, kind):
        if degree not in (1, 2):
            raise ValueError("unsupported scale degree")
        index = len(versions)
        versions.append(
            {
                "id": index,
                "node": node,
                "level": level,
                "degree": degree,
                "effective_level": level + degree - 2,
                "kind": kind,
                "target": ceiling + degree - 2,
            }
        )
        current[node] = index
        return index

    def refresh(indices):
        nonlocal refresh_calls
        before = [current[index] for index in indices]
        after = [version(index, refreshed, 2, "refresh") for index in indices]
        actions.append({"kind": "refresh", "nodes": indices, "inputs": before, "outputs": after})
        refresh_calls += 2

    for line in log.splitlines():
        if line.startswith("refresh_batch "):
            if pending is not None:
                raise ValueError("unexpected refresh inside a node")
            refresh(list(map(int, re.search(r"nodes=([0-9,]+)", line)[1].strip(",").split(","))))
        elif line.startswith("level_input "):
            node = int(re.search(r"node=(\d+)", line)[1])
            metadata = program["nodes"][node]
            parents = [
                tuple(map(int, p.split(":")))
                for p in re.search(r"parents=([^ ]*)", line)[1].strip(",").split(",")
                if p
            ]
            if [p[0] for p in parents] != metadata["parents"] or pending is not None:
                raise ValueError("trace dependency mismatch")
            for parent, level, degree in parents:
                previous = versions[current[parent]]
                if (previous["level"], previous["degree"]) != (level, degree):
                    if (level, degree) != (refreshed, 2):
                        raise ValueError(f"unexplained level change at {node}: {parent}")
                    refresh([parent])
            pending = {
                "kind": "node",
                "node": node,
                "op": metadata["op"],
                "inputs": [current[p] for p in metadata["parents"]],
                "bootstraps": int(re.search(r"refreshed=(\d+)", line)[1]),
            }
        elif line.startswith("level_output "):
            node, level, degree, boots = map(
                int,
                re.match(
                    r"level_output node=(\d+) level=(\d+) degree=(\d+) bootstraps=(\d+)", line
                ).groups(),
            )
            if pending is None or node != pending["node"] or boots != pending["bootstraps"]:
                raise ValueError("unmodeled internal bootstrap or incomplete trace")
            out = version(node, level, degree, pending["op"])
            if pending["op"] in ("input", "public", "feedback"):
                cost = None
            else:
                cost = versions[out]["effective_level"] - max(
                    versions[i]["effective_level"] for i in pending["inputs"]
                )
                if cost < 0:
                    raise ValueError("negative observed operation depth")
            pending.update(output=out, cost=cost)
            actions.append(pending)
            pending = None
    if pending is not None or any(node not in current for node in program["outputs"]):
        raise ValueError("trace is incomplete")

    for action in reversed(actions):
        if action["kind"] == "refresh":
            continue
        if action["cost"] is not None:
            required = versions[action["output"]]["target"] - action["cost"]
            for parent in action["inputs"]:
                versions[parent]["target"] = min(versions[parent]["target"], required)
    summary = defaultdict(Counter)
    for value in versions:
        value["drop"] = value["target"] - value["effective_level"]
        if value["drop"] < 0:
            raise ValueError("observed schedule exceeds available depth")
    for action in actions:
        if action["kind"] != "node":
            continue
        stats = summary[action["op"]]
        stats["nodes"] += 1
        if action["cost"] is None:
            continue
        old = max(versions[i]["level"] for i in action["inputs"])
        new = max(versions[i]["target"] - versions[i]["degree"] + 2 for i in action["inputs"])
        stats["sum_input_towers_before"] += towers - old
        stats["sum_input_towers_after"] += towers - new
        stats["nodes_with_less_input_modulus"] += new > old
        stats["sum_input_towers_removed"] += new - old
    return {
        "scope": (
            "Optimistic level slack under the observed order and refresh groups; "
            "tower counts are not time predictions"
        ),
        "program_sha256": program["sha256"],
        "ceiling": ceiling,
        "refreshed": refreshed,
        "refresh_calls": refresh_calls,
        "nodes": sum(a["kind"] == "node" for a in actions),
        "summary": dict(summary),
        "actions": actions,
        "versions": versions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--metadata", action="store_true", help="program is already extracted metadata"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = (
        json.loads(args.program.read_text()) if args.metadata else program_metadata(args.program)
    )
    result = analyze(metadata, args.trace.read_text())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("actions", "versions")},
            indent=2,
        )
    )
