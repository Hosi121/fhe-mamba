#!/usr/bin/env python3
"""Extract one observed elementary operation for fresh-ciphertext isolation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from fhemamba.diagnostics.analysis import elementary, max_difference, read_operations


def extract_refresh_group(metadata, observations, program, order):
    # Verify the entire frozen program before using its public bounds.
    read_operations(program, set(), metadata["program_sha256"])
    before, after, selected, last_event = [], {}, False, None
    digest = hashlib.sha256()
    for i, line in enumerate(observations.open("rb")):
        digest.update(line)
        row = json.loads(line)
        if row["event"] == "before_refresh":
            if last_event != "before_refresh":
                before, after = [], {}
            before.append(row)
        elif row["event"] == "after_refresh" and before:
            after[row["node"]] = row
            selected |= i == order
            if selected and len(after) == len(before):
                break
        elif i == order:
            raise ValueError("group isolation requires an after_refresh observation")
        last_event = row["event"]
    if not selected or len(before) != 2 or set(after) != {x["node"] for x in before}:
        raise ValueError("complete two-input refresh group not observed")
    if sum(len(x["values"]) for x in before) != 65536:
        raise ValueError("partial group cannot establish full refresh packing")
    expected = [v for x in before for v in x["values"]]
    actual = [v for x in before for v in after[x["node"]]["values"]]
    if any(v is None or not math.isfinite(v) for v in expected + actual):
        raise ValueError("nonfinite group observation")
    wanted, bounds = {x["node"] for x in before}, {}
    with program.open() as stream:
        next(stream)
        for i in range(max(wanted) + 1):
            line = next(stream)
            if i in wanted:
                bounds[i] = float(line.split()[2])
    fields = [
        "fhemamba-operation-case-v1",
        "refresh_group",
        "65536",
        "65536",
        repr(max(bounds.values())),
        "2",
    ]
    for x in before:
        fields.extend(map(str, [x["level"], x["degree"], len(x["values"])]))
        fields.extend(map(repr, x["values"]))
    fields.extend(["2", *(repr(bounds[x["node"]]) for x in before), "65536"])
    fields.extend(map(repr, expected))
    content = (" ".join(fields) + "\n").encode()
    return content, {
        "diagnostic_only": True,
        "original_program_sha256": metadata["program_sha256"],
        "observation_prefix_sha256": digest.hexdigest(),
        "observation_order": order,
        "operation": "refresh_group",
        "parent_nodes": [x["node"] for x in before],
        "inputs": [
            {k: x[k] for k in ("node", "event", "level", "degree")}
            | {"size": len(x["values"]), "bound": bounds[x["node"]]}
            for x in before
        ],
        "original_local_error": max_difference(actual, expected),
        "case_sha256": hashlib.sha256(content).hexdigest(),
        "scope": "Both full-packing companions retained; original RNS noise and keys not retained",
    }


def extract(metadata, observations, program, order):
    operations = read_operations(
        program, set(map(int, metadata["references"])), metadata["program_sha256"]
    )
    latest, digest = {}, hashlib.sha256()
    selected = None
    for i, line in enumerate(observations.open("rb")):
        digest.update(line)
        row = json.loads(line)
        if i == order:
            selected = row
            break
        latest[row["node"]] = row
    if selected is None:
        raise ValueError("observation order not present")
    node = selected["node"]
    operation = operations.get(node)
    if selected["event"] == "after_refresh":
        before = latest.get(node)
        if before is None or before["event"] != "before_refresh":
            raise ValueError("refresh has no matching prior observation")
        parents, op, data = [node], "refresh", []
        expected = before["values"]
    elif selected["event"] == "node_output" and operation:
        parents, op, data = operation["parents"], operation["operation"], operation["data"]
        if op == "sum":
            raise ValueError("native isolation does not support sum")
        if any(p not in latest for p in parents):
            raise ValueError("operation has an unobserved input")
        expected = elementary(operation, [latest[p]["values"] for p in parents])
    else:
        raise ValueError("unsupported event/operation for isolation")
    inputs = [latest[p] for p in parents]
    if any(v is None or not math.isfinite(v) for x in inputs for v in x["values"]):
        raise ValueError("nonfinite observed input")
    # Read the original public bound rather than estimating it from plaintext.
    with program.open() as stream:
        next(stream)
        for _ in range(node):
            next(stream)
        bound = float(next(stream).split()[2])
    fields = [
        "fhemamba-operation-case-v1",
        op,
        "65536",
        str(len(expected)),
        repr(bound),
        str(len(inputs)),
    ]
    for x in inputs:
        fields.extend(map(str, [x["level"], x["degree"], len(x["values"])]))
        fields.extend(map(repr, x["values"]))
    fields.append(str(len(data)))
    fields.extend(map(repr, data))
    fields.append(str(len(expected)))
    fields.extend(map(repr, expected))
    content = (" ".join(fields) + "\n").encode()
    return content, {
        "diagnostic_only": True,
        "original_program_sha256": metadata["program_sha256"],
        "observation_prefix_sha256": digest.hexdigest(),
        "observation_order": order,
        "original_node": node,
        "operation": op,
        "event": selected["event"],
        "parent_nodes": parents,
        "inputs": [{k: x[k] for k in ("node", "event", "level", "degree")} for x in inputs],
        "original_local_error": max_difference(selected["values"], expected),
        "case_sha256": hashlib.sha256(content).hexdigest(),
        "scope": (
            "Fresh-encryption isolation; original RNS noise and refresh-group companions "
            "are not retained"
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="fhemamba diagnose extract", description=__doc__)
    for name in ["metadata", "observations", "program", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--order", type=int, required=True)
    p.add_argument(
        "--refresh-group", action="store_true", help="extract both full-packing companions"
    )
    a = p.parse_args(argv)
    export = extract_refresh_group if a.refresh_group else extract
    content, info = export(json.loads(a.metadata.read_text()), a.observations, a.program, a.order)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "case.txt").write_bytes(content)
    (a.output / "case.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
