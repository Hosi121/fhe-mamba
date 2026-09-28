#!/usr/bin/env python3
"""Summarize packed diagnostic errors and separate input drift from local arithmetic error."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def chebyshev(x, coefficients):
    b1 = b2 = 0.0
    for c in reversed(coefficients[1:]):
        b1, b2 = 2 * x * b1 - b2 + c, b1
    return x * b1 - b2 + coefficients[0]


def max_difference(a, b):
    if any(x is None or not math.isfinite(x) for x in [*a, *b]):
        return None
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def read_operations(path, selected, expected_sha256, *, extra_operations=()):
    """Read selected elementary operators, preserving original node IDs."""
    digest, operations = hashlib.sha256(), {}
    supported = {"add", "mul", "addp", "mulp", "gather", "scatter", "repeat", "sum"}
    supported.update(extra_operations)
    with path.open("rb") as stream:
        header = stream.readline()
        digest.update(header)
        fields = header.split()
        if fields[0] != b"fhemamba-packed-v2":
            raise ValueError("operator analysis requires a v2 program")
        for node in range(int(fields[3])):
            line = stream.readline()
            digest.update(line)
            op = line.split(None, 1)[0].decode()
            if node not in selected or op not in supported:
                continue
            row = line.split()
            count = int(row[3])
            operations[node] = {
                "operation": op,
                "size": int(row[1]),
                "parents": list(map(int, row[4 : 4 + count])),
                "data": list(map(float, row[5 + count :])),
            }
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("operator analysis program digest mismatch")
    return operations


def elementary(node, values):
    op, data = node["operation"], node["data"]
    x = values[0]
    if op in {"add", "mul", "addp", "mulp"}:
        y = data if op.endswith("p") else values[1]
        return [a + b if op.startswith("add") else a * b for a, b in zip(x, y, strict=True)]
    if op == "gather":
        return [x[int(i)] for i in data]
    if op == "scatter":
        result = [0.0] * node["size"]
        for i, value in zip(data, x, strict=True):
            result[int(i)] += value
        return result
    if op == "repeat":
        outer, inner, repeat = map(int, data)
        return [
            value
            for g in range(outer)
            for _ in range(repeat)
            for value in x[g * inner : (g + 1) * inner]
        ]
    if op == "sum":
        width = int(data[0])
        return [math.fsum(x[i : i + width]) for i in range(0, len(x), width)]
    raise ValueError("unsupported elementary diagnostic operator")


def analyze(metadata, manifest, observations, program=None):
    references = metadata["references"]
    sites = {(r["step"], name): int(node) for node, r in references.items() for name in r["names"]}
    latest, records, digest = {}, [], hashlib.sha256()
    operations = (
        read_operations(program, set(map(int, references)), metadata["program_sha256"])
        if program
        else {}
    )
    for order, line in enumerate(observations.open("rb")):
        digest.update(line)
        row = json.loads(line)
        node = row["node"]
        info = references[str(node)]
        values = row.pop("values")
        if len(values) != info["size"]:
            raise ValueError("diagnostic observation width differs from reference")
        row.update(order=order, step=info["step"], names=info["names"])
        previous = latest.get(node)
        if row["event"] == "after_refresh" and previous is not None:
            row["refresh_max_abs_change"] = max_difference(values, previous)
        if row["event"] == "node_output" and node in operations:
            operation = operations[node]
            parents = operation["parents"]
            if all(parent in latest for parent in parents):
                inputs = [latest[parent] for parent in parents]
                if all(v is not None and math.isfinite(v) for x in inputs for v in x):
                    row["local_operation"] = {
                        "operation": operation["operation"],
                        "parent_nodes": parents,
                        "max_abs_error_on_observed_inputs": max_difference(
                            values, elementary(operation, inputs)
                        ),
                    }
        for name in info["names"]:
            key = name.removesuffix(".output")
            if (
                row["event"] != "node_output"
                or not name.endswith(".output")
                or key not in manifest["polynomials"]
            ):
                continue
            parent = sites[info["step"], key + ".input"]
            if parent not in latest:
                raise ValueError(f"missing observed input for {key}, step {info['step']}")
            x = latest[parent]
            p = manifest["polynomials"][key]
            if any(v is None or not math.isfinite(v) for v in x):
                row["polynomial_input_non_finite"] = True
                continue
            expected = [
                chebyshev((2 * v - p["lo"] - p["hi"]) / (p["hi"] - p["lo"]), p["coefficients"])
                for v in x
            ]
            row["polynomial"] = {
                "site": key,
                "degree": p["degree"],
                "input_node": parent,
                "input_min": min(x),
                "input_max": max(x),
                "domain": [p["lo"], p["hi"]],
                "outside_domain_count": sum(v < p["lo"] or v > p["hi"] for v in x),
                "max_abs_error_vs_polynomial_on_observed_input": max_difference(values, expected),
            }
        latest[node] = values
        records.append(row)
    # This threshold is diagnostic only: intermediate quantities have different
    # units/scales. The 0.001 qualification gate applies to final hidden outputs.
    return {
        "diagnostic_only": True,
        "observations_sha256": digest.hexdigest(),
        "observation_count": len(records),
        "observed_node_count": len(latest),
        "records": records,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="fhemamba diagnose analyze", description=__doc__)
    for name in ["metadata", "manifest", "observations", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--program", type=Path, help="also analyze elementary operators on observed inputs"
    )
    a = p.parse_args(argv)
    result = analyze(
        json.loads(a.metadata.read_text()),
        json.loads(a.manifest.read_text()),
        a.observations,
        a.program,
    )
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
