#!/usr/bin/env python3
"""Propagate selected observed errors through the frozen CPU circuit.

Diagnostic only: all other inputs stay at their CPU reference values and
later encrypted arithmetic/refresh errors are excluded. No FHE value changes.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from fhemamba.diagnostics.analysis import elementary, read_operations


def read_references(path, metadata):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata["references_sha256"]:
        raise ValueError("counterfactual reference digest mismatch")
    if data[:8] != b"FHEMDG01":
        raise ValueError("invalid reference magic")
    count = struct.unpack_from("<Q", data, 8)[0]
    offset, refs = 16, {}
    for _ in range(count):
        node, size = struct.unpack_from("<II", data, offset)
        offset += 8
        if node in refs or size != metadata["references"][str(node)]["size"]:
            raise ValueError("reference node/width mismatch")
        refs[node] = np.frombuffer(data, dtype="<f8", count=size, offset=offset)
        offset += size * 8
        if not np.isfinite(refs[node]).all():
            raise ValueError("nonfinite CPU reference")
    if offset != len(data) or count != len(metadata["references"]):
        raise ValueError("reference count/trailing data mismatch")
    return refs


def propagate(metadata, program, refs, injections, target):
    if not injections or target not in refs or any(n not in refs or n > target for n in injections):
        raise ValueError("injections and target require frozen references in node order")
    for node, values in injections.items():
        if len(values) != len(refs[node]) or not np.isfinite(values).all():
            raise ValueError("injection width/nonfinite value")
    extra = ("cheb", "linear", "linear_ref", "input", "public", "feedback")
    nodes = read_operations(
        program,
        set(range(min(injections), target + 1)),
        metadata["program_sha256"],
        extra_operations=extra,
    )
    affected = set(injections)
    for node, op in nodes.items():
        if any(parent in affected for parent in op["parents"]):
            affected.add(node)
    if target not in affected:
        raise ValueError("target is not downstream of the injected nodes")
    weight_ids = {
        int(nodes[n]["data"][0]) for n in affected if nodes[n]["operation"] == "linear_ref"
    }
    weights = (
        read_operations(
            program, weight_ids, metadata["program_sha256"], extra_operations=("linear",)
        )
        if weight_ids
        else {}
    )

    def run(replacements):
        values = dict(refs)
        for node in sorted(affected):
            op = nodes[node]
            if node in replacements:
                out = np.asarray(replacements[node], dtype=np.float64)
            else:
                parents = [values[p] for p in op["parents"]]
                if op["operation"] in {"linear", "linear_ref"}:
                    data = (
                        weights[int(op["data"][0])]["data"]
                        if op["operation"] == "linear_ref"
                        else op["data"]
                    )
                    out = np.asarray(data).reshape(op["size"], len(parents[0])) @ parents[0]
                elif op["operation"] == "cheb":
                    lo, hi, *coefficients = op["data"]
                    out = np.polynomial.chebyshev.chebval(
                        (2 * parents[0] - lo - hi) / (hi - lo), coefficients
                    )
                else:
                    out = np.asarray(elementary(op, parents))
            if out.shape != refs[node].shape or not np.isfinite(out).all():
                raise ValueError("counterfactual output width/nonfinite value")
            values[node] = out
        return values

    control = run({n: refs[n] for n in injections})
    control_error = max(float(np.max(np.abs(control[n] - refs[n]))) for n in affected)
    if control_error > 1e-7:
        raise ValueError("CPU counterfactual control differs from frozen references")
    changed = run(injections)
    result = {
        "diagnostic_only": True,
        "program_sha256": metadata["program_sha256"],
        "injected_nodes": sorted(injections),
        "target_node": target,
        "affected_node_count": len(affected),
        "control_max_abs_error": control_error,
        "target_max_abs_error_vs_reference": float(np.max(np.abs(changed[target] - refs[target]))),
        "scope": (
            "CPU propagation of only the injected errors; other inputs use frozen references "
            "and subsequent encrypted-operation/refresh errors are excluded"
        ),
    }
    return result, changed[target]


def main(argv=None):
    p = argparse.ArgumentParser(prog="fhemamba diagnose propagate", description=__doc__)
    for name in ("metadata", "references", "program", "observations", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--inject-order", type=int, nargs="+", required=True)
    p.add_argument("--target-node", type=int, required=True)
    a = p.parse_args(argv)
    metadata = json.loads(a.metadata.read_text())
    refs = read_references(a.references, metadata)
    selected, target, digest = {}, None, hashlib.sha256()
    for order, line in enumerate(a.observations.open("rb")):
        digest.update(line)
        row = json.loads(line)
        if order in a.inject_order:
            if row["node"] in selected or row["event"] != "after_refresh":
                raise ValueError("injection requires unique after-refresh nodes")
            selected[row["node"]] = row["values"]
        if row["node"] == a.target_node and row["event"] == "node_output":
            target = row["values"]
    if len(selected) != len(a.inject_order):
        raise ValueError("injection observation order missing")
    result, values = propagate(metadata, a.program, refs, selected, a.target_node)
    result["observations_sha256"] = digest.hexdigest()
    result["references_sha256"] = metadata["references_sha256"]
    result["injection_orders"] = a.inject_order
    if target is not None:
        if len(target) != len(values) or any(v is None for v in target):
            raise ValueError("invalid observed target")
        result["target_max_abs_error_vs_observed"] = float(np.max(np.abs(values - target)))
        result["observed_target_max_abs_error_vs_reference"] = float(
            np.max(np.abs(refs[a.target_node] - target))
        )
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
