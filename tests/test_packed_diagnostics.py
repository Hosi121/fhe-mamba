"""Diagnostic references must bind to the original arithmetic, not a new graph."""

import hashlib
import importlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    modules = {
        "packed_analysis": "analysis",
        "export_references": "references",
        "verify_prefix": "prefix",
        "extract_case": "cases",
        "counterfactual": "counterfactual",
    }
    return importlib.import_module("fhemamba.diagnostics." + modules[name])


def test_rejected_decryption_stays_missing_in_refresh_analysis(tmp_path):
    analyzer = load("packed_analysis")
    metadata = {"references": {"0": {"step": 24, "size": 2, "names": ["mul"]}}}
    rows = [
        {
            "node": 0,
            "event": "before_refresh",
            "max_abs_error": None,
            "decryption_error": "ckks_approximation_error_too_high",
            "values": [None, None],
        },
        {"node": 0, "event": "after_refresh", "values": [1.0, 2.0]},
    ]
    path = tmp_path / "observations.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = analyzer.analyze(metadata, {"polynomials": {}}, path)
    assert result["records"][0]["max_abs_error"] is None
    assert result["records"][1]["refresh_max_abs_change"] is None


@pytest.mark.parametrize("perturbation", [0.0, 0.1])
def test_counterfactual_propagates_frozen_linear_and_polynomial_error(
    tmp_path, monkeypatch, perturbation
):
    tool = load("counterfactual")
    program = tmp_path / "program.txt"
    program.write_text(
        "fhemamba-packed-v2 8 64 5 1\n"
        "input 2 8 0 2 1 2\n"
        "linear 2 64 1 0 4 1 2 3 4\n"
        "linear_ref 2 64 1 0 1 1\n"
        "cheb 2 64 1 2 4 -16 16 0 16\n"
        "sum 1 64 1 3 1 2\n"
        "4 1 16 16\n"
    )
    values = [[1, 2], [5, 11], [5, 11], [5, 11], [16]]
    data = b"FHEMDG01" + struct.pack("<Q", len(values))
    data += b"".join(
        struct.pack("<II", n, len(v)) + np.asarray(v, dtype="<f8").tobytes()
        for n, v in enumerate(values)
    )
    references = tmp_path / "references.bin"
    references.write_bytes(data)
    meta = {
        "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "references_sha256": hashlib.sha256(data).hexdigest(),
        "references": {str(n): {"size": len(v)} for n, v in enumerate(values)},
    }
    refs = tool.read_references(references, meta)
    result, output = tool.propagate(meta, program, refs, {0: [1 + perturbation, 2]}, 4)
    assert result["control_max_abs_error"] == 0
    assert result["target_max_abs_error_vs_reference"] == pytest.approx(4 * perturbation)
    assert output[0] == pytest.approx(16 + 4 * perturbation)
    with pytest.raises(ValueError, match="width/nonfinite"):
        tool.propagate(meta, program, refs, {0: [float("nan"), 2]}, 4)
    program.write_text(program.read_text() + "\n")
    with pytest.raises(ValueError, match="program digest"):
        tool.propagate(meta, program, refs, {0: [1, 2]}, 4)
    references.write_bytes(data + b"x")
    with pytest.raises(ValueError, match="reference digest"):
        tool.read_references(references, meta)


def test_reference_prefix_is_identical_and_rejects_changed_program(tmp_path):
    exporter, verifier = load("export_references"), load("verify_prefix")
    refs = tmp_path / "references.bin"
    p = exporter.ReferenceProgram({}, refs, [0])
    p.step = 0
    x = p.input(np.array([[1.0, 2.0]]))
    y = p.checkpoint(x * 2 + 1, (0, "layer_output"))
    p.output(y, y.value)
    program = tmp_path / "program.txt"
    p.write(program)
    p.close()
    metadata = {
        "node_count": len(p.nodes),
        "node_prefix_sha256": p.node_hash.hexdigest(),
        "program_sha256": exporter.digest(program),
    }
    assert verifier.verify(program, metadata)["passed"]
    with refs.open("rb") as stream:
        assert stream.read(8) == exporter.MAGIC
        assert struct.unpack("<QII", stream.read(16)) == (1, y.index, 2)
        np.testing.assert_array_equal(np.frombuffer(stream.read(), dtype="<f8"), [3, 5])
    program.write_text(program.read_text().replace("mulp", "addp"))
    with pytest.raises(ValueError, match="nodes differ"):
        verifier.verify(program, metadata)


def test_unselected_steps_are_not_observed(tmp_path):
    p = load("export_references").ReferenceProgram({}, tmp_path / "ref.bin", [1], True)
    p.step = 0
    p.checkpoint(p.input(np.array([1.0])), (0, "layer_output"))
    p.close()
    assert not p.references


def test_node_range_filters_observations_without_changing_the_graph(tmp_path):
    exporter = load("export_references")
    p = exporter.ReferenceProgram({}, tmp_path / "ref.bin", [0], True, [(1, 2), (3, 4)])
    p.step = 0
    x = p.input(np.array([1.0]))
    for _ in range(3):
        x = x + 1
    p.close()
    assert len(p.nodes) == 4
    assert sorted(p.references) == [1, 3]


def test_frozen_bounds_allow_roundoff_but_never_changed_arithmetic(tmp_path):
    exporter = load("export_references")
    original = tmp_path / "program.txt"
    original.write_text(
        "fhemamba-packed-v2 32768 100000000 1 1\ninput 1 8.0000000000000018 0 1 1\n0 1 1 1\n"
    )
    p = exporter.ReferenceProgram({}, tmp_path / "ok.bin", [0], original=original)
    p.step = 0
    p.input(np.array([1.0]))
    p.close()
    assert p.bound_rebindings == 1
    assert p.node_bounds[0] == 8.000000000000002
    assert p.max_relative_bound_drift < 1e-12
    p = exporter.ReferenceProgram({}, tmp_path / "bad.bin", [0], original=original)
    try:
        with pytest.raises(ValueError, match="arithmetic/constant"):
            p.input(np.array([1.1]))
    finally:
        p.close()
    original.write_text(original.read_text().replace("8.0000000000000018", "9"))
    p = exporter.ReferenceProgram({}, tmp_path / "bound.bin", [0], original=original)
    try:
        with pytest.raises(ValueError, match="bound drift"):
            p.input(np.array([1.0]))
    finally:
        p.close()


def test_analyzer_separates_input_drift_from_local_polynomial_error(tmp_path):
    analyzer = load("packed_analysis")
    meta = {
        "references": {
            "0": {"step": 24, "size": 2, "names": ["3:norm.input"]},
            "1": {"step": 24, "size": 2, "names": ["3:norm.output"]},
        }
    }
    manifest = {
        "polynomials": {"3:norm": {"lo": -1, "hi": 1, "degree": 2, "coefficients": [1, 2, 3]}}
    }
    x = [-0.2, 0.3]
    y = np.polynomial.chebyshev.chebval(x, [1, 2, 3])
    y[1] += 0.125
    rows = [
        {"node": 0, "event": "node_output", "values": [0, 0]},
        {"node": 0, "event": "before_refresh", "values": [0, 0]},
        {"node": 0, "event": "after_refresh", "values": x},
        {"node": 1, "event": "node_output", "values": y.tolist()},
    ]
    path = tmp_path / "observations.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = analyzer.analyze(meta, manifest, path)
    assert result["records"][2]["refresh_max_abs_change"] == pytest.approx(0.3)
    assert result["records"][3]["polynomial"][
        "max_abs_error_vs_polynomial_on_observed_input"
    ] == pytest.approx(0.125)
    rng = np.random.default_rng(37)
    coefficients = rng.normal(size=1024)
    for v in [-1, -0.5, 0, 0.3, 1]:
        assert analyzer.chebyshev(v, coefficients) == pytest.approx(
            np.polynomial.chebyshev.chebval(v, coefficients), abs=1e-9
        )


def test_elementary_diagnostics_use_observed_parents_and_frozen_program(tmp_path):
    analyzer = load("packed_analysis")
    program = tmp_path / "program.txt"
    program.write_text(
        "fhemamba-packed-v2 8 64 2 1\ninput 2 8 0 2 1 2\nmulp 2 16 1 0 2 2 2\n1 2 2 4 2 4\n"
    )
    meta = {
        "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "references": {
            "0": {"step": 0, "size": 2, "names": ["input"]},
            "1": {"step": 0, "size": 2, "names": ["mulp"]},
        },
    }
    observations = tmp_path / "observations.jsonl"
    observations.write_text(
        json.dumps({"node": 0, "event": "node_output", "values": [1.1, 2.2]})
        + "\n"
        + json.dumps({"node": 1, "event": "node_output", "values": [2.2, 4.45]})
        + "\n"
    )
    result = analyzer.analyze(meta, {"polynomials": {}}, observations, program)
    assert result["records"][1]["local_operation"][
        "max_abs_error_on_observed_inputs"
    ] == pytest.approx(0.05)
    program.write_text(program.read_text().replace("mulp", "addp"))
    with pytest.raises(ValueError, match="digest mismatch"):
        analyzer.analyze(meta, {"polynomials": {}}, observations, program)
    cases = [
        ("add", [], [[1, 2], [3, 4]], 2, [4, 6]),
        ("mul", [], [[1, 2], [3, 4]], 2, [3, 8]),
        ("addp", [3, 4], [[1, 2]], 2, [4, 6]),
        ("gather", [2, 0], [[1, 2, 3]], 2, [3, 1]),
        ("scatter", [2, 0], [[1, 2]], 4, [2, 0, 1, 0]),
        ("repeat", [2, 2, 3], [[1, 2, 3, 4]], 12, [1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]),
        ("sum", [2], [[1, 2, 3, 4]], 2, [3, 7]),
    ]
    for op, data, values, size, expected in cases:
        assert (
            analyzer.elementary({"operation": op, "data": data, "size": size}, values) == expected
        )


def test_isolation_extracts_latest_inputs_and_rejects_missing_history(tmp_path, monkeypatch):
    extract = load("extract_case").extract
    program = tmp_path / "program.txt"
    program.write_text(
        "fhemamba-packed-v2 8 64 2 1\ninput 2 8 0 2 1 2\nmulp 2 16 1 0 2 2 2\n1 2 2 4 2 4\n"
    )
    meta = {
        "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "references": {"0": {}, "1": {}},
    }
    rows = [
        {"node": 0, "event": "node_output", "level": 0, "degree": 1, "values": [1, 2]},
        {"node": 0, "event": "before_refresh", "level": 34, "degree": 2, "values": [1, 2]},
        {"node": 0, "event": "after_refresh", "level": 18, "degree": 2, "values": [1.1, 2.1]},
        {"node": 1, "event": "node_output", "level": 19, "degree": 2, "values": [2.2, 4.3]},
    ]
    observations = tmp_path / "observations.jsonl"
    observations.write_text("".join(json.dumps(r) + "\n" for r in rows))
    content, info = extract(meta, observations, program, 3)
    assert info["original_local_error"] == pytest.approx(0.1)
    assert info["inputs"][0]["event"] == "after_refresh"
    assert info["inputs"][0]["level"] == 18
    assert info["case_sha256"] == hashlib.sha256(content).hexdigest()
    _, info = extract(meta, observations, program, 2)
    assert info["operation"] == "refresh"
    assert info["original_local_error"] == pytest.approx(0.1)
    observations.write_text(json.dumps(rows[-1]) + "\n")
    with pytest.raises(ValueError, match="unobserved input"):
        extract(meta, observations, program, 0)
    with pytest.raises(ValueError, match="not present"):
        extract(meta, observations, program, 4)


def test_refresh_isolation_requires_all_full_packing_companions(tmp_path, monkeypatch):
    extract = load("extract_case").extract_refresh_group
    program = tmp_path / "program.txt"
    p = load("export_references").ReferenceProgram({}, tmp_path / "refs.bin", [0])
    p.step = 0
    a, b = p.input(np.full(32768, 0.1)), p.input(np.full(32768, 0.2))
    p.output(b, b.value)
    p.write(program)
    p.close()
    meta = {"program_sha256": hashlib.sha256(program.read_bytes()).hexdigest()}
    rows = [
        {
            "node": a.index,
            "event": "before_refresh",
            "level": 35,
            "degree": 2,
            "values": [0.1] * 32768,
        },
        {
            "node": b.index,
            "event": "before_refresh",
            "level": 35,
            "degree": 2,
            "values": [0.2] * 32768,
        },
        {
            "node": a.index,
            "event": "after_refresh",
            "level": 18,
            "degree": 2,
            "values": [0.11] * 32768,
        },
        {
            "node": b.index,
            "event": "after_refresh",
            "level": 18,
            "degree": 2,
            "values": [0.22] * 32768,
        },
    ]
    observations = tmp_path / "observations.jsonl"
    observations.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, info = extract(meta, observations, program, 2)
    assert info["parent_nodes"] == [0, 1]
    assert info["original_local_error"] == pytest.approx(0.02)
    assert [x["bound"] for x in info["inputs"]] == [1, 1.6]
    rows[1]["values"] = [0.2]
    observations.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="partial group"):
        extract(meta, observations, program, 2)
