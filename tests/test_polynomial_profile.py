import importlib
from copy import deepcopy

import pytest


@pytest.fixture
def profile(monkeypatch):
    return importlib.import_module("fhemamba.diagnostics.polynomials").summarize


@pytest.fixture
def inputs():
    recipe = {
        "kind": "inv_sqrt",
        "lo": 1.0,
        "hi": 4.0,
        "degree": 1,
        "coefficients": [0.75, -0.25],
        "grid_max_abs_error": 0.1,
    }
    program = {
        "sha256": "frozen-program",
        "outputs": [2],
        "nodes": [
            {"op": "input", "parents": []},
            {"op": "cheb", "parents": [0], "data": [1.0, 4.0, 0.75, -0.25]},
            {"op": "cheb", "parents": [1], "data": [1.0, 4.0, 0.75, -0.25]},
            {"op": "cheb", "parents": [0], "data": [1.0, 4.0, 0.75, -0.25]},
        ],
    }
    manifest = {"files_sha256": {"program.txt": "frozen-program"}, "polynomials": {"norm": recipe}}
    native = {
        "passed": True,
        "profile_evaluation": True,
        "nodes": 4,
        "evaluated_nodes": 3,
        "eval_seconds": 10.0,
        "polynomial_stats": [
            {"node": 1, "seconds": 1.0, "bootstrap_seconds": 0.0, "bootstraps": 0},
            {"node": 2, "seconds": 2.0, "bootstrap_seconds": 0.5, "bootstraps": 2},
        ],
        "operation_stats": {
            "cheb": {"nodes": 2, "seconds": 3.0, "bootstrap_seconds": 0.5, "bootstraps": 2}
        },
    }
    return program, manifest, native


def test_profile_excludes_dead_nodes_and_matches_exact_coefficients(profile, inputs):
    result = profile(*inputs)
    group = result["groups"]["inv_sqrt"]
    assert result["live_polynomial_nodes"] == 2
    assert group["degrees"] == {1: 2}
    assert group["sites"]["norm"]["interval_ratio"] == 4.0
    assert group["seconds"] == 3.0
    assert group["evaluation_fraction"] == 0.3
    assert group["bootstraps"] == 2
    inputs[1]["polynomials"]["norm"]["coefficients"][1] += 1e-15
    with pytest.raises(ValueError, match="missing or ambiguous"):
        profile(*inputs)


def test_profile_rejects_wrong_program_or_ambiguous_function(profile, inputs):
    program, manifest, native = inputs
    manifest["files_sha256"]["program.txt"] = "other-program"
    with pytest.raises(ValueError, match="identity"):
        profile(*inputs)
    manifest["files_sha256"]["program.txt"] = program["sha256"]
    manifest["polynomials"]["other"] = deepcopy(manifest["polynomials"]["norm"])
    manifest["polynomials"]["other"]["kind"] = "exp"
    with pytest.raises(ValueError, match="missing or ambiguous"):
        profile(program, manifest, native)


@pytest.mark.parametrize("problem", ["partial", "duplicate", "dead", "total", "nonfinite"])
def test_profile_rejects_unusable_timing_evidence(profile, inputs, problem):
    native = inputs[2]
    if problem == "partial":
        native["polynomial_stats"].pop()
    elif problem == "duplicate":
        native["polynomial_stats"].append(native["polynomial_stats"][0])
    elif problem == "dead":
        native["polynomial_stats"][0]["node"] = 3
    elif problem == "total":
        native["operation_stats"]["cheb"]["seconds"] = 2.0
    else:
        native["polynomial_stats"][0]["seconds"] = float("nan")
    with pytest.raises(ValueError, match="polynomial"):
        profile(*inputs)
