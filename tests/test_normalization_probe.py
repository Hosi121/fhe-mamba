from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _probe_inputs(tmp_path):
    binary = tmp_path / "probe"
    # Pinned FIDESlib has been observed to exit zero after a CUDA fatal error.
    binary.write_text("#!/bin/sh\necho 'Cuda failure: synthetic test'\nexit 0\n")
    binary.chmod(0o755)
    binary.with_suffix(".build.json").write_text(
        json.dumps(
            {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "library_sha256": {}}
        )
    )
    recipe = tmp_path / "l00_rms_invsqrt.txt"
    recipe.write_text("fhemamba-invsqrt-v1\n0.01 1 1 1\n1.5 0.5\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "operators": [
                    {"file": recipe.name, "sha256": hashlib.sha256(recipe.read_bytes()).hexdigest()}
                ],
                "bundle_sha256": "b" * 64,
                "repo_commit": "a" * 40,
                "version": "0.5.0",
            }
        )
    )
    return binary, recipe


def _run(binary, recipe, output, *extra):
    return subprocess.run(
        [
            sys.executable,
            "experiments/run_normalization_probe.py",
            "--binary",
            str(binary),
            "--recipe",
            str(recipe),
            "--mode",
            "balanced",
            "--output",
            str(output),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_zero_native_exit_without_measurements_is_failure(tmp_path):
    binary, recipe = _probe_inputs(tmp_path)
    output = tmp_path / "result.json"
    result = _run(binary, recipe, output)
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert report["process_returncode"] == 0
    assert report["passed"] is False
    assert report["status"] == "failed"
    assert "before producing measurements" in report["failure"]
    assert "Cuda failure" in output.with_suffix(".log").read_text()


def test_modified_recipe_rejected_before_native_execution(tmp_path):
    binary, recipe = _probe_inputs(tmp_path)
    recipe.write_text(recipe.read_text() + "tampered\n")
    output = tmp_path / "result.json"
    result = _run(binary, recipe, output)
    assert result.returncode == 2
    assert "recipe differs" in result.stderr
    assert not output.with_suffix(".log").exists()


def test_unsupported_limb_budget_rejected_before_native_execution(tmp_path):
    binary, recipe = _probe_inputs(tmp_path)
    output = tmp_path / "result.json"
    result = _run(binary, recipe, output, "--depth", "56")
    assert result.returncode == 2
    assert "depth 4..44" in result.stderr
    assert not Path(output).exists()


def test_vector_fixture_must_match_recipe(tmp_path):
    binary, recipe = _probe_inputs(tmp_path)
    directory = tmp_path / "vectors"
    directory.mkdir()
    fixture = directory / "fixture.txt"
    fixture.write_text("fhemamba-vector-rms-v1 1 1 0.01 1 0\n")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "operators": [
                    {
                        "file": fixture.name,
                        "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        "recipe_sha256": "f" * 64,
                    }
                ]
            }
        )
    )
    output = tmp_path / "result.json"
    result = _run(binary, recipe, output, "--input-mode", "vector", "--fixture", str(fixture))
    assert result.returncode == 2
    assert "different normalization recipe" in result.stderr
    assert not output.with_suffix(".log").exists()


def test_vector_input_requires_fixture(tmp_path):
    binary, recipe = _probe_inputs(tmp_path)
    result = _run(binary, recipe, tmp_path / "result.json", "--input-mode", "vector")
    assert result.returncode == 2
    assert "must be supplied together" in result.stderr


def test_continuation_options_require_vector_mode(tmp_path):
    binary, recipe = _probe_inputs(tmp_path)
    result = _run(binary, recipe, tmp_path / "result.json", "--internal-refresh", "meta")
    assert result.returncode == 2
    assert "require vector mode" in result.stderr


def test_meta_alpha_rejected_even_without_output_refresh(tmp_path):
    binary, recipe = _probe_inputs(tmp_path)
    result = _run(binary, recipe, tmp_path / "result.json", "--meta-alpha", "21")
    assert result.returncode == 2
    assert "meta-alpha must" in result.stderr


def test_input_pressure_rejects_exhausted_seed_budget(tmp_path):
    binary, recipe = _probe_inputs(tmp_path)
    result = _run(binary, recipe, tmp_path / "result.json", "--input-level", "40")
    assert result.returncode == 2
    assert not (tmp_path / "result.log").exists()
