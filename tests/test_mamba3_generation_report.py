"""A completed process is not proof of encrypted autoregressive generation."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generation_case(tmp_path, monkeypatch):
    from fhemamba.benchmarks import generation as reporter

    payload, run = tmp_path / "payload", tmp_path / "run"
    payload.mkdir()
    run.mkdir()
    baseline = ROOT / "results/b300/2026-09-27/gpu-plaintext-fft/full-fft-candidate-r3"
    native = json.loads((baseline / "native.json").read_text())
    manifest = {
        "schema": "fhemamba-mamba3-lm-v1",
        "generated_tokens": 4,
        "tokens": 5,
        "prompt_ids": [1, 2],
        "prompt": "test",
        "layers": 12,
        "complete_backbone": True,
        "exact_token_ids": native["generated_token_ids"],
        "polynomial_token_ids": native["generated_token_ids"],
        "files_sha256": {
            "program.txt": "program",
            "fixture.npz": "fixture",
            "client_head.f32": "head",
        },
        "client": "actual client selection and encrypted feedback",
    }
    (payload / "manifest.json").write_text(json.dumps(manifest))
    record = {
        "schema_version": 1,
        "state": "completed",
        "exit_code": 0,
        "inputs": {
            "manifest": {"sha256": reporter.digest(payload / "manifest.json")},
            "program": {"sha256": "program"},
            "fixture": {"sha256": "fixture"},
            "client_head": {"sha256": "head"},
        },
    }
    (run / "run.json").write_text(json.dumps(record))
    (run / "native.json").write_text(json.dumps(native))
    return reporter, payload, run, native


def test_generation_report_counts_actual_generated_tokens(generation_case):
    reporter, payload, run, native = generation_case
    result = reporter.validate_generation(payload, run, security="128-classic")
    assert result["server_evaluations"] == 5
    assert result["generated_tokens"] == 4
    assert result["seconds_per_generated_token"] == native["eval_seconds"] / 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_abs_error_vs_exact", 0.002),
        ("max_abs_error_vs_polynomial", float("nan")),
        ("generated_token_ids", [1, 2, 3, 4]),
        ("client_output_decrypt_count", 3),
        ("evaluation_decryptions", 1),
        ("diagnostic_only", True),
        ("bootstrap_passes", 1),
        ("ring_dimension", 32768),
        ("exact_tolerance", 0.01),
        ("eval_seconds", float("inf")),
    ],
)
def test_generation_report_rejects_false_success(generation_case, field, value):
    reporter, payload, run, native = generation_case
    native[field] = value
    (run / "native.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="generation"):
        reporter.validate_generation(payload, run, security="128-classic")


def test_generation_report_checks_each_output_not_only_aggregate(generation_case):
    reporter, payload, run, native = generation_case
    native["per_output_errors"][-1]["exact"] = 0.01
    (run / "native.json").write_text(json.dumps(native))
    with pytest.raises(ValueError, match="output exceeds"):
        reporter.validate_generation(payload, run)


def test_generation_report_requires_a_result_even_after_exit_zero(generation_case):
    reporter, payload, run, _ = generation_case
    (run / "native.json").unlink()
    with pytest.raises(FileNotFoundError):
        reporter.validate_generation(payload, run)


def test_generation_report_binds_payload(generation_case):
    reporter, payload, run, _ = generation_case
    manifest = json.loads((payload / "manifest.json").read_text())
    manifest["prompt"] = "changed"
    (payload / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different payload"):
        reporter.validate_generation(payload, run)
