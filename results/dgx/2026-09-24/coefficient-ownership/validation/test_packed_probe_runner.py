import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packed_runner", ROOT / "experiments/run_packed_probe.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fixture_files(tmp_path, body):
    payload = tmp_path / "payload"
    payload.mkdir()
    digests = {}
    for name in ("fixture.npz", "program.txt"):
        (payload / name).write_text("fixture")
        digests[name] = hashlib.sha256(b"fixture").hexdigest()
    (payload / "manifest.json").write_text(
        json.dumps(
            {
                "files_sha256": digests,
                "architecture": "mamba3",
                "tokens": 2,
                "output_names": ["output"],
            }
        )
    )
    binary = tmp_path / "native"
    binary.write_text("#!/usr/bin/env python3\n" + body)
    binary.chmod(0o755)
    return binary, payload


def test_timeout_is_bounded_and_recorded(tmp_path):
    binary, payload = fixture_files(tmp_path, "import time\ntime.sleep(20)\n")
    result = RUNNER.run(binary, payload, tmp_path / "result", timeout=0.05)
    assert result["timed_out"] is True
    assert result["passed"] is False
    assert result["wall_seconds"] < 5
    assert (tmp_path / "result/run.json").exists()


def test_tampered_payload_rejected_before_execution(tmp_path):
    binary, payload = fixture_files(tmp_path, "raise RuntimeError('must not run')\n")
    (payload / "program.txt").write_text("different fixture")
    with pytest.raises(ValueError, match="digest differs"):
        RUNNER.run(binary, payload, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize(("error", "passed"), [(0.0001, True), (0.1, False), (float("nan"), False)])
def test_runner_checks_errors_independently_of_native_passed(tmp_path, error, passed):
    result = {
        "schema": "fhemamba-packed-result-v1",
        "encrypted": True,
        "passed": True,
        "max_abs_error_vs_polynomial": error,
        "max_abs_error_vs_exact": error,
        "per_output_errors": [{}],
        "non_finite": 0,
    }
    body = (
        "import json, sys\nfrom pathlib import Path\n"
        f"Path(sys.argv[2]).write_text({json.dumps(result)!r})\n"
    )
    binary, payload = fixture_files(tmp_path, body)
    actual = RUNNER.run(binary, payload, tmp_path / "result")
    assert actual["passed"] is passed
    assert actual["binary_sha256"] == RUNNER.digest(binary)


@pytest.mark.parametrize("tokens", [[5, 6], [5, 9]])
def test_generation_gate_checks_actual_client_tokens(tmp_path, tokens):
    result = {
        "schema": "fhemamba-packed-result-v1",
        "encrypted": True,
        "passed": True,
        "max_abs_error_vs_polynomial": 1e-5,
        "max_abs_error_vs_exact": 1e-5,
        "per_output_errors": [{}],
        "non_finite": 0,
        "evaluation_decryptions": 0,
        "client_output_decrypt_count": 2,
        "generated_token_ids": tokens,
    }
    body = (
        "import json,sys\nfrom pathlib import Path\n"
        "assert '--client-head' in sys.argv\n"
        f"Path(sys.argv[2]).write_text({json.dumps(result)!r})\n"
    )
    binary, payload = fixture_files(tmp_path, body)
    (payload / "client_head.f32").write_bytes(b"head")
    manifest = json.loads((payload / "manifest.json").read_text())
    manifest.update(
        schema="fhemamba-mamba3-lm-v1",
        exact_token_ids=[5, 6],
        generated_tokens=2,
        complete_backbone=True,
        reference_text="example",
    )
    manifest["files_sha256"]["client_head.f32"] = RUNNER.digest(payload / "client_head.f32")
    (payload / "manifest.json").write_text(json.dumps(manifest))
    actual = RUNNER.run(binary, payload, tmp_path / "result")
    assert actual["passed"] is (tokens == [5, 6])


def test_budget_charges_failed_runs_and_preserves_limit(tmp_path):
    binary, payload = fixture_files(tmp_path, "import time\ntime.sleep(20)\n")
    ledger = tmp_path / "budget.json"
    actual = RUNNER.run(
        binary, payload, tmp_path / "result", timeout=0.05, budget_file=ledger, budget_seconds=30
    )
    budget = json.loads(ledger.read_text())
    assert actual["timed_out"]
    assert 0 < actual["wall_seconds"] <= budget["used_seconds"] < 5
    assert actual["campaign_used_seconds"] == budget["used_seconds"]
    with pytest.raises(ValueError, match="cannot change"):
        RUNNER.run(binary, payload, tmp_path / "next", budget_file=ledger, budget_seconds=60)
    budget["used_seconds"] = 16
    ledger.write_text(json.dumps(budget))
    with pytest.raises(ValueError, match="exhausted"):
        RUNNER.run(binary, payload, tmp_path / "next", budget_file=ledger, budget_seconds=30)
    assert not (tmp_path / "next").exists()


def test_budget_caps_native_timeout_and_keeps_routing_option(tmp_path, monkeypatch):
    calls = []

    def fake_run(binary, payload, output, **kwargs):
        calls.append(kwargs)
        output.mkdir()
        return {"passed": True}

    monkeypatch.setattr(RUNNER, "_run", fake_run)
    ledger = tmp_path / "budget.json"
    RUNNER.run(
        None,
        None,
        tmp_path / "result",
        timeout=1000,
        legacy_routing=True,
        budget_file=ledger,
        budget_seconds=60,
    )
    assert calls[0]["timeout"] == 45
    assert calls[0]["legacy_routing"] is True


@pytest.mark.parametrize("kwargs", [{"budget_seconds": 30}, {"budget_file": "unused"}])
def test_budget_requires_both_path_and_limit(kwargs):
    with pytest.raises(ValueError, match=r"requires|budget is required"):
        RUNNER.run(None, None, None, **kwargs)


@pytest.mark.parametrize(
    "options",
    [
        {"bootstrap_passes": 0},
        {"batch_refresh": True},
        {"planned_refresh": True, "batch_refresh": True, "bootstrap_passes": 1},
        {"planned_refresh": True, "legacy_routing": True},
    ],
)
def test_invalid_refresh_options_do_not_start_a_run(tmp_path, options):
    binary, payload = fixture_files(tmp_path, "raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match=r"requires|must be"):
        RUNNER.run(binary, payload, tmp_path / "result", **options)
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("budgeted", [False, True])
def test_refresh_options_reach_native_and_are_recorded(tmp_path, budgeted):
    body = (
        "import sys\n"
        "assert '--planned-refresh' in sys.argv\n"
        "assert '--batch-refresh' in sys.argv\n"
        "assert '--trace-levels' in sys.argv\n"
        "assert '--profile-evaluation' in sys.argv\n"
        "assert '--inplace-ops' in sys.argv\n"
        "assert '--cache-plaintexts' in sys.argv\n"
        "assert '--fast-plaintext-upload' in sys.argv\n"
        "assert '--gpu-plaintext-ntt' in sys.argv\n"
        "assert '--direct-plaintext-upload' in sys.argv\n"
        "assert '--move-plaintext-coefficients' in sys.argv\n"
        "assert '--naf-rotations' in sys.argv\n"
        "assert '--reuse-dead-inputs' in sys.argv\n"
        "assert '--compact-weights' in sys.argv\n"
    )
    binary, payload = fixture_files(tmp_path, body)
    budget = {"budget_file": tmp_path / "budget.json", "budget_seconds": 60} if budgeted else {}
    result = RUNNER.run(
        binary,
        payload,
        tmp_path / "result",
        planned_refresh=True,
        batch_refresh=True,
        trace_levels=True,
        profile_evaluation=True,
        inplace_ops=True,
        cache_plaintexts=True,
        fast_plaintext_upload=True,
        gpu_plaintext_ntt=True,
        direct_plaintext_upload=True,
        move_plaintext_coefficients=True,
        naf_rotations=True,
        reuse_dead_inputs=True,
        compact_weights=True,
        **budget,
    )
    assert result["returncode"] == 0
    expected_flags = [
        "--trace-levels",
        "--planned-refresh",
        "--batch-refresh",
        "--profile-evaluation",
        "--inplace-ops",
        "--cache-plaintexts",
        "--fast-plaintext-upload",
        "--gpu-plaintext-ntt",
        "--direct-plaintext-upload",
        "--move-plaintext-coefficients",
        "--naf-rotations",
        "--reuse-dead-inputs",
        "--compact-weights",
    ]
    assert result["command"][-len(expected_flags) :] == expected_flags
    assert result["passed"] is False  # No native result; flags cannot bypass the gate.
