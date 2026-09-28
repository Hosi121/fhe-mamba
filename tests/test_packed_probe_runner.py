import hashlib
import json

import pytest

from fhemamba.benchmarks import packed as runner


def native_result(error, **fields):
    """Independent native-report fixture; each gate changes only its own evidence."""
    return {
        "schema": "fhemamba-packed-result-v1",
        "encrypted": True,
        "passed": True,
        "max_abs_error_vs_polynomial": error,
        "max_abs_error_vs_exact": error,
        "per_output_errors": [{}],
        "non_finite": 0,
        **fields,
    }


def fixture_files(tmp_path, body="", *, native=None):
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
    if native is not None:
        body = "import sys\nfrom pathlib import Path\n" + body
        body += f"Path(sys.argv[2]).write_text({json.dumps(native)!r})\n"
    binary.write_text("#!/usr/bin/env python3\n" + body)
    binary.chmod(0o755)
    return binary, payload


def test_tampered_payload_rejected_before_execution(tmp_path):
    binary, payload = fixture_files(tmp_path, "raise RuntimeError('must not run')\n")
    (payload / "program.txt").write_text("different fixture")
    with pytest.raises(ValueError, match="digest differs"):
        runner.run(binary, payload, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("limit", [-1, 1000001, True, 1.5])
def test_invalid_frontier_live_limit_rejected_before_execution(tmp_path, limit):
    binary, payload = fixture_files(tmp_path, "raise RuntimeError('must not run')\n")
    with pytest.raises(ValueError, match="frontier live limit"):
        runner.run(binary, payload, tmp_path / "result", frontier_live_limit=limit)
    assert not (tmp_path / "result").exists()


def test_frontier_live_limit_requires_frontier_schedule(tmp_path):
    binary, payload = fixture_files(tmp_path, "raise RuntimeError('must not run')\n")
    with pytest.raises(ValueError, match="requires frontier refresh"):
        runner.run(binary, payload, tmp_path / "result", frontier_live_limit=128)


def test_frontier_live_limit_is_forwarded_and_verified(tmp_path):
    result = native_result(1e-5, frontier_refresh=True, frontier_live_limit=128)
    binary, payload = fixture_files(
        tmp_path,
        "assert sys.argv[sys.argv.index('--frontier-live-limit')+1] == '128'\n",
        native=result,
    )
    actual = runner.run(
        binary,
        payload,
        tmp_path / "result",
        planned_refresh=True,
        batch_refresh=True,
        frontier_refresh=True,
        frontier_live_limit=128,
        budget_file=tmp_path / "budget.json",
        budget_seconds=30,
    )
    assert actual["passed"]


@pytest.mark.parametrize(("error", "passed"), [(0.0001, True), (0.1, False), (float("nan"), False)])
def test_runner_checks_errors_independently_of_native_passed(tmp_path, error, passed):
    result = native_result(error)
    binary, payload = fixture_files(tmp_path, native=result)
    actual = runner.run(binary, payload, tmp_path / "result")
    assert actual["passed"] is passed
    assert actual["binary_sha256"] == runner.digest(binary)


@pytest.mark.parametrize("tokens", [[5, 6], [5, 9]])
def test_generation_gate_checks_actual_client_tokens(tmp_path, tokens):
    result = native_result(
        1e-5, evaluation_decryptions=0, client_output_decrypt_count=2, generated_token_ids=tokens
    )
    binary, payload = fixture_files(tmp_path, "assert '--client-head' in sys.argv\n", native=result)
    (payload / "client_head.f32").write_bytes(b"head")
    manifest = json.loads((payload / "manifest.json").read_text())
    manifest.update(
        schema="fhemamba-mamba3-lm-v1",
        exact_token_ids=[5, 6],
        generated_tokens=2,
        complete_backbone=True,
        reference_text="example",
    )
    manifest["files_sha256"]["client_head.f32"] = runner.digest(payload / "client_head.f32")
    (payload / "manifest.json").write_text(json.dumps(manifest))
    actual = runner.run(binary, payload, tmp_path / "result")
    assert actual["passed"] is (tokens == [5, 6])


def test_budget_charges_failed_runs_and_preserves_limit(tmp_path):
    binary, payload = fixture_files(tmp_path, "import time\ntime.sleep(20)\n")
    ledger = tmp_path / "budget.json"
    actual = runner.run(
        binary, payload, tmp_path / "result", timeout=0.05, budget_file=ledger, budget_seconds=30
    )
    budget = json.loads(ledger.read_text())
    assert actual["timed_out"] is True
    assert actual["passed"] is False
    assert (tmp_path / "result/run.json").exists()
    assert 0 < actual["wall_seconds"] <= budget["used_seconds"] < 5
    assert actual["campaign_used_seconds"] == budget["used_seconds"]
    with pytest.raises(ValueError, match="cannot change"):
        runner.run(binary, payload, tmp_path / "next", budget_file=ledger, budget_seconds=60)
    budget["used_seconds"] = 16
    ledger.write_text(json.dumps(budget))
    with pytest.raises(ValueError, match="exhausted"):
        runner.run(binary, payload, tmp_path / "next", budget_file=ledger, budget_seconds=30)
    assert not (tmp_path / "next").exists()


@pytest.mark.parametrize("passed", [True, False])
def test_budget_forwards_options_and_preserves_result(tmp_path, monkeypatch, passed):
    # Budgeting is a transparent wrapper: test it once, not for every evidence defect.
    calls = []

    def fake_run(binary, payload, output, **kwargs):
        calls.append((binary, payload, output, kwargs))
        output.mkdir()
        return {"passed": passed, "native_sha256": "evidence"}

    monkeypatch.setattr(runner, "_run", fake_run)
    ledger = tmp_path / "budget.json"
    binary, payload, output = (tmp_path / name for name in ("binary", "payload", "result"))
    options = {"legacy_routing": True, "security": "128-classic", "security_digits": 4}
    result = runner.run(
        binary,
        payload,
        output,
        timeout=1000,
        budget_file=ledger,
        budget_seconds=60,
        **options,
    )
    assert calls == [(binary, payload, output, {"timeout": 45, **options})]
    assert result["passed"] is passed
    assert result["native_sha256"] == "evidence"
    assert result["campaign_budget_seconds"] == 60
    assert result["campaign_used_seconds"] == json.loads(ledger.read_text())["used_seconds"]
    assert json.loads((output / "run.json").read_text()) == result


@pytest.mark.parametrize("kwargs", [{"budget_seconds": 30}, {"budget_file": "unused"}])
def test_budget_requires_both_path_and_limit(kwargs):
    with pytest.raises(ValueError, match=r"requires|budget is required"):
        runner.run(None, None, None, **kwargs)


def test_unknown_option_does_not_touch_budget_or_output(tmp_path):
    ledger = tmp_path / "budget.json"
    output = tmp_path / "result"
    with pytest.raises(TypeError, match="unexpected keyword"):
        runner.run(
            None,
            None,
            output,
            budget_file=ledger,
            budget_seconds=60,
            misspelled_option=True,
        )
    assert not ledger.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    "options",
    [
        {"bootstrap_passes": 0},
        {"plaintext_cache_capacity": 1024},
        {"cache_plaintexts": True, "plaintext_cache_capacity": -1},
        {"cache_plaintexts": True, "plaintext_cache_capacity": True},
        {"prefetch_plaintexts": True},
        {"prefetch_workers": 2},
        {"prefetch_workers": 0},
        {"prefetch_workers": 3},
        {"prefetch_workers": 1.5},
        {"batch_plaintext_rns": True, "fuse_plaintext_rns_ntt": True},
        {"gpu_plaintext_fft": True},
        {"gpu_plaintext_fft": True, "gpu_plaintext_rns": True, "batch_plaintext_rns": True},
        {"gpu_plaintext_fft": True, "gpu_plaintext_rns": True, "fuse_plaintext_rns_ntt": True},
        {"frontier_refresh": True},
        {"frontier_refresh": True, "planned_refresh": True},
        {"s2c_first": True},
        {"s2c_first": True, "planned_refresh": True},
        {"gpu_dual_ring": True},
        {"gpu_dual_ring": True, "planned_refresh": True, "batch_refresh": True},
        {"batch_refresh": True},
        {"merge_refresh_correction": True},
        {"planned_refresh": True, "merge_refresh_correction": True, "bootstrap_passes": 1},
        {"planned_refresh": True, "batch_refresh": True, "bootstrap_passes": 1},
        {"planned_refresh": True, "legacy_routing": True},
    ],
)
def test_invalid_refresh_options_do_not_start_a_run(tmp_path, options):
    binary, payload = fixture_files(tmp_path, "raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match=r"requires|must be|choose one"):
        runner.run(binary, payload, tmp_path / "result", **options)
    assert not (tmp_path / "result").exists()


def test_refresh_options_reach_native_and_are_recorded(tmp_path):
    expected_flags = [
        "--trace-levels",
        "--planned-refresh",
        "--batch-refresh",
        "--profile-evaluation",
        "--inplace-ops",
        "--cache-plaintexts",
        "--indexed-mask-cache",
        "--fast-plaintext-upload",
        "--gpu-plaintext-ntt",
        "--direct-plaintext-upload",
        "--move-plaintext-coefficients",
        "--borrow-plaintext-upload",
        "--bsgs-routing-stages",
        "--naf-rotations",
        "--reuse-dead-inputs",
        "--reuse-public-ciphertexts",
        "--compact-weights",
        "--frontier-refresh",
        "--s2c-first",
        "--gpu-plaintext-rns",
        "--gpu-addend-rns",
        "--fuse-plaintext-rns-ntt",
        "--prefetch-plaintexts",
        "--hoist-rotations",
        "--share-chebyshev",
        "--gpu-dual-ring",
        "--prefetch-workers",
        "2",
        "--plaintext-cache-capacity",
        "128",
    ]
    body = "import sys\n" + f"assert sys.argv[-{len(expected_flags)}:] == {expected_flags!r}\n"
    binary, payload = fixture_files(tmp_path, body)
    result = runner.run(
        binary,
        payload,
        tmp_path / "result",
        planned_refresh=True,
        batch_refresh=True,
        trace_levels=True,
        profile_evaluation=True,
        inplace_ops=True,
        cache_plaintexts=True,
        indexed_mask_cache=True,
        plaintext_cache_capacity=128,
        fast_plaintext_upload=True,
        gpu_plaintext_ntt=True,
        direct_plaintext_upload=True,
        move_plaintext_coefficients=True,
        borrow_plaintext_upload=True,
        bsgs_routing_stages=True,
        naf_rotations=True,
        reuse_dead_inputs=True,
        reuse_public_ciphertexts=True,
        compact_weights=True,
        frontier_refresh=True,
        s2c_first=True,
        gpu_plaintext_rns=True,
        gpu_addend_rns=True,
        fuse_plaintext_rns_ntt=True,
        prefetch_plaintexts=True,
        prefetch_workers=2,
        hoist_rotations=True,
        share_chebyshev=True,
        gpu_dual_ring=True,
        budget_file=tmp_path / "budget.json",
        budget_seconds=60,
    )
    assert result["returncode"] == 0
    assert result["command"][-len(expected_flags) :] == expected_flags
    assert result["passed"] is False  # No native result; flags cannot bypass the gate.


# Each row isolates one native evidence failure; no acceptance predicate is reused.
_EVIDENCE_CASES = {
    "merge": (
        {"planned_refresh": True, "merge_refresh_correction": True},
        {
            "merge_refresh_correction": True,
            "merged_refreshes": 2,
            "merge_refresh_fallbacks": 1,
            "bootstraps": 6,
        },
        [
            ("missing_flag", "merge_refresh_correction", None),
            ("missing_count", "merged_refreshes", None),
            ("bad_total", "bootstraps", 4),
            ("negative", "merge_refresh_fallbacks", -1),
            ("boolean", "merged_refreshes", True),
        ],
    ),
    "fft": (
        {"gpu_plaintext_rns": True, "gpu_plaintext_fft": True},
        {
            "gpu_plaintext_fft": True,
            "gpu_fft_encodes": 8,
            "gpu_fft_fallbacks": 2,
            "host_encodes": 10,
        },
        [
            ("flag", "gpu_plaintext_fft", False),
            ("missing", "gpu_fft_encodes", None),
            ("empty", "gpu_fft_encodes", 0),
            ("negative", "gpu_fft_fallbacks", -1),
            ("boolean", "gpu_fft_encodes", True),
            ("total", "host_encodes", 1),
        ],
    ),
    "classical128": (
        {"security": "128-classic", "security_digits": 4},
        {
            "security": "128-classic",
            "ring_dimension": 131072,
            "refresh_ring_dimension": 131072,
            "gpu_dual_ring": False,
            "evaluation_decryptions": 0,
            "security_audit": {
                "passed": True,
                "ring_dimension": 131072,
                "qp_bits": 3376,
                "guideline_max_qp_bits": 3523,
                "error_sigma": 3.19,
                "library_classical128": True,
                "uniform_ternary": True,
                "hybrid": True,
                "hybrid_digits": 4,
            },
        },
        [
            ("missing_audit", "security_audit", None),
            ("not_set", "security_audit.library_classical128", False),
            ("sparse", "security_audit.uniform_ternary", False),
            ("oversized", "security_audit.qp_bits", 3524),
            ("wrong_bound", "security_audit.guideline_max_qp_bits", 3576),
            ("small_refresh_ring", "refresh_ring_dimension", 65536),
            ("dual_ring", "gpu_dual_ring", True),
            ("low_sigma", "security_audit.error_sigma", 3.0),
            ("nan_sigma", "security_audit.error_sigma", float("nan")),
            ("wrong_digits", "security_audit.hybrid_digits", 6),
            ("evaluator_decrypts", "evaluation_decryptions", 1),
        ],
    ),
}


@pytest.mark.parametrize(
    ("options", "evidence", "path", "value"),
    [
        pytest.param(options, evidence, path, value, id=f"{name}-{defect}")
        for name, (options, evidence, defects) in _EVIDENCE_CASES.items()
        for defect, path, value in [("valid", None, None), *defects]
    ],
)
def test_requested_mode_requires_native_evidence(tmp_path, options, evidence, path, value):
    from copy import deepcopy

    native = native_result(1e-5 if "security" in options else 1e-6, **deepcopy(evidence))
    if path:
        target = native
        *parents, key = path.split(".")
        for parent in parents:
            target = target[parent]
        if value is None:
            del target[key]
        else:
            target[key] = value
    flags = ["--" + name.replace("_", "-") for name in options]
    body = f"assert all(flag in sys.argv for flag in {flags!r})\n"
    if "security" in options:
        body += "assert sys.argv[sys.argv.index('--security') + 1] == '128-classic'\n"
        body += "assert sys.argv[sys.argv.index('--security-digits') + 1] == '4'\n"
    binary, payload = fixture_files(tmp_path, body, native=native)
    actual = runner.run(binary, payload, tmp_path / "result", **options)
    assert actual["returncode"] == 0
    assert actual["passed"] is (path is None)


@pytest.mark.parametrize(
    "options",
    [
        {"security": "128-quantum"},
        {"gpu_addend_rns": True},
        {"security": "128-classic", "gpu_dual_ring": True},
        {"security_digits": 4},
        {"security": "128-classic", "security_digits": 0},
        {"security": "128-classic", "security_digits": True},
    ],
)
def test_invalid_security_options_rejected_before_execution(tmp_path, options):
    binary, payload = fixture_files(tmp_path, "raise RuntimeError('must not run')\n")
    with pytest.raises(ValueError, match=r"security|classical-128|GPU addend RNS"):
        runner.run(binary, payload, tmp_path / "result", **options)
    assert not (tmp_path / "result").exists()
