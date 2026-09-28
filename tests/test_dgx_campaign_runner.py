from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cli_environment_overrides_are_recorded(campaign) -> None:
    campaign.configure(
        {
            "name": "portable-spark",
            "defaults": {"PT_CACHE_GIB": "5"},
            "experiments": [{"name": "candidate", "env": {"LOGARITHMIC_REPLICATION": "1"}}],
        }
    )
    campaign.run(
        "--dry-run",
        "--env",
        "INPUT_CHAIN=/payloads/with spaces",
        "--env",
        "PT_CACHE_GIB=7",
        check=True,
        capture_output=True,
    )
    environment = campaign.report["experiments"][0]["environment"]
    assert environment["INPUT_CHAIN"] == "/payloads/with spaces"
    assert environment["PT_CACHE_GIB"] == "7"
    assert environment["LOGARITHMIC_REPLICATION"] == "1"


def _write_fake_runner(path: Path, *, write_artifact: bool = True) -> None:
    artifact_block = (
        """
counter_file = os.environ.get("COUNTER_FILE")
if counter_file:
    counter = Path(counter_file)
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
results = Path(os.environ["RESULTS_DIR"])
results.mkdir(parents=True, exist_ok=True)
for layer in os.environ["LAYERS"].split():
    output = results / f'm2_chain_{os.environ["RUN_TAG"]}_l{layer}_t{os.environ["TOKENS"]}.json'
    error = float(os.environ.get("FAKE_ERROR", "0.01"))
    passed = error <= 0.05
    tokens = int(os.environ.get("FAKE_TOKENS", os.environ["TOKENS"]))
    selected_ids = [10, 11]
    expected_ids = [10, 12] if os.environ.get("FAKE_TOKEN_MISMATCH") == "1" else selected_ids
    decrypt_ok = [1] * tokens
    if os.environ.get("FAKE_DECRYPT_FAILURE") == "1":
        decrypt_ok[-1] = 0
    if os.environ.get("FAKE_DECRYPT_MALFORMED") == "1":
        decrypt_ok[-1] = "false"
    output.write_text(json.dumps({
        "version": os.environ.get("FAKE_VERSION", os.environ["ARTIFACT_VERSION"]),
        "repo_commit": os.environ.get("FAKE_REPO_COMMIT", os.environ["REPO_COMMIT"]),
        "binary_sha256": os.environ.get(
            "FAKE_BINARY_SHA256", os.environ.get("BINARY_SHA256", "f" * 64)
        ),
        "input_payload_sha256": os.environ.get("INPUT_CHAIN_SHA256"),
        "stage": "stage1-mamba2-decode-fideslib",
        "backend": "fideslib-gpu",
        "encrypted": True,
        "config": {"input_mode": "test-campaign"},
        "status": "passed" if passed else "failed",
        "passed": passed,
        "parameters": {
            "n_layers_loaded": int(os.environ.get("FAKE_LAYERS", layer)),
            "tokens": tokens,
            "fideslib_sync_profile": os.environ.get(
                "FAKE_SYNC_PROFILE", os.environ.get("FIDESLIB_SYNC_PROFILE", "full")
            ),
        },
        "measurements": {
            "max_abs_error": error,
            "per_token_decrypt_ok": decrypt_ok,
            "autoregressive_selected_ids": selected_ids,
            "autoregressive_expected_ids": expected_ids,
            "autoregressive_tokens_match": selected_ids == expected_ids,
        },
        "measurement_scope": {
            "zero_intermediate_decrypts": os.environ.get("FAKE_INTERMEDIATE_DECRYPT") != "1",
            "full_layer_chain": int(os.environ.get("FAKE_LAYERS", layer)) == 24,
            "layers_loaded": int(os.environ.get("FAKE_LAYERS", layer)),
            "tokens": tokens,
            "full_model_correctness_claimed": False,
            "claim": "Synthetic campaign-runner test artifact.",
        },
    }))
"""
        if write_artifact
        else ""
    )
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        f"{artifact_block}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


class Campaign:
    """Real CLI execution with fresh fake-native artifacts for each test."""

    def __init__(self, root):
        self.results = root / "results"
        self.runner = root / "fake_runner.py"
        self.manifest = root / "campaign.json"
        self.output = root / "campaign-result.json"
        _write_fake_runner(self.runner)

    def configure(self, spec):
        defaults = {"RESULTS_DIR": str(self.results), "TOKENS": "2", **spec.get("defaults", {})}
        self.manifest.write_text(json.dumps({**spec, "defaults": defaults}), encoding="utf-8")

    @property
    def command(self):
        return [
            sys.executable,
            "experiments/execution/run_dgx_campaign.py",
            "--manifest",
            str(self.manifest),
            "--runner",
            str(self.runner),
            "--output-json",
            str(self.output),
        ]

    @property
    def report(self):
        return json.loads(self.output.read_text(encoding="utf-8"))

    def run(self, *arguments, check=False, **options):
        return subprocess.run([*self.command, *arguments], cwd=ROOT, check=check, **options)


@pytest.fixture
def campaign(tmp_path):
    return Campaign(tmp_path)


def test_campaign_continues_on_candidate_failure_and_applies_promotion_gate(campaign) -> None:
    campaign.configure(
        {
            "name": "test-campaign",
            "experiments": [
                {"name": "bad-proxy", "env": {"LAYERS": "8", "FAKE_ERROR": "0.2"}},
                {
                    "name": "deep-run",
                    "when": {
                        "experiment": "bad-proxy",
                        "max_abs_error_lte": 0.05,
                        "all_tokens_decrypt": True,
                    },
                    "env": {"LAYERS": "24", "FAKE_ERROR": "0.01"},
                },
            ],
        }
    )
    completed = campaign.run(check=False)
    payload = campaign.report
    assert completed.returncode == 0
    assert payload["passed"] is True
    assert payload["measurements"]["infrastructure_failures"] == 0
    assert payload["experiments"][0]["candidate_passed"] is False
    assert payload["experiments"][0]["infrastructure_ok"] is True
    assert payload["experiments"][1]["state"] == "skipped"
    assert not (campaign.results / "m2_chain_deep-run_l24_t2.json").exists()


def test_campaign_fails_fast_when_runner_produces_no_artifact(campaign) -> None:
    _write_fake_runner(campaign.runner, write_artifact=False)
    campaign.configure({"experiments": [{"name": "missing", "env": {"LAYERS": "2"}}]})
    completed = campaign.run(check=False)
    payload = campaign.report
    assert completed.returncode == 1
    assert payload["passed"] is False
    assert payload["measurements"]["infrastructure_failures"] == 1
    assert "missing artifact" in payload["experiments"][0]["issues"][0]


def test_campaign_terminates_timed_out_process_group(campaign) -> None:
    campaign.runner.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    campaign.runner.chmod(0o755)
    campaign.configure(
        {
            "timeout_seconds": 0.1,
            "experiments": [{"name": "timeout", "env": {"LAYERS": "2"}}],
        }
    )
    completed = campaign.run(check=False, timeout=5)
    payload = campaign.report
    assert completed.returncode == 1
    assert payload["experiments"][0]["timed_out"] is True
    assert payload["experiments"][0]["returncode"] == 124
    assert "exceeded timeout" in payload["experiments"][0]["issues"][0]


def test_campaign_resume_reuses_complete_artifact(tmp_path: Path, campaign) -> None:
    counter = tmp_path / "counter.txt"
    campaign.configure(
        {
            "defaults": {"COUNTER_FILE": str(counter), "BINARY_SHA256": "f" * 64},
            "experiments": [{"name": "resume", "env": {"LAYERS": "2"}}],
        }
    )
    campaign.run(check=True)
    campaign.run("--resume", check=True)
    payload = campaign.report
    assert counter.read_text(encoding="utf-8") == "1"
    assert payload["experiments"][0]["state"] == "resumed"
    assert payload["experiments"][0]["candidate_passed"] is True


def test_campaign_resume_rejects_changed_effective_environment(tmp_path: Path, campaign) -> None:
    counter = tmp_path / "counter.txt"
    manifest_payload = {
        "defaults": {
            "RESULTS_DIR": str(tmp_path / "results"),
            "TOKENS": "2",
            "COUNTER_FILE": str(counter),
            "BINARY_SHA256": "f" * 64,
            "COMPLEX_STATE_PAIRING": "1",
        },
        "experiments": [{"name": "resume", "env": {"LAYERS": "2"}}],
    }
    campaign.configure(manifest_payload)
    campaign.run(check=True)
    manifest_payload["defaults"]["COMPLEX_STATE_PAIRING"] = "0"
    campaign.configure(manifest_payload)
    campaign.run("--resume", check=True)
    record = campaign.report["experiments"][0]
    assert counter.read_text(encoding="utf-8") == "2"
    assert record["state"] == "executed"
    assert "effective environment changed: COMPLEX_STATE_PAIRING" in record["resume_rejections"]


def test_campaign_resume_hashes_and_rechecks_current_binary(tmp_path: Path, campaign) -> None:
    binary = tmp_path / "stage1_mamba2_decode_fideslib"
    binary.write_bytes(b"first binary")
    counter = tmp_path / "counter.txt"
    campaign.configure(
        {
            "defaults": {"COUNTER_FILE": str(counter), "BINARY_PATH": str(binary)},
            "experiments": [{"name": "resume", "env": {"LAYERS": "2"}}],
        }
    )
    campaign.run(check=True)
    binary.write_bytes(b"second binary")
    campaign.run("--resume", check=True)
    payload = campaign.report
    record = payload["experiments"][0]
    artifact = json.loads(Path(record["artifact_paths"][0]).read_text(encoding="utf-8"))
    assert counter.read_text(encoding="utf-8") == "2"
    assert record["state"] == "executed"
    assert any("BINARY_SHA256" in issue for issue in record["resume_rejections"])
    assert artifact["binary_sha256"] == record["environment"]["BINARY_SHA256"]


@pytest.mark.parametrize(
    ("override", "issue_fragment", "infrastructure_ok"),
    [
        ({}, None, True),
        ({"FAKE_ERROR": "0.2"}, "maximum error", True),
        ({"FAKE_ERROR": "nan"}, "maximum error", False),
        ({"FAKE_DECRYPT_FAILURE": "1"}, "successful decryption", True),
        ({"FAKE_DECRYPT_MALFORMED": "1"}, "successful decryption", True),
        ({"FAKE_TOKEN_MISMATCH": "1"}, "token IDs do not match", True),
        ({"FAKE_INTERMEDIATE_DECRYPT": "1"}, "zero intermediate decrypts", True),
        ({"FAKE_SYNC_PROFILE": "bootstrap-lifetime"}, "sync profile", False),
    ],
)
def test_promoted_campaign_enforces_acceptance_contract(
    override: dict[str, str], issue_fragment: str | None, infrastructure_ok: bool, campaign
) -> None:
    campaign.configure(
        {
            "acceptance": {
                "layers": 24,
                "tokens": 2,
                "max_abs_error_lte": 0.05,
                "all_tokens_decrypt": True,
                "autoregressive_tokens_match": True,
                "zero_intermediate_decrypts": True,
                "required_sync_profile": "full",
            },
            "defaults": {"LAYERS": "24", "FIDESLIB_SYNC_PROFILE": "full", **override},
            "experiments": [{"name": "promoted"}],
        }
    )
    completed = campaign.run(check=False)
    payload = campaign.report
    passed = issue_fragment is None
    assert completed.returncode == (0 if passed else 1)
    assert payload["passed"] is passed
    assert payload["infrastructure_ok"] is infrastructure_ok
    assert payload["promotion_passed"] is passed
    assert payload["acceptance"]["evaluated"] is True
    if passed:
        assert payload["acceptance"]["issues"] == []
    else:
        assert any(issue_fragment in issue for issue in payload["acceptance"]["issues"])


@pytest.mark.parametrize(
    ("section", "key", "stale_value", "issue_fragment"),
    [
        (None, "repo_commit", "stale-commit", "repo_commit mismatch"),
        (None, "binary_sha256", "e" * 64, "binary_sha256 mismatch"),
        (None, "input_payload_sha256", "e" * 64, "input payload hash mismatch"),
        (None, "version", "0.0.0-stale", "version mismatch"),
        ("parameters", "tokens", 99, "token count mismatch"),
        ("parameters", "fideslib_sync_profile", "none", "sync profile mismatch"),
    ],
)
def test_campaign_resume_rejects_stale_artifact_identity(
    tmp_path: Path,
    section: str | None,
    key: str,
    stale_value: object,
    issue_fragment: str,
    campaign,
) -> None:
    counter = tmp_path / "counter.txt"
    campaign.configure(
        {
            "defaults": {
                "COUNTER_FILE": str(counter),
                "BINARY_SHA256": "f" * 64,
                "FIDESLIB_SYNC_PROFILE": "full",
                "INPUT_CHAIN_SHA256": "a" * 64,
            },
            "experiments": [{"name": "resume", "env": {"LAYERS": "2"}}],
        }
    )
    campaign.run(check=True)
    artifact_path = campaign.results / "m2_chain_resume_l2_t2.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    target = artifact if section is None else artifact[section]
    target[key] = stale_value
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    campaign.run("--resume", check=True)
    payload = campaign.report
    record = payload["experiments"][0]
    assert counter.read_text(encoding="utf-8") == "2"
    assert record["state"] == "executed"
    assert any(issue_fragment in issue for issue in record["resume_rejections"])
    assert record["issues"] == []


@pytest.mark.parametrize(
    ("processes", "utilization", "gpu_index", "issue"),
    [
        pytest.param("123, 4096", 99, None, "GPU remained occupied", id="occupied"),
        pytest.param("", 94, None, "utilization=94.0%", id="unreported-utilization"),
        pytest.param("", 0, 3, None, id="selected-gpu"),
    ],
)
def test_campaign_gpu_preflight(tmp_path, campaign, processes, utilization, gpu_index, issue):
    calls = tmp_path / "nvidia-smi-calls.txt"
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n"
        'case "$*" in\n'
        f"  *--query-compute-apps=*) printf '%s\\n' {shlex.quote(processes)} ;;\n"
        f"  *--query-gpu=*) printf '{utilization}\\n' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    counter = tmp_path / "counter.txt"
    campaign.configure(
        {
            "gpu_preflight": {
                "required": True,
                "nvidia_smi": str(nvidia_smi),
                "poll_seconds": 0.01,
                "timeout_seconds": 0,
                "max_utilization_percent": 5,
                **({"gpu_index": gpu_index} if gpu_index is not None else {}),
            },
            "defaults": {"COUNTER_FILE": str(counter)},
            "experiments": [{"name": "preflight", "env": {"LAYERS": "2"}}],
        }
    )
    completed = campaign.run(check=False)
    record = campaign.report["experiments"][0]
    if issue is None:
        assert completed.returncode == 0
        assert record["candidate_passed"] is True
        assert counter.read_text() == "1"
    else:
        assert completed.returncode == 1
        assert not counter.exists()
        assert record["state"] == "preflight-failed"
        assert issue in record["issues"][0]
    recorded_calls = calls.read_text(encoding="utf-8").splitlines()
    assert len(recorded_calls) == 2
    prefix = "--query-" if gpu_index is None else f"-i {gpu_index} --query-"
    assert all(call.startswith(prefix) for call in recorded_calls)


def test_campaign_sighup_terminates_active_runner_group(tmp_path: Path, campaign) -> None:
    campaign.runner.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        "Path(os.environ['PID_FILE']).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    campaign.runner.chmod(0o755)
    pid_file = tmp_path / "runner.pid"
    campaign.configure(
        {
            "defaults": {"PID_FILE": str(pid_file)},
            "experiments": [{"name": "hangup", "env": {"LAYERS": "2"}}],
        }
    )
    process = subprocess.Popen(campaign.command, cwd=ROOT)
    try:
        # Wait for readiness: parallel native builds can delay startup.
        # This test checks signal handling, not startup latency.
        deadline = time.monotonic() + 30
        while not pid_file.exists() and process.poll() is None and (time.monotonic() < deadline):
            time.sleep(0.02)
        assert pid_file.exists()
        runner_pid = int(pid_file.read_text(encoding="utf-8"))
        os.kill(process.pid, signal.SIGHUP)
        assert process.wait(timeout=7) == 128 + signal.SIGHUP
        with pytest.raises(ProcessLookupError):
            os.kill(runner_pid, 0)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=7)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if pid_file.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text(encoding="utf-8")), signal.SIGKILL)
