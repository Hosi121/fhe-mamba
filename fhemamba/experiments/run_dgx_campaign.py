#!/usr/bin/env python3
"""Run resumable DGX benchmark campaigns without treating tolerance misses as infra failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba.artifacts import validate_benchmark_artifact

from fhemamba import __version__


class CampaignInterruptedError(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(f"campaign interrupted by signal {signum}")
        self.signum = signum


def _raise_campaign_signal(signum: int, _frame: Any) -> None:
    raise CampaignInterruptedError(signum)


def _read_object(path: Path) -> dict[str, Any]:
    def reject_non_finite(value: str) -> Any:
        raise ValueError(f"non-finite JSON number {value!r} in {path}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_non_finite,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _timeout_seconds(value: Any) -> float | None:
    if value is None:
        return None
    timeout = float(value)
    if timeout < 0:
        raise ValueError("timeout_seconds must be non-negative")
    return timeout or None


def _artifact_paths(env: dict[str, str]) -> list[Path]:
    results_dir = Path(env.get("RESULTS_DIR", str(Path.home() / "fhemamba" / "results")))
    layers = env.get("LAYERS", "5 8 12 24").split()
    tokens = env.get("TOKENS", "1")
    run_tag = env["RUN_TAG"]
    return [results_dir / f"m2_chain_{run_tag}_l{layer}_t{tokens}.json" for layer in layers]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _artifact_expectation(
    env: dict[str, str],
    *,
    version: str,
    repo_commit: str,
) -> dict[str, Any]:
    # Campaign identity is derived from the running checkout, not from
    # user-editable manifest defaults.
    env["ARTIFACT_VERSION"] = version
    env["REPO_COMMIT"] = repo_commit
    binary_sha256 = env.get("BINARY_SHA256")
    binary_path_value = env.get("BINARY_PATH") or env.get("BINARY")
    if binary_path_value:
        binary_path = Path(binary_path_value)
        if binary_path.is_file():
            computed_sha256 = _file_sha256(binary_path)
            if binary_sha256 is not None and binary_sha256.lower() != computed_sha256:
                raise ValueError(f"BINARY_SHA256 does not match current binary {binary_path}")
            binary_sha256 = computed_sha256
            env["BINARY_SHA256"] = binary_sha256
    if binary_sha256 is not None and not _is_sha256(binary_sha256):
        raise ValueError("BINARY_SHA256 must be a 64-digit hexadecimal SHA-256")
    try:
        layers = [int(value) for value in env.get("LAYERS", "5 8 12 24").split()]
        tokens = int(env.get("TOKENS", "1"))
    except ValueError as exc:
        raise ValueError("campaign LAYERS and TOKENS must be integers") from exc
    return {
        "version": env["ARTIFACT_VERSION"],
        "repo_commit": env["REPO_COMMIT"],
        "binary_sha256": binary_sha256,
        "layers": layers,
        "tokens": tokens,
        "sync_profile": env.get("FIDESLIB_SYNC_PROFILE"),
    }


def _artifact_int(payload: dict[str, Any], key: str) -> int | None:
    parameters = payload.get("parameters", {})
    scope = payload.get("measurement_scope", {})
    for source in (parameters, scope):
        value = source.get(key) if isinstance(source, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _validate_artifact_identity(
    payload: dict[str, Any],
    path: Path,
    *,
    expected: dict[str, Any],
    expected_layer: int,
    require_binary_match: bool,
) -> list[str]:
    issues: list[str] = []
    expected_versions = {expected["version"], f"{expected['version']}+{path.stem}"}
    if payload.get("version") not in expected_versions:
        issues.append(
            f"artifact version mismatch for {path}: expected one of "
            f"{sorted(expected_versions)!r}, got {payload.get('version')!r}"
        )
    if payload.get("repo_commit") != expected["repo_commit"]:
        issues.append(
            f"artifact repo_commit mismatch for {path}: "
            f"expected {expected['repo_commit']!r}, got {payload.get('repo_commit')!r}"
        )
    binary_sha256 = payload.get("binary_sha256")
    if not _is_sha256(binary_sha256):
        issues.append(f"artifact binary_sha256 is not a valid SHA-256: {path}")
    expected_binary = expected.get("binary_sha256")
    if require_binary_match and not expected_binary:
        issues.append(
            f"cannot safely resume {path}: set BINARY_PATH or BINARY_SHA256 "
            "to identify the current binary"
        )
    elif expected_binary and binary_sha256 != expected_binary:
        issues.append(
            f"artifact binary_sha256 mismatch for {path}: "
            f"expected {expected_binary!r}, got {binary_sha256!r}"
        )
    layers = _artifact_int(payload, "n_layers_loaded")
    if layers is None:
        layers = _artifact_int(payload, "layers_loaded")
    if layers != expected_layer:
        issues.append(
            f"artifact layer count mismatch for {path}: expected {expected_layer}, got {layers!r}"
        )
    tokens = _artifact_int(payload, "tokens")
    if tokens != expected["tokens"]:
        issues.append(
            f"artifact token count mismatch for {path}: "
            f"expected {expected['tokens']}, got {tokens!r}"
        )
    expected_sync = expected.get("sync_profile")
    parameters = payload.get("parameters", {})
    actual_sync = parameters.get("fideslib_sync_profile") if isinstance(parameters, dict) else None
    if expected_sync is not None and actual_sync != expected_sync:
        issues.append(
            f"artifact sync profile mismatch for {path}: "
            f"expected {expected_sync!r}, got {actual_sync!r}"
        )
    return issues


def _load_artifacts(
    paths: list[Path],
    *,
    expected: dict[str, Any],
    require_binary_match: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    artifacts: list[dict[str, Any]] = []
    issues: list[str] = []
    expected_layers = expected["layers"]
    if len(expected_layers) != len(paths):
        raise ValueError("artifact paths and expected layers must have the same length")
    for path, expected_layer in zip(paths, expected_layers, strict=True):
        if not path.is_file():
            issues.append(f"missing artifact: {path}")
            continue
        try:
            payload = _read_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"invalid artifact {path}: {exc}")
            continue
        if "status" not in payload or "passed" not in payload:
            issues.append(f"artifact lacks status/passed: {path}")
            continue
        validation = validate_benchmark_artifact(payload, require_commit=True)
        issues.extend(
            f"invalid artifact {path} at {issue.path}: {issue.message}"
            for issue in validation.errors
        )
        issues.extend(
            _validate_artifact_identity(
                payload,
                path,
                expected=expected,
                expected_layer=expected_layer,
                require_binary_match=require_binary_match,
            )
        )
        measurements = payload.get("measurements", {})
        timing = payload.get("timing", {})
        scope = payload.get("measurement_scope", {})
        parameters = payload.get("parameters", {})
        artifacts.append(
            {
                "path": str(path),
                "version": payload.get("version"),
                "repo_commit": payload.get("repo_commit"),
                "binary_sha256": payload.get("binary_sha256"),
                "stage": payload.get("stage"),
                "status": payload["status"],
                "passed": payload["passed"],
                "parameters": {
                    key: parameters.get(key)
                    for key in ("n_layers_loaded", "tokens", "fideslib_sync_profile")
                    if key in parameters
                },
                "measurements": {
                    key: measurements.get(key)
                    for key in (
                        "max_abs_error",
                        "per_token_max_abs_error",
                        "per_token_decrypt_ok",
                        "autoregressive_selected_ids",
                        "autoregressive_expected_ids",
                        "autoregressive_tokens_match",
                        "executed_bootstrap_count",
                        "peak_rss_gib",
                    )
                    if key in measurements
                },
                "timing": {
                    key: timing.get(key)
                    for key in ("eval_seconds", "total_seconds")
                    if key in timing
                },
                "measurement_scope": {
                    key: scope.get(key)
                    for key in (
                        "zero_intermediate_decrypts",
                        "full_layer_chain",
                        "layers_loaded",
                        "tokens",
                    )
                    if key in scope
                },
                "validation": validation.to_json_dict(),
            }
        )
    return artifacts, issues


def _max_error(artifacts: list[dict[str, Any]]) -> float:
    errors: list[float] = []
    for artifact in artifacts:
        value = artifact.get("measurements", {}).get("max_abs_error")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return float("inf")
        error = float(value)
        if not math.isfinite(error) or error < 0:
            return float("inf")
        errors.append(error)
    return max(errors, default=float("inf"))


def _all_decrypt(artifacts: list[dict[str, Any]]) -> bool:
    if not artifacts:
        return False
    for artifact in artifacts:
        values = artifact.get("measurements", {}).get("per_token_decrypt_ok")
        if (
            not isinstance(values, list)
            or not values
            or not all(value is True or (type(value) is int and value == 1) for value in values)
        ):
            return False
    return True


def _artifact_layers(artifact: dict[str, Any]) -> int | None:
    parameters = artifact.get("parameters", {})
    scope = artifact.get("measurement_scope", {})
    for source, key in ((parameters, "n_layers_loaded"), (scope, "layers_loaded")):
        value = source.get(key) if isinstance(source, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _artifact_tokens(artifact: dict[str, Any]) -> int | None:
    parameters = artifact.get("parameters", {})
    scope = artifact.get("measurement_scope", {})
    for source in (parameters, scope):
        value = source.get("tokens") if isinstance(source, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _validate_acceptance_config(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("manifest acceptance must be an object")
    allowed = {
        "layers",
        "tokens",
        "max_abs_error_lte",
        "all_tokens_decrypt",
        "autoregressive_tokens_match",
        "zero_intermediate_decrypts",
        "required_sync_profile",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"manifest acceptance has unknown fields: {', '.join(unknown)}")
    for key in ("layers", "tokens"):
        if key in value and (
            not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] <= 0
        ):
            raise ValueError(f"manifest acceptance {key} must be a positive integer")
    if "max_abs_error_lte" in value:
        threshold = value["max_abs_error_lte"]
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or float(threshold) < 0
        ):
            raise ValueError("manifest acceptance max_abs_error_lte must be non-negative")
    for key in (
        "all_tokens_decrypt",
        "autoregressive_tokens_match",
        "zero_intermediate_decrypts",
    ):
        if key in value and not isinstance(value[key], bool):
            raise ValueError(f"manifest acceptance {key} must be boolean")
    if "required_sync_profile" in value and (
        not isinstance(value["required_sync_profile"], str) or not value["required_sync_profile"]
    ):
        raise ValueError("manifest acceptance required_sync_profile must be a non-empty string")
    return value


def _evaluate_acceptance(
    acceptance: dict[str, Any] | None,
    experiments: list[dict[str, Any]],
    *,
    complete: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if acceptance is None:
        return {"required": False, "evaluated": False, "passed": None, "issues": []}
    if not complete or dry_run:
        return {"required": True, "evaluated": False, "passed": None, "issues": []}

    issues: list[str] = []
    if any(item.get("infrastructure_ok") is not True for item in experiments):
        issues.append("one or more experiments have an infrastructure failure")
    artifacts = [
        artifact
        for experiment in experiments
        for artifact in experiment.get("artifacts", [])
        if isinstance(artifact, dict)
    ]
    if not artifacts:
        issues.append("campaign produced no artifacts")
    elif not all(
        artifact.get("status") == "passed" and artifact.get("passed") is True
        for artifact in artifacts
    ):
        issues.append("one or more candidate artifacts did not pass")

    expected_layers = acceptance.get("layers")
    if expected_layers is not None:
        mismatches = [
            artifact.get("path", "<unknown>")
            for artifact in artifacts
            if _artifact_layers(artifact) != expected_layers
        ]
        if mismatches:
            issues.append(
                f"artifacts do not report the required {expected_layers} layers: "
                + ", ".join(mismatches)
            )

    expected_tokens = acceptance.get("tokens")
    if expected_tokens is not None:
        mismatches = [
            artifact.get("path", "<unknown>")
            for artifact in artifacts
            if _artifact_tokens(artifact) != expected_tokens
        ]
        if mismatches:
            issues.append(
                f"artifacts do not report the required {expected_tokens} tokens: "
                + ", ".join(mismatches)
            )

    threshold = acceptance.get("max_abs_error_lte")
    maximum_error = _max_error(artifacts)
    if threshold is not None and maximum_error > float(threshold):
        rendered_error = maximum_error if math.isfinite(maximum_error) else "missing"
        issues.append(f"maximum error {rendered_error} exceeds {threshold}")

    if acceptance.get("all_tokens_decrypt"):
        if not _all_decrypt(artifacts):
            issues.append("not every artifact reports successful decryption for every token")
        elif expected_tokens is not None and any(
            len(artifact.get("measurements", {}).get("per_token_decrypt_ok", [])) != expected_tokens
            for artifact in artifacts
        ):
            issues.append("per-token decrypt telemetry length does not match required tokens")

    if acceptance.get("autoregressive_tokens_match"):
        for artifact in artifacts:
            measurements = artifact.get("measurements", {})
            selected = measurements.get("autoregressive_selected_ids")
            expected_ids = measurements.get("autoregressive_expected_ids")
            if (
                measurements.get("autoregressive_tokens_match") is not True
                or not isinstance(selected, list)
                or not selected
                or selected != expected_ids
            ):
                issues.append(
                    "autoregressive token IDs do not match for "
                    + str(artifact.get("path", "<unknown>"))
                )

    if acceptance.get("zero_intermediate_decrypts") and any(
        artifact.get("measurement_scope", {}).get("zero_intermediate_decrypts") is not True
        for artifact in artifacts
    ):
        issues.append("one or more artifacts do not prove zero intermediate decrypts")

    expected_sync = acceptance.get("required_sync_profile")
    if expected_sync is not None:
        mismatches = [
            artifact.get("path", "<unknown>")
            for artifact in artifacts
            if artifact.get("parameters", {}).get("fideslib_sync_profile") != expected_sync
        ]
        if mismatches:
            issues.append(
                f"artifacts do not use required sync profile {expected_sync!r}: "
                + ", ".join(mismatches)
            )

    return {
        "required": True,
        "evaluated": True,
        "passed": not issues,
        "criteria": acceptance,
        "issues": issues,
    }


def _promotion_satisfied(
    condition: dict[str, Any] | None,
    completed: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    if condition is None:
        return True, "unconditional"
    source_name = condition.get("experiment")
    if not isinstance(source_name, str) or source_name not in completed:
        return False, f"promotion source is unavailable: {source_name!r}"
    source = completed[source_name]
    artifacts = source.get("artifacts", [])
    if source.get("infrastructure_ok") is not True:
        return False, f"promotion source has an infrastructure failure: {source_name}"
    threshold = condition.get("max_abs_error_lte")
    if threshold is not None and _max_error(artifacts) > float(threshold):
        return False, f"{source_name} max error exceeds {threshold}"
    if condition.get("all_tokens_decrypt") and not _all_decrypt(artifacts):
        return False, f"{source_name} does not decrypt every token"
    if condition.get("passed") and not all(
        artifact.get("passed") is True for artifact in artifacts
    ):
        return False, f"{source_name} did not pass"
    return True, "promotion gate passed"


def _repo_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip() or "working-tree"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty.stdout.strip():
        digest = hashlib.sha256()
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--no-ext-diff"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        digest.update(tracked_diff.stdout)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        for encoded_path in untracked.stdout.split(b"\0"):
            if not encoded_path:
                continue
            digest.update(b"\0untracked\0")
            digest.update(encoded_path)
            path = root / os.fsdecode(encoded_path)
            try:
                digest.update(path.read_bytes())
            except OSError as exc:
                digest.update(f"<unreadable:{exc}>".encode())
        commit += f"-dirty.{digest.hexdigest()[:16]}"
    return commit


def _resume_context_issues(
    previous: dict[str, Any] | None,
    *,
    campaign_name: str,
    experiment_name: str,
    repo_commit: str,
    environment: dict[str, str],
    artifact_paths: list[Path],
) -> list[str]:
    if previous is None:
        return ["no prior campaign report is available to verify the effective environment"]
    issues: list[str] = []
    if previous.get("stage") != "fhemamba-dgx-campaign-report":
        issues.append("prior output is not a DGX campaign report")
    if previous.get("campaign") != campaign_name:
        issues.append(
            f"prior campaign name mismatch: expected {campaign_name!r}, "
            f"got {previous.get('campaign')!r}"
        )
    if previous.get("repo_commit") != repo_commit:
        issues.append(
            f"prior campaign repo_commit mismatch: expected {repo_commit!r}, "
            f"got {previous.get('repo_commit')!r}"
        )
    prior_records = previous.get("experiments")
    if not isinstance(prior_records, list):
        issues.append("prior campaign report has no experiment records")
        return issues
    matches = [
        item
        for item in prior_records
        if isinstance(item, dict) and item.get("name") == experiment_name
    ]
    if len(matches) != 1:
        issues.append(f"prior campaign report has {len(matches)} records for {experiment_name!r}")
        return issues
    prior_record = matches[0]
    prior_environment = prior_record.get("environment")
    if not isinstance(prior_environment, dict):
        issues.append(f"prior experiment {experiment_name!r} has no effective environment")
    elif prior_environment != environment:
        changed_keys = sorted(set(prior_environment) | set(environment))
        changed_keys = [
            key for key in changed_keys if prior_environment.get(key) != environment.get(key)
        ]
        issues.append("effective environment changed: " + ", ".join(changed_keys))
    expected_paths = [str(path) for path in artifact_paths]
    if prior_record.get("artifact_paths") != expected_paths:
        issues.append(f"artifact paths changed for experiment {experiment_name!r}")
    return issues


def _run_runner(
    runner: Path,
    *,
    root: Path,
    env: dict[str, str],
    timeout_seconds: float | None,
) -> tuple[int, bool]:
    process = subprocess.Popen(
        [str(runner)],
        cwd=root,
        env=env,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds), False
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return 124, True
    except BaseException:
        _terminate_process_group(process)
        raise


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    except ProcessLookupError:
        process.wait()


def _nvidia_smi_command(nvidia_smi: str, gpu_index: int | None, *args: str) -> list[str]:
    command = [nvidia_smi]
    if gpu_index is not None:
        command.extend(["-i", str(gpu_index)])
    command.extend(args)
    return command


def _gpu_processes(
    nvidia_smi: str,
    gpu_index: int | None,
) -> tuple[list[dict[str, int]], str | None]:
    completed = subprocess.run(
        _nvidia_smi_command(
            nvidia_smi,
            gpu_index,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"{nvidia_smi} exited {completed.returncode}"
        return [], f"GPU preflight failed: {message}"
    processes = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0]:
            continue
        try:
            processes.append({"pid": int(fields[0]), "used_memory_mib": int(fields[1])})
        except ValueError:
            return [], f"GPU preflight returned an invalid row: {line!r}"
    return processes, None


def _gpu_utilization(nvidia_smi: str, gpu_index: int | None) -> tuple[float, str | None]:
    completed = subprocess.run(
        _nvidia_smi_command(
            nvidia_smi,
            gpu_index,
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"{nvidia_smi} exited {completed.returncode}"
        return 0.0, f"GPU utilization preflight failed: {message}"
    values = []
    for line in completed.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            return 0.0, f"GPU utilization preflight returned an invalid row: {line!r}"
    if not values:
        return 0.0, "GPU utilization preflight returned no GPUs"
    return max(values), None


def _mem_available_gib(meminfo_path: str) -> tuple[float, str | None]:
    try:
        fields = Path(meminfo_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return 0.0, f"memory preflight failed: {exc}"
    for line in fields:
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024**2, None
                except ValueError:
                    break
    return 0.0, f"memory preflight found no valid MemAvailable in {meminfo_path}"


def _wait_for_idle_gpu(config: dict[str, Any] | None) -> tuple[float, str | None]:
    if not config or not bool(config.get("required")):
        return 0.0, None
    nvidia_smi = str(config.get("nvidia_smi", "nvidia-smi"))
    gpu_index_value = config.get("gpu_index")
    gpu_index = None if gpu_index_value is None else int(gpu_index_value)
    poll_seconds = float(config.get("poll_seconds", 30))
    timeout_seconds = float(config.get("timeout_seconds", 7200))
    max_utilization = float(config.get("max_utilization_percent", 5))
    min_mem_available_gib = float(config.get("min_mem_available_gib", 0))
    stable_polls = int(config.get("stable_polls", 1))
    meminfo_path = str(config.get("meminfo_path", "/proc/meminfo"))
    if (
        poll_seconds <= 0
        or timeout_seconds < 0
        or max_utilization < 0
        or min_mem_available_gib < 0
        or stable_polls <= 0
        or (gpu_index is not None and gpu_index < 0)
    ):
        raise ValueError("GPU preflight thresholds and polling values are invalid")
    started_at = time.monotonic()
    idle_polls = 0
    while True:
        processes, issue = _gpu_processes(nvidia_smi, gpu_index)
        if issue:
            return time.monotonic() - started_at, issue
        utilization, issue = _gpu_utilization(nvidia_smi, gpu_index)
        if issue:
            return time.monotonic() - started_at, issue
        mem_available_gib, issue = _mem_available_gib(meminfo_path)
        if issue:
            return time.monotonic() - started_at, issue
        idle = (
            not processes
            and utilization <= max_utilization
            and mem_available_gib >= min_mem_available_gib
        )
        idle_polls = idle_polls + 1 if idle else 0
        if idle_polls >= stable_polls:
            return time.monotonic() - started_at, None
        elapsed = time.monotonic() - started_at
        if elapsed >= timeout_seconds:
            occupied = ", ".join(
                f"pid={item['pid']} memory={item['used_memory_mib']}MiB" for item in processes
            )
            detail = occupied or "no process rows"
            return elapsed, (
                f"GPU remained occupied after {timeout_seconds}s: {detail}; "
                f"utilization={utilization}% mem_available={mem_available_gib:.1f}GiB"
            )
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds - elapsed)))


def _campaign_payload(
    *,
    name: str,
    version: str,
    repo_commit: str,
    started_at: float,
    experiments: list[dict[str, Any]],
    acceptance: dict[str, Any] | None,
    complete: bool,
    dry_run: bool,
) -> dict[str, Any]:
    infra_failures = sum(item.get("infrastructure_ok") is False for item in experiments)
    executed = sum(item.get("state") in {"executed", "resumed"} for item in experiments)
    skipped = sum(item.get("state") == "skipped" for item in experiments)
    candidate_passes = sum(item.get("candidate_passed") is True for item in experiments)
    candidate_failures = sum(item.get("candidate_passed") is False for item in experiments)
    acceptance_result = _evaluate_acceptance(
        acceptance,
        experiments,
        complete=complete,
        dry_run=dry_run,
    )
    infrastructure_ok = infra_failures == 0
    promotion_ok = acceptance_result["passed"] if acceptance_result["required"] else None
    completed_ok = infrastructure_ok and (promotion_ok is not False)
    if not complete:
        status = "running"
    elif dry_run:
        status = "dry-run"
    else:
        status = "passed" if completed_ok else "failed"
    return {
        "version": version,
        "stage": "fhemamba-dgx-campaign-report",
        "repo_commit": repo_commit,
        "backend": "orchestration",
        "encrypted": False,
        "config": {"input_mode": "campaign-orchestration"},
        "status": status,
        "passed": completed_ok if complete else False,
        "campaign": name,
        "complete": complete,
        "infrastructure_ok": infrastructure_ok,
        "promotion_passed": promotion_ok,
        "acceptance": acceptance_result,
        "experiments": experiments,
        "measurements": {
            "experiments_total": len(experiments),
            "experiments_executed_or_resumed": executed,
            "experiments_skipped": skipped,
            "infrastructure_failures": infra_failures,
            "candidate_passes": candidate_passes,
            "candidate_failures": candidate_failures,
        },
        "timing": {"campaign_seconds": time.monotonic() - started_at},
        "measurement_scope": {
            "campaign_orchestration_only": True,
            "full_model_correctness_claimed": False,
            "candidate_failures_are_not_infrastructure_failures": True,
            "claim": (
                "Campaign execution and artifact collection only; candidate artifacts carry "
                "their own encrypted-correctness claims."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("fhemamba/experiments/run_dgx_layer_ladder.sh"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGHUP, _raise_campaign_signal)
    signal.signal(signal.SIGTERM, _raise_campaign_signal)
    root = Path(__file__).resolve().parents[2]
    manifest = _read_object(args.manifest)
    name = str(manifest.get("name", args.manifest.stem))
    version = str(manifest.get("version", __version__))
    repo_commit = _repo_commit(root)
    acceptance = _validate_acceptance_config(manifest.get("acceptance"))
    default_timeout = _timeout_seconds(manifest.get("timeout_seconds", 0))
    gpu_preflight = manifest.get("gpu_preflight")
    if gpu_preflight is not None and not isinstance(gpu_preflight, dict):
        raise ValueError("manifest gpu_preflight must be an object")
    defaults = manifest.get("defaults", {})
    experiments_spec = manifest.get("experiments", [])
    if not isinstance(defaults, dict) or not isinstance(experiments_spec, list):
        raise ValueError("manifest defaults must be an object and experiments must be a list")

    started_at = time.monotonic()
    records: list[dict[str, Any]] = []
    completed_by_name: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    previous_campaign: dict[str, Any] | None = None
    previous_campaign_issue: str | None = None
    if args.resume:
        if args.output_json.is_file():
            try:
                previous_campaign = _read_object(args.output_json)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                previous_campaign_issue = f"cannot read prior campaign report: {exc}"
        else:
            previous_campaign_issue = "prior campaign report does not exist"

    def write_campaign(*, complete: bool) -> dict[str, Any]:
        payload = _campaign_payload(
            name=name,
            version=version,
            repo_commit=repo_commit,
            started_at=started_at,
            experiments=records,
            acceptance=acceptance,
            complete=complete,
            dry_run=args.dry_run,
        )
        _write_json(args.output_json, payload)
        return payload

    for spec in experiments_spec:
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise ValueError("each experiment must be an object with a string name")
        experiment_name = spec["name"]
        if experiment_name in seen:
            raise ValueError(f"duplicate experiment name: {experiment_name}")
        seen.add(experiment_name)

        condition = spec.get("when")
        if condition is not None and not isinstance(condition, dict):
            raise ValueError(f"experiment {experiment_name} when must be an object")
        if args.dry_run:
            promoted, promotion_reason = True, "dry-run does not evaluate promotion gates"
        else:
            promoted, promotion_reason = _promotion_satisfied(condition, completed_by_name)
        if not promoted:
            record = {
                "name": experiment_name,
                "state": "skipped",
                "reason": promotion_reason,
                "infrastructure_ok": True,
                "candidate_passed": False,
                "artifacts": [],
            }
            records.append(record)
            completed_by_name[experiment_name] = record
            write_campaign(complete=len(records) == len(experiments_spec))
            continue

        overrides = spec.get("env", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"experiment {experiment_name} env must be an object")
        campaign_env = {str(key): str(value) for key, value in defaults.items()}
        campaign_env.update({str(key): str(value) for key, value in overrides.items()})
        campaign_env.setdefault("RUN_TAG", experiment_name)
        artifact_paths = _artifact_paths(campaign_env)
        expected_artifact = _artifact_expectation(
            campaign_env,
            version=version,
            repo_commit=repo_commit,
        )

        artifacts: list[dict[str, Any]] = []
        issues: list[str] = []
        resume_rejections: list[str] = []
        state = "dry-run" if args.dry_run else "executed"
        returncode: int | None = None
        timed_out = False
        duration = 0.0
        gpu_wait_seconds = 0.0
        if args.resume and not args.dry_run:
            if previous_campaign_issue is not None:
                resume_rejections.append(previous_campaign_issue)
            else:
                resume_rejections.extend(
                    _resume_context_issues(
                        previous_campaign,
                        campaign_name=name,
                        experiment_name=experiment_name,
                        repo_commit=repo_commit,
                        environment=campaign_env,
                        artifact_paths=artifact_paths,
                    )
                )
            resume_artifacts, artifact_rejections = _load_artifacts(
                artifact_paths,
                expected=expected_artifact,
                require_binary_match=True,
            )
            resume_rejections.extend(artifact_rejections)
            if not resume_rejections and len(resume_artifacts) == len(artifact_paths):
                artifacts = resume_artifacts
                state = "resumed"
        if state not in {"resumed", "dry-run"}:
            gpu_wait_seconds, preflight_issue = _wait_for_idle_gpu(gpu_preflight)
            if preflight_issue:
                state = "preflight-failed"
                issues.append(preflight_issue)
            else:
                env = os.environ.copy()
                env.update(campaign_env)
                experiment_start = time.monotonic()
                timeout_value = spec.get("timeout_seconds", default_timeout)
                timeout_seconds = _timeout_seconds(timeout_value)
                returncode, timed_out = _run_runner(
                    args.runner,
                    root=root,
                    env=env,
                    timeout_seconds=timeout_seconds,
                )
                duration = time.monotonic() - experiment_start
                artifacts, issues = _load_artifacts(
                    artifact_paths,
                    expected=expected_artifact,
                )
                if timed_out:
                    issues.insert(0, f"runner exceeded timeout of {timeout_seconds} seconds")

        infrastructure_ok = not issues
        candidate_passed = bool(artifacts) and all(
            artifact.get("status") == "passed" and artifact.get("passed") is True
            for artifact in artifacts
        )
        max_error = _max_error(artifacts)
        record = {
            "name": experiment_name,
            "state": state,
            "reason": promotion_reason,
            "returncode": returncode,
            "timed_out": timed_out,
            "seconds": duration,
            "gpu_wait_seconds": gpu_wait_seconds,
            "environment": campaign_env,
            "artifact_paths": [str(path) for path in artifact_paths],
            "artifacts": artifacts,
            "issues": issues,
            "resume_rejections": resume_rejections,
            "infrastructure_ok": infrastructure_ok,
            "candidate_passed": candidate_passed,
            "max_abs_error": max_error if math.isfinite(max_error) else None,
            "all_tokens_decrypt": _all_decrypt(artifacts),
        }
        records.append(record)
        completed_by_name[experiment_name] = record
        complete = len(records) == len(experiments_spec)
        payload = write_campaign(complete=complete)
        if not infrastructure_ok and not bool(spec.get("continue_on_infrastructure_failure")):
            if not complete:
                payload = write_campaign(complete=True)
            return 1

    payload = write_campaign(complete=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignInterruptedError as exc:
        print(f"run_dgx_campaign.py: {exc}", file=sys.stderr)
        raise SystemExit(128 + exc.signum) from exc
    except KeyboardInterrupt:
        print("run_dgx_campaign.py: interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"run_dgx_campaign.py: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
