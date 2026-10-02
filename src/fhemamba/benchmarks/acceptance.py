"""Shared acceptance gates for Mamba-2 campaigns and local generation."""

from __future__ import annotations

import math
from typing import Any


def artifact_int(payload: dict[str, Any], key: str) -> int | None:
    parameters = payload.get("parameters", {})
    scope = payload.get("measurement_scope", {})
    for source in (parameters, scope):
        value = source.get(key) if isinstance(source, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def max_error(artifacts: list[dict[str, Any]]) -> float:
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


def all_decrypt(artifacts: list[dict[str, Any]]) -> bool:
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


def artifact_layers(artifact: dict[str, Any]) -> int | None:
    parameters = artifact.get("parameters", {})
    scope = artifact.get("measurement_scope", {})
    for source, key in ((parameters, "n_layers_loaded"), (scope, "layers_loaded")):
        value = source.get(key) if isinstance(source, dict) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def evaluate_acceptance(
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

    expected_tokens = acceptance.get("tokens")
    for key, read in (
        ("layers", artifact_layers),
        ("tokens", lambda row: artifact_int(row, "tokens")),
    ):
        expected = acceptance.get(key)
        if expected is not None:
            mismatches = [
                artifact.get("path", "<unknown>")
                for artifact in artifacts
                if read(artifact) != expected
            ]
            if mismatches:
                issues.append(
                    f"artifacts do not report the required {expected} {key}: "
                    + ", ".join(mismatches)
                )

    threshold = acceptance.get("max_abs_error_lte")
    maximum_error = max_error(artifacts)
    if threshold is not None and maximum_error > float(threshold):
        rendered_error = maximum_error if math.isfinite(maximum_error) else "missing"
        issues.append(f"maximum error {rendered_error} exceeds {threshold}")

    if acceptance.get("all_tokens_decrypt"):
        if not all_decrypt(artifacts):
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
