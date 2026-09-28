"""Compare repeated measurements under an explicit, reusable contract."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from .io import field as _field
from .io import file_identity, read_object


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _same(actual: Any, expected: Any) -> bool:
    """JSON equality that also distinguishes booleans from numeric values."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _same(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_samples(paths: list[Path], contract: dict[str, Any]) -> list[float]:
    """Validate even an unpaired successful run, without claiming a comparison."""
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported comparison schema_version")
    allowed = {"schema_version", "metric", "equal", "require", "maximum", "minimum_reduction"}
    if set(contract) - allowed:
        raise ValueError("unknown comparison contract field")
    if not paths:
        raise ValueError("at least one sample is required")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("a sample cannot occur twice or in both arms")
    metric = contract.get("metric", "eval_seconds")
    equal = contract.get("equal", [])
    required = contract.get("require", {})
    maximum = contract.get("maximum", {})
    if (
        not isinstance(metric, str)
        or not isinstance(equal, list)
        or any(not isinstance(name, str) for name in equal)
    ):
        raise ValueError("metric/equal must name report fields")
    if not isinstance(required, dict) or not isinstance(maximum, dict):
        raise ValueError("require/maximum must map field names to values")
    threshold = _number(contract.get("minimum_reduction", 0), "minimum_reduction")
    if not 0 <= threshold < 1:
        raise ValueError("minimum_reduction must be in [0, 1)")
    reports = [read_object(path) for path in paths]
    timings = []
    for report in reports:
        for name in equal:
            actual, expected = _field(report, name), _field(reports[0], name)
            if not _same(actual, expected):
                raise ValueError(f"comparison condition changed: {name}")
        for name, expected in required.items():
            actual = _field(report, name)
            if not _same(actual, expected):
                raise ValueError(f"requirement failed: {name}")
        for name, bound in maximum.items():
            value = _number(_field(report, name), name)
            if not 0 <= value <= _number(bound, name):
                raise ValueError(f"error/limit exceeded: {name}")
        timing = _number(_field(report, metric), metric)
        if timing <= 0:
            raise ValueError("timing must be positive")
        timings.append(timing)
    return timings


def compare(
    baseline: list[Path], candidate: list[Path], contract: dict[str, Any]
) -> dict[str, Any]:
    """Validate every sample before deriving a descriptive mean reduction."""
    if not baseline or not candidate:
        raise ValueError("baseline and candidate must both contain samples")
    timings = validate_samples(baseline + candidate, contract)
    metric = contract.get("metric", "eval_seconds")
    threshold = contract.get("minimum_reduction", 0)
    groups = (timings[: len(baseline)], timings[len(baseline) :])
    means = [statistics.mean(group) for group in groups]
    reduction = 1 - means[1] / means[0]
    return {
        "schema_version": 1,
        "passed": reduction >= threshold,
        "metric": metric,
        "baseline": {"samples": groups[0], "mean": means[0]},
        "candidate": {"samples": groups[1], "mean": means[1]},
        "reduction": reduction,
        "contract": contract,
        "inputs": [{"path": str(path), **file_identity(path)} for path in baseline + candidate],
        "scope": "Descriptive matched comparison; no statistical significance claim.",
    }
