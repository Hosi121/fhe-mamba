"""Run an argv-based job with a deadline, optional lock and durable event.

Linux process groups and flock match the native CUDA build environments. Docker,
NUMA placement and SSH are caller choices, not hard-coded machine assumptions.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from .io import file_identity, read_object, write_json

_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve(value: Any, settings: dict[str, str]) -> Any:
    """Expand only explicit settings, never the ambient shell environment."""
    if isinstance(value, str):
        return _VARIABLE.sub(lambda match: settings[match[1]], value)
    if isinstance(value, list):
        return [resolve(item, settings) for item in value]
    if isinstance(value, dict):
        return {key: resolve(item, settings) for key, item in value.items()}
    return value


def _positive_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    return float(value)


@contextmanager
def _resource_lock(path: Path | None, deadline: float):
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("deadline expired waiting for the resource lock") from None
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _stop_group(process: subprocess.Popen) -> None:
    # The group may contain children even if the original command has exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=2)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_job(
    spec_path: Path,
    output: Path,
    *,
    settings: dict[str, str] | None = None,
    lock_path: Path | None = None,
    events_path: Path | None = None,
) -> dict[str, Any]:
    """Run once, refusing to overwrite evidence; report failure before returning.

    The timeout covers lock waiting and execution. The inherited environment is
    used to run the process but is never copied into the report. Only explicit
    overrides are recorded. Completion events describe process exit, not model
    correctness; use the workload's acceptance gates separately.
    """
    output = output.resolve()
    substitutions = dict(settings or {})
    if any(not isinstance(value, str) for value in substitutions.values()):
        raise ValueError("settings must contain string values")
    substitutions["output"] = str(output)
    spec = resolve(read_object(spec_path), substitutions)
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported job schema_version")
    unknown = set(spec) - {"schema_version", "command", "cwd", "env", "inputs", "timeout_seconds"}
    if unknown:
        raise ValueError(f"unknown job fields: {sorted(unknown)}")
    command = spec.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or "\0" in arg for arg in command)
    ):
        raise ValueError("command must be a non-empty argv list of strings")
    timeout = _positive_seconds(spec.get("timeout_seconds"))
    environment = spec.get("env", {})
    if not isinstance(environment, dict) or any(
        not isinstance(value, str) or "\0" in value or "=" in key or "\0" in key
        for key, value in environment.items()
    ):
        raise ValueError("env must map environment variable names to strings")
    cwd = Path(spec.get("cwd", Path.cwd())).resolve()
    if not cwd.is_dir():
        raise ValueError(f"working directory does not exist: {cwd}")
    inputs = spec.get("inputs", {})
    if not isinstance(inputs, dict) or any(not isinstance(v, str) for v in inputs.values()):
        raise ValueError("inputs must map names to file paths")
    identities = {key: file_identity(cwd / path) for key, path in inputs.items()}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {
        "schema_version": 1,
        "state": "waiting",
        "started_at_unix": time.time(),
        "spec": spec,
        "inputs": identities,
        "exit_code": None,
    }
    write_json(output / "run.json", record)
    process = None

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"interrupted by signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with _resource_lock(lock_path, started + timeout):
            if time.monotonic() >= started + timeout:
                raise TimeoutError("deadline expired waiting for the resource lock")
            record.update(state="running", lock_wait_seconds=time.monotonic() - started)
            write_json(output / "run.json", record)
            try:
                with (output / "run.log").open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env={**os.environ, **environment},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    remaining = max(0, started + timeout - time.monotonic())
                    record["exit_code"] = process.wait(timeout=remaining)
                    record["state"] = "completed" if process.returncode == 0 else "failed"
            finally:
                # Release the resource lock only after all job children stop.
                if process is not None:
                    _stop_group(process)
                    process = None
    except (TimeoutError, subprocess.TimeoutExpired):
        record.update(state="timed_out", error="job deadline expired")
    except OSError as exc:
        record.update(state="failed", error=str(exc))
    finally:
        if process is not None:
            _stop_group(process)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        record["wall_seconds"] = time.monotonic() - started
        write_json(output / "run.json", record)
        event = {
            "event": "job_finished",
            "output": str(output),
            "state": record["state"],
            "exit_code": record["exit_code"],
            "at_unix": time.time(),
        }
        write_json(output / "completion.json", event)
        if events_path is not None:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as events:
                fcntl.flock(events, fcntl.LOCK_EX)
                events.write(json.dumps(event, allow_nan=False) + "\n")
    return record
