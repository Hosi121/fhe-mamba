"""Bounded subprocesses whose descendants cannot outlive their owning job."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress


def run_process(command, *, timeout=None, terminate_grace=2, **options) -> tuple[int, bool]:
    """Return (exit code, timed out), reaping the group on every exit path.

    Callers own signal handlers, output records and resource locks. Cleanup
    finishes before returning or propagating cancellation, so a caller can
    safely release its lock. The deadline includes process creation.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    process = subprocess.Popen(command, start_new_session=True, **options)
    timed_out = False
    try:
        process.wait(timeout=None if deadline is None else max(0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        # A completed group leader can still have running descendants.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=terminate_grace)
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    return process.returncode, timed_out
