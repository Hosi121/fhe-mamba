import datetime
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
name = sys.argv[2]
output = root / name
output.mkdir(exist_ok=False)
binary = root / sys.argv[3]
command = [str(binary), str(output / "native.json")]
os.sched_setaffinity(0, set(range(15, 20)))
ledger_path = root / "optimization-cycle2-budget-20260924.json"
with ledger_path.open("a+") as ledger:
    fcntl.flock(ledger, fcntl.LOCK_EX)
    ledger.seek(0)
    value = ledger.read()
    budget = (
        json.loads(value) if value else {"limit_seconds": 21600, "used_seconds": 0.0, "runs": []}
    )
    assert budget["limit_seconds"] == 21600
    remaining = budget["limit_seconds"] - budget["used_seconds"]
    if remaining <= 15:
        raise RuntimeError("review window exhausted")
    started = time.monotonic()
    record = {
        "command": command,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "passed": False,
    }
    try:
        with (output / "native.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            record["timed_out"] = False
            try:
                record["returncode"] = process.wait(timeout=min(300, remaining - 15))
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record["returncode"] = process.returncode
    finally:
        record["wall_seconds"] = time.monotonic() - started
        budget["used_seconds"] += record["wall_seconds"]
        budget["runs"].append({"output": str(output), "charged_seconds": record["wall_seconds"]})
        ledger.seek(0)
        ledger.truncate()
        json.dump(budget, ledger, indent=2)
        ledger.flush()
        os.fsync(ledger.fileno())
    native = output / "native.json"
    if native.exists():
        row = json.loads(native.read_text())
        record["native_sha256"] = hashlib.sha256(native.read_bytes()).hexdigest()
        record["passed"] = (
            record["returncode"] == 0
            and not record["timed_out"]
            and row["passed"]
            and row["exact_rns_cases"] == 160
            and math.isfinite(row["max_abs_error"])
            and row["max_abs_error"] < 1e-6
        )
    record["campaign_used_seconds"] = budget["used_seconds"]
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    sys.exit(0 if record["passed"] else 1)
