"""Replay the bounded cache experiment against an existing campaign ledger."""

import argparse
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
args = parser.parse_args()
root = args.root.resolve()
os.sched_setaffinity(0, set(range(15, 20)))
assert os.environ["OMP_NUM_THREADS"] == "4"
(root / "cache-affinity.json").write_text(
    json.dumps({"allowed_cpus": sorted(os.sched_getaffinity(0)), "omp_num_threads": 4}, indent=2)
    + "\n"
)
for name in ("a1", "b1", "b2", "a2"):
    ledger = root / "optimization-budget-20260924.json"
    budget = json.loads(ledger.read_text())
    if budget["limit_seconds"] - budget["used_seconds"] < 65:
        raise RuntimeError("Insufficient remaining budget for another prefix plus timeout reserve")
    out = root / ("cache-prefix-" + name)
    command = [
        "python3",
        str(root / "run_packed_probe.py"),
        "--binary",
        str(root / "packed_fideslib-cache"),
        "--payload",
        str(root / "lm-layer1-prefix"),
        "--output",
        str(out),
        "--planned-refresh",
        "--batch-refresh",
        "--inplace-ops",
        "--profile-evaluation",
        "--timeout",
        "65",
        "--budget-file",
        str(ledger),
        "--budget-seconds",
        "7200",
    ]
    if name.startswith("b"):
        command.append("--cache-plaintexts")
    subprocess.run(command, check=True)
    native = json.loads((out / "native.json").read_text())
    print(
        name,
        {key: native[key] for key in ("eval_seconds", "host_encodes", "plaintext_cache_hits")},
        flush=True,
    )
