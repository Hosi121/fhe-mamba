import json
import subprocess
from pathlib import Path

root = Path("/home/kataiwa/fhemamba/mamba3-20260924")
probes = [
    ("full-prefix", "deepopt-full-prefix-payload", root / "packed_fideslib-inplace", 600, False),
    ("layer1", "lm-layer1", root / "packed_fideslib-inplace", 200, False),
    ("synthetic", "payload", root / "packed_fideslib-inplace", 150, False),
    ("inplace-profile", "lm-layer1-prefix", root / "microkernels/profile-inplace", 180, True),
]
for name, payload, binary, timeout, profile in probes:
    out = root / ("micro-" + name)
    cmd = [
        "python3",
        str(root / "run_packed_probe.py"),
        "--binary",
        str(binary),
        "--payload",
        str(root / payload),
        "--output",
        str(out),
        "--planned-refresh",
        "--batch-refresh",
        "--inplace-ops",
        "--timeout",
        str(timeout),
        "--budget-file",
        str(root / "optimization-budget-20260924.json"),
        "--budget-seconds",
        "7200",
    ]
    if profile:
        cmd += ["--profile-evaluation"]
    subprocess.run(cmd, check=True)
    x = json.loads((out / "native.json").read_text())
    print(name, x["eval_seconds"], x["max_abs_error_vs_exact"], flush=True)
