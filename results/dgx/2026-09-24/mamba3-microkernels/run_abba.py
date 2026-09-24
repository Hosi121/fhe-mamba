import json
import subprocess
from pathlib import Path

root = Path("/home/kataiwa/fhemamba/mamba3-20260924")
for name in ("a1", "b1", "b2", "a2"):
    out = root / ("micro-prefix-" + name)
    cmd = [
        "python3",
        str(root / "run_packed_probe.py"),
        "--binary",
        str(root / "packed_fideslib-inplace"),
        "--payload",
        str(root / "lm-layer1-prefix"),
        "--output",
        str(out),
        "--planned-refresh",
        "--batch-refresh",
        "--timeout",
        "180",
        "--budget-file",
        str(root / "optimization-budget-20260924.json"),
        "--budget-seconds",
        "7200",
    ]
    if name.startswith("b"):
        cmd += ["--inplace-ops"]
    subprocess.run(cmd, check=True)
    native = json.loads((out / "native.json").read_text())
    print(name, native["eval_seconds"], native["max_abs_error_vs_exact"], flush=True)
