"""Serial matched probes; every attempt shares the cycle-2 review ledger."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("binary")
parser.add_argument("payload")
parser.add_argument("label")
parser.add_argument("--modes", default="a,b,c,c,b,a")
parser.add_argument("--timeout", type=float, default=150)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--cache", action="store_true")
args = parser.parse_args()
root = args.root.resolve()
spec = importlib.util.spec_from_file_location("packed_runner", root / "run_packed_probe.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
os.sched_setaffinity(0, set(range(15, 20)))
assert os.environ["OMP_NUM_THREADS"] == "4"
(root / (args.label + "-affinity.json")).write_text(
    json.dumps(
        {
            "allowed_cpus": sorted(os.sched_getaffinity(0)),
            "omp_num_threads": 4,
            "modes": args.modes,
            "profile": args.profile,
            "cache": args.cache,
        },
        indent=2,
    )
    + "\n"
)
for index, mode in enumerate(args.modes.split(","), start=1):
    assert mode in ("a", "b", "c")
    name = f"{args.label}-{mode}{index}"
    output = root / name
    result = runner.run(
        root / args.binary,
        root / args.payload,
        output,
        planned_refresh=True,
        batch_refresh=True,
        inplace_ops=True,
        cache_plaintexts=args.cache,
        profile_evaluation=args.profile,
        fast_plaintext_upload=mode in ("b", "c"),
        gpu_plaintext_ntt=mode == "c",
        timeout=args.timeout,
        budget_file=root / "optimization-cycle2-budget-20260924.json",
        budget_seconds=21600,
    )
    print(name, json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)
    native = json.loads((output / "native.json").read_text())
    print(
        name,
        {
            k: native[k]
            for k in (
                "eval_seconds",
                "host_encoding_seconds",
                "plaintext_upload_seconds",
                "max_abs_error_vs_exact",
                "generated_token_ids",
            )
        },
        flush=True,
    )
