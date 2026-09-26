"""Record a standalone DGX ring-switch probe, including failed runs."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("qualification", "abba", "baab"), required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        "taskset",
        "-c",
        "15-19",
        str(args.binary.resolve()),
        str((args.output / "probe.json").resolve()),
        args.mode,
    ]
    source = Path(__file__).parent
    repo = source.parent.parent
    sources = list(source.glob("*")) + list(
        (repo / "native/fideslib_stage0/src").glob("fideslib_*ring*")
    )
    record = {
        "state": "running",
        "command": command,
        "baseline_commit": args.commit,
        "binary_sha256": sha(args.binary),
        "source_sha256": {str(p.relative_to(repo)): sha(p) for p in sorted(sources) if p.is_file()},
        "environment": {
            "OMP_NUM_THREADS": "4",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        },
        "started_at_unix": time.time(),
    }

    def save():
        tmp = args.output / "run.tmp"
        tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output / "run.json")

    save()
    try:
        with (args.output / "run.log").open("w") as log:
            process = subprocess.run(
                command,
                env=dict(os.environ, OMP_NUM_THREADS="4"),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        record["exit_code"] = process.returncode
        record["state"] = "completed" if process.returncode == 0 else "failed"
    except Exception as error:
        record.update(state="failed", error=repr(error))
    finally:
        record["wall_seconds"] = time.time() - record["started_at_unix"]
        save()
    print(json.dumps(record), flush=True)
    return 0 if record["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
