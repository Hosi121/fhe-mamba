"""Watch the serialized comparison with inotify; emit only durable events."""

import argparse
import ctypes
import json
import os
import select
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--controller-pid", required=True, type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(root), 0x00000108) < 0:
        raise OSError(ctypes.get_errno(), "cannot watch campaign directory")
    seen = set()

    def emit(event, **data):
        record = {"event": event, "observed_at_unix": time.time(), **data}
        line = json.dumps(record, allow_nan=False)
        with (root / "hook-events.jsonl").open("a") as out:
            out.write(line + "\n")
        print(line, flush=True)

    emit("hook_attached", controller_pid=args.controller_pid)
    try:
        while True:
            status = json.loads((root / "full-status.json").read_text())
            samples_file = root / "full-samples.json"
            samples = json.loads(samples_file.read_text()) if samples_file.exists() else []
            for sample in samples:
                if sample["name"] not in seen:
                    seen.add(sample["name"])
                    emit("sample_completed", **sample)
            if status["state"] != "running":
                emit("campaign_finished", status=status, samples=samples)
                return 0 if status["state"] == "completed" else 1
            if not Path(f"/proc/{args.controller_pid}").is_dir():
                emit("controller_lost", status=status)
                return 1
            ready, _, _ = select.select([fd], [], [], 30)
            if ready:
                while True:
                    try:
                        os.read(fd, 65536)
                    except BlockingIOError:
                        break
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
