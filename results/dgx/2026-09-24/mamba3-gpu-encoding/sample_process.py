import datetime
import json
from pathlib import Path

rows = []
for process in Path("/proc").iterdir():
    if not process.name.isdigit():
        continue
    try:
        command = (process / "cmdline").read_bytes().decode().split("\0")
        if Path(command[0]).name != "packed_fideslib-gpu-ntt-final":
            continue
        status = dict(line.split(":", 1) for line in (process / "status").read_text().splitlines())
        environment = (process / "environ").read_bytes().split(b"\0")
        rows.append(
            {
                "pid": int(process.name),
                "command": [arg for arg in command if arg],
                "status": {
                    k: status[k].strip()
                    for k in ("Cpus_allowed_list", "Threads", "VmRSS", "VmPeak")
                },
                "omp_num_threads": next(
                    v.split(b"=", 1)[1].decode()
                    for v in environment
                    if v.startswith(b"OMP_NUM_THREADS=")
                ),
            }
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
assert len(rows) == 1, rows
print(
    json.dumps(
        {
            "sample_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "process": rows[0],
        },
        indent=2,
    )
)
