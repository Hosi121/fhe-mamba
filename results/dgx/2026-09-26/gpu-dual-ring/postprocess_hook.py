"""After the full comparison, check the final binary's unchanged default path."""

import ctypes
import hashlib
import json
import math
import os
import select
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for_full():
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(ROOT), 0x188) < 0:
        raise OSError(ctypes.get_errno(), "cannot watch comparison")
    try:
        while True:
            status = read(ROOT / "full-status.json")
            if status["state"] != "running":
                assert status["state"] == "completed", status
                return
            ready, _, _ = select.select([fd], [], [], 30)
            if ready:
                while True:
                    try:
                        os.read(fd, 65536)
                    except BlockingIOError:
                        break
    finally:
        os.close(fd)


def main():
    wait_for_full()
    binary = ROOT / "spark/kernel/packed_fideslib"
    reference = read(ROOT / "prefix-baseline-command.json")
    output = ROOT / "default-path-regression"
    output.mkdir()
    command = reference["command"].copy()
    command[3], command[5] = str(binary), str(output / "native.json")
    env = dict(os.environ, OMP_NUM_THREADS="4")
    before = time.time()
    print("Full comparison complete; checking the final binary without the new flag.", flush=True)
    with (output / "run.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
    save(output / "run.json", {"command": command, "exit_code": result.returncode,
         "started_at_unix": before, "wall_seconds": time.time() - before,
         "binary_sha256": sha(binary), "environment": {
             "OMP_NUM_THREADS": "4", "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]}})
    assert result.returncode == 0
    data = read(output / "native.json")
    assert data["passed"] and not data["gpu_dual_ring"]
    assert (data["ring_dimension"], data["slots"]) == (65536, 32768)
    assert data["generated_token_ids"] == [6864, 6864]
    assert data["non_finite"] == data["evaluation_decryptions"] == 0
    assert data["bootstraps"] == 16 and data["bootstrap_passes"] == 2
    assert data["ring_switch_up_calls"] == data["ring_switch_down_calls"] == 0
    assert data["polynomial_tolerance"] == data["exact_tolerance"] == .001
    for key in ("max_abs_error_vs_exact", "max_abs_error_vs_polynomial"):
        assert math.isfinite(data[key]) and 0 <= data[key] <= .001
    assert sha(binary) == read(ROOT / "full-1-candidate/run.json")["binary_sha256"]
    negative = []
    for name, slots, width, s2c in (("missing-s2c", 32768, 2, False),
                                    ("wrong-slots", 16384, 2, True),
                                    ("excess-width", 32768, 16385, True)):
        fixture = output / f"{name}.txt"
        values = " ".join("0" for _ in range(width))
        fixture.write_text(f"fhemamba-packed-v2 {slots} 64 1 1\n"
                           f"input {width} 1 0 {width} {values}\n"
                           f"0 {width} {values} {values}\n")
        cmd = [str(binary), str(fixture), str(output / f"{name}.json"), ".001", ".001",
               "--gpu-dual-ring", "--planned-refresh", "--batch-refresh"]
        if s2c:
            cmd.append("--s2c-first")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
        log = result.stdout + result.stderr
        (output / f"{name}.log").write_text(log)
        expected = ("GPU dual ring requires full refresh packing and logical widths <=16384"
                    if s2c else "GPU dual ring requires S2C-first planned two-pass refresh")
        assert result.returncode != 0 and expected in log, (name, result.returncode, log)
        negative.append({"name": name, "command": cmd, "exit_code": result.returncode,
                         "expected_message": expected, "passed": True})
    save(output / "rejections.json", negative)
    prior = read(ROOT / "prior-dependencies.json")["files_sha256"]
    for name, digest in prior.items():
        assert sha(Path(name)) == digest, name
    save(ROOT / "canonical-dependencies-after.json", {"passed": True, "files_sha256": prior})
    backend = read(ROOT / "prior-backend-source.json")["files_sha256"]
    for name, digest in backend.items():
        assert sha(ROOT / "spark/source-8f75cf9c2329fd3b" / name) == digest, name
    save(ROOT / "backend-source-verification.json", {
         "passed": True, "identical_files": len(backend), "files_sha256": backend,
         "reference": "results/dgx/2026-09-25/structural-four/backend-source-verification.json"})
    native = read(ROOT / "full-source-manifest.json")
    for name, digest in native.items():
        assert sha(ROOT / "repo/native/fideslib_stage0" / name) == digest, name
    save(ROOT / "post-validation.json", {"passed": True, "default_path_passed": True,
         "early_rejections": len(negative), "native_source_unchanged": True,
         "finished_at_unix": time.time()})
    print("PASS: default prefix, three early rejections and unchanged source/dependencies.", flush=True)


if __name__ == "__main__":
    main()
