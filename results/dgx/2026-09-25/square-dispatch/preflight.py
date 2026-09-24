"""Freeze the existing baseline, source snapshots, dependencies and new binaries."""
import hashlib
import json
import subprocess
from pathlib import Path

from runlib import ROOT, save, sha

old = json.loads((ROOT / "baseline-target-provenance.json").read_text())
final = json.loads((ROOT / "baseline-final-target-provenance.json").read_text())
previous = ROOT.parent / "owned-arithmetic-20260924"
baseline = {
    str(previous / "build-final/packed_fideslib"): final["binary_sha256"],
    str(previous / "build/stage1_mamba2_decode_fideslib"):
        old["candidate_binaries_sha256"]["stage1_mamba2_decode_fideslib"],
}
dependencies = {p: h for p, h in old["dependencies_verified"].items()
                if "borrowed-plaintext-20260924/build" not in p}
for path, value in {**baseline, **dependencies}.items():
    assert sha(path) == value, path
manifest = json.loads((ROOT / "compiled-sources.json").read_text())
base = json.loads((ROOT / "baseline-sources.json").read_text())
assert base["files"] == json.loads((previous / "compiled-final-sources.json").read_text())["files"]
assert all(sha(ROOT / "source" / p) == h for p, h in manifest["files"].items())

def command(*args):
    return subprocess.check_output(args, text=True).strip()

backend = "/home/kataiwa/fhemamba/spark/source-2a70798e869944af"
assert hashlib.sha256(command("git", "-C", backend, "diff").encode()).hexdigest() == old["backend_diff_sha256"]
assert command("git", "-C", backend, "rev-parse", "HEAD") == old["backend_commit"]
assert command("g++", "--version") == old["compiler"]
assert command("/usr/local/cuda-13.0/bin/nvcc", "--version") == old["cuda"]
assert command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader") == old["hardware"]
data = {
    "baseline_binaries_sha256": baseline,
    "candidate_binaries_sha256": {name: sha(ROOT / "build" / name) for name in
        ("packed_fideslib", "stage1_mamba2_decode_fideslib", "owned_arithmetic_probe")},
    "dependencies_sha256": dependencies,
    "source_manifests_sha256": {name: sha(ROOT / name) for name in
        ("baseline-sources.json", "compiled-sources.json")},
    "source_archives_sha256": {name: sha(ROOT / name) for name in
        ("baseline-sources.tar.gz", "compiled-sources.tar.gz")},
    "hardware": old["hardware"], "compiler": old["compiler"], "cuda": old["cuda"],
    "backend_commit": old["backend_commit"], "backend_diff_sha256": old["backend_diff_sha256"],
    "compile_commands": json.loads((ROOT / "build/compile_commands.json").read_text()),
    "changed_sources": [p for p, h in manifest["files"].items() if h != base["files"][p]],
    "gpu_before": command("nvidia-smi"),
    "scope": "Separate immutable baseline/candidate binaries; exact RNS probes then short ABBA model controls.",
}
save(ROOT / "target-provenance.json", data)
print("preflight passed", flush=True)
