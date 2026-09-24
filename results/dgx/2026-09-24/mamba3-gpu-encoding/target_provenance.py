"""Read-only build identity capture, run after measured GPU work completes."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path("/home/kataiwa/fhemamba/mamba3-20260924")
backend = Path("/home/kataiwa/fhemamba/spark/source-2a70798e869944af")
openfhe = Path("/home/kataiwa/fhe-deps/FIDESlib/deps/openfhe-src")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


files = [
    root / "packed_fideslib-gpu-ntt-v2",
    root / "packed_fideslib-gpu-ntt-final",
    root / "packed_plaintext_probe-gpu-ntt-final",
    root / "run_packed_probe.py",
    Path("/home/kataiwa/fhemamba/spark/install-2a70798e869944af/lib/fideslib.a"),
    Path("/home/kataiwa/fhe-deps/openfhe-fides/lib/libOPENFHEcore_static.a"),
    Path("/home/kataiwa/fhe-deps/openfhe-fides/lib/libOPENFHEpke_static.a"),
    backend / "api/CryptoContext.cpp",
    backend / "api/Plaintext.cpp",
    backend / "src/CKKS/Plaintext.cu",
    backend / "src/CKKS/RNSPoly.cpp",
    backend / "src/CKKS/LimbPartition.cu",
    backend / "src/CKKS/Limb.cu",
    backend / "src/CKKS/openfhe-interface/RawCiphertext.cu",
    openfhe / "src/pke/lib/encoding/ckkspackedencoding.cpp",
]
compile_commands = json.loads((root / "build/compile_commands.json").read_text())
selected = [
    r
    for r in compile_commands
    if Path(r["file"]).name
    in ("packed_fideslib.cpp", "fideslib_plaintext_encoder.cpp", "packed_plaintext_probe.cpp")
]
result = {
    "hardware": command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"),
    "kernel": command("uname", "-a"),
    "compiler": command("g++", "--version"),
    "cuda": command("/usr/local/cuda-13.0/bin/nvcc", "--version"),
    "backend_commit": command("git", "-C", str(backend), "rev-parse", "HEAD"),
    "backend_status": command("git", "-C", str(backend), "status", "--short"),
    "backend_diff_sha256": hashlib.sha256(
        command("git", "-C", str(backend), "diff").encode()
    ).hexdigest(),
    "source_and_binary_sha256": {str(p): sha(p) for p in files},
    "compile_commands": selected,
    "model_source_sha256": {
        str(p.relative_to(root / "source")): sha(p)
        for p in sorted((root / "source/native/fideslib_stage0").rglob("*"))
        if p.is_file() and p.suffix in (".cpp", ".hpp", ".txt")
    },
}
print(json.dumps(result, indent=2))
