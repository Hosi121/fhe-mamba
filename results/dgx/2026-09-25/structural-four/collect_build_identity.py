"""Collect identities before removing this study's isolated build trees."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/home/kataiwa/fhemamba/structural-four-20260925")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((ROOT / "full-status.json").read_text())["state"] == "completed"
    output = ROOT / "build-identity"
    output.mkdir(exist_ok=True)
    metadata = {
        "native-cmake-cache.txt": ROOT / "backend/spark/kernel/CMakeCache.txt",
        "cheddar-cmake-cache.txt": ROOT / "cheddar-build/CMakeCache.txt",
        "cheddar-compile-commands.json": ROOT / "cheddar-build/compile_commands.json",
        "native-link.txt": ROOT / "backend/spark/kernel/CMakeFiles/packed_fideslib.dir/link.txt",
        "cheddar-probe-link.txt": ROOT / "cheddar-build/CMakeFiles/composite_probe.dir/link.txt",
    }
    for name, source in metadata.items():
        shutil.copy2(source, output / name)
    environment = {}
    for name, command in {
        "compiler": ["g++", "--version"],
        "cuda": ["/usr/local/cuda-13.0/bin/nvcc", "--version"],
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
    }.items():
        environment[name] = subprocess.check_output(command, text=True)
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    files = {}
    for directory in (ROOT / "backend/spark/install-8f75cf9c2329fd3b", ROOT / "cheddar-build"):
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and (path.name.endswith(".a") or ".so" in path.name)
                and "/_deps/" not in str(path)
            ):
                files[str(path)] = sha(path)
    for name, path in {
        "native": ROOT / "backend/spark/kernel/packed_fideslib",
        "cheddar": ROOT / "cheddar-build/composite_probe",
    }.items():
        files[str(path)] = sha(path)
        dependencies = subprocess.check_output(["ldd", str(path)], text=True)
        (output / f"{name}-ldd.txt").write_text(dependencies)
        if name == "native":
            assert str(ROOT) not in dependencies, dependencies
        else:
            for line in dependencies.splitlines():
                if str(ROOT) in line:
                    assert str(ROOT / "cheddar-build/libcheddar.so") in line, line
    (output / "libraries-sha256.json").write_text(json.dumps(files, indent=2) + "\n")
    sources = [
        ROOT / "cheddar" / name
        for name in (
            "CMakeLists.txt",
            "composite_probe.cpp",
            "composite_params.hpp",
            "stock32_params.hpp",
        )
    ]
    (output / "alternative-source-sha256.json").write_text(
        json.dumps({str(p): sha(p) for p in sources}, indent=2) + "\n"
    )
    print(json.dumps({"libraries_and_binaries": len(files), "metadata_files": len(metadata)}))


if __name__ == "__main__":
    main()
