#!/usr/bin/env python3
"""Build an isolated FIDESlib numerical probe and bind its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--fideslib-prefix", type=Path, required=True)
    parser.add_argument("--openfhe-prefix", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=(
            "stage1_normalization_probe",
            "stage1_vector_rms_probe",
            "stage1_coefficient_encoding_probe",
            "stage1_subring_encoding_probe",
        ),
        default="stage1_normalization_probe",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    binary = args.build_dir.resolve() / args.target
    manifest = binary.with_suffix(".build.json")
    sources = [
        "native/fideslib_stage0/CMakeLists.txt",
        f"native/fideslib_stage0/src/{args.target}.cpp",
        "fhemamba/src/fhemamba/_version.py",
    ]
    test_target = "test_stage1_normalization"
    if args.target not in {"stage1_coefficient_encoding_probe", "stage1_subring_encoding_probe"}:
        sources.append("native/fideslib_stage0/include/stage1_normalization.hpp")
    else:
        test_target = None
    if args.target == "stage1_subring_encoding_probe":
        sources.append("native/fideslib_stage0/src/fideslib_periodic_encoder.hpp")
    if args.target == "stage1_vector_rms_probe":
        sources.append("native/fideslib_stage0/include/stage1_vector_rms.hpp")
        test_target = "test_stage1_vector_rms"
    hashes = {name: sha256(root / name) for name in sources}
    if binary.exists():
        if not manifest.exists():
            parser.error("existing binary has no provenance; use a new build directory")
        old = json.loads(manifest.read_text())
        if old["binary_sha256"] != sha256(binary):
            parser.error("existing binary differs from its manifest")
        history = args.build_dir / "history" / old["binary_sha256"]
        history.mkdir(parents=True, exist_ok=True)
        for source, name in ((binary, binary.name), (manifest, "build.json")):
            if not (history / name).exists():
                shutil.copy2(source, history / name)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(root / "native/fideslib_stage0"),
            "-B",
            str(args.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
            f"-DCMAKE_PREFIX_PATH={args.fideslib_prefix};{args.openfhe_prefix}",
            f"-Dfideslib_DIR={args.fideslib_prefix}/share/fideslib/cmake",
            "-DFHE_STAGE0_BUILD_TESTS=ON",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(args.build_dir),
            "--target",
            args.target,
            *([test_target] if test_target else []),
            "-j",
            "4",
        ],
        check=True,
    )
    if test_target:
        subprocess.run([str(args.build_dir / test_target)], check=True)
    if hashes != {name: sha256(root / name) for name in sources}:
        raise RuntimeError("sources changed during compilation")
    libraries = {}
    for word in subprocess.check_output(["ldd", str(binary)], text=True).split():
        if word.startswith("/") and Path(word).is_file():
            path = Path(word).resolve()
            libraries[str(path)] = sha256(path)
    report = {
        "binary_sha256": sha256(binary),
        "source_sha256": hashes,
        "library_sha256": libraries,
        "static_library_sha256": {
            str(p.resolve()): sha256(p)
            for prefix in (args.fideslib_prefix, args.openfhe_prefix)
            for p in (prefix / "lib").glob("*.a")
        },
        "cmake_cache_sha256": sha256(args.build_dir / "CMakeCache.txt"),
        "architecture": platform.machine(),
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "builder_source_sha256": sha256(Path(__file__)),
    }
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    archive = args.build_dir / "history" / report["binary_sha256"]
    archive.mkdir(parents=True, exist_ok=True)
    if not (archive / "source.tar.gz").exists():
        with tarfile.open(archive / "source.tar.gz", "w:gz") as tar:
            for name in sources:
                tar.add(root / name, arcname=name)
    print(f"built {binary}; SHA-256 {report['binary_sha256']}")


if __name__ == "__main__":
    main()
