"""Remove only this completed study's redundant DGX build trees."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/home/kataiwa/fhemamba/gpu-rns-20260925")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size(path):
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def main():
    assert ROOT.resolve() == ROOT
    assert read(ROOT / "status.json")["state"] == "completed"
    assert read(ROOT / "post-validation.json")["state"] == "completed"
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            executable = (entry / "exe").resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        assert not executable.is_relative_to(ROOT), (entry.name, executable)

    build = read(ROOT / "build.json")
    retained = {
        "packed_fideslib": build["candidate_binary_sha256"],
        "packed_rns_probe": build["probe_binary_sha256"],
        "packed_plaintext_probe": read(ROOT / "existing-plaintext-mamba3/run.json")[
            "binary_sha256"
        ],
    }
    env = dict(os.environ)
    env.update(read(ROOT / "existing-plaintext-mamba3/run.json")["environment"])
    libraries = {}
    for name, digest in retained.items():
        path = ROOT / "native-build" / name
        assert sha(path) == digest, path
        libraries[name] = subprocess.check_output(["ldd", str(path)], env=env, text=True)
        assert "not found" not in libraries[name]
        assert str(ROOT) not in libraries[name], libraries[name]

    manifest = read(ROOT / "native-source-manifest.json")
    for name, digest in manifest.items():
        assert sha(ROOT / "native-source" / name) == digest, name
    dependencies = read(ROOT / "dependency-verification.json")
    assert dependencies["passed"]
    for name, digest in dependencies["files_sha256"].items():
        assert sha(Path(name)) == digest, name

    before = size(ROOT)
    removed = {}
    paths = [
        ROOT / name for name in ["source", "build", "install", "native-source", "legacy-build"]
    ]
    paths += [p for p in (ROOT / "native-build").iterdir() if p.name not in retained]
    for path in paths:
        assert path.is_relative_to(ROOT)
        assert path != ROOT
        removed[str(path.relative_to(ROOT))] = size(path)
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    for name, digest in retained.items():
        assert sha(ROOT / "native-build" / name) == digest
    for name, digest in dependencies["files_sha256"].items():
        assert sha(Path(name)) == digest, name
    result = {
        "passed": True,
        "removed_paths_logical_bytes": removed,
        "before_logical_bytes": before,
        "after_logical_bytes": size(ROOT),
        "retained_executable_sha256": retained,
        "retained_executable_ldd": libraries,
        "canonical_dependency_files_unchanged": len(dependencies["files_sha256"]),
        "note": (
            "Raw results, source archive, manifests and measured executables retained; "
            "no canonical dependency or payload removed."
        ),
    }
    (ROOT / "cleanup.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "retained_executable_ldd"}, indent=2))


if __name__ == "__main__":
    main()
