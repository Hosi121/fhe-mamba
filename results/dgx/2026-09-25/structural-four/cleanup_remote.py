"""Remove only this study's temporary trees after the completed comparison.

Measured executables and the Cheddar shared library remain at their original
paths. Canonical dependencies and other campaigns are never cleanup targets.
"""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path("/home/kataiwa/fhemamba/structural-four-20260925")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((ROOT / "full-status.json").read_text())["state"] == "completed"
    assert json.loads((ROOT / "dependency-verification-after.json").read_text())["passed"]
    native = ROOT / "backend/spark/kernel/packed_fideslib"
    composite = ROOT / "cheddar-build/composite_probe"
    lib = ROOT / "cheddar-build/libcheddar.so"
    ring = ROOT / "small-ring-dense-probe"
    full = json.loads((ROOT / "full-1-candidate/run.json").read_text())
    cheddar = json.loads((ROOT / "composite-final-64/run.json").read_text())
    ring_record = json.loads((ROOT / "small-ring-dense-15/run.json").read_text())
    assert sha(native) == full["binary_sha256"]
    assert sha(composite) == cheddar["binary_sha256"]
    assert sha(lib) == cheddar["library_sha256"]
    assert sha(ring) == ring_record["binary_sha256"]
    # Metadata/source archives must already have been collected by the root agent.
    assert (ROOT / "archive-collected.json").exists()
    retained = [
        native,
        ROOT / "backend/spark/kernel/rotation_batch_probe",
        composite,
        lib,
        ring,
        ROOT / "small-ring-probe",
    ]
    identities = {str(p): sha(p) for p in retained}
    removed = []

    def remove(path):
        assert path.is_relative_to(ROOT)
        assert path != ROOT
        assert path.exists()
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))

    # Keep original executable locations so recorded commands/RPATH remain useful.
    for directory, keep in [
        (ROOT / "backend/spark/kernel", {"packed_fideslib", "rotation_batch_probe"}),
        (ROOT / "cheddar-build", {"composite_probe", "libcheddar.so"}),
    ]:
        for child in directory.iterdir():
            if child.name not in keep:
                remove(child)
    for child in (ROOT / "backend/spark").iterdir():
        if child.name != "kernel":
            remove(child)
    for name in ("repo", "cheddar", "cheddar-private-deps"):
        remove(ROOT / name)
    assert identities == {str(p): sha(p) for p in retained}
    deps = json.loads((ROOT / "dependency-verification-after.json").read_text())["files_sha256"]
    assert all(sha(Path(p)) == digest for p, digest in deps.items())
    result = {
        "passed": True,
        "removed": removed,
        "retained_sha256": identities,
        "canonical_dependencies_unchanged": True,
        "worktrees_created": 0,
    }
    (ROOT / "cleanup.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "removed_paths": len(removed), "retained_binaries": len(retained)}
        )
    )


if __name__ == "__main__":
    main()
