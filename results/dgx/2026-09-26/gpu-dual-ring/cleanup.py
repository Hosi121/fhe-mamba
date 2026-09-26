"""Archive and verify this study's own source copies before removing builds."""

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def archive_directory(path, name):
    files = {
        str(p.relative_to(path)): sha(p)
        for p in sorted(path.rglob("*"))
        if ".git" not in p.relative_to(path).parts and p.is_file() and not p.is_symlink()
    }
    links = {
        str(p.relative_to(path)): str(p.readlink())
        for p in sorted(path.rglob("*"))
        if ".git" not in p.relative_to(path).parts and p.is_symlink()
    }
    with tarfile.open(ROOT / name, "w:gz") as out:
        out.add(
            path, arcname=".", filter=lambda info: None if ".git" in Path(info.name).parts else info
        )
    with tarfile.open(ROOT / name) as archive:
        captured = {
            p.name.removeprefix("./"): hashlib.sha256(archive.extractfile(p).read()).hexdigest()
            for p in archive
            if p.isfile()
        }
        captured_links = {p.name.removeprefix("./"): p.linkname for p in archive if p.issym()}
    assert captured == files
    assert captured_links == links
    return {"archive": name, "sha256": sha(ROOT / name), "files_sha256": files, "symlinks": links}


def main():
    assert Path("/home/kataiwa/fhemamba/gpu-dual-ring-20260926") == ROOT
    assert read(ROOT / "full-status.json")["state"] == "completed"
    assert read(ROOT / "post-validation.json")["passed"]
    assert read(ROOT / "backend-source-verification.json")["passed"]
    source = ROOT / "spark/source-8f75cf9c2329fd3b"
    lineage = {
        "commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    }
    patch = subprocess.check_output(["git", "-C", str(source), "diff", "--binary"])
    (ROOT / "backend-source.patch").write_bytes(patch)
    lineage["patch_sha256"] = hashlib.sha256(patch).hexdigest()
    assert lineage["patch_sha256"] == (source / ".source-diff").read_text().strip()
    archives = [
        archive_directory(source, "backend-sources.tar.gz"),
        archive_directory(ROOT / "repo", "repo-snapshot.tar.gz"),
    ]
    for original, target in (
        ("probe-build/CMakeCache.txt", "probe-cmake-cache.txt"),
        ("probe-build/CMakeFiles/gpu_dual_ring.dir/link.txt", "probe-link.txt"),
        ("probe-build/CMakeFiles/gpu_dual_ring.dir/flags.make", "probe-flags.make"),
        ("spark/kernel/packed_fideslib.build.json", "initial-packed-build.json"),
    ):
        path = ROOT / original
        if path.is_file():
            shutil.copy2(path, ROOT / target)
    keep = [
        ROOT / "spark/kernel/packed_fideslib",
        ROOT / "prefix-packed_fideslib",
        ROOT / "probe-build/gpu_dual_ring",
    ]
    retained = {str(p): sha(p) for p in keep}
    assert retained[str(keep[0])] == read(ROOT / "full-1-candidate/run.json")["binary_sha256"]
    assert retained[str(keep[1])] == read(ROOT / "prefix-1-candidate/run.json")["binary_sha256"]
    assert retained[str(keep[2])] == read(ROOT / "micro-qualification/run.json")["binary_sha256"]
    paths = [
        ROOT / "repo",
        source,
        ROOT / "spark/install-8f75cf9c2329fd3b",
        ROOT / "spark/fideslib-8f75cf9c2329fd3b",
        ROOT / "spark/build.lock",
    ]
    for directory in (ROOT / "spark/kernel", ROOT / "probe-build"):
        paths += [p for p in directory.iterdir() if p not in keep]
    removed = []
    for path in paths:
        assert path.is_relative_to(ROOT)
        assert path != ROOT
        assert path not in keep
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        removed.append(str(path))
    for name, digest in retained.items():
        assert sha(Path(name)) == digest
    canonical = read(ROOT / "canonical-dependencies-after.json")
    for name, digest in canonical["files_sha256"].items():
        assert sha(Path(name)) == digest, name
    result = {
        "passed": True,
        "removed": removed,
        "retained_sha256": retained,
        "archives": archives,
        "backend_lineage": lineage,
        "canonical_dependencies_unchanged": True,
        "worktrees_created": 0,
    }
    (ROOT / "cleanup.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "removed_paths": len(removed), "retained_executables": len(retained)}
        )
    )


if __name__ == "__main__":
    main()
