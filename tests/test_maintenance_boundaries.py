import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TREES = (
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "experiments",
)
CHECKED_SUFFIXES = {".py", ".sh", ".sbatch"}


def test_active_tree_does_not_import_compatibility_package() -> None:
    offenders = []
    for tree in ACTIVE_TREES:
        for path in tree.rglob("*"):
            if (
                path.is_file()
                and path.suffix in CHECKED_SUFFIXES
                and "fhe_native_mamba3" in path.read_text(encoding="utf-8")
            ):
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_compatibility_source_package_is_absent() -> None:
    assert not (ROOT / "src" / "fhe_native_mamba3").exists()


def test_results_do_not_expose_operational_scripts_or_logs() -> None:
    """Historical operational files belong in provenance, not the public API."""
    operational = {".py", ".sh", ".log", ".jsonl", ".make", ".pyc"}
    assert [
        str(path.relative_to(ROOT))
        for path in (ROOT / "results").rglob("*")
        if path.is_file() and path.suffix in operational
    ] == []


def test_published_evidence_is_present_in_git() -> None:
    """A local-only record can pass hash verification and still break a clone."""
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    missing = []
    for path in (ROOT / "results").rglob("publication.json"):
        manifest = json.loads(path.read_text())
        files = [path, path.parent / manifest["bundle"]["path"]]
        files.extend(
            path.parent / record["path"]
            for record in manifest["files"]
            if record["location"] == "file"
        )
        missing.extend(
            str(file.relative_to(ROOT))
            for file in files
            if str(file.relative_to(ROOT)) not in tracked
        )
    assert missing == [], f"Published evidence must be staged in Git: {missing}"


def test_local_settings_and_private_notes_are_ignored() -> None:
    completed = subprocess.run(
        [
            "git",
            "check-ignore",
            "config/local/settings.json",
            ".local/notes.md",
            "runs/job/run.json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(completed.stdout.splitlines()) == 3


def test_public_text_has_no_personal_home_directories() -> None:
    pattern = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+")
    trees = ("src", "native", "experiments", "scripts", "config", "docs", "results")
    suffixes = {".py", ".cpp", ".hpp", ".cu", ".cuh", ".sh", ".md", ".json", ".env"}
    offenders = []
    for tree in trees:
        for path in (ROOT / tree).rglob("*"):
            if path.suffix not in suffixes or "local" in path.parts or not path.is_file():
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
