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
