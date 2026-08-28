import re
from importlib import metadata
from pathlib import Path

import pytest

import fhemamba

ROOT = Path(__file__).resolve().parents[1]


def test_source_tree_version_matches_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    assert match is not None
    assert fhemamba.__version__ == match.group(1)


def test_installed_distribution_metadata_matches_active_package() -> None:
    distribution = metadata.distribution("fhemamba")
    assert distribution.metadata["Name"] == "fhemamba"
    assert distribution.version == fhemamba.__version__

    entry_points = metadata.entry_points(group="console_scripts", name="fhemamba")
    assert [(entry.name, entry.value) for entry in entry_points] == [
        ("fhemamba", "fhemamba.cli:main")
    ]
    assert list(metadata.entry_points(group="console_scripts", name="fhe-mamba3")) == []
    with pytest.raises(metadata.PackageNotFoundError):
        metadata.distribution("fhe-native-mamba3")


def test_runtime_artifact_versions_use_the_package_version_source() -> None:
    common = (ROOT / "fhemamba/experiments/dgx_mamba2_common.sh").read_text()
    cmake = (ROOT / "native/fideslib_stage0/CMakeLists.txt").read_text()
    native_sources = [
        (ROOT / "native/fideslib_stage0/src/fideslib_client_server_probe.cpp").read_text(),
        (ROOT / "native/fideslib_stage0/src/fideslib_ctpt_probe.cpp").read_text(),
    ]

    assert "_version.py" in common
    assert "_version.py" in cmake
    assert "FHEMAMBA_VERSION" in cmake
    assert all("FHEMAMBA_VERSION" in source for source in native_sources)
    assert all(f'"{fhemamba.__version__}"' not in text for text in [common, *native_sources])
