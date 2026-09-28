import re
from importlib import metadata
from pathlib import Path

import fhemamba

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_and_installed_entry_point() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    assert match is not None
    distribution = metadata.distribution("fhemamba")
    assert distribution.metadata["Name"] == "fhemamba"
    assert distribution.version == fhemamba.__version__ == match.group(1)

    entry_points = metadata.entry_points(group="console_scripts", name="fhemamba")
    assert [(entry.name, entry.value) for entry in entry_points] == [
        ("fhemamba", "fhemamba.cli:main")
    ]
