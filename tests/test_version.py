import re
from pathlib import Path

import fhemamba

ROOT = Path(__file__).resolve().parents[1]


def test_source_tree_version_matches_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    assert match is not None
    assert fhemamba.__version__ == match.group(1)
