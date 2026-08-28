from __future__ import annotations

import subprocess
import sys

from fhemamba import __version__


def test_cli_reports_active_package_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "fhemamba.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == __version__
