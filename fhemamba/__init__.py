"""Source-tree import shim for the active ``fhemamba`` package.

The repository stores tests, experiments, and results beside ``src/fhemamba``.
Without this shim, running Python from the repository root resolves the outer
directory as an empty namespace package before the editable-install finder can
load the real package.
"""

from pathlib import Path

_SOURCE_PACKAGE = Path(__file__).resolve().parent / "src" / "fhemamba"
__path__ = [str(_SOURCE_PACKAGE)]

from ._version import __version__  # noqa: E402

__all__ = ["__version__"]
