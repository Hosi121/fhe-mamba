"""Opt out of unused vision dependencies in text-only Transformers commands.

Call explicitly before importing Transformers. A preloaded torchvision module
is left alone; otherwise Python reports it as unavailable. This keeps optional
vision-extension compatibility from preventing text-only checkpoint loading.
"""

from __future__ import annotations

import sys


def block_broken_torchvision() -> None:
    if "torchvision" not in sys.modules:
        sys.modules["torchvision"] = None  # type: ignore[assignment]
