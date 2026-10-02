"""Share byte-identical client heads without writable aliases between payloads."""

from __future__ import annotations

import os
import shutil
import struct
import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np

from fhemamba.benchmarks.io import file_sha256


def share_client_head(source: Path, destination: Path) -> None:
    """Create a new relative link to a verified, read-only sibling blob store.

    Existing destinations (including dangling links) are never overwritten.
    Copying through a temporary file keeps source and shared storage independent.
    """
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    expected = file_sha256(source)
    store = destination.parent.parent / ".client-heads"
    if store.is_symlink():
        raise ValueError("client-head store must not be a symlink")
    store.mkdir(exist_ok=True)
    blob = store / f"{expected}.f32"
    if not os.path.lexists(blob):
        with tempfile.NamedTemporaryFile(dir=store, prefix=".pending-") as temporary:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
            pending = Path(temporary.name)
            if file_sha256(pending) != expected:
                raise ValueError("client head changed while copying")
            pending.chmod(0o444)
            # Publish without replacing another writer's completed blob.
            with suppress(FileExistsError):
                os.link(pending, blob)
    if blob.is_symlink() or not blob.is_file() or blob.stat().st_mode & 0o222:
        raise ValueError("client-head blob must be a read-only regular file")
    if file_sha256(blob) != expected:
        raise ValueError("client-head blob checksum mismatch")
    destination.symlink_to(os.path.relpath(blob, destination.parent))


def write_client_head(destination: Path, weight: np.ndarray) -> None:
    """Serialize the existing little-endian FP32 format, then publish a new link."""
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    weight = np.asarray(weight, dtype="<f4")
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".head-") as stream:
        stream.write(struct.pack("<II", *weight.shape))
        stream.flush()
        weight.tofile(stream)
        stream.flush()
        share_client_head(Path(stream.name), destination)
