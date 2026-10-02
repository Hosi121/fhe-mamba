import os
import struct

import numpy as np
import pytest

from fhemamba.workloads.client_head import share_client_head, write_client_head


def test_identical_heads_share_bytes_and_keep_native_format(tmp_path):
    weights = np.arange(12, dtype=np.float64).reshape(3, 4) / 7
    paths = [tmp_path / name / "client_head.f32" for name in ("first", "second")]
    for path in paths:
        path.parent.mkdir()
        write_client_head(path, weights)
    expected = struct.pack("<II", 3, 4) + weights.astype("<f4").tobytes()
    assert paths[0].read_bytes() == expected
    assert all(path.is_symlink() and not os.path.isabs(os.readlink(path)) for path in paths)
    assert paths[0].resolve() == paths[1].resolve()
    assert paths[0].stat().st_mode & 0o222 == 0
    assert len(list((tmp_path / ".client-heads").iterdir())) == 1


def test_reexport_refuses_to_overwrite_shared_head(tmp_path):
    first, second = [tmp_path / name / "client_head.f32" for name in ("first", "second")]
    first.parent.mkdir()
    second.parent.mkdir()
    write_client_head(first, np.ones((2, 3)))
    share_client_head(first, second)
    before = second.read_bytes()
    with pytest.raises(FileExistsError):
        write_client_head(first, np.zeros((2, 3)))
    with pytest.raises(FileExistsError):
        share_client_head(second, first)
    assert first.read_bytes() == second.read_bytes() == before


def test_source_stays_independent_and_corrupt_store_is_rejected(tmp_path):
    source = tmp_path / "source.f32"
    source.write_bytes(b"original")
    first, second = [tmp_path / name / "client_head.f32" for name in ("first", "second")]
    first.parent.mkdir()
    second.parent.mkdir()
    share_client_head(source, first)
    source.write_bytes(b"changed")
    assert first.read_bytes() == b"original"
    source.write_bytes(b"original")
    blob = first.resolve()
    blob.chmod(0o644)
    blob.write_bytes(b"corrupted")
    blob.chmod(0o444)
    with pytest.raises(ValueError, match="checksum"):
        share_client_head(source, second)
    assert not os.path.lexists(second)


def test_dangling_destination_is_not_followed_or_replaced(tmp_path):
    source = tmp_path / "source.f32"
    source.write_bytes(b"head")
    target = tmp_path / "missing"
    destination = tmp_path / "head.f32"
    destination.symlink_to(target)
    with pytest.raises(FileExistsError):
        share_client_head(source, destination)
    assert destination.is_symlink()
    assert not target.exists()
