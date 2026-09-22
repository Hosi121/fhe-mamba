from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "manage_dgx_build", ROOT / "experiments/manage_dgx_build.py"
)
assert spec is not None
assert spec.loader is not None
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


def test_payload_hash_binds_contents_and_names(tmp_path: Path) -> None:
    (tmp_path / "chain.json").write_text("{}")
    values = tmp_path / "state.bin"
    values.write_bytes(b"first")
    first = manager.payload_sha256(tmp_path)
    values.write_bytes(b"second")
    second = manager.payload_sha256(tmp_path)
    assert first != second
    values.rename(tmp_path / "other.bin")
    assert manager.payload_sha256(tmp_path) != second
    (tmp_path / "chain.json").unlink()
    with pytest.raises(ValueError, match=r"chain\.json"):
        manager.payload_sha256(tmp_path)


@pytest.mark.parametrize("changed", ["binary", "library", "source"])
def test_spark_provenance_rejects_stale_build(tmp_path: Path, monkeypatch, changed: str) -> None:
    binary = tmp_path / "spark/kernel/stage1_mamba2_decode_fideslib"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary-v1")
    library = tmp_path / "libopenfhe.so"
    library.write_bytes(b"library-v1")
    monkeypatch.setattr(manager, "source_sha256", lambda: "source-v1")
    metadata = {
        "schema_version": 1,
        "platform": "dgx-spark",
        "source_sha256": "source-v1",
        "binary_sha256": manager.sha256(binary),
        "shared_libraries": {str(library): manager.sha256(library)},
    }
    binary.with_suffix(".build.json").write_text(json.dumps(metadata))
    assert manager.validate(tmp_path) == metadata
    if changed == "binary":
        binary.write_bytes(b"binary-v2")
    elif changed == "library":
        library.write_bytes(b"library-v2")
    else:
        monkeypatch.setattr(manager, "source_sha256", lambda: "source-v2")
    with pytest.raises(ValueError, match=r"changed|mismatch"):
        manager.validate(tmp_path)
