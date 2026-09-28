import hashlib
import importlib.util
from pathlib import Path

import pytest


def test_archive_identity_checks_each_duplicate_and_long_named_member(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "ntt_rebuild", Path(__file__).parents[1] / "experiments/ntt_lazy/rebuild.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def member(name, data):
        header = f"{name:<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(data):<10}`\n".encode()
        return header + data + (b"\n" if len(data) % 2 else b"")

    a, b = tmp_path / "a.a", tmp_path / "b.a"
    prefix = b"!<arch>\n" + member("/", b"ignored symbol index")
    prefix += member("//", b"a_very_long_object_name.cu.o/\n")
    prefix += member("/0", b"long") + member("same.o/", b"one")
    a.write_bytes(prefix + member("same.o/", b"two"))
    b.write_bytes(prefix + member("same.o/", b"bad"))
    assert module.archive_members(a) == [
        (name, hashlib.sha256(data).hexdigest())
        for name, data in [
            ("a_very_long_object_name.cu.o", b"long"),
            ("same.o", b"one"),
            ("same.o", b"two"),
        ]
    ]
    assert module.archive_changes(a, b) == ["same.o"]
    b.write_bytes(prefix + member("other.o/", b"two"))
    with pytest.raises(ValueError, match="names/order"):
        module.archive_changes(a, b)
    b.write_bytes(a.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        module.archive_members(b)
