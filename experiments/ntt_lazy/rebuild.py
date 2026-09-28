#!/usr/bin/env python3
"""Rebuild one CUDA archive member and its device link in an isolated directory.

Compiler flags and device-link flags come from an existing CMake build. The
original archive and source tree are read-only inputs. All other archive
members must remain identical, including member order and duplicate names.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from fhemamba.benchmarks.io import file_sha256 as sha


def archive_members(path):
    """Hash every ar payload in order, including repeated member names."""
    members, names = [], b""
    with path.open("rb") as stream:
        if stream.read(8) != b"!<arch>\n":
            raise ValueError("expected an ordinary ar archive")
        while header := stream.read(60):
            if len(header) != 60 or header[58:] != b"`\n":
                raise ValueError("invalid ar member header")
            name = header[:16].decode("ascii").strip()
            size = int(header[48:58])
            if size < 0:
                raise ValueError("invalid ar member size")
            payload = stream.read(size)
            if len(payload) != size or (size % 2 and stream.read(1) != b"\n"):
                raise ValueError("truncated ar member")
            if name == "//":
                names = payload
                continue
            if name in {"/", "/SYM64/"}:
                continue
            if name.startswith("/"):
                offset = int(name[1:])
                end = names.find(b"/\n", offset)
                if offset < 0 or end < offset:
                    raise ValueError("invalid ar long name")
                name = names[offset:end].decode()
            elif name.startswith("#1/"):
                length = int(name[3:])
                if not 0 < length <= size:
                    raise ValueError("invalid ar embedded name")
                name, payload = payload[:length].rstrip(b"\0").decode(), payload[length:]
            else:
                name = name.removesuffix("/")
            members.append((name, hashlib.sha256(payload).hexdigest()))
    return members


def archive_changes(baseline, candidate):
    before, after = archive_members(baseline), archive_members(candidate)
    if [name for name, _ in before] != [name for name, _ in after]:
        raise ValueError("archive member names/order changed")
    return [a[0] for a, b in zip(before, after, strict=True) if a != b]


def build(archive, backend, cmake_build, patch, output, container=None):
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(argv, cwd=output, **kwargs):
        commands.append({"argv": list(map(str, argv)), "cwd": str(cwd)})
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if container:
            return subprocess.run(
                ["docker", "exec", "--workdir", str(cwd), container, *map(str, argv)],
                cwd=output,
                check=True,
                **kwargs,
            )
        return subprocess.run(argv, cwd=cwd, check=True, **kwargs)

    shutil.copytree(backend / "src", output / "src")
    source = output / "src/NTT.cu"
    before = {
        str(p.relative_to(output)): sha(p) for p in (output / "src").rglob("*") if p.is_file()
    }
    run(["git", "apply", "--check", str(patch)])
    run(["git", "apply", str(patch)])
    after = {str(p.relative_to(output)): sha(p) for p in (output / "src").rglob("*") if p.is_file()}
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if changed != ["src/NTT.cu", "src/NTThelper.cuh"]:
        raise ValueError(f"unexpected source changes: {changed}")
    compile_records = json.loads((cmake_build / "compile_commands.json").read_text())
    (record,) = [r for r in compile_records if r["file"].endswith("/NTT.cu")]
    command = shlex.split(record["command"])
    command[command.index(record["file"])] = str(source)
    command[command.index("-o") + 1] = str(output / "NTT.cu.o")
    run(command, cwd=Path(record["directory"]))
    names = subprocess.check_output(["ar", "t", archive], text=True).splitlines()
    cuda_members = [n for n in names if n.endswith(".cu.o")]
    if len(cuda_members) != len(set(cuda_members)) or names.count("cmake_device_link.o") != 1:
        raise ValueError("archive has ambiguous CUDA members")
    for name in cuda_members:
        if name == "NTT.cu.o":
            continue
        with (output / name).open("wb") as stream:
            run(["ar", "p", str(archive), name], stdout=stream)
    link = shlex.split((cmake_build / "CMakeFiles/fideslib.dir/dlink.txt").read_text())
    for i in reversed(range(len(link) - 1)):
        if link[i] == "--options-file":
            response = link[i + 1]
            if response.endswith("deviceObjects1.rsp"):
                link[i : i + 2] = [str(output / n) for n in cuda_members]
            else:
                link[i + 1] = str(cmake_build / response)
    link[link.index("-o") + 1] = str(output / "cmake_device_link.o")
    run(link)
    candidate = output / "fideslib.a"
    shutil.copyfile(archive, candidate)
    run(["ar", "r", str(candidate), str(output / "NTT.cu.o"), str(output / "cmake_device_link.o")])
    run(["ranlib", str(candidate)])
    if names != subprocess.check_output(["ar", "t", candidate], text=True).splitlines():
        raise ValueError("archive member names/order changed")
    changed_members = archive_changes(archive, candidate)
    if sorted(changed_members) != ["NTT.cu.o", "cmake_device_link.o"]:
        raise ValueError(f"unexpected archive changes: {changed_members}")
    result = {
        "passed": True,
        "baseline_sha256": sha(archive),
        "candidate_sha256": sha(candidate),
        "patch_sha256": sha(patch),
        "source_before": before,
        "source_after": after,
        "changed_sources": changed,
        "changed_archive_members": changed_members,
    }
    (output / "identity.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "backend", "cmake-build", "patch", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--container", help="execute compiler/archive commands in this retained build container"
    )
    a = p.parse_args()
    build(
        a.archive.resolve(),
        a.backend.resolve(),
        a.cmake_build.resolve(),
        a.patch.resolve(),
        a.output.resolve(),
        a.container,
    )
