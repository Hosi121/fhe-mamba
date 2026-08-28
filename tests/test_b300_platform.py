from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "fhemamba" / "experiments" / "manage_b300_build_metadata.py"


def _platform_values() -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for raw_line in (ROOT / "config" / "b300-platform.env").read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    )


def _expectation(binary: Path, image_id: str = "sha256:image") -> list[str]:
    return [
        "--platform-config-version",
        "1",
        "--platform-config-sha256",
        "b" * 64,
        "--image",
        "fhemamba-b300:cuda12.8-fideslib",
        "--image-id",
        image_id,
        "--cuda-version",
        "12.8",
        "--fideslib-repository",
        "https://github.com/CAPS-UMU/FIDESlib.git",
        "--fideslib-commit",
        "cd171f20f510eeca04c71d7b0034ef073829f761",
        "--fideslib-arch",
        "100-real",
        "--fideslib-sm",
        "100",
        "--sync-profile",
        "full",
        "--variant",
        "sm100",
        "--patch-set-sha256",
        "a" * 64,
        "--binary-relative-path",
        "build/fideslib-stage0-sm100/stage1_mamba2_decode_fideslib",
        "--binary",
        str(binary),
    ]


def test_b300_platform_resolves_variants_and_distinct_patch_sets() -> None:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1/scripts/b300_platform.sh"; '
            'b300_load_platform "$1"; '
            "for profile in full bootstrap-lifetime lifetime none; do "
            'printf "%s %s %s\\n" "$profile" "$(b300_variant "$profile")" '
            '"$(b300_patchset_sha256 "$1" "$profile")"; done',
            "bash",
            str(ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split() for line in completed.stdout.splitlines()]
    assert [row[:2] for row in rows] == [
        ["full", "sm100"],
        ["bootstrap-lifetime", "sm100-bootstrap-lifetime"],
        ["lifetime", "sm100-lifetime"],
        ["none", "sm100-none"],
    ]
    assert len({row[2] for row in rows}) == 4
    assert all(len(row[2]) == 64 for row in rows)


def test_b300_image_defaults_match_authoritative_platform() -> None:
    platform = _platform_values()
    dockerfile = (ROOT / "docker" / "b300-fideslib.Dockerfile").read_text()
    image_builder = (ROOT / "scripts" / "build_b300_fideslib_image.sh").read_text()
    launcher = (ROOT / "scripts" / "launch_b300_fideslib_build.sh").read_text()
    runner = (ROOT / "scripts" / "run_b300_mamba2.sh").read_text()

    assert f"ARG CUDA_IMAGE={platform['B300_CUDA_BASE_IMAGE']}" in dockerfile
    for script in (image_builder, launcher, runner):
        assert "b300_load_platform" in script
        assert "${B300_IMAGE}" in script


def test_b300_build_metadata_validates_and_attaches(tmp_path: Path) -> None:
    binary = tmp_path / "stage1_mamba2_decode_fideslib"
    binary.write_bytes(b"native-binary")
    metadata = tmp_path / "build.json"
    artifact = tmp_path / "artifact.json"
    expectation = _expectation(binary)
    subprocess.run(
        [
            sys.executable,
            str(MANAGER),
            "write",
            *expectation,
            "--output",
            str(metadata),
            "--nvcc",
            "Cuda compilation tools, release 12.8",
            "--cxx-path",
            "/usr/bin/g++",
            "--cxx-version",
            "g++ 13.3.0",
            "--cmake-version",
            "cmake version 3.28.3",
        ],
        check=True,
    )
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    artifact.write_text(
        json.dumps({"binary_sha256": binary_sha256, "passed": True}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(MANAGER),
            "attach",
            *expectation,
            "--metadata",
            str(metadata),
            "--artifact",
            str(artifact),
        ],
        check=True,
    )

    payload = json.loads(artifact.read_text())
    provenance = payload["build_provenance"]
    assert provenance["image"]["id"] == "sha256:image"
    assert provenance["fideslib"]["patch_set_sha256"] == "a" * 64
    assert provenance["binary"]["sha256"] == binary_sha256
    assert provenance["toolchain"]["cxx_version"] == "g++ 13.3.0"


def test_b300_build_metadata_rejects_image_identity_mismatch(tmp_path: Path) -> None:
    binary = tmp_path / "binary"
    binary.write_bytes(b"binary")
    metadata = tmp_path / "metadata.json"
    payload = {
        "schema_version": 1,
        "platform_config_version": "1",
        "platform_config_sha256": "b" * 64,
        "image": {"reference": "fhemamba-b300:cuda12.8-fideslib", "id": "old"},
        "cuda": {"configured_version": "12.8"},
        "fideslib": {
            "commit": "cd171f20f510eeca04c71d7b0034ef073829f761",
            "repository": "https://github.com/CAPS-UMU/FIDESlib.git",
            "arch": "100-real",
            "sm": "100",
            "sync_profile": "full",
            "variant": "sm100",
            "patch_set_sha256": "a" * 64,
        },
        "binary": {
            "relative_path": "build/fideslib-stage0-sm100/stage1_mamba2_decode_fideslib",
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
        "toolchain": {
            "cxx_path": "/usr/bin/g++",
            "cxx_version": "g++ 13.3.0",
            "cmake_version": "cmake version 3.28.3",
        },
    }
    payload["cuda"]["nvcc"] = "Cuda compilation tools, release 12.8"
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(MANAGER),
            "validate",
            *_expectation(binary, image_id="new"),
            "--metadata",
            str(metadata),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "image.id mismatch" in completed.stderr
