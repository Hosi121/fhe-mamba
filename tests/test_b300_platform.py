from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = [sys.executable, "-m", "fhemamba", "benchmark", "b300-build"]


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
            "-eu",
            "-c",
            'source "$1/scripts/b300_platform.sh"; '
            'b300_load_platform "$1"; '
            "for profile in full bootstrap-lifetime lifetime none; do "
            'b300_select_patches "$1" "$profile"; '
            'printf "%s %s %s" "$profile" "$(b300_variant "$profile")" '
            '"$(b300_patchset_sha256 "$1" "$profile")"; '
            'printf " %s" "${B300_PATCH_PATHS[@]##*/}"; printf "\\n"; done',
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
    required = {
        "fideslib-v2.1.0-linear-transform-api.patch",
        "fideslib-v2.1.0-conjugate-api.patch",
        "fideslib-v2.1.0-ckks-data-type-api.patch",
    }
    assert all(required <= set(row[3:]) for row in rows)


def test_historical_mamba2_campaign_resolves_promoted_options(tmp_path: Path) -> None:
    manifest_path = ROOT / "experiments/manifests/b300_autoregressive_prompt2_generate4.json"
    manifest = json.loads(manifest_path.read_text())
    output = tmp_path / "dry-run.json"
    subprocess.run(
        [
            sys.executable,
            "experiments/execution/run_dgx_campaign.py",
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
    )
    defaults = json.loads(output.read_text())["experiments"][0]["environment"]
    expected = {
        "LAYERS": "24",
        "TOKENS": "5",
        "SECURITY": "not-set",
        "IMAGE": "fhemamba-b300:cuda12.8-fideslib",
        "B300_CUDA_VERSION": "12.8",
        "FIDESLIB_ARCH": "100-real",
        "FIDESLIB_SM": "100",
        "FIDESLIB_VARIANT": "sm100",
        "FIDESLIB_SYNC_PROFILE": "full",
        "FUSED_REPLICATED_LINEAR_TRANSFORM": "1",
        "FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE": "out-proj",
        "COMPLEX_STATE_PAIRING": "1",
        "SHARED_HEAD_EXPANSION": "0",
        "STATE_REFRESH_INTERVAL": "1",
        "PT_CACHE_GIB": "65",
    }
    assert {name: defaults[name] for name in expected} == expected
    assert len(defaults["B300_PLATFORM_CONFIG_SHA256"]) == 64
    assert defaults["BINARY_PATH"].endswith(
        "/build/fideslib-stage0-sm100/stage1_mamba2_decode_fideslib"
    )
    assert manifest["gpu_preflight"]["gpu_index"] == int(defaults["GPU_DEVICE"])
    assert manifest["gpu_preflight"]["min_mem_available_gib"] > 120.24
    assert manifest["acceptance"]["max_abs_error_lte"] == 0.05
    assert manifest["acceptance"]["all_tokens_decrypt"] is True
    assert manifest["acceptance"]["zero_intermediate_decrypts"] is True


def test_b300_build_metadata_validates_and_attaches(tmp_path: Path) -> None:
    binary = tmp_path / "stage1_mamba2_decode_fideslib"
    binary.write_bytes(b"native-binary")
    metadata = tmp_path / "build.json"
    artifact = tmp_path / "artifact.json"
    expectation = _expectation(binary)
    subprocess.run(
        [
            *MANAGER,
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
            *MANAGER,
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
            *MANAGER,
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
