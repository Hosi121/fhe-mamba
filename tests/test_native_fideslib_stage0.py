import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_b300_build_exposes_fideslib_conjugation() -> None:
    patch = (
        ROOT / "native" / "fideslib_stage0" / "patches" / "fideslib-v2.1.0-conjugate-api.patch"
    ).read_text()
    platform_helper = (ROOT / "scripts" / "b300_platform.sh").read_text()

    assert "EvalConjugate" in patch
    assert "result_gpu->conjugate(*input_gpu)" in patch
    assert "fideslib-v2.1.0-conjugate-api.patch" in platform_helper


def test_b300_build_exposes_ckks_complex_data_type() -> None:
    patch = (
        ROOT / "native" / "fideslib_stage0" / "patches" / "fideslib-v2.1.0-ckks-data-type-api.patch"
    ).read_text()
    platform_helper = (ROOT / "scripts" / "b300_platform.sh").read_text()
    probe = (ROOT / "native" / "fideslib_stage0" / "src" / "stage1_bootstrap_probe.cpp").read_text()

    assert "SetCKKSDataType" in patch
    assert "params.SetCKKSDataType(data_type_openfhe)" in patch
    assert "fideslib-v2.1.0-ckks-data-type-api.patch" in platform_helper
    assert "SetCKKSDataType(config.complex_pair ? COMPLEX : REAL)" in probe


def test_b300_sync_profiles_keep_experimental_builds_isolated() -> None:
    build_script = (ROOT / "scripts" / "build_b300_fideslib.sh").read_text()
    launch_script = (ROOT / "scripts" / "launch_b300_fideslib_build.sh").read_text()
    runner = (ROOT / "scripts" / "run_b300_mamba2.sh").read_text()
    cmake = (ROOT / "native" / "fideslib_stage0" / "CMakeLists.txt").read_text()

    assert 'source "${REPO_DIR}/scripts/b300_platform.sh"' in build_script
    assert "flock -n 9" in build_script
    assert "build/fideslib-sources/" in build_script
    assert "git clone --no-hardlinks --no-checkout" in build_script
    assert 'checkout --detach "${B300_FIDESLIB_COMMIT}"' in build_script
    assert '--env FIDESLIB_SOURCE_DIR="/workspace/src/${FIDESLIB_SOURCE_NAME}"' in launch_script
    assert '--env B300_SYNC_PROFILE="${B300_SYNC_PROFILE}"' in launch_script
    assert "manage_b300_build_metadata.py" in runner
    assert 'validate "${metadata_expectation[@]}"' in runner
    assert "set(CMAKE_CXX_COMPILER" not in cmake
    assert '-DCMAKE_CXX_COMPILER="${CXX_COMPILER}"' in build_script
    assert "fideslib-stage0-${FIDESLIB_VARIANT}" in runner
    assert 'FUSED_REPLICATED_LINEAR_TRANSFORM="${FUSED_REPLICATED_LINEAR_TRANSFORM:-1}"' in runner
    assert (
        'FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE="${FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE:-out-proj}"'
        in runner
    )
    assert 'COMPLEX_STATE_PAIRING="${COMPLEX_STATE_PAIRING:-1}"' in runner
    assert 'if [[ "${SHARED_HEAD_EXPANSION}" == "1" ]]' in runner
    assert 'PT_CACHE_GIB="${PT_CACHE_GIB:-${default_pt_cache_gib}}"' in runner


def test_b300_long_horizon_manifest_pins_promoted_path(tmp_path: Path) -> None:
    manifest_path = ROOT / "experiments/manifests/b300_autoregressive_prompt2_generate4.json"
    manifest = json.loads(manifest_path.read_text())
    output = tmp_path / "dry-run.json"
    subprocess.run(
        [
            sys.executable,
            "experiments/run_dgx_campaign.py",
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
    acceptance = manifest["acceptance"]
    preflight = manifest["gpu_preflight"]
    runner = (ROOT / "scripts" / "run_b300_mamba2.sh").read_text()

    assert defaults["LAYERS"] == "24"
    assert defaults["TOKENS"] == "5"
    assert defaults["SECURITY"] == "not-set"
    assert defaults["IMAGE"] == "fhemamba-b300:cuda12.8-fideslib"
    assert defaults["B300_CUDA_VERSION"] == "12.8"
    assert len(defaults["B300_PLATFORM_CONFIG_SHA256"]) == 64
    assert defaults["FIDESLIB_ARCH"] == "100-real"
    assert defaults["FIDESLIB_SM"] == "100"
    assert defaults["FIDESLIB_VARIANT"] == "sm100"
    assert defaults["FIDESLIB_SYNC_PROFILE"] == "full"
    assert defaults["BINARY_PATH"].endswith(
        "/build/fideslib-stage0-sm100/stage1_mamba2_decode_fideslib"
    )
    assert defaults["FUSED_REPLICATED_LINEAR_TRANSFORM"] == "1"
    assert defaults["FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE"] == "out-proj"
    assert defaults["COMPLEX_STATE_PAIRING"] == "1"
    assert defaults["SHARED_HEAD_EXPANSION"] == "0"
    assert defaults["STATE_REFRESH_INTERVAL"] == "1"
    assert defaults["PT_CACHE_GIB"] == "65"
    assert preflight["gpu_index"] == int(defaults["GPU_DEVICE"])
    assert preflight["min_mem_available_gib"] > 120.24
    assert acceptance["max_abs_error_lte"] == 0.05
    assert acceptance["all_tokens_decrypt"] is True
    assert acceptance["zero_intermediate_decrypts"] is True
    assert "m2_chain_${RUN_TAG}_l${LAYERS}_t${TOKENS}.json" in runner
    assert '--env ARTIFACT_VERSION="${ARTIFACT_VERSION:-}"' in runner
    assert 'repo_commit="${REPO_COMMIT}"' in runner
    assert 'binary_sha256="${BINARY_SHA256:-' in runner


def test_autoregressive_output_decrypts_are_not_intermediate_debug_decrypts() -> None:
    source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_decode_fideslib.cpp"
    ).read_text()
    zero_decrypt_scope = source.split('out << "\\"zero_intermediate_decrypts\\":"', 1)[1]
    zero_decrypt_scope = zero_decrypt_scope.split(
        'out << "\\"autoregressive_client_loop_simulation\\":"', 1
    )[0]

    assert "args.autoregressive_client_loop" not in zero_decrypt_scope
    assert '"\\"autoregressive_client_output_decrypt_count\\":"' in source
    assert '"\\"client_output_decrypts_are_protocol_boundary\\":"' in source


def test_mamba2_decode_wires_complex_state_pairing() -> None:
    source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_decode_fideslib.cpp"
    ).read_text()
    config = (ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_config.cpp").read_text()
    runner = (ROOT / "experiments" / "dgx_mamba2_common.sh").read_text()

    assert 'arg == "--complex-state-pairing"' in config
    assert "SetCKKSDataType(args.complex_state_pairing ? COMPLEX : REAL)" in source
    assert "maybe_bootstrap_pair" in source
    assert "EvalConjugate(refreshed)" in source
    assert "paired_state_bootstrap_count" in source
    assert '--complex-state-pairing "$COMPLEX_STATE_PAIRING"' in runner


def test_mamba2_decode_wires_shared_head_expansion() -> None:
    source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_decode_fideslib.cpp"
    ).read_text()
    config = (ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_config.cpp").read_text()
    runner = (ROOT / "experiments" / "dgx_mamba2_common.sh").read_text()

    assert 'arg == "--shared-head-expansion"' in config
    assert "place_all_heads" in source
    assert "extract_shared_head_group" in source
    assert "shared_head_expansion" in source
    assert '--shared-head-expansion "$SHARED_HEAD_EXPANSION"' in runner


def test_replicated_bsgs_uses_hit_first_plaintext_handles() -> None:
    source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_decode_fideslib.cpp"
    ).read_text()
    plan = (ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_plan.hpp").read_text()

    assert "resolve_hit_first_handle" in plan
    assert "replicated_in_proj_table" in source
    assert "replicated_out_proj_table" in source
    assert "resolve_replicated_plain" in source
    assert '"\\"replicated_eval_mask_builds\\":"' in source
    assert '"\\"replicated_mask_bytes_materialized\\":"' in source
    assert '"\\"replicated_mask_build_seconds\\":"' in source


def test_mamba2_native_kernel_is_repo_owned() -> None:
    source = ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_decode_fideslib.cpp"
    config = ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_config.cpp"
    cmake = ROOT / "native" / "fideslib_stage0" / "CMakeLists.txt"

    assert source.exists()
    assert config.exists()
    assert re.search(
        r"add_executable\s*\(\s*stage1_mamba2_decode_fideslib\b",
        cmake.read_text(),
    )
    source_text = source.read_text()
    for process_role in ("client-init", "server-eval", "client-decrypt"):
        assert f'args.process_role == "{process_role}"' in source_text
    assert "write_runtime_failure_payload" in source_text


def test_retired_native_executables_are_absent() -> None:
    native_root = ROOT / "native" / "fideslib_stage0"
    cmake = (native_root / "CMakeLists.txt").read_text()
    retired = (
        "stage0_static_mimo",
        "stage1_rank_gate_fideslib",
        "stage1_rank_gate_payload_eval",
        "stage1_rotation_probe",
        "stage1_tail_fideslib",
        "stage1_tail_payload_eval",
    )

    for name in retired:
        assert name not in cmake
        assert not (native_root / "src" / f"{name}.cpp").exists()
