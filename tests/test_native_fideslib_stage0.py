import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_b300_build_exposes_fideslib_conjugation() -> None:
    patch = (
        ROOT / "native" / "fideslib_stage0" / "patches" / "fideslib-v2.1.0-conjugate-api.patch"
    ).read_text()
    build_script = (ROOT / "scripts" / "build_b300_fideslib.sh").read_text()

    assert "EvalConjugate" in patch
    assert "result_gpu->conjugate(*input_gpu)" in patch
    assert "fideslib-v2.1.0-conjugate-api.patch" in build_script


def test_b300_build_exposes_ckks_complex_data_type() -> None:
    patch = (
        ROOT / "native" / "fideslib_stage0" / "patches" / "fideslib-v2.1.0-ckks-data-type-api.patch"
    ).read_text()
    build_script = (ROOT / "scripts" / "build_b300_fideslib.sh").read_text()
    probe = (ROOT / "native" / "fideslib_stage0" / "src" / "stage1_bootstrap_probe.cpp").read_text()

    assert "SetCKKSDataType" in patch
    assert "params.SetCKKSDataType(data_type_openfhe)" in patch
    assert "fideslib-v2.1.0-ckks-data-type-api.patch" in build_script
    assert "SetCKKSDataType(config.complex_pair ? COMPLEX : REAL)" in probe


def test_b300_sync_profiles_keep_experimental_builds_isolated() -> None:
    build_script = (ROOT / "scripts" / "build_b300_fideslib.sh").read_text()
    launch_script = (ROOT / "scripts" / "launch_b300_fideslib_build.sh").read_text()
    runner = (ROOT / "scripts" / "run_b300_mamba2.sh").read_text()

    assert 'B300_SYNC_PROFILE="${B300_SYNC_PROFILE:-full}"' in build_script
    assert 'test -e "${FIDESLIB_DIR}/.git"' in build_script
    assert "bootstrap-lifetime)" in build_script
    assert 'remove_patch_if_applied "${keyswitch_sync_patch}"' in build_script
    assert 'BUILD_VARIANT="sm${FIDESLIB_SM}-${B300_SYNC_PROFILE}"' in build_script
    assert '--env FIDESLIB_DIR="/workspace/src/${FIDESLIB_SOURCE_NAME}"' in launch_script
    assert '--env B300_SYNC_PROFILE="${B300_SYNC_PROFILE}"' in launch_script
    assert 'FIDESLIB_VARIANT="${FIDESLIB_VARIANT:-sm${FIDESLIB_SM}}"' in runner
    assert 'inferred_sync_profile="${FIDESLIB_VARIANT#sm${FIDESLIB_SM}-}"' in runner
    assert "fideslib-stage0-${FIDESLIB_VARIANT}" in runner
    assert 'FUSED_REPLICATED_LINEAR_TRANSFORM="${FUSED_REPLICATED_LINEAR_TRANSFORM:-1}"' in runner
    assert (
        'FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE="${FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE:-out-proj}"'
        in runner
    )
    assert 'COMPLEX_STATE_PAIRING="${COMPLEX_STATE_PAIRING:-1}"' in runner
    assert 'if [[ "${SHARED_HEAD_EXPANSION}" == "1" ]]' in runner
    assert 'PT_CACHE_GIB="${PT_CACHE_GIB:-${default_pt_cache_gib}}"' in runner


def test_b300_long_horizon_manifest_pins_promoted_path() -> None:
    manifest_path = ROOT / "fhemamba" / "experiments" / "b300_autoregressive_prompt2_generate4.json"
    manifest = json.loads(manifest_path.read_text())
    defaults = manifest["defaults"]
    acceptance = manifest["acceptance"]
    preflight = manifest["gpu_preflight"]
    runner = (ROOT / "scripts" / "run_b300_mamba2.sh").read_text()

    assert defaults["LAYERS"] == "24"
    assert defaults["TOKENS"] == "5"
    assert defaults["SECURITY"] == "not-set"
    assert defaults["FIDESLIB_SYNC_PROFILE"] == "full"
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


def test_mamba2_decode_wires_complex_state_pairing() -> None:
    source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_decode_fideslib.cpp"
    ).read_text()
    config = (ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_config.cpp").read_text()
    runner = (ROOT / "fhemamba" / "experiments" / "dgx_mamba2_common.sh").read_text()

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
    runner = (ROOT / "fhemamba" / "experiments" / "dgx_mamba2_common.sh").read_text()

    assert 'arg == "--shared-head-expansion"' in config
    assert "place_all_heads" in source
    assert "extract_shared_head_group" in source
    assert "shared_head_expansion" in source
    assert '--shared-head-expansion "$SHARED_HEAD_EXPANSION"' in runner


def test_mamba2_native_kernel_is_repo_owned() -> None:
    source = ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_decode_fideslib.cpp"
    config = ROOT / "native" / "fideslib_stage0" / "src" / "stage1_mamba2_config.cpp"
    cmake = ROOT / "native" / "fideslib_stage0" / "CMakeLists.txt"

    assert source.exists()
    assert config.exists()
    assert "add_executable(stage1_mamba2_decode_fideslib" in cmake.read_text()
    source_text = source.read_text()
    assert "full_one_layer_polynomial_output_checked" in source_text
    assert "ciphertext_recurrent_state_chain" in source_text
    assert "write_runtime_failure_payload" in source_text
