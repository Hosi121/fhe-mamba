#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_DIR}/scripts/b300_platform.sh"
b300_load_platform "${REPO_DIR}"
ROOT_DIR="${ROOT_DIR:-/home/kataiwa/fhemamba-b300}"
IMAGE="${IMAGE:-${B300_IMAGE}}"
GPU_DEVICE="${GPU_DEVICE:-3}"
FIDESLIB_SM="${FIDESLIB_SM:-${B300_FIDESLIB_SM}}"
FIDESLIB_SYNC_PROFILE="${FIDESLIB_SYNC_PROFILE:-${B300_DEFAULT_SYNC_PROFILE}}"
resolved_variant="$(b300_variant "${FIDESLIB_SYNC_PROFILE}")"
FIDESLIB_VARIANT="${FIDESLIB_VARIANT:-${resolved_variant}}"
expected_binary_path="${ROOT_DIR}/build/fideslib-stage0-${FIDESLIB_VARIANT}/stage1_mamba2_decode_fideslib"
BINARY_PATH="${BINARY_PATH:-${expected_binary_path}}"
b300_assert_equal image "${IMAGE}" "${B300_IMAGE}"
b300_assert_equal FIDESlib-SM "${FIDESLIB_SM}" "${B300_FIDESLIB_SM}"
b300_assert_equal FIDESlib-variant "${FIDESLIB_VARIANT}" "${resolved_variant}"
b300_assert_equal binary-path "${BINARY_PATH}" "${expected_binary_path}"
image_id="$(b300_image_id "${IMAGE}")"
patch_set_sha256="$(b300_patchset_sha256 "${REPO_DIR}" "${FIDESLIB_SYNC_PROFILE}")"
binary_relative_path="build/fideslib-stage0-${FIDESLIB_VARIANT}/stage1_mamba2_decode_fideslib"
metadata_path="${BINARY_PATH}.build.json"
platform_config_sha256="$(sha256sum "${REPO_DIR}/config/b300-platform.env" | cut -d' ' -f1)"
LAYERS="${LAYERS:-24}"
TOKENS="${TOKENS:-1}"
# This B300-specific combination passes the 24-layer/three-token gate while
# avoiding the all-scope path's larger first-token cost.
FUSED_REPLICATED_LINEAR_TRANSFORM="${FUSED_REPLICATED_LINEAR_TRANSFORM:-1}"
FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE="${FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE:-out-proj}"
AUTOREGRESSIVE_CLIENT_LOOP="${AUTOREGRESSIVE_CLIENT_LOOP:-0}"
NORMALIZED_STATE_META_BTS="${NORMALIZED_STATE_META_BTS:-0}"
COMPLEX_STATE_PAIRING="${COMPLEX_STATE_PAIRING:-1}"
SHARED_HEAD_EXPANSION="${SHARED_HEAD_EXPANSION:-0}"
STATE_REFRESH_INTERVAL="${STATE_REFRESH_INTERVAL:-1}"
DEBUG_LAYER_ERRORS="${DEBUG_LAYER_ERRORS:-0}"
DEBUG_NORMALIZED_STATE_BOOTSTRAP_RANGE="${DEBUG_NORMALIZED_STATE_BOOTSTRAP_RANGE:-0}"
DEBUG_RECURRENCE_TOKEN="${DEBUG_RECURRENCE_TOKEN:--1}"
DEBUG_RECURRENCE_LAYER="${DEBUG_RECURRENCE_LAYER:--1}"
# The promoted path registers 64.824 GiB. Shared head expansion adds 0.571 GiB
# of masks, so keep its opt-in comparison fully cached as well.
if [[ "${SHARED_HEAD_EXPANSION}" == "1" ]]; then
  default_pt_cache_gib=66
else
  default_pt_cache_gib=65
fi
PT_CACHE_GIB="${PT_CACHE_GIB:-${default_pt_cache_gib}}"
PT_CACHE_LEVEL="${PT_CACHE_LEVEL:-0}"
PT_CACHE_WEIGHT_LEVEL="${PT_CACHE_WEIGHT_LEVEL:-20}"
PT_MISS_CONSUMPTION_LEVEL="${PT_MISS_CONSUMPTION_LEVEL:-1}"
ENCODE_THREADS="${ENCODE_THREADS:-8}"
INPUT_CHAIN="${INPUT_CHAIN:-${ROOT_DIR}/payloads/m2_chain_payload_sqnewton_wiki512_t2}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"
META_BTS_RESIDUAL_LAYERS="${META_BTS_RESIDUAL_LAYERS:-21,22,23}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG:-}"
if [[ -z "${OUTPUT_JSON:-}" ]]; then
  if [[ -n "${RUN_TAG}" ]]; then
    # run_dgx_campaign.py predicts this path before starting the runner. Keep the
    # campaign and direct B300 launch contracts identical when RUN_TAG is set.
    OUTPUT_JSON="${RESULTS_DIR}/m2_chain_${RUN_TAG}_l${LAYERS}_t${TOKENS}.json"
  else
    OUTPUT_JSON="${RESULTS_DIR}/m2_chain_b300-sm${FIDESLIB_SM}-l${LAYERS}-t${TOKENS}-${RUN_ID}.json"
  fi
fi

if [[ "${GPU_DEVICE}" != "2" && "${GPU_DEVICE}" != "3" ]]; then
  echo "GPU_DEVICE must be 2 or 3" >&2
  exit 2
fi
if [[ ! -x "${BINARY_PATH}" ]]; then
  echo "missing sm${B300_FIDESLIB_SM} stage binary; run launch_b300_fideslib_build.sh first" >&2
  exit 2
fi

metadata_expectation=(
  --metadata "${metadata_path}"
  --platform-config-version "${B300_PLATFORM_VERSION}"
  --platform-config-sha256 "${platform_config_sha256}"
  --image "${B300_IMAGE}"
  --image-id "${image_id}"
  --cuda-version "${B300_CUDA_VERSION}"
  --fideslib-repository "${B300_FIDESLIB_REPOSITORY}"
  --fideslib-commit "${B300_FIDESLIB_COMMIT}"
  --fideslib-arch "${B300_FIDESLIB_ARCH}"
  --fideslib-sm "${B300_FIDESLIB_SM}"
  --sync-profile "${FIDESLIB_SYNC_PROFILE}"
  --variant "${FIDESLIB_VARIANT}"
  --patch-set-sha256 "${patch_set_sha256}"
  --binary-relative-path "${binary_relative_path}"
  --binary "${BINARY_PATH}"
)
python3 "${REPO_DIR}/fhemamba/experiments/manage_b300_build_metadata.py" \
  validate "${metadata_expectation[@]}"

mkdir -p "${RESULTS_DIR}"
if [[ -n "${REPO_COMMIT:-}" ]]; then
  repo_commit="${REPO_COMMIT}"
else
  repo_commit="$(git -C "${ROOT_DIR}/cipher" rev-parse HEAD 2>/dev/null || echo working-tree)"
  if [[ -n "$(git -C "${ROOT_DIR}/cipher" status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
    repo_commit="${repo_commit}-dirty"
  fi
fi
actual_binary_sha256="$(sha256sum "${BINARY_PATH}" | cut -d' ' -f1)"
binary_sha256="${BINARY_SHA256:-${actual_binary_sha256}}"
b300_assert_equal binary-SHA-256 "${binary_sha256}" "${actual_binary_sha256}"
container_name="fhemamba-b300-${RUN_ID}"

docker run --rm \
  --name "${container_name}" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  --shm-size=32g \
  --volume "${ROOT_DIR}:/workspace" \
  --workdir /workspace \
  --env FHEMAMBA_REMOTE_ROOT=/workspace \
  --env BINARY="${BINARY_PATH/#${ROOT_DIR}/\/workspace}" \
  --env INPUT_CHAIN="${INPUT_CHAIN/#${ROOT_DIR}/\/workspace}" \
  --env RESULTS_DIR=/workspace/results \
  --env META_BTS_RESIDUAL_LAYERS="${META_BTS_RESIDUAL_LAYERS}" \
  --env OUTPUT_JSON="${OUTPUT_JSON/#${ROOT_DIR}/\/workspace}" \
  --env ARTIFACT_VERSION="${ARTIFACT_VERSION:-}" \
  --env REPO_COMMIT="${repo_commit}" \
  --env LAYERS="${LAYERS}" \
  --env TOKENS="${TOKENS}" \
  --env FUSED_REPLICATED_LINEAR_TRANSFORM="${FUSED_REPLICATED_LINEAR_TRANSFORM}" \
  --env FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE="${FUSED_REPLICATED_LINEAR_TRANSFORM_SCOPE}" \
  --env FIDESLIB_SYNC_PROFILE="${FIDESLIB_SYNC_PROFILE}" \
  --env AUTOREGRESSIVE_CLIENT_LOOP="${AUTOREGRESSIVE_CLIENT_LOOP}" \
  --env NORMALIZED_STATE_META_BTS="${NORMALIZED_STATE_META_BTS}" \
  --env COMPLEX_STATE_PAIRING="${COMPLEX_STATE_PAIRING}" \
  --env SHARED_HEAD_EXPANSION="${SHARED_HEAD_EXPANSION}" \
  --env STATE_REFRESH_INTERVAL="${STATE_REFRESH_INTERVAL}" \
  --env DEBUG_LAYER_ERRORS="${DEBUG_LAYER_ERRORS}" \
  --env DEBUG_NORMALIZED_STATE_BOOTSTRAP_RANGE="${DEBUG_NORMALIZED_STATE_BOOTSTRAP_RANGE}" \
  --env DEBUG_RECURRENCE_TOKEN="${DEBUG_RECURRENCE_TOKEN}" \
  --env DEBUG_RECURRENCE_LAYER="${DEBUG_RECURRENCE_LAYER}" \
  --env PT_CACHE_GIB="${PT_CACHE_GIB}" \
  --env PT_CACHE_LEVEL="${PT_CACHE_LEVEL}" \
  --env PT_CACHE_WEIGHT_LEVEL="${PT_CACHE_WEIGHT_LEVEL}" \
  --env PT_MISS_CONSUMPTION_LEVEL="${PT_MISS_CONSUMPTION_LEVEL}" \
  --env ENCODE_THREADS="${ENCODE_THREADS}" \
  --env BINARY_SHA256="${binary_sha256}" \
  --env LD_LIBRARY_PATH="/workspace/install/fideslib-${FIDESLIB_VARIANT}/lib:/workspace/install/openfhe-fides/lib:/workspace/install/openfhe-fides/lib64" \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    unset CUDA_LAUNCH_BLOCKING
    source /workspace/cipher/fhemamba/experiments/dgx_mamba2_common.sh
    init_dgx_mamba2_defaults
    build_dgx_mamba2_args "${LAYERS}" "${TOKENS}"
    "${BINARY}" "${DGX_MAMBA2_ARGS[@]}" \
      --output-json "${OUTPUT_JSON}" \
      --artifact-version "${ARTIFACT_VERSION}" \
      --repo-commit "${REPO_COMMIT}" \
      --binary-sha256 "${BINARY_SHA256}"
  '

python3 "${REPO_DIR}/fhemamba/experiments/manage_b300_build_metadata.py" \
  attach "${metadata_expectation[@]}" --artifact "${OUTPUT_JSON}"

echo "output_json=${OUTPUT_JSON}"
echo "build_metadata=${metadata_path}"
