#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
REPO_DIR="${ROOT_DIR}/cipher"
source "${REPO_DIR}/scripts/b300_platform.sh"
b300_load_platform "${REPO_DIR}"

FIDESLIB_SOURCE_DIR="${FIDESLIB_SOURCE_DIR:-${ROOT_DIR}/src/FIDESlib}"
B300_SYNC_PROFILE="${B300_SYNC_PROFILE:-${B300_DEFAULT_SYNC_PROFILE}}"
BUILD_VARIANT="$(b300_variant "${B300_SYNC_PROFILE}")"
PATCH_SET_SHA256="$(b300_patchset_sha256 "${REPO_DIR}" "${B300_SYNC_PROFILE}")"
PATCH_SET_SHORT="${PATCH_SET_SHA256:0:16}"
FIDESLIB_DIR="${ROOT_DIR}/build/fideslib-sources/${BUILD_VARIANT}-${PATCH_SET_SHORT}"
OPENFHE_PREFIX="${OPENFHE_PREFIX:-${ROOT_DIR}/install/openfhe-fides}"
FIDESLIB_PREFIX="${FIDESLIB_PREFIX:-${ROOT_DIR}/install/fideslib-${BUILD_VARIANT}}"
FIDESLIB_BUILD_DIR="${FIDESLIB_BUILD_DIR:-${ROOT_DIR}/build/fideslib-${BUILD_VARIANT}}"
STAGE_BUILD_DIR="${STAGE_BUILD_DIR:-${ROOT_DIR}/build/fideslib-stage0-${BUILD_VARIANT}}"
BUILD_JOBS="${BUILD_JOBS:-32}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/logs/fideslib-build-${BUILD_VARIANT}.log}"
B300_IMAGE_ID="${B300_IMAGE_ID:-}"
B300_PLATFORM_CONFIG_SHA256="${B300_PLATFORM_CONFIG_SHA256:-}"
CXX_COMPILER="${CXX_COMPILER:-$(command -v g++)}"
BINARY_PATH="${STAGE_BUILD_DIR}/stage1_mamba2_decode_fideslib"
BINARY_RELATIVE_PATH="build/fideslib-stage0-${BUILD_VARIANT}/stage1_mamba2_decode_fideslib"
METADATA_PATH="${BINARY_PATH}.build.json"

if [[ -z "${B300_IMAGE_ID}" ]]; then
  echo "B300_IMAGE_ID is required; launch through launch_b300_fideslib_build.sh" >&2
  exit 2
fi
actual_platform_config_sha256="$(sha256sum "${REPO_DIR}/config/b300-platform.env" | cut -d' ' -f1)"
b300_assert_equal platform-config-SHA-256 \
  "${actual_platform_config_sha256}" "${B300_PLATFORM_CONFIG_SHA256}"
if [[ "${B300_SYNC_PROFILE}" == "full" ]]; then
  b300_assert_equal variant "${BUILD_VARIANT}" "${B300_DEFAULT_VARIANT}"
  b300_assert_equal binary-path "${BINARY_RELATIVE_PATH}" "${B300_BINARY_RELATIVE_PATH}"
fi
if [[ ! -x "${CXX_COMPILER}" ]]; then
  echo "CXX compiler is not executable: ${CXX_COMPILER}" >&2
  exit 2
fi

mkdir -p \
  "$(dirname "${LOG_FILE}")" \
  "${ROOT_DIR}/build/fideslib-sources" \
  "${ROOT_DIR}/install"
exec > >(tee -a "${LOG_FILE}") 2>&1

exec 9>"${ROOT_DIR}/build/fideslib-${BUILD_VARIANT}.lock"
if ! flock -n 9; then
  echo "another ${BUILD_VARIANT} build holds the profile lock" >&2
  exit 3
fi

cuda_release="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -1)"
if [[ -z "${cuda_release}" ]]; then
  echo "cannot determine CUDA release" >&2
  exit 1
fi
b300_assert_equal CUDA-version "${cuda_release}" "${B300_CUDA_VERSION}"

echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "platform_config_version=${B300_PLATFORM_VERSION}"
echo "platform_config_sha256=${actual_platform_config_sha256}"
echo "image=${B300_IMAGE}"
echo "image_id=${B300_IMAGE_ID}"
echo "cuda_version=${cuda_release}"
echo "fideslib_repository=${B300_FIDESLIB_REPOSITORY}"
echo "fideslib_commit=${B300_FIDESLIB_COMMIT}"
echo "fideslib_arch=${B300_FIDESLIB_ARCH}"
echo "fideslib_sm=${B300_FIDESLIB_SM}"
echo "b300_sync_profile=${B300_SYNC_PROFILE}"
echo "build_variant=${BUILD_VARIANT}"
echo "patch_set_sha256=${PATCH_SET_SHA256}"
echo "cxx_compiler=${CXX_COMPILER}"
echo "build_jobs=${BUILD_JOBS}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader
nvcc --version
"${CXX_COMPILER}" --version
cmake --version

test -e "${FIDESLIB_SOURCE_DIR}/.git"
git config --global --add safe.directory "${FIDESLIB_SOURCE_DIR}"
if ! git -C "${FIDESLIB_SOURCE_DIR}" cat-file -e "${B300_FIDESLIB_COMMIT}^{commit}"; then
  echo "source repository lacks pinned FIDESlib commit ${B300_FIDESLIB_COMMIT}" >&2
  exit 2
fi

snapshot_metadata="${FIDESLIB_DIR}/.fhemamba-patch-set-sha256"
snapshot_diff_metadata="${FIDESLIB_DIR}/.fhemamba-source-diff-sha256"
if [[ -d "${FIDESLIB_DIR}" ]]; then
  snapshot_commit="$(git -C "${FIDESLIB_DIR}" rev-parse HEAD 2>/dev/null || true)"
  if [[ -f "${snapshot_metadata}" ]]; then
    snapshot_patch_set="$(<"${snapshot_metadata}")"
  else
    snapshot_patch_set=""
  fi
  if [[ -f "${snapshot_diff_metadata}" ]]; then
    expected_snapshot_diff="$(<"${snapshot_diff_metadata}")"
  else
    expected_snapshot_diff=""
  fi
  actual_snapshot_diff="$(git -C "${FIDESLIB_DIR}" diff --binary | sha256sum | cut -d' ' -f1)"
  b300_assert_equal snapshot-commit "${snapshot_commit}" "${B300_FIDESLIB_COMMIT}"
  b300_assert_equal snapshot-patch-set "${snapshot_patch_set}" "${PATCH_SET_SHA256}"
  b300_assert_equal snapshot-diff "${actual_snapshot_diff}" "${expected_snapshot_diff}"
  echo "using_existing_source_snapshot=${FIDESLIB_DIR}"
else
  snapshot_temp="${FIDESLIB_DIR}.tmp.$$"
  if [[ -e "${snapshot_temp}" ]]; then
    echo "temporary source snapshot already exists: ${snapshot_temp}" >&2
    exit 2
  fi
  cleanup_snapshot() {
    if [[ -d "${snapshot_temp}" ]]; then
      rm -rf -- "${snapshot_temp}"
    fi
  }
  trap cleanup_snapshot EXIT
  git clone --no-hardlinks --no-checkout "${FIDESLIB_SOURCE_DIR}" "${snapshot_temp}"
  git -C "${snapshot_temp}" checkout --detach "${B300_FIDESLIB_COMMIT}"
  b300_select_patches "${REPO_DIR}" "${B300_SYNC_PROFILE}"
  for patch_path in "${B300_PATCH_PATHS[@]}"; do
    git -C "${snapshot_temp}" apply --check "${patch_path}"
    git -C "${snapshot_temp}" apply "${patch_path}"
    echo "patch_applied=$(basename "${patch_path}")"
  done
  printf '%s\n' "${PATCH_SET_SHA256}" >"${snapshot_temp}/.fhemamba-patch-set-sha256"
  git -C "${snapshot_temp}" diff --binary | sha256sum | cut -d' ' -f1 \
    >"${snapshot_temp}/.fhemamba-source-diff-sha256"
  mv "${snapshot_temp}" "${FIDESLIB_DIR}"
  trap - EXIT
  echo "source_snapshot_created=${FIDESLIB_DIR}"
fi

if [[ ! -f "${OPENFHE_PREFIX}/lib/OpenFHE/OpenFHEConfig.cmake" && \
      ! -f "${OPENFHE_PREFIX}/lib/cmake/OpenFHE/OpenFHEConfig.cmake" ]]; then
  exec 8>"${ROOT_DIR}/build/openfhe-fides.lock"
  flock 8
  if [[ ! -f "${OPENFHE_PREFIX}/lib/OpenFHE/OpenFHEConfig.cmake" && \
        ! -f "${OPENFHE_PREFIX}/lib/cmake/OpenFHE/OpenFHEConfig.cmake" ]]; then
    (
      cd "${FIDESLIB_DIR}/deps"
      ./build.sh "${OPENFHE_PREFIX}"
    )
  fi
  flock -u 8
else
  echo "using_existing_openfhe=${OPENFHE_PREFIX}"
fi

cmake \
  -S "${FIDESLIB_DIR}" \
  -B "${FIDESLIB_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DCUDA_PATH=/usr/local/cuda \
  -DFIDESLIB_ARCH="${B300_FIDESLIB_ARCH}" \
  -DOPENFHE_INSTALL_PREFIX="${OPENFHE_PREFIX}" \
  -DFIDESLIB_INSTALL_PREFIX="${FIDESLIB_PREFIX}" \
  -DFIDESLIB_INSTALL_OPENFHE=OFF \
  -DFIDESLIB_COMPILE_TESTS=OFF \
  -DFIDESLIB_COMPILE_BENCHMARKS=OFF
cmake --build "${FIDESLIB_BUILD_DIR}" --target fideslib gpu-test -j "${BUILD_JOBS}"
cmake --build "${FIDESLIB_BUILD_DIR}" --target install -j "${BUILD_JOBS}"

"${FIDESLIB_BUILD_DIR}/gpu-test"

cmake \
  -S "${REPO_DIR}/native/fideslib_stage0" \
  -B "${STAGE_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -Dfideslib_DIR="${FIDESLIB_PREFIX}/share/fideslib/cmake" \
  -DCMAKE_PREFIX_PATH="${FIDESLIB_PREFIX};${OPENFHE_PREFIX}"
cmake --build "${STAGE_BUILD_DIR}" \
  --target stage1_mamba2_decode_fideslib stage1_bootstrap_probe fideslib_client_server_probe \
  -j "${BUILD_JOBS}"

python3 "${REPO_DIR}/fhemamba/experiments/manage_b300_build_metadata.py" write \
  --output "${METADATA_PATH}" \
  --platform-config-version "${B300_PLATFORM_VERSION}" \
  --platform-config-sha256 "${actual_platform_config_sha256}" \
  --image "${B300_IMAGE}" \
  --image-id "${B300_IMAGE_ID}" \
  --cuda-version "${B300_CUDA_VERSION}" \
  --fideslib-repository "${B300_FIDESLIB_REPOSITORY}" \
  --fideslib-commit "${B300_FIDESLIB_COMMIT}" \
  --fideslib-arch "${B300_FIDESLIB_ARCH}" \
  --fideslib-sm "${B300_FIDESLIB_SM}" \
  --sync-profile "${B300_SYNC_PROFILE}" \
  --variant "${BUILD_VARIANT}" \
  --patch-set-sha256 "${PATCH_SET_SHA256}" \
  --binary-relative-path "${BINARY_RELATIVE_PATH}" \
  --binary "${BINARY_PATH}" \
  --nvcc "$(nvcc --version | tail -1)" \
  --cxx-path "${CXX_COMPILER}" \
  --cxx-version "$("${CXX_COMPILER}" --version | head -1)" \
  --cmake-version "$(cmake --version | head -1)"

echo "binary=${BINARY_PATH}"
echo "binary_sha256=$(sha256sum "${BINARY_PATH}" | cut -d' ' -f1)"
echo "build_metadata=${METADATA_PATH}"
echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
