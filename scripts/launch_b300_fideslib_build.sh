#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_DIR}/scripts/b300_platform.sh"
b300_load_platform "${REPO_DIR}"
ROOT_DIR="${ROOT_DIR:-/home/kataiwa/fhemamba-b300}"
IMAGE="${IMAGE:-${B300_IMAGE}}"
GPU_DEVICE="${GPU_DEVICE:-3}"
FIDESLIB_SOURCE_NAME="${FIDESLIB_SOURCE_NAME:-FIDESlib}"
B300_SYNC_PROFILE="${B300_SYNC_PROFILE:-${B300_DEFAULT_SYNC_PROFILE}}"
BUILD_VARIANT="$(b300_variant "${B300_SYNC_PROFILE}")"
CONTAINER_NAME="${CONTAINER_NAME:-fhemamba-b300-build-${BUILD_VARIANT}}"
b300_assert_equal image "${IMAGE}" "${B300_IMAGE}"
image_id="$(b300_image_id "${IMAGE}")"
platform_config_sha256="$(sha256sum "${REPO_DIR}/config/b300-platform.env" | cut -d' ' -f1)"

if [[ "${GPU_DEVICE}" != "2" && "${GPU_DEVICE}" != "3" ]]; then
  echo "GPU_DEVICE must be 2 or 3" >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/build" "${ROOT_DIR}/install" "${ROOT_DIR}/logs"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  state="$(docker inspect --format '{{.State.Status}}' "${CONTAINER_NAME}")"
  if [[ "${state}" == "running" ]]; then
    running_image_id="$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")"
    b300_assert_equal running-container-image "${running_image_id}" "${image_id}"
    echo "container_already_running=${CONTAINER_NAME}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

container_id="$({
  docker run \
    --detach \
    --name "${CONTAINER_NAME}" \
    --gpus "device=${GPU_DEVICE}" \
    --ipc=host \
    --shm-size=32g \
    --volume "${ROOT_DIR}:/workspace" \
    --workdir /workspace \
    --env ROOT_DIR=/workspace \
    --env FIDESLIB_SOURCE_DIR="/workspace/src/${FIDESLIB_SOURCE_NAME}" \
    --env B300_SYNC_PROFILE="${B300_SYNC_PROFILE}" \
    --env B300_IMAGE_ID="${image_id}" \
    --env B300_PLATFORM_CONFIG_SHA256="${platform_config_sha256}" \
    --env BUILD_JOBS="${BUILD_JOBS:-32}" \
    "${IMAGE}" \
    bash /workspace/cipher/scripts/build_b300_fideslib.sh
})"

echo "container_id=${container_id}"
echo "container_name=${CONTAINER_NAME}"
echo "host_gpu=${GPU_DEVICE}"
echo "image=${IMAGE}"
echo "image_id=${image_id}"
echo "cuda_version=${B300_CUDA_VERSION}"
echo "platform_config_sha256=${platform_config_sha256}"
echo "fideslib_arch=${B300_FIDESLIB_ARCH}"
echo "fideslib_sm=${B300_FIDESLIB_SM}"
echo "fideslib_commit=${B300_FIDESLIB_COMMIT}"
echo "fideslib_source_name=${FIDESLIB_SOURCE_NAME}"
echo "b300_sync_profile=${B300_SYNC_PROFILE}"
echo "build_variant=${BUILD_VARIANT}"
echo "log=${ROOT_DIR}/logs/fideslib-build-${BUILD_VARIANT}.log"
