#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_DIR}/scripts/b300_platform.sh"
b300_load_platform "${REPO_DIR}"
IMAGE="${IMAGE:-${B300_IMAGE}}"
CUDA_IMAGE="${CUDA_IMAGE:-${B300_CUDA_BASE_IMAGE}}"
b300_assert_equal image "${IMAGE}" "${B300_IMAGE}"
b300_assert_equal CUDA-base-image "${CUDA_IMAGE}" "${B300_CUDA_BASE_IMAGE}"

docker build \
  --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" \
  --file "${REPO_DIR}/docker/b300-fideslib.Dockerfile" \
  --tag "${IMAGE}" \
  "${REPO_DIR}"

echo "image=${IMAGE}"
echo "image_id=$(b300_image_id "${IMAGE}")"
echo "cuda_version=${B300_CUDA_VERSION}"
echo "platform_config_version=${B300_PLATFORM_VERSION}"
