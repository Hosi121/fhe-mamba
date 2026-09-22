#!/usr/bin/env bash
# Shared B300 platform resolution. Source only.

b300_load_platform() {
  local repo_dir="$1"
  local config_path="${repo_dir}/config/b300-platform.env"
  if [[ ! -f "${config_path}" ]]; then
    echo "missing B300 platform config: ${config_path}" >&2
    return 2
  fi
  # shellcheck source=../config/b300-platform.env
  source "${config_path}"
  local required=(
    B300_PLATFORM_VERSION
    B300_IMAGE
    B300_CUDA_BASE_IMAGE
    B300_CUDA_VERSION
    B300_FIDESLIB_REPOSITORY
    B300_FIDESLIB_COMMIT
    B300_FIDESLIB_ARCH
    B300_FIDESLIB_SM
    B300_DEFAULT_SYNC_PROFILE
    B300_DEFAULT_VARIANT
    B300_BINARY_RELATIVE_PATH
  )
  local name
  for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
      echo "B300 platform config lacks ${name}" >&2
      return 2
    fi
  done
  if [[ ! "${B300_FIDESLIB_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "B300_FIDESLIB_COMMIT must be a full lowercase commit SHA" >&2
    return 2
  fi
  if [[ "${B300_FIDESLIB_ARCH%%-*}" != "${B300_FIDESLIB_SM}" ]]; then
    echo "B300 FIDESlib architecture and SM disagree" >&2
    return 2
  fi
  if [[ "${B300_DEFAULT_VARIANT}" != "sm${B300_FIDESLIB_SM}" ]]; then
    echo "B300 default variant must be the full-sync SM variant" >&2
    return 2
  fi
}

b300_variant() {
  local profile="$1"
  case "${profile}" in
    full)
      printf 'sm%s\n' "${B300_FIDESLIB_SM}"
      ;;
    bootstrap-lifetime|lifetime|none)
      printf 'sm%s-%s\n' "${B300_FIDESLIB_SM}" "${profile}"
      ;;
    *)
      echo "B300 sync profile must be full, bootstrap-lifetime, lifetime, or none" >&2
      return 2
      ;;
  esac
}

b300_select_patches() {
  local repo_dir="$1"
  local profile="$2"
  local cuda_major="${B300_CUDA_VERSION%%.*}"
  local patch_dir="${repo_dir}/native/fideslib_stage0/patches"
  B300_PATCH_PATHS=()
  case "${profile}" in
    full)
      B300_PATCH_PATHS+=(
        "${patch_dir}/fideslib-v2.1.0-bootstrap-stage-sync.patch"
        "${patch_dir}/fideslib-v2.1.0-b300-ciphertext-lifetime-sync.patch"
        "${patch_dir}/fideslib-v2.1.0-b300-keyswitch-stage-sync.patch"
      )
      ;;
    bootstrap-lifetime)
      B300_PATCH_PATHS+=(
        "${patch_dir}/fideslib-v2.1.0-bootstrap-stage-sync.patch"
        "${patch_dir}/fideslib-v2.1.0-b300-ciphertext-lifetime-sync.patch"
      )
      ;;
    lifetime)
      B300_PATCH_PATHS+=(
        "${patch_dir}/fideslib-v2.1.0-b300-ciphertext-lifetime-sync.patch"
      )
      ;;
    none) ;;
    *)
      echo "invalid B300 sync profile: ${profile}" >&2
      return 2
      ;;
  esac
  B300_PATCH_PATHS+=(
    "${patch_dir}/fideslib-v2.1.0-linear-transform-api.patch"
    "${patch_dir}/fideslib-v2.1.0-conjugate-api.patch"
    "${patch_dir}/fideslib-v2.1.0-ckks-data-type-api.patch"
  )
  if ((cuda_major >= 13)); then
    B300_PATCH_PATHS+=(
      "${patch_dir}/fideslib-v2.1.0-cuda13-graph-api.patch"
      "${patch_dir}/fideslib-v2.1.0-cuda13-cccl-include.patch"
    )
  fi
  local patch_path
  for patch_path in "${B300_PATCH_PATHS[@]}"; do
    if [[ ! -f "${patch_path}" ]]; then
      echo "missing B300 patch: ${patch_path}" >&2
      return 2
    fi
  done
}

b300_patchset_sha256() {
  local repo_dir="$1"
  local profile="$2"
  b300_select_patches "${repo_dir}" "${profile}"
  {
    printf 'platform=%s\n' "${B300_PLATFORM_VERSION}"
    printf 'fideslib=%s\n' "${B300_FIDESLIB_COMMIT}"
    printf 'cuda=%s\n' "${B300_CUDA_VERSION}"
    printf 'sync=%s\n' "${profile}"
    local patch_path
    for patch_path in "${B300_PATCH_PATHS[@]}"; do
      printf '%s ' "$(basename "${patch_path}")"
      sha256sum "${patch_path}" | cut -d' ' -f1
    done
  } | sha256sum | cut -d' ' -f1
}

b300_assert_equal() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "B300 ${name} mismatch: expected ${expected}, got ${actual}" >&2
    return 2
  fi
}

b300_image_id() {
  local image="$1"
  docker image inspect --format '{{.Id}}' "${image}"
}
