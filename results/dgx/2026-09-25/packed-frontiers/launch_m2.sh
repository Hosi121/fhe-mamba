#!/usr/bin/env bash
set -euo pipefail
root=/home/kataiwa/fhemamba/layout-batch-20260925
source "$root/source/experiments/dgx_mamba2_common.sh"
init_dgx_mamba2_defaults
build_dgx_mamba2_args "$2" "$3"
exec "$BINARY" "${DGX_MAMBA2_ARGS[@]}" \
  --artifact-version "$ARTIFACT_VERSION" --repo-commit "$REPO_COMMIT" \
  --binary-sha256 "$BINARY_SHA256" --output-json "$1"
