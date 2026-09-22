#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${FHEMAMBA_REMOTE_ROOT:-$HOME/fhemamba}"
expected_binary="$ROOT/spark/kernel/stage1_mamba2_decode_fideslib"
BINARY="${BINARY:-$expected_binary}"
[[ "$BINARY" == "$expected_binary" ]] || { echo 'Use the validated Spark binary' >&2; exit 2; }
manager="$REPO_DIR/fhemamba/experiments/manage_dgx_build.py"
export LD_LIBRARY_PATH
LD_LIBRARY_PATH="$(python3 "$manager" library-path --root "$ROOT"):/usr/local/cuda-13.0/lib64"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
FIDESLIB_SYNC_PROFILE=full
INPUT_CHAIN="${INPUT_CHAIN:-$ROOT/payloads/mamba2-130m}"
source "$REPO_DIR/fhemamba/experiments/dgx_mamba2_common.sh"
init_dgx_mamba2_defaults
LAYERS="${LAYERS:-24}"
TOKENS="${TOKENS:-1}"
[[ "$LAYERS" =~ ^[1-9][0-9]*$ && "$TOKENS" =~ ^[1-9][0-9]*$ ]] || {
  echo 'Spark runner expects one positive layer count and token count' >&2; exit 2;
}
RUN_TAG="${RUN_TAG:-spark-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_JSON="${OUTPUT_JSON:-$RESULTS_DIR/m2_chain_${RUN_TAG}_l${LAYERS}_t${TOKENS}.json}"
mkdir -p "$RESULTS_DIR" "$ROOT/logs"
actual_sha="$(sha256sum "$BINARY" | cut -d' ' -f1)"
[[ "${BINARY_SHA256:-$actual_sha}" == "$actual_sha" ]] || { echo 'Binary hash mismatch' >&2; exit 2; }
build_dgx_mamba2_args "$LAYERS" "$TOKENS"
payload_sha="$(python3 "$manager" payload-hash --payload "$INPUT_CHAIN")"
[[ "${INPUT_CHAIN_SHA256:-$payload_sha}" == "$payload_sha" ]] || {
  echo 'Input payload changed after campaign validation' >&2; exit 2;
}
status=0
if [[ -f "$OUTPUT_JSON" ]]; then
  mkdir -p "$RESULTS_DIR/archive"
  old_sha="$(sha256sum "$OUTPUT_JSON" | cut -d' ' -f1)"
  mv -- "$OUTPUT_JSON" "$RESULTS_DIR/archive/$(basename "$OUTPUT_JSON" .json)-${old_sha}.json"
fi
"$BINARY" "${DGX_MAMBA2_ARGS[@]}" \
  --artifact-version "$ARTIFACT_VERSION" --repo-commit "$REPO_COMMIT" \
  --binary-sha256 "$actual_sha" --output-json "$OUTPUT_JSON" \
  > >(tee "$ROOT/logs/$(basename "$OUTPUT_JSON" .json).log") 2>&1 || status=$?
# Attach to failed numerical gates too; a failed experiment is still evidence.
if [[ -s "$OUTPUT_JSON" ]]; then
  python3 "$manager" attach --root "$ROOT" --artifact "$OUTPUT_JSON" --payload-sha256 "$payload_sha"
else
  echo "No backend artifact: $OUTPUT_JSON" >&2
  exit 1
fi
exit "$status"
