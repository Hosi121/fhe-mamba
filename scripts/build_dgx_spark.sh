#!/usr/bin/env bash
# Build in an isolated prefix; never patch the existing DGX installation.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/config/dgx-spark.env"
ROOT="${FHEMAMBA_REMOTE_ROOT:-$HOME/fhemamba}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-${SPARK_CUDA_VERSION}}"
export PATH="$CUDA_HOME/bin:$PATH"
SOURCE="${FIDESLIB_SOURCE_DIR:-$HOME/fhe-deps/FIDESlib}"
OPENFHE_PREFIX="${OPENFHE_PREFIX:-$HOME/fhe-deps/openfhe-fides}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SPARK_DIR="$ROOT/spark"
mkdir -p "$SPARK_DIR"
exec 9>"$SPARK_DIR/build.lock"
flock -n 9 || { echo 'Another Spark build is running' >&2; exit 2; }
[[ "$(uname -m)" == aarch64 ]] || { echo 'Spark requires aarch64' >&2; exit 2; }
cuda_release="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
[[ "$cuda_release" == "$SPARK_CUDA_VERSION" ]] || { echo 'CUDA release mismatch' >&2; exit 2; }
[[ -f "$OPENFHE_PREFIX/lib/OpenFHE/OpenFHEConfig.cmake" ]] || {
  echo "OpenFHE is required at $OPENFHE_PREFIX (see docs/dgx-spark.md)" >&2; exit 2;
}
patch_names=(
  bootstrap-stage-sync b300-ciphertext-lifetime-sync b300-keyswitch-stage-sync
  linear-transform-api conjugate-api ckks-data-type-api cuda13-graph-api cuda13-cccl-include
  s2c-first-bootstrap
)
patches=()
for name in "${patch_names[@]}"; do
  patches+=("$REPO_DIR/native/fideslib_stage0/patches/fideslib-v2.1.0-${name}.patch")
done
patch_hash="$(cat "$REPO_DIR/config/dgx-spark.env" "${patches[@]}" | sha256sum | cut -d' ' -f1)"
snapshot="$SPARK_DIR/source-${patch_hash:0:16}"
if [[ ! -d "$snapshot" ]]; then
  temp_snapshot="${snapshot}.tmp.$$"
  trap 'rm -rf -- "$temp_snapshot"' EXIT
  git clone --no-hardlinks --no-checkout "$SOURCE" "$temp_snapshot"
  git -C "$temp_snapshot" checkout --detach "$SPARK_FIDESLIB_COMMIT"
  for patch in "${patches[@]}"; do
    git -C "$temp_snapshot" apply --check "$patch"
    git -C "$temp_snapshot" apply "$patch"
  done
  git -C "$temp_snapshot" diff --binary | sha256sum | cut -d' ' -f1 > "$temp_snapshot/.source-diff"
  mv "$temp_snapshot" "$snapshot"
  trap - EXIT
fi
[[ "$(git -C "$snapshot" rev-parse HEAD)" == "$SPARK_FIDESLIB_COMMIT" ]]
[[ "$(git -C "$snapshot" diff --binary | sha256sum | cut -d' ' -f1)" == "$(cat "$snapshot/.source-diff")" ]]
prefix="$SPARK_DIR/install-${patch_hash:0:16}"
cmake -S "$snapshot" -B "$SPARK_DIR/fideslib-${patch_hash:0:16}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCUDA_PATH="$CUDA_HOME" -DFIDESLIB_ARCH="$SPARK_FIDESLIB_ARCH" \
  -DOPENFHE_INSTALL_PREFIX="$OPENFHE_PREFIX" -DFIDESLIB_INSTALL_PREFIX="$prefix" \
  -DFIDESLIB_INSTALL_OPENFHE=OFF -DFIDESLIB_COMPILE_TESTS=OFF \
  -DFIDESLIB_COMPILE_BENCHMARKS=OFF
cmake --build "$SPARK_DIR/fideslib-${patch_hash:0:16}" --target install -j "$BUILD_JOBS"
cmake -S "$REPO_DIR/native/fideslib_stage0" -B "$SPARK_DIR/kernel" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_PREFIX_PATH="$prefix;$OPENFHE_PREFIX" \
  -Dfideslib_DIR="$prefix/share/fideslib/cmake" -DFHE_STAGE0_BUILD_TESTS=ON \
  -DFHE_STAGE0_GPU_RNS=ON -DFHE_STAGE0_GPU_DUAL_RING=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$SPARK_FIDESLIB_ARCH"
cmake --build "$SPARK_DIR/kernel" -j "$BUILD_JOBS"
ctest --test-dir "$SPARK_DIR/kernel" --output-on-failure
python3 "$REPO_DIR/experiments/manage_dgx_build.py" write \
  --root "$ROOT" --fideslib-prefix "$prefix" --openfhe-prefix "$OPENFHE_PREFIX" \
  --patch-set-sha256 "$patch_hash"
