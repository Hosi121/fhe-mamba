#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:?Pass the prepared experiment root on DGX Spark}"
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64
taskset -c 15-19 python3 "$ROOT_DIR/run_packed_probe.py" \
  --binary "$ROOT_DIR/packed_fideslib-cache" \
  --payload "$ROOT_DIR/payload" \
  --output "$ROOT_DIR/cache-synthetic" \
  --planned-refresh --batch-refresh --inplace-ops --cache-plaintexts \
  --profile-evaluation --timeout 63 \
  --budget-file "$ROOT_DIR/optimization-budget-20260924.json" --budget-seconds 7200
