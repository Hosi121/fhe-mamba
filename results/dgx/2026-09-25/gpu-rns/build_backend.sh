#!/usr/bin/env bash
set -euo pipefail
root=/home/kataiwa/fhemamba/gpu-rns-20260925
export PATH=/usr/local/cuda-13.0/bin:$PATH
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64
finish() {
  local code=$?
  python3 - "$root/build-completion.json" "$code" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1])
t = p.with_suffix('.tmp')
t.write_text(json.dumps({'exit_code': int(sys.argv[2]), 'finished_at_unix': time.time()}) + '\n')
t.replace(p)
PY
}
trap finish EXIT
cmake -S "$root/source" -B "$root/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCUDA_PATH=/usr/local/cuda-13.0 \
  -DFIDESLIB_ARCH=121-real -DOPENFHE_INSTALL_PREFIX=/home/kataiwa/fhe-deps/openfhe-fides \
  -DFIDESLIB_INSTALL_PREFIX="$root/install" -DFIDESLIB_INSTALL_OPENFHE=OFF \
  -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF
cmake --build "$root/build" --target install -j 6
