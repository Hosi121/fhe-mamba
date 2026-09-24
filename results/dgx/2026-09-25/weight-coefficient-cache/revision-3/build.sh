#!/usr/bin/env bash
set -euo pipefail
root=/home/kataiwa/fhemamba/weight-preparation-20260925/revision-3
cmake -S "$root/source/native/fideslib_stage0" -B "$root/build" \
  -DFHE_STAGE0_BUILD_KERNEL=ON -DFHE_STAGE0_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_PREFIX_PATH=/home/kataiwa/fhemamba/spark/install-2a70798e869944af;/home/kataiwa/fhe-deps/openfhe-fides' \
  -Dfideslib_DIR=/home/kataiwa/fhemamba/spark/install-2a70798e869944af/share/fideslib/cmake
cmake --build "$root/build" -j 4 --target packed_fideslib weight_plaintext_probe
