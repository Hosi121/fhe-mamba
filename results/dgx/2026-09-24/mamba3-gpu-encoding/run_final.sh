#!/usr/bin/env bash
set -euo pipefail
root=/home/kataiwa/fhemamba/mamba3-20260924
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64
python3 "$root/run_ntt_probe.py" "$root" gpu-ntt-probe-final packed_plaintext_probe-gpu-ntt-final
python3 "$root/run_encoding_comparison.py" "$root" packed_fideslib-gpu-ntt-final lm-layer1 gpu-encoding-layer --modes c --timeout 300 --profile
python3 "$root/run_encoding_comparison.py" "$root" packed_fideslib-gpu-ntt-final payload gpu-encoding-synthetic --modes c --timeout 180 --profile --cache
python3 "$root/run_encoding_comparison.py" "$root" packed_fideslib-gpu-ntt-final lm-layer1-prefix gpu-encoding-final-prefix --modes a,c,c,a --profile
python3 "$root/run_encoding_comparison.py" "$root" packed_fideslib-gpu-ntt-final lm-full gpu-encoding-full --modes a,c --timeout 2400 --profile
