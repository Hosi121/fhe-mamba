# Exact inverse-FFT plan qualification

The [OpenFHE patch](../../native/fideslib_stage0/patches/openfhe-v1.4.2-inverse-fft-plan.patch)
precomputes the original inverse-FFT twiddle lookup sequence once per public
geometry. It preserves butterfly operations, bit reversal, normalization,
CKKS scaling and rounding. Unsupported plan geometries use the original loop.
It caches no input values or encoded weights.

Use separate baseline and patched installations of the pinned **OpenFHE 1.4.2,
64-bit, math backend 4** dependency. Preserve the existing FIDESlib integration
patches. Apply this additional patch to an isolated source checkout, then rebuild
and install OpenFHE into a new prefix. Rebuild FIDESlib and the native executors
against that prefix: they link OpenFHE statically, so changing
`LD_LIBRARY_PATH` alone cannot enable the optimization. Keep the same compiler,
flags and OpenMP configuration for both builds; the measured code uses no
`fast-math` or machine-specific ISA flags.

For example, starting from an isolated copy of the existing FIDESlib-patched
OpenFHE source (including its populated submodules), build the additional patch
into a new installation. Replace the example paths with absolute paths:

```bash
git -C /path/to/isolated/openfhe apply \
  "$(pwd)/native/fideslib_stage0/patches/openfhe-v1.4.2-inverse-fft-plan.patch"
cmake -S /path/to/isolated/openfhe -B runs/openfhe-ifft \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/patched \
  -DNATIVE_SIZE=64 -DMATHBACKEND=4 -DWITH_NATIVEOPT=OFF -DWITH_OPENMP=ON \
  -DBUILD_STATIC=ON -DBUILD_SHARED=ON -DBUILD_UNITTESTS=OFF \
  -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build runs/openfhe-ifft --target install -j4
```

Build these probes once against each installation:

```bash
cmake -S experiments/openfhe_ifft -B runs/ifft-base \
  -DCMAKE_BUILD_TYPE=Release -DOpenFHE_DIR=/path/to/base/lib/OpenFHE
cmake --build runs/ifft-base -j4
cmake -S experiments/openfhe_ifft -B runs/ifft-candidate \
  -DCMAKE_BUILD_TYPE=Release -DOpenFHE_DIR=/path/to/patched/lib/OpenFHE
cmake --build runs/ifft-candidate -j4

runs/ifft-base/fft_probe write runs/fft-golden.bin runs/fft-base.json
runs/ifft-candidate/fft_probe compare runs/fft-golden.bin runs/fft-candidate.json
runs/ifft-base/encoding_probe write runs/encoding-golden.bin runs/encoding-base.json
runs/ifft-candidate/encoding_probe compare runs/encoding-golden.bin runs/encoding-candidate.json
```

The FFT probe checks **249 cases**, including all powers of two through 32,768,
signed zero, subnormals, large finite values, infinity, a NaN payload and an
uncached geometry. The encoder checks **251,658,240 integer coefficient words**
plus metadata in real and complex contexts, rings 32,768 and 65,536, full and
partial packing, three levels and two scale degrees. All 192 successful encodes
and 48 expected small-scale errors must agree. Golden streams need about
1.9 GiB of temporary disk space and are compared word-for-word, including EOF.

`fft_probe bench unused OUTPUT.json` measures warm inverse-FFT calls after
copying each input into fresh scratch. `encoding_probe bench unused OUTPUT.json`
measures full coefficient-root plaintext construction. Use independent ABBA
processes with fixed affinity for either benchmark. These timings do not
replace cold, trained-model comparisons; FFT-table construction is included
in the model's setup time.

The B300 study retains its exact commands, static-archive overlay build,
source hashes, profiler summaries and model controls under
[`results/b300/2026-09-26/nsight-ifft/`](../../results/b300/2026-09-26/nsight-ifft/).

## GPU coefficients

Enable `-DFHE_IFFT_GPU_PROBE=ON` to build `gpu_fft_probe` (CUDA required).
The probe uses the same
[`GpuSpecialInverseFFT`](../../native/fideslib_stage0/src/gpu_special_fft.cu)
as the native plaintext encoder. It compares 204 CPU/GPU transforms bitwise,
153 rounded coefficient arrays, and 51 expected small-scale rejections over
powers of two through 65,536 slots. The optional shared-memory tail is measured
separately and is not the executor's default.

```bash
cmake -S experiments/openfhe_ifft -B runs/gpu-fft \
  -DCMAKE_BUILD_TYPE=Release -DOpenFHE_DIR=/path/to/openfhe/lib/OpenFHE \
  -DFHE_IFFT_GPU_PROBE=ON -DCMAKE_CUDA_ARCHITECTURES=103
cmake --build runs/gpu-fft --target gpu_fft_probe -j4
runs/gpu-fft/gpu_fft_probe runs/gpu-fft/result.json
```

Use the architecture for your GPU. Build the native executor with
`FHE_STAGE0_GPU_RNS=ON`, then qualify its full RNS/NTT path with
`packed_rns_probe OUTPUT --classical128 --ring 131072 --compact-addends --gpu-fft`
and again with `--mamba2`. The packed model runner accepts
`--gpu-plaintext-rns --gpu-plaintext-fft`. The FFT path is opt-in, supports up
to 65,536 slots with full packing on one GPU, and uses the CPU encoder for unsupported
scale degrees, ranges, levels or sparse packing. It retains the original
rounding order and uses no reduced-precision arithmetic.

CPU producers still prepare masks; encoding and device registration run on
the evaluator thread. Rounded coefficients remain on the GPU through RNS
expansion and NTT. `gpu_fft_encodes`, `gpu_fft_fallbacks` and `gpu_fft_seconds`
report usage and total GPU preparation wall time. The legacy
`host_encoding_seconds` field measures time spent in encoding calls, including
this GPU path; it must not be interpreted as CPU-only time with this flag.
