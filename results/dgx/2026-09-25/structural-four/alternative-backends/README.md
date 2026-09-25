# Alternative representation probes

These implement candidates 3 and 4 from the [four-candidate study](../../../../../docs/research/2026-09-25-structural-four.md).
They are isolated experiments, not enabled in the Mamba executor. Both report
`security=not-set`; Cheddar's upstream client is for testing only. No weights
or secret keys are distributed here.

## 32-bit composite RNS (Cheddar)

The selected nominal 59-bit profile **fails** the 1e-6 numerical gate. Its
64-bit-word control passes. The additional upstream 40-bit 32-bit control also
passes, but is not a matched precision/depth model candidate. A failed 32-bit
run normally exits 3; do not treat its timing as a speedup.

Requirements used: DGX Spark GB10 (CUDA architecture 121), CUDA 13.0, CMake,
C++17 and GMP headers/library. The build patch pins RMM and preserves the
full path returned by `find_library(GMP)`. The recorded installation extracted
`libgmp-dev` into a private directory and linked the existing system GMP
library; it did not modify system packages. Consult `../cheddar-configure/run.json`
for those exact paths and compiler flags. For a normal GMP development install:

```bash
# Run from the repository root; choose a new, disposable build directory.
probe_sources="$PWD/results/dgx/2026-09-25/structural-four/alternative-backends"
probe_work="$(mktemp -d)"
git clone https://github.com/scale-snu/cheddar-fhe.git "$probe_work/cheddar"
git -C "$probe_work/cheddar" checkout 8df8b26ce5411a68b68f0e7c2fb7e9a2f05f3e94
git -C "$probe_work/cheddar" apply "$probe_sources/cheddar/build.patch"
cp "$probe_sources/cheddar/composite_probe.cpp" \
   "$probe_sources/cheddar/composite_params.hpp" \
   "$probe_sources/cheddar/stock32_params.hpp" "$probe_work/cheddar/"
cmake -S "$probe_work/cheddar" -B "$probe_work/build" \
  -DBUILD_UNITTEST=OFF -DBUILD_TESTS=OFF -DUSE_GMP=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
cmake --build "$probe_work/build" --target composite_probe -j 6
OMP_NUM_THREADS=4 taskset -c 15-19 "$probe_work/build/composite_probe" \
  64 "$probe_work/composite64.json"
OMP_NUM_THREADS=4 taskset -c 15-19 "$probe_work/build/composite_probe" \
  32 "$probe_work/composite32.json"
OMP_NUM_THREADS=4 taskset -c 15-19 "$probe_work/build/composite_probe" \
  32 "$probe_work/stock32.json" --stock32
```

Run commands independently: the expected exit 3 must not prevent the control
from running. The patch's architecture 121 targets Spark; a different GPU
requires a different build and is outside the recorded measurement.
`generate_composite_params.py` deterministically regenerates
`cheddar/composite_params.hpp`. `stock32_params.hpp` retains the parameter
arrays from the pinned upstream `parameters/bootparam_40.json`; its MIT notice is in
`cheddar/LICENSE`. `../cheddar-dependency-pins.json` records resolved RMM,
Thrust and spdlog revisions. `pins.json` and `source-manifest.json` bind the
selected dependencies and adapter source.

`--diagnose` decrypts retained subcircuit outputs on the test client after the
complete encrypted circuit, never using them as evaluator inputs. It changes
timing and is not used for the final paired qualification. Its log localizes
the 59-bit 32-bit failure to at least the first ciphertext square.
`failed-default-key-level.cpp` records an earlier adapter error, before explicit
rotation-key levels repaired `QSize mismatch`; it is not the final probe.

## Smaller ordinary ring (Lattigo)

The CPU probe computes a square at N=32,768, refreshes in N=65,536 through an
encrypted ring switch, returns to the smaller ring and squares again. The
N=65,536 control uses identical residual primes and refresh parameters. Both
pass 1e-6 across four logical widths, including padding; both take about 55 s
per refresh boundary. The CPU path is rejected for model integration. This
is not a benchmark of GPU ring switching.

```bash
# Run in this directory. Go 1.24+ is required; recorded compiler: go1.26.8.
cd small-ring
ring_probe_work="$(mktemp -d)"
GOTOOLCHAIN=go1.26.8 CGO_ENABLED=0 go build -o "$ring_probe_work/probe" .
taskset -c 15-19 "$ring_probe_work/probe" --logn 15 --output "$ring_probe_work/ring15.json"
taskset -c 15-19 "$ring_probe_work/probe" --logn 16 --output "$ring_probe_work/ring16.json"
```

For an x86-64 build host and the recorded ARM64 target, add
`GOOS=linux GOARCH=arm64` and transfer the executable to Spark. `go.mod` and
`go.sum` pin Lattigo v6.2.0 and transitive modules. The initial default K=16
configuration fails for both dense-secret controls; its source is in
`small-ring-attempt1/`. The qualified source uses continuous cosine modular
reduction, K=512, degree 127, six double-angle steps, no sparse ephemeral
secret, a 28-bit reserved correction prime and the same 59-bit scale.

Preserve generated reports and source identities, then remove the temporary
build directories and executable when finished. The original campaign's
cleanup record distinguishes retained measured binaries from removed trees.
