# Mamba-3 microkernel investigation

This measurement follows the [24.95-minute full run](../mamba3-depth-batching/).
It measures local kernels and short encrypted workloads separately; its
microkernel ratios are not full-model speedups.

- `mask-abba.json`, `mask-baab.json`, `mask-comparison.json`: all timed CPU
  samples, allocation included, and median comparisons on Cortex-X925 CPU 15.
- `bench_mask.cpp`: old scalar-modulo loop and the production contiguous-copy
  helper, plus bitwise parity. Compile from the repository root:
  `g++ -O3 -std=c++20 -Inative/fideslib_stage0/include results/dgx/2026-09-24/mamba3-microkernels/bench_mask.cpp -o /tmp/bench_mask`.
  Run `taskset -c 15 /tmp/bench_mask abba` and `taskset -c 15 /tmp/bench_mask baab`
  on the measured Spark (CPU numbering is host-specific).
- `copy_dispatch.cpp`, `copy-dispatch.txt`, `*-assembly.txt`: resolved glibc
  dispatch and emitted code. The nonoverlapping large-copy path uses 128-bit
  NEON `ldp/stp q` instructions; small-copy paths also contain SVE instructions.
  The source has no architecture-specific intrinsics or relaxed FP arithmetic.
- `prefix-payload/`, `prefix-derivation.json`: input manifest, reference fixture
  and exact recipe for the 127-node prefix.
- `prefix-profile/`: 127-node trained prefix, preserving the existing 0.001
  exact/polynomial gates; Nsight wrapper, not native executable, is hashed by
  the generic runner. `provenance.json` binds the actual native executable.
- `profile-packed`: exact Nsight invocation. It captures CUDA evaluation only;
  setup and reference decryption are outside the capture range. Its absolute
  paths are the measured host's paths and must be adjusted for reproduction.
- `micro-profile*.nsys-rep.gz`, `trace-digests.json`: complete compressed Nsight
  captures and hashes, retained so the summaries can be regenerated.
- `profile_cuda_*.csv`: raw Nsight Systems summaries. CUDA API wait times
  overlap device execution and must not be added to kernel times.
- `profile-sources.tar.gz`: source of the initial profiled executable.
- `compiled-sources.tar.gz`, `compiled-sources.json`: source of the later
  executable with optional scratch-value in-place operations.
- `prefix-a1/`, `prefix-b1/`, `prefix-b2/`, `prefix-a2/` and
  `inplace-comparison.json`: matched short-prefix ABBA and input/binary hashes.
- `full-prefix/`, `layer1/`, `synthetic/`: wider numerical validation; the first
  covers all 12 layers once, the second carries one layer through three
  evaluations and two selected tokens, and the last checks four mixer steps
  including carried state.
- `inplace-profile/`, `inplace-profile_cuda_*.csv`: second trace with private
  scratch reuse enabled; instrumentation is excluded from ABBA timings.
- `generation-a1/`, `generation-b1/`, `generation-b2/`, `generation-a2/` and
  `generation-comparison.json`: one-layer, three-evaluation client-generation
  ABBA; both modes use the affinity recorded in `generation-affinity.json`.
- `budget.json`: all 24 campaign attempts, including the earlier full baseline,
  failed one-pass probe, setup and profiling, share the original 7200 s cap.
- `arm-contract.log`: native bit-copy and routing contract on the target ARM CPU.
- `checks.log`, `native-tests.log`: complete local checks after the
  microkernel changes, including the 15 native CPU contracts.

The CPU mask permutation preserves every coefficient bit. In-place GPU
operations may mutate only newly cloned scratch ciphertexts. Original DAG
values, kernel arithmetic, operand order and synchronization are preserved.
All encrypted probes retain the feasibility context's `security=not-set`.

The campaign used 6926.77 of its original 7200 seconds. No GPU job remains
running. Reproducible large derived prefixes and transient profiling exports
were removed after preserving manifests, derivation recipes, hashes and
compressed captures. Frozen native executables and source archives remain.
