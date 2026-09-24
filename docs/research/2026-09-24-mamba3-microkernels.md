# Mamba-3 SIMD mask copies and GPU scratch reuse

Replacing scalar coefficient permutation with contiguous copies gives a
**12.4× local speedup** at 32768 coefficients on DGX Spark's Cortex-X925.
Reusing private GPU ciphertext scratch values gives a smaller **1.8% reduction**
on a matched 127-node trained prefix. A separate one-layer generation-loop
ABBA measures **2.2% less time**, with two actual tokens matching in every run. These are separate measurements from the
[46.20 → 24.95 minute full-generation result](2026-09-24-mamba3-depth-batching.md).
The full five-evaluation run predates both microkernel changes.

[Raw measurements and source](../../results/dgx/2026-09-24/mamba3-microkernels/).

## SIMD path and correctness

Both replicated matrix BSGS and small-diagonal routing call
`rotate_plaintext_mask`. The old loop computed a modulo for every coefficient;
the replacement computes the split once and copies two contiguous spans.
It performs no floating-point arithmetic, preserving coefficient bits,
including negative zero and NaN payloads. Native contracts cover empty arrays,
short tails, positive/negative wraparound, and 32768-element masks.

The emitted code calls `memmove` twice. Runtime symbol resolution on the
measured aarch64 host selects glibc code containing NEON `ldp/stp q` for large
nonoverlapping copies and SVE instructions for small copies. This uses the
platform library's SIMD implementation without new intrinsics, a CPU-specific
build requirement, or fast-math. The CPU helper is shared with Mamba-2, but a
Mamba-2 encrypted runtime improvement was not measured here.

| Coefficients | Scalar median | Contiguous-copy median | Local speedup |
| ---: | ---: | ---: | ---: |
| 24 | 64 ns | 48 ns | 1.33× |
| 768 | 1136 ns | 112 ns | 10.14× |
| 1024 | 1376 ns | 144 ns | 9.56× |
| 32768 | 43168 ns | 3488 ns | 12.38× |

These are the ABBA medians, with 100 timed samples and 10 warmups per block.
A separate BAAB process reports 42608 → 3424 ns (12.44×) for 32768 coefficients.
Allocation is timed in both variants; output validation/destruction is outside
the timer. Both functions are noinline, so the compiler cannot specialize the
old modulo loop for a constant size. GCC 13.3.0 uses `-O3 -std=c++20`; CPU 15 is
a Cortex-X925 with the performance governor. No GPU run or target build ran
concurrently with the CPU benchmark.

## What the encrypted profile says

Nsight Systems 2025.3.2 captures evaluation through `cudaProfilerStart/Stop`.
The 127-node trained prefix passes the unchanged exact/polynomial 0.001 gates.
Instrumented evaluation takes 21.43 s; 888 host encoding calls consume 7.05 s
(32.9%). Current mask preparation takes only 0.031 s. The local 12.4× copy
result therefore does not imply a large whole-model gain.

The recorded CUDA kernel durations total about 9.47 s. The copy kernel alone
accounts for 1.99 s (21.0% of that total), and hoisted rotation/key switching
for 1.51 s. NTT/INTT variants account for another 3.23 s (34.1%). CUDA API synchronization time overlaps GPU execution; it must not
be added to those device times. The trace excludes key setup and reference
decryption and is a short-prefix profile, not a full-generation decomposition.

## Removing redundant GPU copies

`--inplace-ops` reuses ciphertexts already privately cloned for level alignment
in addition and multiplication. A composite rotation clones its source once,
then rotates that private result in place. It never mutates original DAG
values and retains the existing synchronization and temporary lifetimes.
The FIDESlib arithmetic kernels and operand order remain the same. No NTT,
modular multiplication or bootstrap kernel implementation is replaced.

A single executable switches this option on/off; both modes already include
the CPU mask improvement. Four independent processes run in ABBA order with
identical payload/context and unchanged operation counts (8 bootstrap calls,
181 ciphertext products, 1440 plaintext products and 1103 rotations):

| Run | In-place scratch | Evaluation (s) |
| --- | --- | ---: |
| A1 | No | 22.7495 |
| B1 | Yes | 22.0737 |
| B2 | Yes | 22.3786 |
| A2 | No | 22.5336 |

A second profile confirms that copy-kernel calls fall from 20454 to 15832
(exactly two calls for each of the 2311 avoided ciphertext clones). Their
recorded kernel time falls from 1.99 to 1.56 s. However, the instrumented
candidate's host encoding takes 8.71 s versus 7.05 s in the earlier profile,
and its total rises to 22.92 s. These sequential traces locate work and
verify call elimination; they do not establish a latency benefit by themselves.
Holding other work fixed, the 0.43 s device-copy saving corresponds to about
2.0% of the earlier 21.43 s evaluation. That conditional estimate explains
why eliminating thousands of copies still gives a modest total gain.

The means are **22.6416 → 22.2261 s (−1.835%, 1.0187×)**. Both candidate runs
remove 2311 explicit ciphertext clones. All four runs pass; the largest exact
error is 0.00002178. Two observations per mode establish a narrow measured
result, not a statistical guarantee across workloads or machines.

## Matched generation-loop comparison

To check whether the small prefix benefit carries into recurrent evaluation,
the same binary also runs a trained one-layer component through three
evaluations and two client-selected tokens. Both modes are pinned to
Cortex-X925 CPUs 15–19 with `OMP_NUM_THREADS=4`; the child's actual allowed CPU
list was verified. This controls core placement on the heterogeneous ARM CPU.
It is a separate matched comparison: its absolute times must not be compared
with earlier unpinned runs to infer the effect of scratch reuse.

| Run | In-place scratch | Evaluation (s) |
| --- | --- | ---: |
| A1 | No | 67.5636 |
| B1 | Yes | 66.1623 |
| B2 | Yes | 66.1234 |
| A2 | No | 67.7492 |

Means are **67.6564 → 66.1428 s (−2.237%, 1.0229×)**.
Every run passes both 0.001 error gates and selects `[6864, 6864]`. Each
candidate avoids 8662 explicit clones; arithmetic/refresh counts and
input/binary hashes match. The option remains opt-in because the complete
12-layer, five-evaluation session has not been repeated with it.

## Wider numerical checks

The same in-place executable also passes these gates, with the original
0.001 exact/polynomial tolerances:

| Scope | Evaluation (s) | Maximum exact error | Output check |
| --- | ---: | ---: | --- |
| All 12 layers, first evaluation | 265.48 | 0.00007770 | Final hidden vector |
| One trained layer, three evaluations | 76.46 | 0.00002282 | Both actual generated IDs `[6864, 6864]` match |
| Four-step synthetic mixer | 38.97 | 0.00000007723 | 20 output/state comparisons |

These are correctness checks and one-off timings, not matched performance
comparisons. The full 12-layer five-evaluation generation has not been rerun
with the microkernel changes. Local verification passes all 280 Python tests
and the 15 native CPU contracts; the routing/bit-copy contract also passes
on the target ARM CPU.

## Reproduction

Use the normal packed runner with the frozen payload and shared budget ledger.
`--inplace-ops` is opt-in. `--profile-evaluation` adds host timing and evaluation
capture markers. The retained `run_abba.py`, `run_validation.py` and Nsight
wrappers record the measured invocations; replace their absolute host paths
before reuse. Profiling overhead is excluded from the ABBA comparison.

The next larger target is host plaintext encoding, followed by the measured
NTT/key-switch/copy paths. Removing device synchronization blindly is unsafe:
these barriers currently protect plaintext and ciphertext buffer lifetimes.
The experiment retains the existing inline-client, `security=not-set` scope.

All 24 campaign attempts consumed 6926.77 of the original 7200-second limit,
including setup, failed probes and profiler postprocessing. No full-session
microkernel speedup is claimed from the component comparisons.
