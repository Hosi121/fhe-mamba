# B300 RNS input fusion and bounded CPU preparation

This study tests two changes in order against the retained inverse-FFT baseline:
RNS expansion batching/input-NTT fusion, then one CPU producer inside a replicated
BSGS linear group. **CPU preparation is adopted as an opt-in path**: its full
ABBA mean is **234.84 → 202.33 s (13.84%)**, or **50.58 s/generated token** for
this four-token request. RNS fusion is not adopted: its separate full block is
168.46 s baseline versus 228.83 s fused. The retained CPU pipeline therefore
uses the original per-limb RNS loader. CPU frequency/placement variation across
blocks prevents treating these numbers as a universal fastest configuration.

## Fixed comparison

The workload is the same trained Mamba-3 SISO 187M checkpoint, 12 layers, five
encrypted evaluations and four generated tokens for `The capital`. The expected
IDs are `[315, 279, 1614, 315]`. The ordinary ring is 32,768, the refresh ring
65,536, depth 44, scale 59, first modulus 60, and both error gates are 0.001.
The two-pass S2C-first refresh, model polynomials, public weights and inverse-FFT
plan remain fixed. No evaluation decryption is introduced. The experimental
`security=not-set` configuration is unchanged; this is not a new security claim.

Runs use B300 GPU 4, CPU affinity 64–68, NUMA 2, and `OMP_NUM_THREADS=4` in the
same CUDA 13.0.1 image. Each timed arm is a fresh process with its own keys.
Setup and post-evaluation validation are outside native `eval_seconds`; complete
container wall time is also retained. A local build never overlaps a timed job
on the B300 host. Completion hooks serialize builds, probes and comparisons.

The first stage uses full ABBA. Its rejection means the second stage also uses
ABBA against the original inverse-FFT baseline. The predeclared plan would have
used original / stage1 / integrated / integrated / stage1 / original had fusion
passed. Prefix admission and full adoption require a mean reduction greater
than 0.5%. All samples, including failures and variable runs, are retained.

## RNS batching and fusion

The original loader uploads one compact coefficient vector, launches a signed
RNS lift once per active modulus, then runs the existing NTT. The batching
variant uses a two-dimensional grid to expand all moduli in one launch. It
preserves the integer reciprocal/remainder arithmetic.

The fused variant reuses two already exported FIDESlib kernels. The first half
of `NTT_RESCALE` broadcasts the compact source and switches its signed residue
to each destination modulus while loading the NTT tile. The second half uses
`NTT_NONE`, so the rescale epilogue never runs. The intermediate expanded
coefficient polynomial is not written and read back. Each compact upload uses
two NTT launches over all active moduli. No dependency patch or extra key is
needed for these kernel halves.

Both paths retain the single-device, full-packing, degree-one admission and
compact coefficient range bound. At most 64 contiguous 64-bit limbs are batched;
unsupported layouts use the per-limb loader. Noncompact inputs keep the prior
borrowed/staged path. The existing synchronization boundaries remain in place,
including completion before the shared compact source may be overwritten.
The device source pointer table is initialized only when its allocation grows.

This corrects one conservative assumption in the preceding design note: the
entire `NTT_RESCALE` operation cannot be used unchanged, but its first half can
be paired with the ordinary second half. Exact integer comparisons test that
composition directly, including positive/negative values, rounding/range
boundaries, small packing, level 44 and degree-two fallback.

Initial probes cover Mamba-2 complex/sparse-ternary and Mamba-3 real/uniform-ternary
parameters at both ring sizes. All eight processes pass 240 exact RNS/metadata
cases and nine encrypted arithmetic cases each: 1,920 exact cases and 72 encrypted
cases in total. The encrypted tolerance remains 1e-6. Ordinary-ring Mamba-3
warm upload medians select fusion over batching (0.12757 versus 0.14550 ms).
These primitive medians are selection evidence, not model latency.

The first prefix ABBA is `12.4250 / 12.0896 / 8.4361 / 12.2886` seconds. The
second candidate also has much faster unchanged CPU encoding, so the apparent
16.95% mean reduction cannot all be attributed to RNS fusion. Upload/NTT time
falls in both candidate runs. The full ABBA is `170.5472 / 228.9167 / 228.7400 / 166.3695` seconds:
168.4584 s baseline versus 228.8284 s fused, a **35.84% regression in this block**.
Fusion is not promoted. CPU encoding in the first baseline takes 63.31 s versus
114.32/114.69 s in the candidates. The inverse-FFT symbol and patched static
archive are retained in both executables. Read-only observations on the later
candidate/baseline show the main thread on core 67 at roughly 2.3 GHz and core
64 at roughly 4.6 GHz respectively; their allowed affinity and policy match.
The observations start late in the block and do not prove the cause of all
variation. The observer also misses main-thread mappings when its single PID
lookup precedes container startup; it does not measure the producer thread.
Its frequency samples include setup and are not an effective-cycle integral.
No governor, CPU clock, GPU clock or host policy was changed. This experiment
does not establish an intrinsic slowdown of fused
RNS arithmetic, but it also does not support a full-model speedup. The second
stage therefore keeps the established per-limb RNS loader.

## CPU preparation overlap

`--prefetch-plaintexts` creates one CPU producer for a replicated-BSGS linear
group after all baby ciphertext levels are known. It constructs masks and
encodes coefficients while the consumer performs upload, multiplication,
addition and rotation in the original order. All CPU plaintext buffers belong
to the produced item. FIDESlib wrapping, registry access, GPU work and serial
counters stay on the evaluator thread.

The producer reads immutable CPU parameters and FFT tables initialized before
thread creation. It uses the same coefficient encoder and floating-point
arithmetic as the serial path. Cache-admissible masks retain the original
serial lookup/encoding policy; dense public-weight diagonals can be prepared
ahead. A level check guards adoption. No arbitrary DAG-level prediction or
plaintext cache expansion is involved.

The queue has capacity two, with at most two ready/in-production items plus
one item being consumed. It blocks the producer when full, propagates errors
and joins before captured weights, levels or encoders can die. The reported
maximum item bytes counts its mask and coefficient array, not process RSS or
encoding scratch. Host encoding time now overlaps GPU work and must not be
added to other timers to reconstruct wall time.

The queue contract tests cover ordering, backpressure, move-only ownership,
producer exceptions, exhaustion and cancellation. Address/undefined-behavior
sanitizers also pass the queue test; this does not claim sanitizer coverage of
the CUDA backend. Final-source exact probes additionally run CPU preparation
concurrently with GPU consumption and the stock CPU oracle. Final-source serial
and prefetch controls pass 2,160 exact cases and 81 encrypted arithmetic cases
across nine processes, including an OpenMP-one-thread control. Combined with
stage one, this is 4,080 exact cases and 153 encrypted cases.

The second-stage prefix ABBA is `12.5033 / 10.4160 / 10.3744 / 8.6636` s,
mean **10.5834 → 10.3952 s (1.78%)**. Both candidate samples are close, but
baseline placement/frequency variation persists. The full ABBA completes as
**234.1951 / 202.6348 / 202.0350 / 235.4805 s**, giving **234.8378 → 202.3349 s
(13.84%)**. Unlike the prefixes and first-stage block, both full baselines and
both candidates are close within this block. This supports adoption for the
tested configuration, not statistical significance or a platform-wide guarantee.
The first full candidate prepares **51,240 coefficients in 240 groups** ahead,
with peak ready count two and maximum item size **393,216 bytes**. Up to three
such payloads add at most **1.125 MiB**, excluding encoding scratch, allocator
retention and backend buffers. Observed total peak RSS is about **26.29 GiB**.
The pipeline's maximum exact-reference error is **4.71227e-5**, and its maximum
polynomial-reference error is **6.66590e-7**, both below the unchanged 0.001
gates. Complete container wall times, including setup and validation, average
**342.55 → 300.69 s**. The 50.58 s/token headline divides evaluation time by
four; it excludes setup and is not a per-step decode latency distribution.

## Critical-path interpretation

The new path changes when public operands are prepared; it does not reduce
the number of encrypted operations or weaken their parameters. All full runs
retain 78,369 host encodes, 75,712 compact uploads, 484 bootstraps and the same
token IDs. The pipeline prepares 51,240 encodes (65.38% by count) in 240 linear
groups; 27,129 remain on the serial paths. These are counts, not CPU-time shares.

The same-block node timers localize most of the observed gain:

| Whole-node wall time | Baseline mean | Pipeline mean | Reduction |
| --- | ---: | ---: | ---: |
| `linear` + `linear_ref` | 101.0845 s | 72.0496 s | 28.72% |
| Evaluation outside those nodes | 133.7534 s | 130.2852 s | 2.59% |
| Complete evaluation | 234.8378 s | 202.3349 s | 13.84% |

Of the 32.50 s reduction, 29.03 s (89.33%) occurs in the changed linear nodes.
That is consistent with overlap helping its intended region. Host encoding
service time only falls from 114.36 to 110.28 s, while upload time increases
from 19.67 to 22.06 s. Adding or subtracting those overlapping timers would
misattribute the gain. Node times also contain some refresh; the mean refresh
time of 44.23 s must not be added to them as a disjoint category.

For one group, let `e_i` be CPU preparation service and `c_i` consumer work
from taking item `i` until the next take, including intervening additions and
rotations. Let `P_i` be preparation completion, `S_i` the consumer start and
`D_i` its completion. A capacity-`q` queue with the implemented admission rule
has this ideal recurrence, with nonpositive indices equal to zero:

```text
P_i = max(P_(i-1), S_(i-q)) + e_i
S_i = max(P_i, D_(i-1))
D_i = S_i + c_i
```

Thus `D_n >= max(sum(e_i) + c_n, e_1 + sum(c_i))`. Even unlimited buffering
cannot beat the slower service stream or remove the group's fill/drain cost.
This model holds services fixed and omits locking, thread creation, scheduling
and resource contention. The aggregate measurements do not provide each
`e_i`/`c_i`, so they cannot predict the optimal queue capacity.

The current mean time inside `pop()` is **33.39 s**. If every other cost stayed
fixed and that entire interval vanished, evaluation would still take at least
**168.94 s**: a conditional **16.50%** additional reduction. This is an upper
bound for removing that waiting category alone, not a prediction for a faster
encoder; `pop()` includes locking/wakeup overhead. The approximately 110 s
encoding timer is not another 110 s removable from the 202 s wall time.

The next useful questions follow from those bounds:

- Separate producer computation, producer backpressure and consumer waiting
  per group before increasing queue capacity. More buffering can absorb bursts
  but cannot increase sustained preparation throughput.
- Test whether public, level-independent mask work can begin during preceding
  GPU work, including refresh. The current implementation starts only after
  baby levels are known and joins at each linear-group boundary. Extending
  this window requires an explicit level/scale and buffer-lifetime contract;
  it is not implemented by this study.
- Revisit fusion with the main thread held to the same physical core in both
  arms and a separately identified producer placement. The existing identical
  allowed mask does not enforce identical thread placement. Retain all samples
  and actual clock observations instead of rescaling timings by guessed clocks.

The unchanged baseline alone ranges from **166.37 to 235.48 s** across these
blocks, a 1.415× ratio. A favorable historical baseline must not be compared
directly against a later candidate, nor should the negative fusion block be
discarded. RNS fusion removes logical intermediate traffic and submissions,
but its full-model benefit remains unestablished under this variability.

The retained analysis records these values and input hashes. The one-off
analyzer has been retired; the conditional bounds above remain separate from
measured latency and physical HBM throughput.

## Diagnostic Nsight confirmation

Three fresh prefixes are profiled after all uninstrumented controls finish.
They confirm dispatch and work counts independently of the adoption timings:

| Prefix trace | Original | RNS fusion | CPU pipeline |
| --- | ---: | ---: | ---: |
| GPU kernels | 308,734 | 219,621 | 308,734 |
| Standalone RNS expansion launches | 82,303 | 0 | 82,303 |
| NTT launches | 137,090 | 130,280 | 137,090 |
| Device synchronizations | 54,601 | 54,601 | 54,601 |
| GPU activity interval union | 2.548 s | 2.389 s | 2.465 s |
| Instrumented evaluation | 14.219 s | 13.826 s | 10.956 s |

Fusion removes all standalone expansion launches plus 6,810 NTT launches by
covering the active moduli together: 28.86% fewer GPU kernels in this prefix.
That dispatch reduction is established even though the full adoption gate
fails. The pipeline retains the same GPU kernel and synchronization counts;
its elapsed-time gain is consistent with supplying the same work sooner.

The GPU activity union covers 17.92% of the original instrumented interval and
22.50% with the pipeline. This is the union of CUPTI-visible kernel/copy/memset
intervals, not SM occupancy, DRAM bandwidth or a full-run utilization estimate.
One instrumented sample per arm, CPU placement variation and profiling overhead
prevent using these times as a second latency comparison. No new Nsight Compute
bandwidth or stall measurement is claimed here. The earlier selected-kernel
Compute records remain in the preceding inverse-FFT study.

The [evidence bundle](../../results/b300/2026-09-26/rns-pipeline/) contains raw
reports, exact commands, negative results, completion events, source archives,
hashes and offline analyzers. The full suite passes 292 Python tests; a focused
25-test runner check includes two subsequently added invalid-option cases.
All 22 CPU native contracts pass. GPU validation totals 4,080 exact cases,
153 encrypted primitive cases and 19 successful model reports, including the
three instrumented prefixes.

## Use and boundaries

Build with `FHE_STAGE0_GPU_RNS=ON` and the architecture matching the target GPU.
The RNS bridge now uses CUDA C++20 to include the pinned FIDESlib NTT headers.
Use one of `--batch-plaintext-rns` or `--fuse-plaintext-rns-ntt`; both imply
compact GPU RNS preparation. Add `--prefetch-plaintexts` for the bounded CPU
producer. The portable packed runner forwards and records these flags. All
new paths remain opt-in.

The shared preparation bridge and primitive probes cover both model parameter
families. The full-model study covers the packed Mamba-3 executor on one frozen
prompt. It does not establish a full Mamba-2 gain, a Spark gain, or arbitrary
prompt accuracy. Logical avoided intermediate bytes are not a physical HBM
traffic measurement.
