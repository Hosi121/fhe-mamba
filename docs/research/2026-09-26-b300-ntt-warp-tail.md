# B300 forward NTT register/shuffle tail — 2026-09-26

**Do not adopt this candidate.** Moving the last five forward-NTT butterfly
stages from shared memory into registers passes exactness checks, but the
matched small ABBA mean rises **8.71157 → 8.91241 s (2.31% slower)**. It fails
the predefined greater-than-0.5% improvement gate, so no full-model comparison
is run. The previously qualified CPU preparation pipeline remains the baseline.

This is one implemented GPU mechanism, followed by diagnosis of its negative
result. It does not establish a general limit on GPU optimization. The
[experiment and patch](../../experiments/ntt_warp_tail/README.md),
[raw evidence](../../results/b300/2026-09-26/ntt-warp-tail/README.md) and offline
verifier retain the attempted mechanism and decision.

## Mechanism and its bound

The pinned FIDESlib forward NTT uses shared-memory Cooley–Tukey butterflies.
After butterfly distance 32, each warp owns 64 coefficients per row. For
distances 16, 8, 4, 2 and 1, the candidate keeps two values per thread for each
of four rows in registers. Lane `tid` exchanges one value with `tid XOR m`,
then applies the existing butterfly with the same twiddle and Shoup quotient.
An upper lane replaces its low value; a lower lane replaces its high value.
Final values are stored at `2*tid` and `2*tid+1` in the existing shared layout.

The fast path requires 64-bit Shoup arithmetic and at least 32 threads/block.
Sub-warp blocks and other arithmetic types retain the original loop. All
32 lanes participate in each exchange. The initial warp memory barrier remains;
shuffle is not a replacement for the shared-memory visibility requirement.
[CUDA shuffle semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).

For this five-stage coefficient subset, the source-level shared loads/stores
fall from **80 to 16 uint64 words/thread**. This is neither an 80% reduction
in all NTT traffic nor measured HBM traffic. Earlier NTT stages, twiddle loads,
global inputs/outputs, butterfly arithmetic and shared-memory allocation remain.
Register pressure and exchange/select instructions can erase the benefit.
The current full-request NTT critical-path fraction is not measured here, so
there is no full-model speed prediction from that byte count.

## Correctness and build isolation

| Check | Result |
| --- | ---: |
| Independent CPU lane/ownership calculation | 192 cases pass |
| Candidate against separately linked original GPU kernels | 720 cases pass |
| Exact GPU output coefficient words compared | 289,415,168 |
| Integrated RNS/metadata checks, Mamba-2/3 patterns and both deployed rings | 960 cases pass |
| Encrypted primitive checks | 36 cases pass |
| Uninstrumented prefix processes | All 4 pass accuracy and token gates |
| Instrumented prefix processes | All 4 pass accuracy and token gates |

The direct oracle covers rings 1,024/2,048/4,096/32,768/65,536; six forward
fusion modes (`NONE`, `RESCALE`, `MULTPT`, `MODDOWN`, `KSK_DOT`, `KSK_DOT_ACC`);
both halves; 1/4/44 moduli; and zero, modulus-minus-one, alternating and mixed
public inputs. Every compared word uses a golden stream written by the
unmodified executable. Secondary outputs that should remain untouched are
also checked. Both unchanged sub-warp and changed full-warp paths are covered.

The build uses FIDESlib `cd171f20f510eeca04c71d7b0034ef073829f761`, whose 91
core source hashes match the earlier retained archive, and OpenFHE 1.4.2
`aa391988d354d4360f390f223a90e0d1b98839d7`. Both arms are built at the same
source/build/install paths with the same toolchain. Only `src/NTT.cu` changes
in the backend; archive-member comparison finds only `NTT.cu.o` and the
resulting CUDA device-link object differ. Native objects are shared and relinked
against the separate backend archives. The 99-file measured source snapshot,
compiler/linker commands and executable hashes are retained.

The first native probe build failed because the API and internal namespaces
both define feature enum names. Explicit `fideslib::` qualification fixes that
compile error. Its failed job and controller attempt remain in the evidence;
all qualification and timing use the successful `native-r2` source snapshot.

## Matched small comparison

| ABBA process | Evaluation (s) |
| --- | ---: |
| `prefix-0-base` | 8.706181909 |
| `prefix-1-candidate` | 8.966841044 |
| `prefix-2-candidate` | 8.857975468 |
| `prefix-3-base` | 8.716956811 |

The unchanged baseline is rebuilt with the adopted CPU prefetch and optional
inverse-FFT plan, plus the original per-modulus RNS upload. Both arms retain
the same prefix/payload, ordinary ring 32,768, refresh ring 65,536, depth 44,
59-bit scale, 60-bit first modulus, two-pass S2C-first refresh, lifetime
barriers and experimental `security=not-set`. Both numerical tolerances remain
0.001; generated IDs remain `[6864, 6864]`, with zero evaluation decryptions.
Every process performs 16 bootstraps and 2,562 prefetched encodes in 12 groups,
with at most two ready items. No precision or cryptographic setting is relaxed.

Both use B300 GPU 4, SM103 code, the same CUDA 13.0.1 image, CPU mask 64–68,
NUMA 2 and four OpenMP threads. No host clock/governor change is made. Builds
finish before any timing, and GPU jobs run serially through a completion hook.
There are only two samples/arm on one prefix: no statistical-significance,
full-depth, full Mamba-2 or other-platform performance claim follows.

### CPU placement control and its limitation

Both arms add the same **measurement-only** producer-affinity patch and set
`OMP_PROC_BIND=true`, `OMP_PLACES={64},{66},{67},{68}`. The main thread is
observed on CPU 64 after initialization. The producer requests CPU 65, but
OpenMP rebinds it to CPU 64 when entering a serialized parallel region.
An independent diagnostic in the same container reproduces the transition:
`65` after `pthread_setaffinity_np` and the OpenMP setters, then `64` after
the serialized region. The observer likewise finds new short-lived threads
on CPU 64, not a verified isolated producer on CPU 65.

This attempted producer-isolation control did not take effect. It is disclosed
rather than treated as a successful separate-core experiment. Both variants
use the same binary-side control/environment, and the negative GPU diagnosis
is also supported by independent kernel measurements. Do not compare this
prefix against the historical 202.33-second full run, or turn this measurement
patch into an adopted CPU optimization. A future affinity experiment must
validate placement after OpenMP initialization before timing.

## Nsight explains the tradeoff

Nsight Compute captures four matching `NTT_RESCALE` launches from fresh
prefixes, with 44 active moduli. The first/second halves use blocks 128/64
and grids `(32,44,1)` / `(64,44,1)`. Means below use two launches/half.

| Metric | First half: base → candidate | Second half: base → candidate |
| --- | ---: | ---: |
| Kernel duration (µs) | 32.096 → 33.904 | 26.448 → 27.504 |
| Registers/thread | 38 → 48 | 42 → 48 |
| Theoretical occupancy | 75% → 62.5% | 62.5% → 62.5% |
| Achieved occupancy | 44.73% → 43.81% | 44.60% → 43.72% |
| L1/TEX throughput / peak active | 67.30% → 49.78% | 53.99% → 33.32% |
| Derived local spill requests | 0 → 0 | 0 → 0 |

Shared allocation is unchanged. Fewer shared accesses come with more live
registers and exchange/select work. The first half loses theoretical residency;
all four matched launches take longer. These measurements support the
register-pressure/instruction-cost explanation, but do not isolate each
contributor causally. Low DRAM throughput does not imply that replacing
shared-memory traffic will speed up an instruction-bound dependency chain.

Nsight Systems records **308,734 kernels in each prefix**. Summed kernel
duration rises **2.94075 → 2.98414 s**; recorded GPU activity interval union
rises **2.38710 → 2.44610 s**. These sums can overlap and instrumented runtime
is excluded from the adoption comparison. Profiling occurs after all small
ABBA samples, so replay/instrumentation does not contaminate that gate.

## Decision and handoff

Keep the kernel patch under `experiments/ntt_warp_tail/`; do not apply it to
the normal FIDESlib dependency or change the adopted CPU pipeline. No second
mechanism or full comparison is attempted after the failed small gate. A
future NTT redesign should account for register residency and butterfly
dependency cost together, and prove a small-model gain before full testing.

The evidence bundle retains samples, the negative decision, source and binary
identities, profiler summaries and the CPU-affinity diagnostic. Large golden
streams, executables and raw profiler recordings are identified by hash and are
not bundled as public measurements. See the [experiment workflow](../experiments.md)
for the current publication format.
