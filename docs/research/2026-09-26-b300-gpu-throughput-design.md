# B300 GPU throughput: feed larger work and avoid intermediate materialization

This is a **design investigation**, using the completed inverse-FFT study's
records and the exact archived FIDESlib source. No new inference implementation,
GPU experiment, dependency rebuild or remote job is performed. The current
baseline remains the optional inverse-FFT plan, **234.84228 s** mean evaluation,
with unchanged experimental security, accuracy and refresh settings.

The next bounded GPU experiment should first test **one RNS expansion launch
across all active moduli**, preserving the existing completion/lifetime boundary.
It offers a concrete reduction in host submissions without changing CKKS
arithmetic. For larger gains, the execution design must also let CPU preparation
and GPU consumption overlap. A bandwidth-only reduction in this small loader
cannot explain a large full-model speedup.

[Derived records](../../results/b300/2026-09-26/gpu-throughput-design/analysis.json)
bind their inputs by SHA-256. The one-off analyzer has been retired; equations
and assumptions below explain the derived counts. Use the
[shared evidence tools](../experiments.md) to verify the retained records.

## What the existing measurements establish

The **pre-inverse-FFT-optimization** Nsight Systems prefix records 308,734 GPU
kernels, 310,311 launch API calls, 54,601 device synchronizations and only
2.54483 s covered by the union of GPU kernel/copy/memset intervals in a
15.77573 s instrumented evaluation. CPU samples point to encoding work.
This supports investigating gaps in GPU work supply. Instrumentation overhead
and CPU/GPU overlap mean these values are neither a current full-run utilization
measurement nor an additive wall-time partition.

Four selected Nsight Compute launches show:

| Metric | Observed range |
| --- | ---: |
| DRAM throughput / peak | 0.23–11.72% |
| SM throughput / peak | 42.49–48.15% |
| Achieved occupancy | 44.19–44.80% |
| Waves per SM | 0.79–0.95 |
| Duration | 25.79–32.29 microseconds |

Source inspection resolves `NTT_MODE=1` as **`NTT_RESCALE`**. These are already
fused rescale/NTT kernels, not generic standalone plaintext NTTs. The first two
launches have grids `(32,44,1)` and `(64,44,1)`, blocks of 128 and 64 threads,
and 38 and 42 registers/thread. Their theoretical occupancies are 75% and
62.5%, respectively. L1/TEX throughput is 67.00% and 53.87%, despite low DRAM
throughput. Low HBM usage alone therefore does not distinguish instruction
dependencies, shared-memory work, occupancy and work-supply limits.

The definition of memory throughput depends on the measured memory unit;
inspect its breakdown rather than interpreting every percentage as HBM use.
[Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#metrics-guide).
The GPU's reported waves below one also do **not** mean fewer than one block
per SM: they refer to waves of concurrently resident blocks.

Blackwell Ultra's published HBM bandwidth is up to **8 TB/s per GPU**. This is
distinct from PCIe host transfers, NVLink between GPUs and the SSH/SSM route.
The study uses one GPU; summing eight GPUs' bandwidth does not accelerate this
unchanged serial request. SKU and sustained bandwidth must be measured rather
than inferred from a maximum specification.
[NVIDIA Blackwell Ultra description](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era).

## Candidate 1: batch RNS expansion, then consider NTT input fusion

The reachable path is `PlaintextPreparation::load` → `load_plaintext` →
`expand_plaintext_rns` → `gpu->c0.NTT(16, true)`. The compact loader uploads one
coefficient vector and then launches a separate expansion kernel for each
active modulus. Each launch has only 128 blocks for N=32,768, with 256 threads
per block. The measured GPU has 148 SMs. A two-dimensional grid over
coefficient blocks and moduli can expose all of that independent work in one
launch while preserving contiguous accesses within each modulus.

The existing `compact_rns_saved_host_bytes` counter sums `(L_i - 1) * N * 8`.
Since each compact encode is uploaded once in these records:

```text
current expansion launches = saved_host_bytes / (8*N) + compact_encodes
batched expansion launches = compact_encodes
```

| Workload | Existing expansion launches | Proposed all-modulus launches | Reduction in this kernel's launches |
| --- | ---: | ---: | ---: |
| Profiled prefix | 82,303 | 3,771 | 95.42% |
| Full current candidate | 1,368,344 | 75,712 | 94.47% |

The prefix derivation exactly matches the CUDA trace. The full figure is a
counter/source derivation, **not a full traced launch count**. It is not a
94% reduction in all GPU work or model latency. The relevant source is
[`fideslib_plaintext_encoder.cpp`](../../native/fideslib_stage0/src/fideslib_plaintext_encoder.cpp)
and [`fideslib_plaintext_rns.cu`](../../native/fideslib_stage0/src/fideslib_plaintext_rns.cu).

Preserve one-device/full-packing/degree-one admission, the exact signed residue
conversion, unsupported-layout fallbacks and final completion semantics.
Launch on a known producer stream and join its completion into the existing
NTT dependency chain. The initial version can retain the current synchronization
boundary; removing it is a separate lifetime change.

A subsequent **RNS expansion + first NTT stage** fusion could load the compact
source directly into NTT shared/register storage, convert it to the destination
modulus there, and avoid materializing the expanded coefficient polynomial.
The backend already has a similar input-modulus conversion for `NTT_RESCALE`,
but that mode's output also rescales; it cannot be reused unchanged for this
new operation. The [subsequent implementation study](2026-09-26-b300-rns-pipeline.md)
found that its exported first half can be paired with `NTT_NONE` for the second
half, preserving the ordinary NTT output without a new backend mode.

That fusion could remove `2 * sum(L_i) * N * 8` logical intermediate read/write
bytes: **717,406,339,072 bytes** for the full request. Dividing by the published
8 TB/s peak gives just **0.08968 s**. This is an ideal traffic calculation,
not a physical-DRAM measurement or speed prediction: cache residency matters.
Its implication is that launch aggregation and removing serialized boundaries
are more promising than treating those bytes alone as the bottleneck.

The current full-run upload/NTT timer averages **19.67122 s out of 234.84228 s**.
Under fixed remaining costs, eliminating that entire category would save
**8.38%**; a batching-only change can remove only a subset. Both fusion and
extra register use can alter neighboring costs, so this is a conditional scope
bound, not a prediction.

Small validation should compare all integer residues before and after NTT
against the current loader at representative levels and both ring sizes,
including negative coefficients, boundary values and fallback cases. A
primitive win must also survive a cold prefix with unchanged numerical and
token gates before any full ABBA follow-up. Include metadata preparation and
upload/NTT completion in timing, not only the new kernel duration.

## Candidate 2: keep the GPU supplied with prepared public operands

The current full run still spends **114.37202 s (48.70%)** in host encoding.
Faster GPU kernels cannot remove CPU work that remains serial on the request's
critical path. Independent public masks/diagonals inside a linear operation
provide a possible preparation queue: CPU workers prepare future operands
while the single GPU owner submits already prepared batches.

Use worker-local encoder/scratch state and initialized immutable tables;
the existing GPU registry and shared preparation counters are not established
as thread-safe. Keep level/scale keys exact. Only precompute final coefficients
when their actual level/degree are known; level-independent mask/FFT work is
a narrower speculative preparation boundary. Start within one known-level
BSGS group rather than reordering client feedback or arbitrary DAG nodes.

Bound queue bytes and use reusable pinned transfer buffers. Submission can
group multiple independent plaintexts, adding a plaintext axis to the already
batched modulus axis. This addresses small work units when few RNS moduli
remain. GPU consumers and the CPU producer require explicit ownership until
the last relevant transfer/kernel completes.
[CUDA transfer/overlap guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-and-overlapping-transfers-with-computation).

GPU-resident prepared public weights are a related compute-for-memory trade:
reuse exact prepared values instead of redoing CPU encoding and upload. The
earlier final-coefficient cache regressed because levels changed; simply
making that cache larger is not a justified speed claim. First inventory the
current actual `(weight, diagonal, context, level, scale, degree)` keys and
saved time per resident byte. Level-independent transformed-input reuse remains
another boundary. Include all first-use preparation in cold-request timing.

This route has a larger possible scope than the loader alone, but its benefit
is unmeasured. It requires queue-wait/producer/consumer timing and peak residency
to determine whether preparation overlaps useful device work or merely moves
the same stall elsewhere.

## Candidate 3: reduce NTT shared-memory traffic and dependency stalls

The active Shoup NTT's inner butterfly stages load and store shared memory and
use `__syncwarp()` for stages within a warp. The source also contains disabled
alternative code. A register/shuffle implementation of those stages could
reduce shared-memory round trips without changing integer modular results.
This is an actual microkernel candidate; simply requesting more HBM bandwidth
does not remove these on-chip accesses.

Before selecting it, collect shared-memory bank conflicts, load/store wavefronts,
integer instruction mix, eligible warps and stall reasons for the current binary
at both low and high levels. Check whether extra registers reduce occupancy
enough to offset the saved shared operations. Preserve lazy-reduction bounds,
Shoup products and all NTT fusion modes in exact output checks. This is not
evidence that the existing kernels are currently bank-conflict limited.

## Existing fusion and graphs constrain the plan

FIDESlib already defines NTT rescale, plaintext multiplication, moddown and
key-switch-dot fusion modes. The source inspected here is bound by the prior
B300 backend manifest; those operations must not be proposed as new fusions.
`LimbPartition::multPt` contains a graph path with `capture=false`, a static
map keyed only by limb count and varying input pointers. Flipping that flag
does not establish a reusable, correct graph across contexts and buffers.

A useful graph scope would be a repeated, preallocated arithmetic block with
stable buffers or explicitly updated kernel arguments. Device-wide
synchronization and synchronous default-stream copies inside capture are
incompatible with this approach.
[CUDA graph capture constraints](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#prohibited-and-unhandled-operations).
The existing B300 ciphertext destructor barrier fixes a last-reader lifetime
race; replace it only with a complete producer/consumer event and pool-reuse
contract. The upload queue can initially retain that guard.

The practical order is a small all-modulus RNS trial, then an independently
qualified preparation/consumption queue if a larger scope is desired. NTT
register/shuffle work should be selected by a fresh profile of the current
inverse-FFT binary. No new speedup is claimed by this design investigation.
