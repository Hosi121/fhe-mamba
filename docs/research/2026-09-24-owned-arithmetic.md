# Reusing private ciphertext operands in both models

The [static audit](2026-09-24-static-waste-audit.md) identified copies of
already-private arithmetic buffers. This experiment implements a shared
ownership-aware operation, verifies exact encrypted arithmetic and compares
the previous and modified executables on both trained model paths.

| Complete workload | Baseline seconds | Candidate seconds | Reduction | Decision |
| --- | --- | --- | --- | --- |
| Mamba-3 | 976.483173 | 949.257737 | 2.79% | Adopt |
| Mamba-2 | 1959.302697 | 1942.987097 | 0.83% | Adopt |

Each complete mode has one fresh-process/key sample, measured baseline then
candidate after the short interleaved controls. This is native evaluation time,
including per-operation plaintext encoding and upload. Context/key setup, payload
export, input loading and external transport are outside that timer. The models
have different sizes, weights and error contracts, so this table does not rank
architectures.

[Raw runs, failed probes, source archives, hashes and comparison script](../../results/dgx/2026-09-24/owned-arithmetic/).

## Mechanism

`owned_arithmetic.hpp` consumes scratch handles by value. A uniquely owned
input is reused; a live alias is cloned before level alignment can mutate it.
If both inputs reference the same dying object, detaching the first leaves
the second safe to consume. The completion callback runs before release of
the right input so queued GPU reads cannot outlive its storage.
`fideslib_owned_arithmetic.hpp` shares add/subtract/multiply dispatch while each
model keeps its existing alignment and synchronization policy.

Mamba-3 passes final-use transform, replicated-linear and Chebyshev accumulator
terms to this helper under the existing `--inplace-ops` option. Shared rotation
babies and polynomial bases keep their original protection. This removes the
two input clones previously made by each accumulator addition. Mamba-2's
normalization and joint-gate adapters already cloned both inputs; the helper
reuses that private storage instead of requesting another result allocation
and copy. General Mamba-2 aligned operators remain available to their other
callers. No new tuning switch is introduced.

The pinned backend's allocating and in-place entry points execute the same
GPU arithmetic. Addition order, polynomial coefficients, input protection,
level alignment, moduli and refresh arithmetic are unchanged. Synchronization
needed for lifetime safety is retained; it is not removed to improve a timer.

## Matched controls

Both models run baseline/candidate/candidate/baseline in separate processes.
The baseline is the previously built executable, revalidated by hash; the
candidate has its own immutable source archive and binary. Compiler, backend
commit and local patches, installed libraries/headers, payload and options are
fixed. Only seven native source/test/build paths differ between snapshots.

| Prefix | ABBA samples (s) | Baseline → candidate mean (s) | Reduction |
| --- | --- | --- | --- |
| Mamba-3 | 14.382654, 14.031546, 14.073827, 14.521491 | 14.452072 → 14.052687 | 2.76% |
| Mamba-2 | 54.141712, 53.752613, 54.010517, 54.233496 | 54.187604 → 53.881565 | 0.56% |

Because the first Mamba-2 difference was small, four reversed controls were
declared before either full pair completed and run after them. Their BAAB
samples are `[53.568722199, 54.15096133, 54.366535799, 54.06051165]` seconds. That block gives
54.258749 → 53.814617
seconds (0.82%). Across all four samples per
mode, means are 54.223176 →
53.848091 seconds
(0.69%). Final Mamba-2 adoption
requires improvement in this reversed block, the combined mean and the full
pair; the decision is **adopt**.

The predefined full-run qualification is a prefix mean reduction exceeding
0.5%. Mamba-3 uses CPUs 15–19, four OpenMP threads, borrowed upload, coefficient
moves, NAF, final-use reuse, compact weights, optimized routing and the existing
64-entry mask cache. Mamba-2 uses CPUs 0–19, eight cache workers and
`CUDA_LAUNCH_BLOCKING=1`; borrowed upload and GPU NTT are on, coefficient moves
stay off. Their existing architecture-specific configurations are not unified
by changing cryptographic or numerical requirements.

### Mamba-3

The 12-layer, five-evaluation request retains IDs `[315,279,1614,315]`,
`The capital of the state of`. The candidate uses
**91,956 ownership-aware additions**, reuses
**183,912 input buffers** and performs
**0 defensive helper clones**.
The addition count matches the preceding static inventory exactly. The other
existing scratch and DAG-final-use paths remain separate counters.

| Full-run measurement | Baseline | Candidate |
| --- | --- | --- |
| Evaluation seconds | 976.483173 | 949.257737 |
| Host encoding seconds | 170.079547 | 169.527331 |
| Upload seconds | 23.144668 | 23.244351 |
| Bootstrap seconds | 514.716264 | 515.042541 |
| Peak RSS GiB | 27.314224 | 27.314308 |
| Maximum exact error | 0.000134531921 | 0.000139599176 |
| Maximum polynomial error | 0.000106559681 | 0.00012519189 |

Both runs retain 726 bootstraps,
50,780 model rotations excluding
refresh, 50,913 encodes and
10,783 cache hits. Counts, refresh schedule,
level/depth parameters, weights and both 0.001 gates are unchanged.

### Mamba-2

The 24-layer, five-evaluation request retains IDs `[273,253,4687,273]`,
`The capital of the Republic of`. The helper executes
**151,315 calls**, reusing 302,630 already-private
operands and adding 0 defensive helper clones.
Thus 151,315 backend result copies are removed. The two input
protection clones at each original normalization/joint-gate call remain.
This count covers additions, subtractions and normalization as well as the
8,280 joint multiplications identified by the earlier static lower bound.

| Full-run measurement | Baseline | Candidate |
| --- | --- | --- |
| Evaluation seconds | 1959.302697 | 1942.987097 |
| Bootstrap seconds | 1082.142805 | 1083.239932 |
| Peak RSS GiB | 37.060959 | 37.060768 |
| Maximum polynomial error | 0.0274292557 | 0.00365089172 |

All operation counts, per-token counts, CKKS levels, selected IDs and the 0.05
polynomial gate are unchanged. Intermediate diagnostic decryptions remain zero.
The 0.56% prefix difference is small; a single full pair does not establish
statistical significance or a universal improvement.

Phase timers can overlap or nest and are not a disjoint decomposition. Raw
maximum RSS is process-wide. No kernel bandwidth or isolated allocation-time
claim is inferred from the model timers.

## Correctness, failed checks and reproducibility

The native exactness probe checks 96 cases in each configured context: real
Mamba-3 and complex Mamba-2, four equal/unequal level pairs, scalar-scaled or
unscaled inputs, add/subtract/multiply and four alias patterns. All **192 cases
match every RNS output coefficient and metadata exactly**, and all original
live inputs remain unchanged. Each probe exercises 96 consumed and 96 cloned
inputs. The encrypted synthetic carried-state regression also passes.

The first probe build failed on mutable FIDESlib handle-reference requirements;
the corrected build uses named mutable handles. The first GPU probe then
incorrectly required repeated decrypted values to be bit-identical. The pinned
OpenFHE REAL CKKS decoder adds Gaussian noise on each decoding; repeated decoding
of the same ciphertext confirmed the difference. The corrected acceptance check
compares the ciphertext itself, with zero tolerance, and records repeat-decoding
as a diagnostic. No decoder behavior or model error gate was relaxed. Both
failed revisions, logs and source manifests are preserved alongside the passing
revision. These were probe/harness failures, not hidden model speed samples.

The initial full Mamba-3 candidate passed numerical and selected-token gates
in 947.589282 seconds against a 978.468744-second baseline. Its coverage check
failed: 91,716 consumed additions versus 91,956 predicted. The implementation
had left the non-BSGS radix-routing accumulator on the old protected-copy path.
A source inventory confirms precisely 240 additions there. That call site now
uses the same consuming helper; the final synthetic, prefix and complete
Mamba-3 comparisons were rerun on a separately frozen executable. The initial
measurements and failed coverage check are preserved under
`initial-routing-coverage/`. They are not substituted for the final comparison.
Mamba-2 and the shared helper/probe source and binaries did not change.

The Python suite passes **286 tests at 87.60% coverage** and includes all
**18 C++ contracts**. The ownership contract checks storage identity, aliased
live-outs through mutating alignment, two dying aliases, identical operands,
right-input lifetime through completion, and null rejection before effects.

Per-file manifests bind both compiled snapshots. Every native run records its
binary, source manifest, input, command, environment and raw-output hash. The
postflight check rereads Mamba-2's complete payload and all frozen sources,
dependencies and binaries after GPU timing; Mamba-3 inputs were also hashed
before each run. The portable comparator checks identical architecture-specific
parameters, operations and levels, then reproduces decisions and timings.
Reporting decodes actual selected IDs and validates generation artifacts.
Collection and postprocessing hooks save durable completion records and submit
a local desktop notification. The starting Git commit alone is not treated as
the identity of the uncommitted measured candidate.

## Separate encoder follow-up inventory

While GPU binaries stayed frozen, a local static mask scan confirmed 4,488
unique public-weight diagonals across 22,440 encodes. Saving one inverse-FFT
complex-double vector per identity would require 2.1914 GiB, versus 52.5938 GiB
for expanded RNS at illustrative level 21. This is a storage calculation, not
an implemented cache or a speed result.

An extracted Encode range scan also replaces per-component logarithms with a
single logarithm after finding the largest absolute component. It preserves
scaled bytes and the range result in 6,356 finite-input local cases, including
neighbors of powers and subnormals. A local Intel x86-64 microbenchmark gives
about 3.51× for that isolated scan. A separate
[actual-encoder CPU study](2026-09-25-encoding-range.md) then checks
167,772,160 integer coefficient words and metadata exactly and measures
2.31–3.92% shorter full plaintext construction across four fixtures. Neither
local measurement establishes DGX or model performance. The production
model/library code does not include this prototype. See
[the detailed follow-up and all raw local samples](../../results/dgx/2026-09-24/owned-arithmetic/next-encoding-candidates.md)
for proposed exact-RNS gates and unsupported cases.

The subsequent [preparation design study](2026-09-25-preparation-design.md)
also follows the actual diagonal access order: five identical 4,488-entry
cycles. Below the full working-set size, ordinary LRU has zero identity hits;
a fixed 2-GiB subset can reuse 16,384 preparations. These are static counts,
not a new cache or speed result. That study records the CPU metadata,
registry-concurrency and buffer-retirement constraints for the next experiment.

Scope remains DGX Spark GB10, one frozen prompt, inline client and
`security=not-set`. This does not validate other prompts, long sessions or a
production security setting. The finite campaign ledger includes the failed
initial probe and every timed control. Persistent encoded-weight caching and
asynchronous preparation remain separate, unimplemented candidates.

Recorded DGX native process wall time is 9092.525 seconds within
the 3-hour internal review interval.
