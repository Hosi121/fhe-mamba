# Reusing exact public-weight coefficients

**Decision: do not adopt.** The cold three-step trained workload regresses
**1.26%**, despite faster isolated encoding and exact coefficient parity.
The prototype is archived; the active implementation is restored to the
shared-ownership version published at `eb66247`.
This study tests one additional mechanism with unchanged cryptographic
parameters, polynomial coefficients, arithmetic order and numerical gates.

## Representation and admission

A normal OpenFHE encode remains the authority on a cold miss. The prototype cache reads
the coefficient-format result, recovers one signed representative per
coefficient from the first modulus, and verifies that expanding those words
reproduces **every word in every RNS limb**. Unsupported representations return
the ordinary plaintext. There is no new FFT, rounding rule or approximation.

An entry stores 65,536 signed 64-bit words, or 512 KiB, plus metadata. On a hit,
the shared encoder creates an independent CPU plaintext with identical scale,
level, noise degree, slots, data type and public packed values, then expands
its RNS limbs. Upload and GPU NTT follow the existing borrowed-upload path.
Returned plaintexts can be mutated without changing the retained coefficients.

The cache belongs to one immutable context and slot count. Its caller supplies
an identity for immutable value bits; the actual level is part of the key.
For the packed program, the identity combines the weight node and diagonal.
The parser enforces compatible dimensions on references, and the fixed context
makes the level determine the degree-one encoding scale. The cache is cleared
at each program boundary. Addends and higher noise degrees are outside this API.

In the archived prototype, `--weight-cache-mib 2048` enables a fixed-admission cache with a **2-GiB numeric
coefficient payload budget**. Metadata, allocator overhead, expanded CPU/GPU
plaintexts and other inference buffers are additional. The default is zero.
GPU plaintext NTT must be enabled. Only weights with at least two live uses
are eligible, so a single-use prefix does not pay admission and roundtrip costs.
Entries are retained for the request; ordinary LRU would thrash on the measured
cyclic access order described in the [preparation design](2026-09-25-preparation-design.md).

The prototype integration uses the architecture-neutral packed executor for
Mamba-3. The shared encoder is also checked with Mamba-2's complex configuration;
Mamba-2's separate weight-cache selector is not changed by this experiment.

## Correctness and the small encoder experiment

The exact probe exercises real and complex CKKS, 1/1,024/32,768 slots, levels
0/21/34/44, zeros, sparse negative masks, dense weights, tiny values and large
unsupported coefficients. It checks CPU coefficients and metadata, ordinary
CPU NTT versus GPU readback, level-key separation, mutation of returned miss
and hit buffers, fixed admission, and zero/sub-entry budgets. Each configuration
has 60 cases: 51 admit and reuse an entry, while nine take the verified fallback.

A local integer contract independently generates modular residues, including
positive/negative boundary values and a mismatch at the last word of the last
limb. The prototype passes 289 Python tests, including 19 C++ executables.
Restoring the production baseline also removes the unused prototype API,
flag and tests; the active suite retains its previous 286 tests.

The final probe also visits 16 actual trained-weight diagonals five times,
with baseline/candidate/candidate/baseline blocks in one process. The candidate
includes its first encode, admission check, all hit materializations and releases.
It records roughly 59–62% less **CPU encoding** time at levels 0, 21 and 34.
This is not a model result or an independent-process timing comparison; it
qualifies the mechanism for model measurements.

## Model comparison protocol

All native workloads use the same DGX Spark GB10, dependencies, CPU affinity
15–19 and four OpenMP threads. No build runs alongside GPU timing. Sources,
executables, libraries, payloads and scripts are bound by SHA-256 hashes.
The full-model baseline is the preserved ownership executable; the candidate
has a separate frozen source archive and build. No operation is moved outside
the evaluation timer to create a saving.

The sequence is: final exact probes; independent-process ABBA for the existing
one-layer single-evaluation prefix; ABBA for the existing one-layer three-step
client workload; carried-state synthetic validation. A repeated-weight prefix
mean improvement above 0.5% qualifies one fresh full baseline/candidate pair.
The full pair retains the 12-layer, five-evaluation workload and both `0.001`
error gates, and must preserve all four generated IDs and arithmetic counts.
A finite three-hour native-process review interval includes every attempt.

## Observed model result and rejection

| Workload | Baseline mean | Candidate mean | Change |
| --- | ---: | ---: | ---: |
| One layer, one evaluation; no reuse | 14.028651 s | 14.033876 s | 0.037% slower |
| One layer, three evaluations; two generated tokens | 47.117472 s | 47.711634 s | **1.261% slower** |

All four samples per workload are retained in ABBA order. The repeated-weight
samples are `[47.065730369, 47.726157777, 47.697110354, 47.169214618]` seconds.
The no-reuse samples are `[14.000514704, 14.021150466, 14.046602015,
14.0567873]`. These few samples do not establish statistical significance;
both repeated-weight candidate runs are slower than both baseline runs, and
the predeclared qualification rule is not met. **No full 12-layer pair is run.**

Both repeated-weight candidate runs have the same counters:

| Counter | Observed |
| --- | ---: |
| Cache misses / admitted entries | 825 |
| Cache hits | 297 |
| Requested weight preparations | 1,122 |
| Hit rate | 26.47% |
| Retained numeric coefficients | 412.5 MiB |
| Capacity bypasses / representation rejections | 0 / 0 |
| Actual host encodes | 2,266 versus 2,563 baseline |

The shared helper is reached and avoids encodes, but it has to store multiple
level-specific encodings of each immutable diagonal. Mean host preparation
increases **9.495001 → 9.871531 seconds**. The initial encode plus compression
and verification costs are not recovered by the 297 hits. Adding capacity
would not fix this observed case: its 412.5-MiB payload is already below the
2-GiB limit, and no entry is evicted or rejected. Peak process RSS remains
approximately 27.14 GiB because the peak occurs in other allocations; the
retained payload is not a process-memory measurement.

All model gates and operation counts pass. Both variants select IDs
`[6864, 6864]` on this **truncated one-layer** workload. The largest observed
CKKS-to-exact error is `0.0000229853`, below the unchanged `0.001` gate.
This is not full-backbone generation evidence. The carried-state synthetic
check also passes, and the single-use paths admit no weight entries.

The 59–62% isolated encoder reduction uses a fixed level across all five
cycles, with 64 hits and 16 misses. The model changes levels, so that
microbenchmark's hit ratio does not transfer. The earlier static inverse-FFT
identity count was explicitly independent of level; it must not be described
as a final-coefficient cache hit prediction.

The next preparation candidate is **reuse before scale and rounding**, such as
the inverse FFT. That intermediate can share across these level variants,
but needs its own integration, exact-RNS checks and cold model comparison.
It remains unimplemented. The negative result here narrows that design choice;
it is not a claim that every public-weight cache is unhelpful.

## Evidence and reproduction

[Raw runs, comparisons, source archives and failed attempts](../../results/dgx/2026-09-25/weight-coefficient-cache/)
are preserved. `compare.py` independently reconstructs the rejection from
native JSON and source/binary hashes. The baseline and all three prototype
source snapshots are included; `revision-3` is the final measured version.
This rejected option is intentionally absent from the active runner and binary.

An initial build lacked the new header's include directory; the CMake
dependency was corrected before GPU execution. Two initial Python failures
were stale expected command arguments; forwarding was correct, and the test
expectation was corrected. The first exact probes use revision 2; revision 3
adds the single-use admission guard and repeats both exact probes before
model timing. All failed and superseded records remain available.

The completion hook collected results and submitted a desktop notification.
Postflight checks revalidated 118 candidate source files, 115 baseline files,
the executables, dependencies and all four payloads. Native process wall time
for every probe and control is **561.510 seconds**, within the three-hour
internal review interval. Shared models and dependencies were reused.

Scope remains one frozen prompt, public weights, inline client and
`security=not-set`. No full-run performance result or statistical significance
is asserted for this rejected prototype.
