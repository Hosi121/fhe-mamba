# Long-generation numerical diagnosis and two optimization trials

The frozen 64-token request fails again at selection 24. Operator-level
diagnostics localize the abrupt SSM error to one two-input batch refresh in
the final layer. Both optimization candidates are implemented; neither is
promoted. The NTT candidate loses performance, and the indexed mask cache
fails one full-request numerical qualification after passing its short screen.

## Fixed conditions and evidence

The model, weights, prompt, polynomial coefficients and original program remain
fixed. The classical-128 profile is `N=131072`, `Q=2656`, `P=720`,
`QP=3376 <= 3523`, four HYBRID digits, uniform ternary secret and sigma 3.19.
Depth 44, scale 59, both S2C-first bootstrap passes, merged correction and the
256-handle admission policy are preserved. Final polynomial/exact hidden
errors must both be at most `0.001`, and generated token IDs must match.

The performance control already enables the previously adopted public-source
ciphertext reuse. The accuracy reproduction keeps the original failed run's
reuse setting. Diagnostic runs explicitly decrypt immutable ciphertext copies;
ordinary qualification runs permit no intermediate decryption. The generation
reporter rejects diagnostic results even if no observer callback fired.

The planned performance comparison uses two fixed GPU/CPU/NUMA placements in
opposite orders. Each pair must improve, the pooled evaluation reduction must
reach 2%, and all numerical/token/count gates must pass. Instrumented timings
and incomplete failed runs cannot meet that gate.

## Numerical diagnosis

The original failure, fresh uninstrumented reproduction and boundary diagnostic
all fail at selection 24. The new uninstrumented run changes polynomial error
from about `2.04e-6` at selection 23 to `0.0300652` at selection 24. The
instrumented run reaches about `0.011841` at the same selection. Fresh keys
change the magnitude; these observations do not establish a failure rate.

The diagnostic reconstructs references without refitting any polynomial. It
verifies all 75,970 prefix nodes against the original full program digest
`464380aef53bfaefe2b4f137a2777299f62e4eb660101bea4b3886ed790ad24e`.
For portable reference computation, the exporter can bind original refresh
bounds while tolerating tiny CPU reduction-rounding differences. Operators,
parents, widths and constants must match exactly; the final canonical prefix
hash still must match byte for byte. This does not modify the executed program.

At step 24, layer 11 (zero-based), the six state tiles have maximum errors of
approximately `1.00e-5`, `6.04e-6`, `4.36e-6`, `4.33e-6`, **0.08172**, and
`5.28e-6`. The bad tile is original node 75905. Its preceding carried state
is accurate at step 23; 32,366 of its 32,768 output slots exceed `0.001` at
step 24. The error then enters the mixer residual and final hidden vector.

The state update is

`S' = d S + (1-m) dt d (v_old outer k_old) + m dt (v outer k)`.

For each product, a conservative input-drift estimate is
`product(M_i + e_i) - product(M_i)`, using observed maximum input errors `e_i`
and CPU-reference magnitudes `M_i`. Summing the three terms gives about
`6.26e-5`, versus the observed `0.08172`. This floating-point estimate excludes
errors introduced by the internal encrypted operations and refresh; it is not
a certified interval proof. It motivates tracing those operations instead of
attributing the failure to high-degree RMSNorm amplification alone.

### First divergent operation

A further diagnostic observes all 524 original nodes in the final layer of
steps 23 and 24 (684 records including refresh events). It again fails at
selection 24, with final polynomial/exact errors `0.0344205 / 0.0344162`.
The first abrupt local SSM error is the **batch refresh of nodes 75898 and
75904**. Each occupies 32,768 slots; together they fill the 65,536-slot ring.
Both enter at level 35, scaling degree 2 and leave at level 18, degree 2.

| Node | Role | Error before refresh | Error after refresh | Public bound |
| --- | --- | ---: | ---: | ---: |
| 75898 | Repeated previous-value update factor | 5.25e-7 | 0.0111181 | 2.23344 |
| 75904 | Current value/key outer-product update | 1.63e-6 | 0.0539785 | 10.84368 |

The following operations behave accurately on their **observed** inputs:
node 75900's multiplication adds `1.66e-6` local error, node 75901's addition
`7.48e-6`, and node 75905's final state addition `9.03e-6`. The last state
nevertheless has `0.0803420` total error because it consumes the damaged
refresh outputs. Thus changing the model polynomial approximation alone does
not address the first observed fault.

"First" here refers to the abrupt error in this SSM update, not the first
internal tensor whose absolute error exceeds `0.001`. Earlier normalization
intermediates have larger magnitudes and can exceed that absolute threshold
while their normalized outputs remain accurate. The before/after refresh
comparison and the local downstream checks establish this localization.

### CPU propagation confirms the final-output effect

Injecting only the two observed after-refresh vectors into the frozen CPU
circuit affects 39 nodes through final hidden node 75969. All other inputs
remain CPU references, and later encrypted-operation/refresh errors are
excluded. The no-error control matches all affected references within
`9.31e-10`. The injected calculation predicts final error **`0.0344204290`**,
versus observed **`0.0344199817`**, and its complete final vector differs from
the observed vector by at most **`7.30e-7`**. Thus these two damaged outputs
account for essentially all of this trace's final error. This does not yet
identify the internal bootstrap mechanism or qualify a repair.

The later full-history replay gives the same attribution: injecting its two
refresh outputs predicts `0.0250857449` final error, versus observed
`0.0250866700`; the maximum difference between complete final vectors is
`1.90e-6`.

### Fresh-input isolation

The maintained [operation extractor and replay probe](../../experiments/packed_diagnostics/README.md)
can isolate this two-input refresh with both observed vectors, their original
levels, scaling degrees and public bounds. The extracted case is bound by
SHA-256 `44b5936b3e9340e65cd71bbefe5f823f37b7c240b4c85aaa0482974045531638`.
Fresh encryption at level 35 / degree 2 gives input error `4.06e-12` and
refresh error **`3.40e-6`**. The fault does not reproduce in that fresh case.
This preserves the logical values and packing, but not the original RNS noise,
keys or complete encrypted history; it is not a minimal failing reproduction.

A separate smoke run validates the original-ciphertext replay harness using
freshly created inputs. It checks 5,242,880 cloned RNS words exactly, then
exercises identical packing, doubled public bounds, separate refresh groups
and unmerged correction. All four controls have maximum change below `1e-5`.
This validates the harness, not a repair of the long request. The full-history
diagnostic uses the same helper immediately after the identified group, with
no diagnostic output substituted into model state and no keys serialized.

The first full-history replay observes only the two target nodes. It stops
at the first target observation, before reaching the refresh controls:
OpenFHE rejects the intermediate decode because its approximation-error
estimate is too high. Its observation file is empty; the completed empty-file
analysis is **not** a successful diagnosis. This does not invalidate the
earlier fine trace, but it limits generalizing its precise failure boundary
to every fresh run.

The diagnostic observer now records this specific decoder rejection as missing
values and continues the unchanged encrypted circuit. It does not relax the
decoder check or the final generation gate. Other exceptions still terminate
the process. If a retained refresh input cannot be decoded, a control can be
compared with its frozen CPU reference, with that basis explicitly reported;
such a comparison cannot separate existing input drift from refresh error.

### Full-history controls and repair proposal

The rejection-aware full-history run completes all 684 observations without
any decoder rejection. The original model still fails at selection 24 with
polynomial/exact errors `0.0250861 / 0.0250837`. It also checks 5,242,880 RNS
words when cloning the retained inputs and completes the four controls:

| Execution | Node 75898 error | Node 75904 error | Bootstrap calls |
| --- | ---: | ---: | ---: |
| Original group in the long request | 0.0111165 | 0.0539748 | 2 |
| Identical group on retained-input clones | 7.89e-7 | 4.66e-6 | 2 |
| Doubled public bounds | 1.44e-6 | 7.92e-6 | 2 |
| Separate refresh groups | 7.47e-7 | 4.04e-6 | 4 |
| Unmerged correction | 7.43e-7 | 4.55e-6 | 2 |

These errors use the decrypted retained input as reference. Both inputs
decoded successfully, so no CPU-reference fallback is involved. The identical
group already works; the result does **not** establish that doubling bounds,
splitting the group or disabling merged correction repairs the original call.
No control output replaces model state, and control timings are diagnostic.

The evidence points toward a difference in execution state or ownership between
the first call and replay, rather than establishing an unavoidable failure for
these logical inputs. Cache contents, allocation, temporary buffers and extra
diagnostic synchronizations differ. The raw clone comparison occurs **after**
the original refresh: it does not prove that retained input residues were
immutable across that call. An unintended input mutation remains another
possibility. No particular cache, encoder or backend race is proven.

The proposed repair trial is to give the target refresh explicit ownership of
its input copies and scratch buffers, and prepare its public masks before the
call. Verify original inputs and cached plaintext residues before/after use,
and compare packed input, first bootstrap, correction bootstrap and extraction
against the accurate replay. Fix the first violated ownership, initialization
or metadata invariant; do not adopt an unconditional second execution as a
workaround. This proposal keeps both bootstrap passes, bounds and cryptographic
parameters. It is **unqualified** until a cold first call works and ordinary
64-token generation passes the original gates without diagnostic decryptions.

The [published evidence](../../results/b300/2026-09-28/long-accuracy/README.md)
includes the failed original call, accurate replay controls, CPU propagation,
source identities and the remaining uncertainty. A small standalone failing
case has not yet been obtained; the fresh-input case passes.

## Candidate 1: delayed modular reduction in the NTT

**Reject this implementation.** The unchanged baseline archive is retained.
The optional [patch and oracle](../../experiments/ntt_lazy/README.md) use the
redundant-residue butterfly from Algorithm 4 of
[Harvey's paper](https://arxiv.org/abs/1205.2926). For 64-bit Shoup arithmetic
and `p < 2^62`, internal residues stay in `[0,4p)`, with canonical reduction
before the existing half boundaries and fused consumers.

Only `NTT.cu.o` and the CUDA device-link member change in the 39-member archive.
An independent payload audit checks every member, including repeated names.
The GPU oracle compares **373,293,056 output words exactly**: 144 fused Q
cases plus 104 scalar cases covering the remaining Q source and all twelve P
primes. The pure integer check also passes 100,000 cases.

The same-device, all-12-layer two-token screen passes numerical/token gates
but takes **186.22032 -> 188.67358 seconds**, or **1.32% longer**. No full
16-token NTT performance claim is made.

Nsight Compute captures four selected mode-1 launches for each half and
variant. These are profiler replay measurements, not full-model averages:

| Metric | First half: baseline -> candidate | Second half: baseline -> candidate |
| --- | ---: | ---: |
| Registers used per thread | 38 -> 44 | 42 -> 44 |
| Registers allocated per thread | 40 -> 48 | 48 -> 48 |
| Register-limited resident blocks | 6 -> 5 | 10 -> 10 |
| Achieved occupancy | 67.35% -> 56.95% | 56.05% -> 56.06% |
| Executed instructions | 79,718,400 -> 78,231,552 | 52,850,688 -> 53,098,496 |
| Eligible warps per active scheduler cycle | 3.62 -> 2.63 | 2.49 -> 2.42 |
| Mean replay duration | 106.50 -> 113.79 us | 76.62 -> 78.26 us |

No register spilling is reported. First-half instruction count falls only
1.87%, while register allocation reduces residency and eligible warps. This
supports an occupancy/latency explanation for the measured slowdown. Merely
counting modular corrections was insufficient to predict GPU performance.

## Candidate 2: indexed mask cache

The optional `--indexed-mask-cache` path hashes runs of identical coefficient
bits and indexes the existing LRU entries by hash and CKKS level. A full bitwise
coefficient comparison still protects against hash collisions. Cache admission,
capacity, eviction policy and arithmetic are intended to remain identical.
Tests cover deliberate collisions, signed zero, nonfinite inputs, eviction,
move semantics and randomized equivalence with the original lookup.

The all-layer two-token screen passes with identical cache/refresh counters:
**184.55882 -> 180.16846 seconds**, a **2.38% reduction**. This is a screening
result, not an adopted full-request speedup. The first full candidate run fails
the initial token's numerical gate with polynomial error `0.169579`.
The corresponding full-request control and diagnostic follow-ups are recorded
separately; a passing repeat cannot erase that failure.

| Full 16-token run | Variant | Evaluation | Polynomial / exact max error | Outcome |
| --- | --- | ---: | ---: | --- |
| A1, first placement | Control | 1,076.02696 s | 7.04e-6 / 6.76e-5 | Pass |
| B2, same placement after A1 | Indexed | 1,052.33206 s | 3.95e-6 / 6.58e-5 | Pass |
| B1, second placement | Indexed | Incomplete | 0.169579 / 0.169599 at first selection | Fail |
| A2, second placement after failed B1 | Control follow-up | 1,073.50648 s | 8.55e-6 / 6.44e-5 | Pass |

The passing A1/B2 pair saves 2.20%, with identical 4,426 bootstrap calls,
275,540 cache hits, 15,205 misses, arithmetic counts and token IDs. The planned
opposite-order group is incomplete because B1 failed. A2 was then run as an
explicit control follow-up. No pooled or ABBA improvement can be claimed.
An instrumented cache follow-up reaches 581 distinct observed nodes without a
large error before its 500-second deadline; it stops during the second server
evaluation, before the failing selection, and does not resolve the failure.

The feature remains disabled by default and unqualified for promotion pending
resolution of the numerical failure. It must not be presented as a new secure
generation speed record.

Neither candidate qualifies, so there is no combined candidate to measure.
The previously adopted public-state reuse, refresh circuit and parameter
profile remain the qualified configuration.

## Validation and scope

The complete local checks pass **437 Python tests and 24 native C++ tests**;
coverage is 85.73%. Additional CUDA checks cover the exact NTT oracle, the
fresh primitive replay, the two-input refresh replay and the clone controls.
The diagnostic modules use a distinct import name so they can coexist with
the other experiment analyzers in one test process.

The numerical investigation does not establish successful 64-token generation,
a failure probability, or a universal internal-tensor tolerance. The `0.001`
requirement applies to final hidden vectors. The current task ends with
operator isolation and a supported repair proposal; a repaired full request
must pass a separate 64-token qualification before any long-generation claim.
