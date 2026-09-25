# Four structural optimization trials: shared work and representation changes

Rotation sharing and same-input Chebyshev basis reuse are adopted as opt-in
paths. Four full processes in ABBA order average **619.4101 → 614.1094 s
(0.86% reduction)**, with unchanged numerical gates and refresh counts. The
nominal 59-bit 32-bit adapter fails its numerical gate; the smaller-ring CPU
route passes accuracy but is not competitive. This campaign does **not**
demonstrate a large full-model speedup.

The agreed stopping rule is one mechanism for each of four candidates, small
qualification, integration of eligible candidates, full inference comparison,
and an adoption decision. All four mechanisms are implemented as either
packed-executor options or standalone encrypted probes. Only the first two
qualify for the full-model comparison. The baseline is `c90c54d`, including compact GPU RNS
plaintext preparation and the S2C-first, two-pass refresh circuit. The frozen
checkpoint, public model polynomials, token checks and both 0.001 model error
gates remain the decision criteria.

## Rotation prefixes and sibling decomposition

`--hoist-rotations` replaces independent baby rotations in packed linear and
routing BSGS with a trie of their existing signed-digit sequences. Identical
prefixes are evaluated once. Multiple edges from the same parent use FIDESlib's
vector `EvalFastRotation`, reaching `rotate_hoisted` and sharing the ModUp
preparation. The scalar API's precompute alone does not implement this reuse.
Every edge uses an existing signed power-of-two key. Giant rotations of different
inputs and refresh internals retain their prior path.

The trie holds each parent until its children finish, and synchronizes each
batch before releasing its storage. Zero and duplicate offsets can alias a
read-only source. Mamba-2 complex/sparse and Mamba-3 real/uniform configuration
probes compare every RNS residue and level/scale-degree metadata, including
negative/wrapped/duplicate offsets, binary and NAF decompositions, and six
level/degree combinations. All 552 cases pass; live inputs remain unchanged.
ABBA microbenchmarks exclude warm-ups and use the same zero-rotation behavior.

| Configuration / consumed level | Scalar, ms | Shared, ms | Reduction |
| --- | ---: | ---: | ---: |
| Mamba-3 real / 18 | 273.496 | 160.098 | 41.46% |
| Mamba-3 real / 34 | 99.322 | 61.710 | 37.87% |
| Mamba-2 complex / 18 | 270.824 | 159.306 | 41.18% |
| Mamba-2 complex / 34 | 98.899 | 61.614 | 37.70% |

These are batches of 32 rotations at offsets `24*i`, with four timed samples
per ABBA block after one warm-up. They are not whole-model gains. The Mamba-2
configuration qualifies the shared primitive; the specialized Mamba-2 executor
and its full-depth timing are unchanged in this study.

## Same-input Chebyshev bases

`--share-chebyshev` groups live polynomial nodes by parent, logical width and
normalization interval. It retains only the requested normalized argument and
Chebyshev basis union; each polynomial still uses its original coefficients,
recursive decomposition and inactive-slot correction. Basis handles remain
read-only; arithmetic detaches aliases through the existing owned helpers.

A weak source identity plus level and scale degree detects replacement by
refresh. The evaluator does not mutate the value/scale of a still-live DAG
input. Replacing or cleaning a value changes its handle; level and degree
changes invalidate its entry. The last member of a group releases its cache.
This leaves DAG liveness, refresh planning and the ready-node scheduler intact.

The production sin/cos coefficient probe reduces ciphertext products 22 to 14
and passes with polynomial error around 3e-12. The first synthetic probe used
roundoff-sized coefficients and failed even on the released baseline; its raw
failure remains separate from the successful real-coefficient probe.
A second fixture forces a refresh between compatible consumers and then
reuses the new basis, checking both invalidation and a subsequent cache hit.
Both runs perform two bootstraps. The sharing run records one invalidation and
one hit, with maximum polynomial error `3.05e-8`. The ordinary sin/cos probe
has one hit, changes 22 to 14 ciphertext products, and takes 0.7384 to 0.6368 s.
Those isolated timings are single samples, not the model adoption criterion.

## Composite 32-bit RNS

The separate Cheddar probe uses two small NTT primes for each nominal 59-bit
ordinary rescale step, and a matched 64-bit-word control. It does not halve
the precision or simply change the default scale to 30 bits. Deterministic
prime generation, the exact parameter arrays and all source/dependency pins
are retained. Ring degree is 65,536 with 32,768 slots. The low-level base scale
is 57 bits with a 62-bit composite first modulus, as required by Cheddar's
bootstrap constructor; ordinary scales converge towards 59 bits.
Two 32-bit residues occupy the same raw bytes as one 64-bit residue per
ordinary rescale step. A smaller machine word therefore does not by itself
halve the memory traffic at the preserved precision; faster modular arithmetic
must pay for the extra towers and conversion work.

The circuit includes a square, public bias/product, a sixteen-diagonal 4x4
BSGS map with nonconstant masks, two encrypted refresh passes with residual
correction, and a final square. Timings include public mask encoding, level
conversion into refresh, both refreshes, residual scaling and synchronization.
Final client validation checks every slot against the plaintext circuit at
1e-6. Setup/key generation is separate. Neither this circuit nor its backend
is a trained-model timing. The small circuit has its own modulus chain and
refresh parameters; it is not the full model's 44-level FIDESlib context.

The nominal 59-bit **32-bit-word profile fails** the 1e-6 gate, whereas its
same-circuit 64-bit-word control passes. A client-only diagnostic, performed
after the complete encrypted circuit and never fed back into it, finds an
invalid result already at the first ciphertext square. The remaining
multiplication/relinearization/rescale compatibility issue is not localized to
refresh. Failed-output timings are not speedups.

The upstream **40-bit 32-bit control passes** the same small circuit and error
gate, with worst error about `4.44e-7`. This rules out a completely broken
32-bit/CUDA installation. It has different scale and usable depth, so it does
not qualify the selected nominal 59-bit profile or the model. No lower model
precision or relaxed model gate is substituted. The result rejects this
concrete high-precision adapter; it does not show that 32-bit RNS is impossible.

The first key-generation attempt failed with `QSize mismatch` for both word
sizes. The pinned test helper's default key level uses its short ephemeral
secret base. Supplying the ordinary input level explicitly repairs that
adapter error before numerical qualification. The initial link failure and
failed key-level source are also preserved.

The upstream client helper is test-only. These are experimental parameters,
not a 128-bit security or production-client claim. The pinned FIDESlib and its
S2C-first circuit are not modified by this alternative backend trial.

## Smaller ordinary ring

The Lattigo v6.2.0 prototype computes at N=32,768, switches encrypted data into
N=65,536 for two-pass refresh, switches back, and resumes multiplication. Its
N=65,536 ordinary-ring control uses the same residual Q primes, scale, input
shapes and refresh parameters. Both use dense ternary secrets without sparse
ephemeral encapsulation and reserve a 28-bit correction prime. There is no
intermediate evaluator decryption.

The initial default modular-reduction interval K=16 was inappropriate after
disabling sparse encapsulation; both ring sizes fail that setup. The repaired
dense configuration uses a continuous cosine approximation, K=512, degree 127
and six double-angle steps. It retains the 59-bit scale and 1e-6 gate. Raw
initial failures are retained and do not count as evidence against ring
switching itself.

This is a CPU encrypted-boundary implementation, not a GPU FIDESlib ring
switcher. Its timing can qualify or reject this concrete CPU path, but cannot
establish the performance of a future GPU ring-switch implementation.


The corrected probes pass all four logical widths (1, 32, 768 and 1,536) for
each ring size; every inactive slot is checked too. Worst errors are
`1.10e-8 / 1.57e-8`, below 1e-6. Mean times across the four different widths are:

| CPU probe component | Ordinary N=65,536, s | Ordinary N=32,768, s |
| --- | ---: | ---: |
| Two ordinary ciphertext products | 0.2577 | 0.1233 |
| Complete encrypted refresh boundary | 54.9766 | 55.2503 |
| Sum | 55.2343 | 55.3735 |

The smaller ring saves 52.17% in ordinary arithmetic, but the measured whole
probe is about 0.25% slower. Each width has one sample; this tiny total difference
is not a statistical claim. More decisively, a roughly 55-second CPU boundary
cannot retain the current GPU refresh speed. This concrete CPU route is not
integrated into model inference. A GPU ring-switch implementation remains
unimplemented and unmeasured. No conclusion about its possible speed follows
from the CPU result.

## Matched model comparisons

Both variants keep GPU plaintext RNS expansion, S2C-first refresh, two physical
bootstrap passes, ring 65,536, 32,768 slots, depth 44, 59-bit scale, refresh
ceiling 35 and refreshed level 18. Inputs, checkpoint weights and polynomial
coefficients are the same frozen Mamba-3 SISO 187M payload. CPU affinity is
15–19 with four OpenMP threads on DGX Spark GB10, CUDA 13.0. Builds and the
alternative-backend experiments finish before model timing starts.

Eight independent one-layer prefixes use the order baseline, rotation sharing,
basis sharing, both, both, basis sharing, rotation sharing, baseline. Every run
passes both 0.001 error gates, produces IDs `[6864, 6864]`, uses 16 bootstraps
and performs no evaluator decryption.

| Prefix mode | Mean native evaluation, s | Reduction |
| --- | ---: | ---: |
| Baseline | 31.00764 | — |
| Rotation sharing | 30.80786 | 0.64% |
| Basis sharing | 30.88909 | 0.38% |
| Both | 30.69157 | 1.02% |

Rotation sharing reduces the prefix rotation count from 2,560 to 2,530. Basis
sharing reduces ciphertext products from 673 to 649 with three basis hits.
The combined path qualifies for the full comparison, but the large rotation
microbenchmark improvement does not transfer directly to model latency.

The final twelve-layer comparison uses four fresh processes in ABBA order:
**620.6718, 613.7746, 614.4441, 618.1485 s**. All four pass both 0.001 gates
and return `[315, 279, 1614, 315]`, producing `The capital of the state of`.

| Full component, mean of two processes | Baseline, s | Both options, s |
| --- | ---: | ---: |
| Native evaluation | 619.4101 | 614.1094 |
| Ordinary evaluation, total minus refresh | 380.9368 | 375.7654 |
| Refresh | 238.4733 | 238.3440 |
| Linear operations, excluding refresh | 97.8304 | 95.6129 |
| Polynomial operations, excluding refresh | 120.5109 | 118.4839 |
| Routing, excluding refresh | 148.0334 | 147.2379 |
| Other evaluation | 14.5620 | 14.4308 |
| Host encoding, included in ordinary categories | 65.2231 | 64.8900 |
| Complete process wall time | 661.8449 | 654.8113 |

The total reduction is **0.8558%**, and ordinary evaluation falls **1.3576%**.
The refresh circuit is unchanged; its 0.1293 s mean difference is observational,
not a new refresh optimization. Category totals subtract nested refresh timers;
host encoding is already included and must not be added again.

| Count per full process | Baseline | Both options |
| --- | ---: | ---: |
| Ciphertext products | 12,005 | 11,540 |
| Plaintext products | 106,541 | 106,481 |
| Rotations | 62,969 | 62,369 |
| Shared basis hits | 0 | 60 |
| Physical bootstraps | 484 | 484 |
| Logical refreshes | 1,335 | 1,335 |
| Evaluated nodes | 7,814 | 7,814 |

The 465 fewer ciphertext products agree with the prior static shared-basis
estimate. They are 3.87% of ciphertext products; 600 fewer rotations are 0.95%
of model/wrapper rotations. These count fractions are not time predictions,
but help explain why a large isolated rotation-batch gain becomes a small
whole-model gain. Most ordinary arithmetic and all refresh work remain.

Maximum errors across both candidate processes are **4.65294e-5** against
the exact reference and **9.65161e-7** against the frozen polynomial reference.
The corresponding baseline maxima are 4.73028e-5 and 1.00672e-6. All values are
finite; no evaluator decryption occurs. These fresh-key error differences do
not establish an accuracy improvement. Peak process RSS is **26.21 GiB** in
both modes. It includes setup and is not a separate GPU-memory measurement.

The candidate mean is **10.235 minutes**, or **153.53 s per generated token**
for this five-evaluation/four-generated-token request. Complete process time
is **163.70 s per generated token**. These are amortized request measurements,
not steady-state single-token latency or throughput under batching.

Both candidate samples are below both baseline samples, supported by the
separate prefix and primitive checks. With two samples per mode and one frozen
prompt, there is no statistical-significance or arbitrary-prompt claim. The
combined full comparison does not isolate each option's individual full-model
contribution. **Adopt both as opt-in; reject the two concrete alternative
backend paths for integration.** The agreed four-mechanism stopping rule is
complete. A substantially larger gain was not demonstrated.


## Reproduction and provenance

The [evidence directory](../../results/dgx/2026-09-25/structural-four/README.md)
contains the source archives, exact process commands, samples, failure controls,
parameter arrays, dependency pins and a checked summary script. The production
implementation is commit `b6817f0`; the final native archive is verified against
that commit. The prefix archive precedes include-only cleanup and is retained
separately. The baseline executable is the released `c90c54d` GPU-RNS candidate.

After the normal [Spark build](../dgx-spark.md#build), add
`--hoist-rotations --share-chebyshev` to the packed executable or
`experiments/run_packed_probe.py`, alongside the flags in `full_controller.py`.
Both switches default off. The rotation mechanism needs no new keys, library
patches or numerical parameters. The same-input basis mechanism retains the
existing polynomial circuit and refresh planner.

The alternative-backend [reproduction recipe](../../results/dgx/2026-09-25/structural-four/alternative-backends/README.md)
uses isolated Cheddar and Lattigo builds; neither modifies the canonical
FIDESlib/OpenFHE installation. Local release checks pass **290 tests, including
21 C++ contracts**, with **87.60% Python coverage**. DGX C++ contracts pass too.
The full comparison verifies all 91 existing backend source files, plus the
17 canonical dependency identities before and after execution.

This remains a fixed-prompt, single-process client loop with public weights
and `security=not-set`. The client selects tokens from final decrypted hidden
vectors. The study does not establish long-context behavior, 128-bit full-chain
security, a separated production server, or a full Mamba-2 speed improvement.
The 32-bit and dual-ring probes have their own numerical circuits and do not
constitute trained-model runs.

Primary implementations: [pinned Cheddar](https://github.com/scale-snu/cheddar-fhe/tree/8df8b26ce5411a68b68f0e7c2fb7e9a2f05f3e94),
[Cheddar paper](https://arxiv.org/html/2407.13055v2),
and [Lattigo v6.2.0 bootstrapping](https://github.com/tuneinsight/lattigo/tree/v6.2.0/circuits/ckks/bootstrapping).
The [first-principles review](2026-09-25-first-principles.md) records the original
hypotheses. These trials replace their unmeasured promise with the bounded
positive and negative results above.
