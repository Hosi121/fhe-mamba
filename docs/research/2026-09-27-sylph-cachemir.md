# Sylph and Cachemir: implications for the classical-128 prototype

Read during the B300 security-prototype qualification. This is a design review,
not a reproduction or a new speed result. The frozen Mamba-3 payload, numerical
tolerances, generated token IDs and evaluator-decryption gate remain the reference.

## Primary-source findings

[Sylph v2](https://arxiv.org/html/2601.18511v2), Sections III-B–D, IV-B, V and
Appendices A/C/D, combines low-modulus linear algebra with encoding/ring
conversions absorbed into bootstrapping. Its slim polynomial evaluator trades
unused slots for parallel branches using a sum-of-squares decomposition. Its
GPU partitioning communicates at low modulus; decode scales less than prefill.
The security discussion uses sparse ternary secrets and includes the auxiliary
modulus in the RLWE instance. Its model calibration changes precision and input
statistics. These are separate assumptions from our dense-secret, frozen-model
prototype. Table VI and Appendix C were checked against the PDF.

[Cachemir v1](https://arxiv.org/html/2602.11470v1), Sections 4–7 and Appendix C,
combines interleaved replicated VMM, extraction-mask fusion and KV layouts. Its
level graph accounts for operation latency and bootstrap placement, including
inside nonlinear modules. Pruning assumes negligible level-drop cost,
input-level-insensitive bootstrap cost and monotone operation costs. Its GPU
numbers compose module measurements. It uses 41-bit scale primes, sparse-secret
encapsulation and a different depth budget; copying its advertised 128-bit
parameters would not validate our uniform-ternary context. The PDF layout and
level-management figures were checked.

The source PDF hashes and frozen-payload screening counts are in
[screen.json](../../results/cpu/2026-09-27/sylph-cachemir-screen/screen.json).
Original PDFs are not redistributed. Paper throughput is not compared directly
with our small Mamba model, single B300, prompt or client protocol.

## What is already present

The current `Evaluator::linear` in
[packed_fideslib.cpp](../../native/fideslib_stage0/src/packed_fideslib.cpp)
uses `resolve_interleaved_replicated_shape`, replica construction, BSGS masks
and rotation hoisting. The depth planner defers a projection's final mask when
every consumer is a gather. These are existing packing and mask-fusion
mechanisms, not new opportunities inferred from reading Cachemir again.

Likewise, S2C-first bootstrap, ready-node scheduling, small-state batching,
compact public weights and overlapped plaintext preparation already exist.
The unresolved question is their joint cost at a secure ring/modulus geometry.

## Changes worth investigating

| Priority | Application to this repository | Evidence needed before adoption |
| --- | --- | --- |
| First | Price the full level schedule, rather than minimizing logical refresh count | Secure-context measurements indexed by consumed level, shape and batch occupancy; count packing, masks and state lifetimes |
| Screened | Tune HYBRID digit decomposition at the audited ring | Four/five/six digits pass the parameter/refresh gates; five digits save only 0.52% on the prefix, six are slower, so retain four |
| Next | Fuse correction/repacking around two-pass refresh | Demonstrate the same scale and output level, then verify the full error gate; an extra level can cancel a local saving |
| Conditional | Slim polynomial evaluation for small vectors | An exact decomposition of the existing polynomial, bounded intermediates, slot/level fit and all packing costs |
| Larger redesign | Coefficient-domain projections and conversions fused into refresh | An independent security argument for every smaller ring and switching key, an exact layout simulator, and measured conversion cost |

For our planner, a useful objective is

`sum(operation_time(level, shape)) + sum(refresh_time(batch, input_level))`.

The existing `PackedDepthPlan` optimizes a count, and the ready scheduler uses
fixed ceiling/refreshed levels. Neither is a measured whole-DAG optimum.
Cachemir's pruning theorem cannot simply be transplanted: our two-pass refresh
contains level-alignment work, packing multiple live branches changes cost,
and GPU launch/graph thresholds can break a monotonic cost approximation.
First retain the actual measured transitions; prune only dominated transitions
that satisfy this executor's constraints. Client feedback and carried-state
dependencies are boundaries, not parallel work.

At the current 3,376-bit QP, simply restoring N=65,536 violates this prototype's
1,747-bit guideline bound. Low-modulus projections therefore need a different
chain and audited switching keys; the earlier experimental ring bridge cannot
be used as the secure implementation of Sylph's idea.

Digit count is a smaller, immediately testable instance of this cost tradeoff.
At 45 Q towers, four digits require twelve P towers in the qualified context.
More digits can shorten basis conversion while increasing key accumulation
and key storage. Counting fewer P bits alone does not predict a speedup. This
motivated secure refresh/prefix trials at five and six digits, with the model
circuit unchanged. The measured times were 18.388, 18.293 and 18.655 seconds
for four, five and six digits. Neither alternative reached the 3% screen for
a full comparison. This rejects the local tuning opportunity, not the broader
level-placement idea.

## Why slim evaluation is promising but not a direct substitution

The refreshed payload contains 785 polynomial nodes before dead-node removal:
245 scalar, 240 width-24, 60 width-32, 120 width-768 and 120 width-1536.
All have odd declared and exact coefficient degrees, from 7 through 1023.
Thus none directly meets the even-degree input condition of the paper's
decomposition. Tiny trailing coefficients were not silently discarded.
The older live-DAG inventory of 761 nodes describes a different count: dead
nodes are excluded there.

This does not rule out an algebraic adaptation. An exact Chebyshev parity
identity is `p(x) = A(T_2(x)) + x*B(T_2(x))`, where `T_2(x)=2*x*x-1`.
With the repository's ordinary constant-term convention, `A_j=c_(2j)`;
`B_0=sum_k (-1)^k*c_(2k+1)` and
`B_j=2*sum_(k>=j) (-1)^(k-j)*c_(2k+1)` for positive j. This avoids a conversion
to monomials and keeps the new argument in [-1,1].

Our [float64 screen](../../results/cpu/2026-09-27/sylph-cachemir-screen/parity.json)
checks all 785 nodes at 130 points each: the maximum sampled difference is
7.11e-15, and the combined coefficient l1 norm grows at most 5.36 times.
Both branches still have odd degree at every node. Thus one parity split is
not enough to meet the sum-of-squares precondition. It also introduces another
product and two branches; intermediate magnitudes and encrypted reconstruction
error need separate tests. The sampled CPU result is not an interval certificate.
A degree-1023, width-24 site has room for 1,024 replicas in 65,536 slots; a
width-1,536, degree-127 site does not. Partial recursion and compatible batching
therefore need different schedules. Root-factorization stability, coefficient
encoding and packing/unpacking can dominate the saved ciphertext products.
These are hypotheses to test, not a claimed logarithmic whole-model speedup.

Reproduce the metadata and parity screens with the frozen public program:

```bash
python experiments/security128/screen_polynomials.py /path/to/program.txt \
  --parity --output screen.json
```

## RMSNorm outliers and the frozen Mamba export

Sylph's model calibration is not an RNS coefficient-layout optimization.
[Section III-A and Table II](https://arxiv.org/html/2601.18511v2#S3.SS1)
report that prefixing reduces the maximum RMSNorm input magnitude from
2,243.97 to 7.65. This is a range measurement, not a normalization speedup.
The prefix supplies precomputed attention-sink state and changes the input
context. Separately, orthogonal rotations redistribute coordinate outliers
and are folded into neighboring projections. For an orthogonal matrix R,
`||Rx||² = ||x||²`: rotation alone cannot narrow the mean-square argument of
the inverse square root. The paper's 12-bit noise/perplexity criterion also
differs from this repository's frozen absolute-error gates.

The current Mamba-3 export applies RMSNorm directly, without sink-prefix
processing or that rotation calibration. Its 245 live inverse-square-root
nodes use degrees 15, 31 and 63 (50, 150 and 45 nodes respectively). The
maximum degree 1,023 belongs to `negative_a`, not RMSNorm. There are 761 live
polynomial nodes in total. These counts are obtained by joining the exact
coefficients to the export manifest with
[`polynomials.py`](../../experiments/level_schedule/polynomials.py).

An inverse square root over [a,b] depends on the ratio b/a as well as its
required error. Multiplying all inputs by a public constant reduces their
magnitude but preserves that ratio; our Chebyshev evaluator already maps the
interval to [-1,1]. Thus simple rescaling does not establish a lower-degree
replacement. The current investigation attributes measured polynomial time
before selecting an exact evaluation/layout change. Altering calibration
intervals requires independent range and full-output validation.

The [GPU preparation comparison](2026-09-27-b300-gpu-plaintext-fft.md) now
provides that attribution. On its 165.373-second candidate, inverse-square-root
nodes take 20.325 seconds (12.29%), including 12.403 seconds of refresh batches
initiated at those nodes. `negative_a` takes 15.741 seconds, including 5.676
seconds of refresh. Reductions and surrounding products are separate, and
batched refresh can serve other values. These figures establish priorities,
not an achievable saving from outlier suppression.

## Scope limits

Mamba has recurrent state rather than Transformer KV attention. The general
lesson is to keep persistent state in the layout its next consumers need;
the KV-specific equations are not an implementation for Mamba. Public-prefix
processing, extra sink tokens, changed model calibration and weaker accuracy
would change the current comparison contract and are not adopted here.

Neither paper certifies our backend. The new prototype independently checks
the actual maximum Q/P context, uniform ternary secret and Gaussian error
against the published classical-128 guideline, before key generation. Earlier
`security=not-set` timings remain experimental timings.
