# B300 even-polynomial refresh — 2026-09-27

**Adopt as an optional dependency patch, with single-pair evidence.**
The full frozen Mamba-3 workload improves **148.36 → 145.96 s (1.62%)**;
refresh improves **37.77 → 34.66 s (8.22%)**. This is **36.49 s/generated
token**, amortized over four generated tokens and five encrypted evaluations.
The [patch and encrypted probe](../../experiments/refresh_even_seed/README.md)
and [raw evidence](../../results/b300/2026-09-27/refresh-even-polynomial/README.md)
are retained. The normal dependency build does not enable it automatically.

## Implementation and scope

The [static screen](2026-09-27-refresh-even-polynomial.md) derived an even
seed from the actual pinned binary64 coefficients by moving the phase shift
into the input. The candidate evaluates `Q(2*(x-1/2048)^2-1)`, where Q has
degree 44. Its original-input degree stays 88. The exact real-arithmetic
seed difference from the previous polynomial is bounded by **2.431e-14**
over the entire original interval [-1,1], including the transformed argument's
small extension beyond 1.

The shared FIDESlib `approxModReductionSparse` entry selects this path only
with S2C-first precomputation, K=512, six double-angle steps and exact equality
with the pinned 89 seed coefficients. Other contexts use the existing
evaluator. Both correction passes use the candidate. The general model
polynomial evaluator, including its maximum degree 1,023, is unchanged.
Setup retains the original coefficient/depth budget.

Static EvalMod products fall **24 → 20 per bootstrap**: general ciphertext
multiplications 12 → 8, with 12 squares unchanged after including the new
preparation square and all six double-angle iterations. The full schedule has
484 calls, so this removes 1,936 general products from the static inventory.
Application operation counters exclude these backend operations and remain
identical. No proportional kernel-time saving is assumed.

The experiment rebuilds baseline and candidate FIDESlib static archives from
the same pinned source snapshot. Of **39 archive members**, only
`ApproxModEval.cu.o` differs; compiler commands are identical. The two model
binaries link the same native objects and patched OpenFHE inverse-FFT core
against those separate archives. The additional header and changed body match
the retained patch byte-for-byte. Both variants have the same measurement-only
producer-affinity helper; it is not part of the portable implementation.

## Encrypted qualification

Each independent process checks four raw seed/EvalMod cases with 32,768 slots,
at input levels 0 and 3. Inputs cover endpoints, phase-shift neighborhoods,
the interior and modular-lift integer neighborhoods. The original polynomial
is evaluated in long-double arithmetic for the client reference.

| Check | Baseline maximum | Candidate maximum | Fixed gate |
| --- | ---: | ---: | ---: |
| Raw seed | 8.312e-11 | 1.038e-10 | 1e-8 |
| Seed plus all double angles | 9.893e-9 | 6.099e-8 | 1e-6 |
| Transfers / two-pass refresh and following product | 9.535e-9 | 8.788e-8 | 1e-6 |

Seed output levels are 7/10 and EvalMod levels 13/16 in both variants, with
matching scale degrees and factors. Each process also passes six exact NTT
ring maps, 30 encrypted transfer cases and 12 refresh circuits, including
four warm-ups. Active widths 1, 32, 768 and 1,536, inactive slots, both ring
directions and a subsequent ciphertext product are covered. All corresponding
post-refresh output levels match.

Observed encrypted errors are larger with the candidate. The seed's real
polynomial bound does not bound encrypted rounding or its amplification.
Acceptance uses the unchanged gates, not bitwise output or equal-error claims.
Keys are freshly generated for each process.

## Model comparison

The predeclared sequence is encrypted qualification, one prefix A/B pair,
then one full A/B pair if the prefix has no >1% regression and refresh is
faster. Full promotion requires at least 1% complete evaluation reduction and
3% refresh reduction, with all numerical and metadata gates passing. These
are engineering screens for this small patch, not statistical tests. There
are no discarded timing samples or automatic repetitions.

One B300 (GPU 4), CPU set 64–68, NUMA node 2, four OpenMP threads, CUDA 13.0.1,
GCC 13.3 and SM103 are fixed. The main thread and two producers are observed
on CPUs 64/65/66 in both full runs. Observation starts during the baseline
full run; prefix placement has the same configuration but no sampled proof.
No compilation overlaps GPU timing. The resident model, prefix, client head
and canonical library hashes are checked before and after the campaign.

The 187M SISO model retains 12 layers, five encrypted evaluations, prompt
`The capital`, and four generated IDs `[315,279,1614,315]`. Rings 32,768/65,536,
depth 44, scale 59, first modulus 60, HYBRID/FLEXIBLEAUTO, uniform-ternary
secret, two-pass S2C-first refresh and correction multiplier 4096 are unchanged.
The useful refresh interval remains ceiling 35 / returned level 18. The
experimental `security=not-set` qualification is unchanged; this is not a new
security certification.

| Metric | Baseline | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Prefix evaluation | 7.64918 s | 7.64149 s | 0.10% |
| Prefix refresh | 1.08579 s | 1.00248 s | 7.67% |
| Full evaluation | **148.35766 s** | **145.96074 s** | **1.62%** |
| Full refresh wrapper | **37.76758 s** | **34.66129 s** | **8.22%** |
| Evaluation minus refresh | 110.59007 s | 111.29945 s | −0.64% |
| Setup | 69.71777 s | 69.34236 s | 0.54% |
| Complete process | 235.52330 s | 232.74314 s | 1.18% |
| Evaluation / four generated tokens | 37.08941 s | **36.49019 s** | 1.62% |
| Peak host RSS | 26.28892 GiB | 26.29184 GiB | — |

Evaluation excludes setup and final client verification. Complete process time
runs from the availability check/Docker invocation through process exit,
excluding external transfer and lock waiting. Evaluation minus refresh is an
arithmetic difference, not a separate kernel measurement. Refresh saves
3.1063 s while other time grows 0.7094 s in this pair; the whole saving is
2.3969 s. The previous 37.20 s/token ABBA result remains historical evidence,
not this pair's baseline.

Both full runs pass the unchanged 0.001 exact and polynomial gates, with
zero nonfinite values, identical tokens and zero evaluator decryptions.
Full polynomial error is **8.132e-7 → 3.100e-6**; exact-reference error is
**4.706e-5 → 4.662e-5**. The candidate prefix polynomial error is 8.293e-6.
The 484 bootstraps, 242/484 ring transfers and ordinary operation counts match.
One pair and one prompt do not establish significance, arbitrary-prompt
quality or full Mamba-2 performance. The shared branch is algorithm-independent,
but its current qualification is this real S2C-first configuration.

## Further candidates and retained diagnosis

The [follow-up inventory; archive member `b300/2026-09-27/refresh-even-polynomial/followup-opportunities.md`](../../results/README.md#archived-provenance)
records three distinct candidates found while this experiment ran:

1. **Combine correction before projection/unpacking.** Integer-residue
   multiplication permits `4096*first+second` without an extra scale level.
   Static opportunity: downward transfers 484→242, wrapper rotations
   12,204→8,136, and output plaintext products 2,668→1,334. Validate metadata,
   dynamic range and key-switch error before timing.
2. **Co-design seed and double angles.** Seven steps plus a degree-54 seed
   (degree 27 after symmetry) estimates 18 products at the same planner depth.
   A 131,073-point plaintext screen passes with 2.77e-11 sampled error; it is
   not a uniform certificate or CKKS qualification. The actual adopted patch
   continues using six steps and the translated existing coefficients.
3. **Carry the internal phase layout across correction passes.** Factoring
   `B=R C R^-1` can remove two of four boundary rotations when combined with
   the first candidate. This is a conditional 484 backend-rotation reduction,
   separate from the wrapper counter. Rotation noise/layout still need testing.

The retained Nsight reanalysis uses existing parallel-prefetch databases,
without running another profile. In 16 complete bootstrap ranges of the
two-worker trace, EvalMod accounts for 0.5592 s of 1.0647 s host inclusive
time. Correlation-attributed GPU activity unions are 0.1692 s for EvalMod and
0.1721 s for C2S. Overlapping/nested timings cannot be added as a wall-time
partition or extrapolated into full-model utilization. Generic transform
hoisting is already implemented. The follow-up inventory ranks correction
merging first, with numerical rejection conditions for each proposal.

Two preparation failures (measurement-patch context and an omitted version
file) were fixed before GPU tests; their logs and input snapshots are retained.
The first archive audit also rejected duplicate object names in the static
library. An occurrence-aware parser resolved that audit without changing any
binary, measurement or acceptance gate. Source/evidence verification and
task-local cleanup are recorded with the results.
