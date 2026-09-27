# Refresh polynomial symmetry — 2026-09-27

This records the static design stage. The later
[B300 implementation study](2026-09-27-b300-refresh-even-polynomial.md)
qualifies the encrypted candidate and measures full evaluation
148.36→145.96 s, with refresh 37.77→34.66 s, in one matched pair.

**Static design screen passed; this stage makes no encrypted speed claim.**
The current refresh evaluates an 88-degree polynomial followed by six
double-angle squares. Moving its phase shift into the input exposes an even
polynomial, which can be evaluated as one square plus a polynomial of degree
44. The proposed binary64 coefficients differ from the existing polynomial
by at most **2.431e-14 on the entire original input interval [-1,1]**, proved
using exact rational arithmetic. The archived backend's static EvalMod
ciphertext product count changes **24 → 20 per bootstrap**, including the
extra square and all six double-angle iterations.

This is a concrete candidate for optimizing the refresh calculation itself.
Its original-variable degree stays 88. It neither establishes a faster GPU
kernel nor qualifies the final CKKS error. The coefficient generator, interval
bound, operation counts and source identities are retained in the
[CPU artifact](../../results/cpu/2026-09-27/refresh-even-polynomial/README.md).

## Current path and objective

The adopted [two-worker B300 result](2026-09-27-b300-parallel-prefetch.md)
averages 148.8043 s for five encrypted evaluations/four generated tokens,
or 37.2011 s/generated token. Its refresh timer averages **37.7026 s
(25.34%)**, including normalization, packing/unpacking, ring transfers,
residual correction and the backend bootstrap. Existing traces do not
provide an isolated EvalMod time for this complete workload. The 16.67%
product-count reduction must not be presented as a refresh or model latency
reduction.

The existing real S2C-first path is:

```text
small-ring normalized/packed input
  → encrypted transfer to the large ring
  → [SlotsToCoeffs → ModRaise → CoeffsToSlots → real EvalMod] = first
  → residual = 4096 × (input − first), with level/scale alignment
  → [SlotsToCoeffs → ModRaise → CoeffsToSlots → real EvalMod] = second
  → two downward transfers and unpack(first + second/4096)
```

There are 242 refresh boundaries and **484 backend bootstrap calls**. The
two full candidate samples have 210 grouped refreshes, 1,334 logical values
refreshed, 242 upward transfers and 484 downward transfers. S2C-first already
uses one real EvalMod branch per backend call; removing a second imaginary
branch is not a new opportunity on this path.

ModRaise changes the decryption relation from a small-modulus message into
a lifted message plus an unknown multiple of that modulus. EvalMod removes
that multiple homomorphically. A small semantic residual does not make the
unknown integer multiple small. Consequently, narrowing the second pass's
modular-reduction domain solely because its plaintext is small is not
justified.

The objective remains complete latency at the existing accuracy gates,
input range, useful output levels, two-pass correction and cryptographic
parameters. The earlier [S2C-first study](2026-09-25-s2c-first.md) saved
115 s in refresh but added 103 s elsewhere through representation/level
effects. This candidate therefore preserves the modulus/depth budget in its
design and requires a complete comparison before adoption.

## Expose the symmetry hidden by the phase shift

The pinned OpenFHE header sets `K_UNIFORM=512`, `R_UNIFORM=6` and provides
89 coefficients. Their function is the scaled, shifted cosine

```text
f(x) = A cos((2π/64)(512x − 1/4)),   A = (2π)^(-1/64).
```

The [OpenFHE explanation](https://openfhe.discourse.group/t/approximate-modular-reduction-during-bootstrapping/2119/2)
confirms this function and the 88-degree/6-iteration uniform-secret case.
These are bootstrap coefficients, separate from the model's maximum
1023-degree transition-rate approximation.

Set `δ=1/2048` and `z=x−δ`. Then `f(x)=A cos(16πz)`, which is even in z.
For Chebyshev polynomials, `T_(2j)(z)=T_j(2z²−1)`. Therefore an even
degree-88 polynomial in z becomes a degree-44 polynomial in `w=2z²−1`.
The phase shift is a scalar addition, and forming w needs one ciphertext
square, a doubling and a scalar subtraction.

The proposed coefficients are derived from the **actual existing binary64
polynomial**, rather than fitting an unrelated cosine approximation:

1. Interpret every existing binary64 coefficient as an exact rational.
   Respect the backend's half-constant convention, `c[0]/2`.
2. Translate `P(z+δ)` exactly in the Chebyshev basis.
3. Retain its even terms. The largest discarded odd coefficient is only
   **6.872e-16**. The original, unshifted odd coefficients are substantial
   and cannot be discarded in this way.
4. Round the resulting 45 coefficients once to binary64, producing Q.
5. Expand `Q(2(x−δ)²−1)` back into the original x basis using exact rational
   arithmetic. If the difference has coefficients d, then
   `|P_new(x)−P_old(x)| ≤ Σ|d[j]|`, since `|T_j(x)|≤1` on [-1,1].

The exact sum gives the upward-rounded bound **2.4307835611064548e-14**.
This includes rounding the candidate coefficients to binary64 and holds
over the whole interval; it is not a sampled maximum. Exact inverse
translation and an independent Clenshaw recurrence at nine rational points
check the transformation machinery.

The new argument w ranges from -1 to **1.0019536018371582**. The bound is
computed in the **original x basis**, so it includes that extension. A bound
that incorrectly assumes `|T_j(w)|≤1` would be invalid. CKKS evaluation near
these endpoints still requires explicit testing. The bound concerns the
seed polynomial only: it excludes encrypted rounding/noise, amplification
by double-angle iterations, and final refresh output error.

## Count the actual evaluator work

The pinned `ComputeDegreesPS` selects `(k,m)=(6,4)` for degree 88 and `(7,3)`
for degree 44. The archived FIDESlib evaluator computes the baby basis,
giant squares, an artificial leading term, and a full recursive tree.
Counting only ciphertext products/squares gives:

| Work per backend bootstrap | Current | Proposed |
| --- | ---: | ---: |
| Build w | 0 | 1 |
| Baby basis | 5 | 6 |
| Giant squares | 3 | 2 |
| Artificial leading-term products | 3 | 2 |
| Recursive products | 7 | 3 |
| Double-angle squares | 6 | 6 |
| Total | **24** | **20** |

More specifically, general ciphertext multiplications decrease from 12 to
8; squares remain 12 after including w and the six double-angle steps.
Across 484 calls this removes **1,936** general ciphertext multiplications
from a static EvalMod inventory of 11,616 products. These counts exclude
scalar/linear-combination work, transforms, synchronization and ring transfer.
They do not imply proportional changes in modular kernels or elapsed time.

The upstream unnormalized polynomial depth estimate is 8 for degree 88,
and 7 for degree 44. Including the new square gives 8 again; adding six
double-angle steps gives 14 in both cases. This is a planner estimate,
not measured FLEXIBLEAUTO output metadata. Setup must continue budgeting
for the extra square; substituting only a shorter coefficient vector in
the setup calculation would incorrectly lower the declared budget.

The path is `PackedEvaluator::bootstrap_components` → GPU `Bootstrap` →
`approxModReductionSparse` → `evalChebyshevSeries` →
`applyDoubleAngleIterations`. The two archived backend source hashes match
the B300 source snapshot. The proposal targets this EvalMod entry point,
not the general model polynomial evaluator.

## Next bounded implementation and alternative ideas

The next implementation candidate should be **this single even-polynomial
EvalMod mechanism**, preserving the existing S2C-first order, both passes,
4096 correction multiplier, K=512, six double-angle steps, uniform-ternary
secret distribution, rings, primes and the two 0.001 model error gates.
The current experimental security qualification remains `not-set`; this
design screen adds no cryptographic certification.

The smallest useful rejection test is an encrypted seed/EvalMod comparison
at identical input levels and scales, including endpoints, phase-shift
neighborhoods, both signs and the modular-lift integer neighborhoods. Then
check the complete two-pass refresh and a subsequent product, inactive
slots, output level/scale, and the ordinary/refresh ring transfers. Neither
phase may lose the useful depth interval. Stage timers should separate
S2C, ModRaise, C2S, EvalMod, and the wrapper; preserve the existing B300
correctness synchronizations when collecting them.

Only a passing small case should reach a frozen model prefix and one full
A/B pair under the established B300 controls. Compare evaluation and
setup-inclusive wall time, IDs, both error gates, refresh counts, memory
and output levels. This note does not start that experiment. It supplies
the coefficients and rejection conditions needed to make it concrete.

Two further ideas remain distinct from this candidate:

- **Asymmetric two-pass precision.** With `v=u+e1` and
  `w=B2(4096(u−v))=−4096e1+e2`, the ideal corrected result is
  `u+e2/4096`. This motivates reducing first-pass work subject to a certified
  residual-domain bound. It does not permit arbitrary first-pass error or
  smaller modular-lift bounds. The mechanism follows the existing
  [Meta-BTS correction](https://eprint.iacr.org/2022/1167); its precision
  allocation has not been qualified here.
- **Combine correction before downward transfer/unpacking.** Form
  `h=4096*first+second` using the backend's integer-residue multiplication,
  then transfer once and unpack with `mask/4096`. In real arithmetic this
  preserves the result and could reduce downward transfers 484→242 and
  duplicate unpacking. The integer operation avoids the extra CKKS scale
  multiplication of a naive implementation, but compatible metadata,
  rounding/noise, dynamic range and unchanged output levels still need
  encrypted validation. This targets wrapper work, not the EvalMod core,
  and should be a separate trial.

EvalRound+, transform-budget changes, and new RNS parameters remain broader
circuit/backend alternatives. Their published gains are not additive with
this S2C-first path. No model/runtime or cryptographic-setting changes were
made during this screen; the artifacts and this note are local and uncommitted.
