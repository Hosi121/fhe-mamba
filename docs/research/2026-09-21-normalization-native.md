# From inverse-square-root certificates to encrypted normalization

This study keeps the frozen 49-site coefficient bundle
`89801adcc149d3183bf758fb8d022f4949b9b048ae5c665ff90a6892632c1c15`.
It tests arithmetic schedules on DGX Spark / GB10. It does **not** lower the
entire new language-model surrogate or complete the encrypted server protocol.

## Correct the arithmetic comparison first

The previous ledger counted the factored update `y*(a-b*v*y*y)` as three
dependent ct-ct products. A balanced implementation evaluates

```text
y_next = a*y + ((-b*v)*y)*(y*y)
```

with the two inner products in parallel. The existing native Newton kernel
already uses this reassociation. Layer 0's same-domain, same-accuracy fixed
Newton baseline therefore has **75 products / depth 50**, not depth 75.
The archived earlier report retains its literal factored-DAG count; it is
not the best Newton baseline. `fixed_newton_cost` now exposes both counts.
The accelerated coefficient sequence needs **36 products / depth 24** in
balanced form, versus depth 26 in the original coupled oracle. These are
ct-ct DAG depths; actual rescaling levels are measured separately below.

## Move public scaling off the coupled critical path

For the original coupled state `u=v*y²`, computing `f=a-b*u` puts a scalar
multiply before `u_next=u*f²`. FIDESlib charges a level for that scalar
product, so the apparent two-ct-product update actually consumes three
levels. Instead keep a negative, publicly weighted residual:

```text
r_i = -B_i*v*y_i²
f_i = a_i + r_i
y_(i+1) = y_i*f_i
r_(i+1) = ((b_(i+1)/b_i)*r_i) * f_i².
```

The public scaling of r and the square of f can run in parallel. With the
actual rounded ratios, `B_0=b_0` and
`B_(i+1)=B_i*float(b_(i+1)/b_i)`; in general `B_i != b_i` exactly.
`certify_weighted_schedule` forms these effective coefficients as exact
rationals and repeats the interval proof. All 49 recipes still certify the
same `1e-7` conditional relative-error target. This does not certify CKKS
rounding or domain membership.

The last correction recomputes `u=(v*y)*y` from the original variance, then
returns `y*(1.5-0.5*u)`. This multiplication order avoids forming `v*y²`
directly; the operations remain ordinary polynomial arithmetic. It needs
three ct-ct products and a scalar product. The weighted scheme trades two
additional final levels for fewer scalar products than balanced evaluation.
This is a public coordinate/scheduling transformation of established
Newton/Goldschmidt methods, not a new convergence family or novelty claim.

## Backend failures are retained

The first depth-56 probe selected ring 262144 at 128-bit classic security,
then hit a CUDA illegal memory access. The pinned FIDESlib has `MAXP=64`
arrays for the combined Q/P bases: the HYBRID key-switching extension matters
in addition to the requested depth. This probe now conservatively restricts
HYBRID-3 configurations to depth ≤44. It does not claim that every other
configuration below that limit is supported.

A first weighted implementation called `EvalSub(double, ciphertext)`.
Inspection of the pinned GPU API found two scalar multiplications by -1
around its addition, yielding the reverse sign and consuming extra levels.
That run exhausts its depth and remains a failure. The negative residual
above avoids the API entirely. The shared FIDESlib installation and full-model
binary are unchanged; this is a workaround in the isolated evaluator.

FIDESlib can exit with status zero after a CUDA fatal error. The runner
therefore requires a successful measurement artifact, checks binary/library
and recipe hashes, and preserves absent-output failures. A local regression
test covers that observed exit-status failure mode.

## Measure the actual normalization output

Standalone inverse-square-root tests are valuable stress cases. At layer 0's
interval `[1e-5,466.1573]`, both original coupled and balanced evaluation pass
the prespecified `1e-4` relative-error threshold. Balanced consumes level 27;
coupled consumes level 40 at identical depth-44 parameters. For the widest
interval `[1e-5,58265.8984]`, balanced has relative error `0.003559`, and the
weighted variant also fails this standalone gate. Those failures remain
recorded; the criterion was not relaxed.

However, an inverse square root is an internal ciphertext in RMSNorm.
The [OpenFHE decoder in the linked build](https://github.com/openfheorg/openfhe-development/blob/aa391988d354d4360f390f223a90e0d1b98839d7/src/pke/lib/encoding/ckkspackedencoding.cpp)
adds Gaussian noise estimated from imaginary-component error when decoding
real CKKS messages. Therefore separately decrypted intermediate errors include
fresh decoder noise and cannot be propagated as exact internal errors.
This observation does not establish an output-security proof for the complete
application, nor justify disabling that noise.

The second, separately labelled test encrypts signed synthetic x, computes
`v=x²+epsilon`, evaluates the inverse-square-root polynomial, and multiplies
by the original encrypted x **before** decrypting. It compares with
`x/sqrt(x²+epsilon)`, whose magnitude is at most one. Its derivative is
`epsilon/(x²+epsilon)^(3/2)`, bounded by `1/sqrt(epsilon)`. In contrast,
standalone `v^(-1/2)` has derivative magnitude `1/(2*v^(3/2))`.
These are different functions and different error gates. Correlated variance
formation and the final product must be included in the actual operation.

Real-decoder noise remains enabled. No conjugation or projection suppresses
the imaginary error estimate. This is a scalar RMS core: feature reduction,
rotations, learned gamma, prior-layer error and refresh are not included.

## Matched widest-interval ABBA probe

All four runs use the same binary and recipe, fresh keys, ring **131072**,
depth **40**, scale **59**, HYBRID with three digits, uniform ternary secret,
and OpenFHE-accepted **128-bit classic** parameters. Each evaluates 8,192
geometrically spaced variances with alternating input signs. There are zero
evaluation decryptions and no bootstrapping. Post-evaluation diagnostics run
in the same benchmark process that holds the secret key.

| Metric | Balanced (two runs) | Weighted (two runs) |
|---|---:|---:|
| Normalized-output maximum absolute error | `2.066e-9 / 1.945e-9` | `2.457e-8 / 2.790e-8` |
| Ct-ct products including variance/output | 47 | 47 |
| Public scalar products | 34 | 19 |
| Final consumed level | 35 | 37 |
| Evaluation seconds | `0.75539 / 0.74997` | `0.66160 / 0.66116` |

Weighted evaluation uses **44.1% fewer scalar products** and is faster in
these two small samples; balanced is shallower and has lower measured error.
This is a tradeoff, not a universal schedule preference. Full-model refresh,
packing, memory and security parameters can reverse the runtime choice.
Raw measurements are in the
[ABBA campaign](../../fhemamba/results/dgx/2026-09-21/normalization/abba/campaign.json).

The subsequent [all-site campaign](../../fhemamba/results/dgx/2026-09-21/normalization/all-sites/campaign.json)
passes **49/49** domains using the same weighted evaluator and fixed depth-40
parameters. Each site gets a fresh key and 8,192 samples: 401,408 scalar
normalization outputs in total. Worst sampled absolute error is
**`9.00146e-8`**, at layer 11's gated domain; consumed levels range from
25 to 37. Each process peaks around 6.33 GiB RSS. This is neither a uniform
CKKS error certificate nor 49 whole-model layer evaluations.

## Remaining model error

The [convolution-SiLU ablation](../../fhemamba/results/payload_exact_conv_ablation_20260921.json)
keeps joint gates, scheduled norms and gate SiLU fixed while making only the
24 convolution SiLUs exact. At the same 4,096-token development window,
maximum logit error falls **4.5000 → 0.17742**; all 4,095 next-token argmax
choices agree, with PPL 17.56699 versus 17.56702. This uses exact nonlinearities
and is a diagnostic, not a new deployable surrogate.

The remaining convolution error is mostly approximation rather than float32
evaluation: a float64 public-domain screen still has error about `0.00315`
in layer 10. A useful next degree-allocation model follows from differentiating
the [cosh product](https://dlmf.nist.gov/4.36.E2):

```text
SiLU(x) = x/2 + 2*x² * sum_(k>=0) 1/(x² + ((2k+1)*pi)²).
```

The odd part is exactly linear; the even part's nearest poles are ±iπ.
After scaling a public radius R to `[-1,1]`, coefficient decay depends on
`asinh(pi/R)`. This suggests public, per-layer degree selection instead of
one degree for every radius. No new SiLU coefficients or uniform error
certificate are promoted by this observation.

The next integration gate is a packed vector RMSNorm with actual reductions,
gamma and refresh, then the matching whole-model circuit. An isolated 128-bit
operator does not establish the full model's 128-bit or protocol gate.
