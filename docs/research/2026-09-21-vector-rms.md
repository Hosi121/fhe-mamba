# Packed vector normalization and the refresh boundary

The preceding scalar study did not test feature reduction or learned gamma.
This follow-up uses the real checkpoint's 768-dimensional block/final norms
and 1,536-dimensional gated norms. The 49 inverse-square-root recipes remain
frozen. No coefficients or text-dependent domains are fitted in this study.

## Algebra and packing

Write `v = mean(x²) + epsilon` and `F(x) = gamma * x / sqrt(v)`.
The exact Jacobian before gamma is

```text
J = v^(-1/2) I - x*x^T / (d*v^(3/2)).
```

Its radial eigenvalue is `epsilon/v^(3/2)`; the tangential eigenvalues are
`1/sqrt(v)`. Thus the scalar experiment's radial cancellation does not describe
all vector perturbations. A global Euclidean Lipschitz bound after gamma is
`max(abs(gamma))/sqrt(epsilon)`. This is an exact-function sensitivity bound,
not a bound on accumulated CKKS error or on the polynomial's derivative.

Also `abs(F_i(x)) <= abs(gamma_i)*sqrt(d)`. If the inverse-square-root
polynomial has the previously certified non-expansion property, the same
bound holds in exact real arithmetic. The probe uses a public global bound
`max(1, 1.1*max(abs(gamma))*sqrt(d))` to scale outputs before refresh. The
10% margin is numerical headroom, not a probabilistic CKKS noise proof.

The [original RMSNorm paper](https://arxiv.org/abs/1910.07467) is prior art for
the normalization and its rescaling properties. The Jacobian and component
bounds above are elementary consequences of that formula, including epsilon.

With 8,192 slots, pad the feature width to a power of two P and let
`L=8192/P`. Slot `j*L + lane` stores feature j of one independent vector.
Rotations by `L,2L,...,8192/2` preserve lanes. Summing these rotated squares
reduces all features and replicates each lane's result without masks or a
second broadcast phase. Padding is zero; division uses the **true width d**.
The 768-dimensional case uses eight lanes and ten rotations; the
1,536-dimensional case uses four lanes and eleven rotations.

These are independent vectors, not successive encrypted autoregressive steps.
The full-model kernel's current packing is different and requires an explicit
layout integration before using this operator there.

The public gamma product can be scheduled before the final ciphertext product:

```text
before: (gamma*x) * p(v)
after:  gamma * (x*p(v)).
```

Both have the same polynomial and multiplication counts. The first puts the
gamma work on the short numerator branch, saving one critical CKKS level.
It does not move gamma inside the variance or modify the model's weights.

## Inputs and gates

The fixture exporter runs the exact checkpoint on a previously unused
128-token WikiText-2 window at offset 180,224. It samples eight positions per
site and records all 49 norms' original inputs and learned gamma. Eight
additional deterministic vectors cover zero, epsilon-scale energy, sparse
energy concentrated at the largest-gamma component, alternating signs and
nonuniform dense inputs. Their variances stay in the frozen public domains.
Each site therefore has 16 vectors; no clipping repairs an out-of-domain input.

Client-side plaintexts are freshly encrypted. The encrypted circuit squares,
reduces, divides by d, adds the checkpoint epsilon, evaluates the polynomial,
and applies gamma. Exact and polynomial references are computed independently
from the original vectors using host double precision, with long-double
variance accumulation. These are mathematical references on the exported
float32 activations/weights, not bitwise Torch float32 outputs.
Normal decoding remains enabled, including its
imaginary-error-based noise. Active and padding outputs must both meet the
prespecified absolute-error threshold `1e-4`. Diagnostics decrypt only after
the corresponding encrypted batch finishes.

The program supports a separate output-refresh mode. This gate includes
refresh error; passing without refresh cannot satisfy it. Key generation,
setup, evaluation and refresh are measured separately. Binary, runtime/static
library, recipe, fixture and runner hashes bind each report to its inputs.

## Initial results and the failed refresh

At the widest block-norm interval, balanced evaluation with gamma before the
final product has maximum absolute error `1.42772e-8`, including stress cases;
on the eight real inputs it is `3.57179e-10`. Gamma after the final product
uses level 37, versus 36 before it, at the same 47 ct-ct / 36 ct-pt products
per batch. These are individual runs with fresh keys, not a runtime speedup
claim. Weighted evaluation uses 47 ct-ct / 21 ct-pt products and level 38,
with error `8.05316e-8`.

The [all-site campaign](../../fhemamba/results/dgx/2026-09-21/vector-rms/all-sites/campaign.json)
passes **49/49** without refresh: **784 vectors / 897,024 active components**.
The worst absolute error is **`6.95634e-5`**, at layer 17's gated norm; on the
checkpoint inputs alone it is **`1.17633e-6`**. At the worst site, encrypted
versus polynomial error is only `1.38776e-7`: the polynomial approximation
amplified by gamma dominates. Consumed levels range from 26 to 38, and peak
RSS is at most 11.67 GiB. Every site gets a fresh key.

The inverse-square-root relative certificate `delta` yields the conservative
absolute output bound `delta*max(abs(gamma))*sqrt(d)`. Thus a uniform `1e-7`
inverse target is not, by itself, a uniform `1e-4` vector-output certificate:
the largest checkpoint gamma is 53.53125. Future coefficient planning should
allocate error by this public sensitivity, leaving room for CKKS and refresh
error. The current report retains the original coefficients and sampled gate.

The first ordinary output-refresh test **fails**: maximum error `0.0123022`,
with padding error `0.0128778`, despite pre-refresh error `3.47700e-8`.
Parameters are ring 131072, depth 40, scale 59, uniform ternary secret,
OpenFHE-accepted 128-bit classic security and bootstrap budgets `[4,4]`.
The process peaks at 42.94 GiB RSS. The two batches take 1.62 seconds for RMS
evaluation plus 1.82 seconds for their two physical bootstraps/scaling; setup
takes 35.98 seconds. All these numbers belong to an isolated benchmark.

The refresh returns level 22 after restoring output scale. That leaves less
depth than another full normalization consumes. Merely connecting two such
operators does not yield an executable repeated model circuit; internal
refresh scheduling and its numerical behavior remain necessary.

## Residual error cancellation

[Meta-BTS](https://eprint.iacr.org/2022/1167) uses the original ciphertext as
an encrypted error reference. For a normalized input u and `b=Bootstrap(u)`,
form `r=2^alpha*(u-b)` and return `b+2^(-alpha)*Bootstrap(r)`. Subject to
alignment and range conditions, the first refresh error cancels. It does not
remove errors already present in u. This mechanism already exists in the
full-model kernel; the new probe tests it on the packed normalization output.

Inspection of the pinned FIDESlib GPU API confirms that its `EvalBootstrap`
implementation ignores the `numIterations` and `precision` arguments. The
probe therefore calls two physical bootstraps explicitly, preserving the
original encrypted reference and reserving a live residual-amplification
level. Unity products/rescales align ciphertexts without simply changing their
metadata. Their cost is reported. This is established error cancellation,
not a new bootstrap construction or a complete output-security proof.

At `alpha=12`, the first Meta-BTS probes pass for the widest block interval
(`2.96439e-6`) and layer 11's gated norm (`5.58101e-5`), but **fail** for the
largest-gamma site, layer 20's gated norm (`1.81119e-4`). Default-decoded
residual maxima in these probes can exceed one; these diagnostics include
decoder noise. No uniform residual-domain or bootstrap-error bound is proved
by these samples. The ordinary-refresh failure and this Meta-BTS failure are
retained.

## Public component scaling around refresh

A single global output bound multiplies every component's refresh error by
the largest gamma. An alternative uses public diagonal coordinates:

```text
B_i = 1.1*sqrt(d)*max(abs(gamma_i), max(abs(gamma))/64)
u = diag(1/B_i) * F(x)
refreshed_output = diag(B_i) * Refresh(u).
```

The public floor keeps the coordinate condition number at most 64, including
zero-gamma components. Zero padding uses the floor; an all-zero gamma vector
uses unit scales. The exact component bound still puts each input coordinate
inside `[-1/1.1,1/1.1]`. These two public diagonal products replace the two
global scalar products; they do not remove imaginary components, disable
decoder noise, change gamma, or alter the inverse-square-root polynomial.

If coordinate errors were isotropic with variance sigma², the restored error
energy would be `sigma²*sum(B_i²)`, compared with `sigma²*d*B_max²` for a
global bound. This is a cost/error model, not a proof that CKKS bootstrap
errors are isotropic. The worst-case infinity bound is unchanged. The actual
benefit must be measured with the same key parameters and output criterion.

The matched layer-20 **global / channel / channel / global** experiment uses
one binary, frozen inputs, `alpha=12`, depth 40 and fresh keys for every run:

| Quantity | Global coordinates, two runs | Component coordinates, two runs |
|---|---:|---:|
| Output maximum absolute error | `1.59564e-4 / 1.57742e-4` | `5.06395e-5 / 5.32236e-5` |
| Gate at `1e-4` | fail / fail | pass / pass |
| Padding maximum error | `1.72464e-4 / 1.44974e-4` | `1.48989e-5 / 1.39172e-5` |
| Physical bootstraps per four-vector batch | 2 | 2 |
| Post-refresh consumed level | 23 | 23 |
| Refresh seconds for all four batches | `6.97424 / 6.97022` | `7.04519 / 7.09883` |

The observed error improves about threefold. There is no measured runtime
speedup: the two diagonal plaintext products replace scalar products and
include their encoding cost. Every batch has the same 12 unity-alignment
products and 16 total refresh ct-pt products. The early JSON field
`refresh_scalar_products` counts all ct-pt products, including the two diagonal
ones; subsequent probe builds label it `refresh_ct_pt_products`. The raw
earlier records retain their original spelling.

The component scales are between 36.0592 and 2307.79. Their active-component
RMS/global ratio is 0.1577, a public coefficient statistic; it is not an
independently validated bootstrap-error distribution. The improvement occurs
with the normal decoder and its noise still enabled.

Fresh-key confirmations with component coordinates also pass at layer 11's
gated norm (`1.81627e-5`), layer 17's gated norm (`5.70365e-5`), layer 20's
gated norm (`5.56426e-5`) and layer 23's block norm (`1.24536e-6`). The
confirmation build changes only the ct-pt counter's label from the ABBA
build; numerical operations are unchanged. These are **four selected sites**,
not a 49-site refresh campaign. All use two physical bootstraps per batch,
and the output remains at consumed level 23 / scale degree 2.

Raw files, including the failed controls, are retained in the
[vector result directory](../../fhemamba/results/dgx/2026-09-21/vector-rms/).
The [input manifest](../../fhemamba/results/vector_rms_probe_inputs_20260921.json)
binds checkpoint, selected token positions, gamma, recipes and fixtures.
Final validation: **220 tests**, **11 native host contracts**, **86.32%**
active-package coverage, and **196** dated artifacts with no validation
errors or warnings. Source snapshots and binary/library hashes are preserved
locally and on Spark.

The full-model binary, its frozen payload and the previously measured
plaintext language-model quality remain unchanged. The vector fixtures do
not include preceding-layer CKKS noise, encrypted token selection or a
separate server process without a secret key.
