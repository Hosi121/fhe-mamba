# SSM algebra and cryptographic design

Reviewed 2026-09-21 against primary papers, official Mamba code, and this
repository. The goal is encrypted language-model computation, including an
eventual encrypted token-selection loop. This note separates algebraic
identities, measured results, and proposals requiring training or backend work.
It supplements the July surveys rather than treating their timings as new
evidence. No cryptographic security reduction or new language-model quality
result is claimed here.

## 1. Which structure is actually useful?

For one Mamba-2 head, use state `S ∈ R^(P×N)` and column vectors `b,c`:

```text
δ_t = softplus(z_t),  a_t = exp(A δ_t),  A < 0
u_t = δ_t x_t
S_t = a_t S_(t-1) + u_t b_t^T
y_t = S_t c_t + D x_t
```

The useful restrictions are the scalar transition and rank-one write, not
simply the absence of attention. The scalar commutes with a change of state
basis; the write admits contraction before materialization. The SSD dual
form also permits parallel prompt processing. Neither representation supplies
unknown future autoregressive tokens. This is the structure of
[Mamba-2/SSD](https://arxiv.org/abs/2405.21060).

Mamba-3 introduces an exponential-trapezoidal update, complex rotation, and
MIMO writes. In its real rotating frame, with `v_t = x_t b_t^T`:

```text
α_t = exp(A_t δ_t)
β_t = (1-λ_t) δ_t α_t,  γ_t = λ_t δ_t
S_t = α_t S_(t-1) + β_t v_(t-1) + γ_t v_t
```

Rotations are absorbed into B/C using accumulated angles. The trapezoidal
term can replace the external convolution; MIMO increases write rank without
enlarging the state. The claimed GPU advantage comes from using spare
arithmetic capacity in a memory-bound kernel. Second-order quadrature accuracy
requires conditions on λ, and is not a statement about CKKS error. See the
[paper, Sections 3.1–3.3](https://arxiv.org/html/2603.15569v1).

The [official module](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py)
also contains BC RMSNorm, B/C biases, sigmoid mixing, softplus step sizes,
piecewise reciprocal activation for input-dependent A, and angle/B/x caches.
Its `is_outproj_norm` option changes the postprocessing. Replacing Mamba-2
requires accounting for these operations, not just deleting its convolution
and gated norm. The checked module is recorded with a source hash in
[`sources`](2026-09-21-primary-sources.json).

**Inference for FHE:** keep Mamba-2 as the compatibility anchor. For a new
model, compare Mamba-3 SISO and a bounded complex polynomial SSM before MIMO.
Count ct-ct products, rotations, levels, bootstrap precision, and *all* carried
ciphertexts. Ordinary FLOP counts and state dimension alone are insufficient.

## 2. Exact candidate: defer the large state update

After a public checkpoint time τ, expanding the recurrence gives

```text
p_t    = product(a_i, i=τ+1..t)
w_j,t  = product(a_i, i=j+1..t)       [empty product = 1]
S_t    = p_t S_τ + sum_j w_j,t u_j b_j^T
y_t    = p_t (S_τ c_t) + sum_j w_j,t u_j (b_j^T c_t)
```

This identity allows an immutable dense encrypted checkpoint plus a short
encrypted factor buffer. The readout can compute `b_j^T c_t` first. A public
window length determines when to merge the buffer back into dense state.
[ReplaySSM](https://tridao.me/blog/2026/replayssm/) uses the same checkpoint and
input-history principle for ordinary GPU inference; this is not a novelty
claim. Our proposed FHE lowering must additionally preserve the actual
polynomial circuit, levels and noise of every factor.

The implementation in [`ssm_algebra.py`](../../fhemamba/src/fhemamba/ssm_algebra.py)
updates products directly. It never divides by a prefix product and never
replaces a product of polynomial decays with an exponential of a sum.
Those distinctions matter: the current head-clipped surrogate has exact zero
decays, and `P(x+y) != P(x)P(y)` for its exponential approximation. A mathematically
valid optimization of the exact exponential can silently change the surrogate.

On factors from the actual 24-layer checkpoint, 64 tokens and windows
2/4/8/16, all 96 comparisons pass. Worst readout difference is `2.843e-13`
and final-state difference `1.137e-13` in float64. Nonzero initial states,
zero decays and partial windows have separate tests. This establishes
reassociation parity, **not encrypted speed or CKKS precision**. The
[`raw report`](../../fhemamba/results/ssm_algebra_20260921.json) records source
and checkpoint hashes and input token IDs.

For `H=24, P=64, N=128`, dense state is 196,608 scalars. An eight-entry
buffer adds at most `H[1+8(P+N+1)] = 37,080` scalars in this oracle; it does
not remove the dense checkpoint. Shared B/C groups can reduce buffer storage
further. Compact vectors do not automatically mean compact ciphertexts:
expansion, masks, cross-head layout and noise levels must be measured.

The saving target is fewer dense **writes and refreshes**, not removal of
dense reads. `S_τ c_t` remains necessary every token. Factor dot products,
their broadcasts and the flush are new work. Compare a window of k steps:

```text
saved dense update/refresh work
  > factor-readout work + factor refreshes + amortized flush/packing work
```

Products updated online still accumulate depth. A balanced recomputation
from the buffered known decays can reduce flush depth, with extra operations.
The native baseline already refreshes according to level availability, so
assuming six saved bootstraps per token would be wrong. Implement slot-exact
layouts for k=2/4/8, preserve the unmodified baseline, then run encrypted
readout/state parity at 2/5/16/64 tokens before timing promotion.

## 3. Exact candidate: exploit state-basis freedom

For any fixed invertible public `T ∈ R^(N×N)`, define

```text
S' = S T^T,   b' = T b,   c' = T^(-T) c
S'_t = a_t S'_(t-1) + u_t b'_t^T,   S'_t c'_t = S_t c_t
```

This follows by substitution because `a_t I` commutes with T. A diagonal T
balances state coordinates; an orthogonal T spreads outliers while preserving
Euclidean norm. This is an exact identity, not low-rank truncation or training.
It is more general than the current one-scale-per-ciphertext normalization.

In the current model B/C follow SiLU. A general basis transform cannot be
folded backward through that nonlinearity into `in_proj`. A diagonal transform
may be folded into the existing B/C expansion masks; a Walsh-Hadamard transform
needs its own encrypted additions/rotations. Both must include inverse changes
to the readout and the initial state.

Smaller stored coordinates alone do not guarantee accuracy: multiplying C by
the inverse transformation can amplify refresh error. If stored-state error
is bounded by ε in each coordinate, a row readout error is bounded by
`ε ||c'||_1`. Choose T using held-out state **and readout sensitivity**, then
optimize the resulting output error at a fixed bootstrap precision. Arbitrary
input-dependent normalization would require encrypted reciprocals and is a
different, more expensive algorithm.

A particularly cheap diagonal candidate has a stronger conditional bound.
For one head, let public calibration bounds satisfy `|S[p,j]| <= M_j`, with
positive floors, and let `M=max_j M_j` be the current headwise bound. Normalize
coordinate j by `1/M_j`, and multiply its readout coefficient by `M_j`. If a
refresh introduces at most ε absolute error per normalized coordinate, its
readout-error contribution is bounded by

```text
coordinate scaling: ε sum_j M_j |c_j|
headwise scaling:   ε M sum_j |c_j|.
```

The former is no larger under those assumptions. Existing expansion masks
can potentially carry these public factors without a new ciphertext product.
This is a concrete next alternative to spending more precision on every
bootstrap. It still needs per-coordinate held-out bounds: calibration maxima
are not universal input guarantees. The comparison concerns added refresh
error only; it does not eliminate old error, coefficient-encoding error, or
noise changes elsewhere in the circuit.

For Mamba-3, unrestricted basis changes generally destroy its cheap block
rotation structure. Pairwise scalar/compatible block transforms remain useful;
the full Mamba-2 freedom should not be assumed there.

### Implemented variant: scale the channel rows

The running kernel actually uses one scale per ciphertext containing **four
heads**, rather than one per head. Its six ciphertext groups are independent
of the checkpoint's `n_groups=1` B/C grouping. A cheaper first implementation
than the state-axis transform above is a left diagonal transform:

```text
M[h,p] > 0 is fixed and public
U[h,p,n] = S[h,p,n] / M[h,p]
U_t = a_t U_(t-1) + (dt_t x_t / M) b_t^T
y[h,p] = M[h,p] sum_n U[h,p,n] c[h,n].
```

`--row-normalized-state true` implements this coordinate map. The `1/M`
factor occupies the existing X extraction mask before state-axis replication;
the M factor occupies the existing readout mask after state-axis reduction.
There are **no additional ciphertext products, plaintext products, rotations,
or levels** in this layout. Debug comparisons restore the actual row scales,
not each group's maximum. Nonzero initial states must be scaled on import;
the current native evaluation starts from zero state.

The row map is also algebraically compatible with right-acting Mamba-3 state
rotations: `(D^-1 S) R = D^-1 (S R)`. This is different from an unrestricted
change of basis on the rotating state axis. Range calibration would need to
respect complex magnitudes or Euclidean pair norms, since a rotation can
increase an individual real coordinate while preserving pair magnitude.
The present native implementation and measurements cover Mamba-2 only.

At equal normalized refresh-error bound epsilon, the added error for row p is
bounded by `epsilon M[h,p] ||c_h||_1`, compared with
`epsilon M_group ||c_h||_1`. This is a conditional improvement to a local bound,
not a promised reduction of final model error. Earlier errors, coefficient
encoding, range escapes, and randomized refresh errors still matter.

The [calibration report](../../fhemamba/results/state_coordinate_calibration_20260921.json)
uses the first 512 WikiText-2 **train** tokens, taking maxima over both exact
and frozen-poly trajectories. The mean row/group scale ratio is 0.095974;
per-layer means range from 0.057 to 0.144. The held-out test split is separate.
The cloned payload preserves all coefficients, weights and evaluation vectors;
only state bounds change. Compare group and row modes using this **same** new
payload (`e788c496…`), since comparison with older calibration would confound
the coordinate map with different bounds. Calibration maxima remain empirical.

The [reference-coverage audit](../../fhemamba/results/state_scale_coverage_20260921.json)
finds an important limitation: the five-step public benchmark stays below
0.8731 times its group scale, but reaches 9.0716 times some row scales.
Of 23,592,960 state coordinates over all layer/token pairs, 1,912 exceed the
row scale's 1.1 margin, versus none for group scales. These references were
used only to audit coverage, never to alter calibration. This does not prove
a bootstrap-domain failure by itself, but invalidates an assumption needed
for the simple equal-normalized-noise comparison. Broader independent
calibration, or shrinkage toward a group bound with a stated amplification
cap, is needed before promoting fine-grained scales.

### A length-independent state bound from coupled gates

For one Mamba-2 head write `c=-A>0`, `a_t=exp(-c*dt_t)`, and suppose
`0<=dt_t<=D` and each coordinate of `x_t b_t^T` is bounded by V. From
`exp(x)>=1+x`,

```text
1-exp(-c*dt) >= c*dt / (1+c*dt)
dt / (1-exp(-c*dt)) <= dt + 1/c <= D + 1/c = K.
```

At dt=0 the ratio has the removable limit 1/c. Therefore
`|S_t| <= a_t |S_(t-1)| + (1-a_t) K V`, and
`|S_t| <= max(|S_0|, K V)` independently of sequence length. Bounding decay
and write separately can miss this coupling and yield an unnecessarily loose
linear-in-length state bound. Deriving useful V and D from public projection
and convolution bounds remains necessary; a finite bound can still be too
loose for efficient CKKS scaling.

This suggests a compile-time constraint on direct polynomial gates. Certify
`0<=a(z)<=1` and `|b(z)|<=K*(1-a(z))` on their common input domain, rather than
only fitting each scalar function independently. The new
`certify_affine_state_invariant` checks these three polynomial inequalities
using exact rational Bernstein bounds. It catches the case a=1 with b>0,
where repeated writes accumulate despite a decay bounded by one. Tests cover
reset, perfect-memory and signed-write endpoints. This is a reusable
certificate primitive; the existing composite-decay fits have not passed it.

A future architecture can enforce the constraint by construction with
`S_t=a_t S_(t-1)+(1-a_t)V_t`, bounded V and a in [0,1]. Choosing
`V_t=x_t b_t^T/c` corresponds to an exact-zero-order-hold write coefficient
instead of the current Euler dt coefficient, so it **changes the model** and
requires adaptation/training. Rewriting the current model exactly would use
`V_t=dt_t/(1-a_t)*x_t b_t^T`, introducing a removable-singularity composite
gain to approximate. The invariant is promising for bounded FHE-native
models; neither construction is a free substitution for the checkpoint.

The coupled form also admits `S_t=V_t+a_t*(S_(t-1)-V_t)`. An error in a
then contributes `delta_a*(S_(t-1)-V_t)`, which vanishes at a constant-input
fixed point. Independently approximated decay and write gates instead inject
`delta_a*S_(t-1)+delta_b*V_t`. This correlation of arithmetic errors is a
potential conditioning benefit, not an assumption that ciphertext noise
vanishes. For rank-one V, the centered form uses one encrypted outer product
and one decay/state product, versus the current two write products plus one
decay/state product. Actual packing, refresh and quality costs still need a
new trained-model comparison. A Mamba-3 rotation cannot simply be inserted in
this centered formula while retaining the same convex-state invariant.

## 4. Complex CKKS slots and Mamba-3

Two real state coordinates can be represented by `z=s₀+i s₁` in a CKKS slot.
The real two-by-two rotation becomes multiplication by `exp(iθ)`:

```text
z_t = a_t exp(iθ_t) [z_(t-1) + (1-λ_t)δ_t v_(t-1)] + λ_t δ_t v_t
readout = Re(sum_j conjugate(c_j) z_j)
```

Here B and C encode their real coordinate pairs with the same sign convention.
The lag write is rotated too; omitting that factor changes trapezoidal
discretization. The new complex oracle is tested against independent real
two-by-two matrix arithmetic. It is a recurrence oracle, not an implementation
of a trained Mamba-3 block. The rotating-frame derivation is explained by the
authors in their [methodological discussion](https://tridao.me/blog/2026/mamba3-part2/).

This changes the earlier blanket decision to defer complex dynamics: **test
direct complex arithmetic and rotating-frame arithmetic against each other**.
CKKS already represents complex messages, so a complex multiply need not be
implemented as four separately packed real multiplies. But constructing
encrypted phases, conjugation keys, slot utilization and complex bootstrap
support remain costs. The repository's existing *pairing of unrelated real
states for refresh* is not a Mamba-3 complex-state implementation.

The rotating frame only rotates small B/C vectors, but accumulates encrypted
angles. Polynomial sine/cosine then need a growing domain or encrypted range
reduction. Direct local rotations avoid that accumulated-angle domain but
multiply the large state. A unit-phasor accumulator trades angle range for
multiplicative depth and radial drift. These are different FHE circuits.

### A useful negative result: low degree can erase memory

Our elementary cubic candidate is

```text
p₃(θ) = 1 - θ²/2 + i(θ - θ³/6)
|p₃(θ)|² = 1 - θ⁴/12 + θ⁶/36 <= 1    when |θ| <= sqrt(3).
```

It is contractive on that interval and takes two sequential powers (`θ²`,
`θ³`). Nevertheless at θ=0.5 its magnitude after 1,024 transitions is about
0.086. An exactly unitary transition would retain magnitude 1. At θ=2 it
expands instead. The quadratic candidate already expands at every nonzero
angle because `|1+iθ-θ²/2|²=1+θ⁴/4`.

Thus a local approximation error or non-expansion certificate cannot certify
long-context memory. For real polynomials p,q, the identity `p²+q²=1` on an
interval implies constant p,q: extend the polynomial identity and inspect
the highest-degree nonnegative squared coefficients. An exactly unitary,
nonconstant polynomial phase is unavailable. Rational Cayley rotations
preserve unit modulus in exact arithmetic, but require encrypted division;
approximating that division brings the error problem back.

Train and evaluate phase **and radial** error over the target horizon; compare
memory retrieval and state-tracking quality at matched encrypted cost. A new
FHE-oriented model may bound each local angle and learn with the actual
polynomial, but it is not an exact checkpoint conversion.

### A different candidate: polynomial shears instead of a polynomial phasor

An additional algebraic route is to relax Euclidean norm preservation while
preserving area. This is a candidate derived here for evaluation, not a
novelty claim or an adopted Mamba-3 conversion. Apply three shears:

```text
x₁ = x - (θ/2)y
y₁ = y + θ x₁
x₂ = x₁ - (θ/2)y₁

[x₂; y₁] = M(θ)[x; y]
M(θ) = [[1-θ²/2, -θ+θ³/4], [θ, 1-θ²/2]].
```

Each shear has determinant 1, so their product has determinant 1 for every θ,
without a reciprocal, square root or polynomial normalization. For a *fixed*
`|θ| < 2`, let `k=1-θ²/4 > 0`. Direct multiplication gives
`Mᵀ diag(1,k) M = diag(1,k)`. The invariant is `x²+k y²`; a fixed coordinate
scaling turns M into a rotation of angle `2 asin(θ/2)`.

The [implemented oracle](../../fhemamba/src/fhemamba/phase_algebra.py) at
θ=0.5 over 1,024 steps keeps Euclidean norm between 1.0000 and 1.0328,
instead of the cubic phasor's collapse to 0.0863. But its frequency error
accumulates to **5.489 radians**. Preserving amplitude does not preserve
the checkpoint's computation.

There is also a decisive switching counterexample, even for small angles.
Three steps at θ=0.5 followed by sixteen at θ=0.1 yield

```text
F = M(0.1)^16 M(0.5)^3
det(F) ≈ 1,   trace(F) = -2.000343128,
spectral_radius(F) = 1.018696078 > 1.
```

Repeating that public schedule grows along an eigenvector. Each individual
matrix has unit-modulus eigenvalues, but the product need not. Thus local
eigenvalue bounds and unit determinant do not establish stability for a
selective SSM. The [raw screen](../../fhemamba/results/phase_schedules_20260921.json)
and independent tests retain both the useful fixed-angle behavior and the
counterexample.

For a separately trained architecture, compare fixed per-head phase, learned
phase with certified damping, and this shear recurrence at matched quality.
Three serial ciphertext products and real-coordinate access may be more
expensive than a direct complex product. CKKS perturbations also break the
exact invariant. This candidate remains unpromoted; it demonstrates why the
search should optimize long-horizon behavior and arithmetic structure together.

## 5. Fuse functions before approximating them

### Recheck the existing head-pruning argument first

The legacy head mask is a model approximation, not a proof that the original
head is memoryless. For `A<0` and `δ in [δ_min,δ_max]`, monotonicity gives

```text
minimum decay = exp(A δ_max)
maximum decay = exp(A δ_min).
```

The current planner deletes a head when the **minimum** falls below `exp(-32)`.
This does not bound its maximum. We corrected the contrary explanation in
`HeadMaskedDecay` and label new exported metadata
`head_pruning_policy=legacy-largest-step-approximation`, without silently
changing the measured polynomial circuit.

The [payload audit](../../fhemamba/results/decay_composition_20260921.json)
finds 47 pruned heads. None has a maximum below `exp(-32)` on its exported
softplus-input interval; those maxima range from 0.97701 to 0.999999745.
They are analytic extrema evaluated in float64 on a shared layer interval,
not evidence that each head actually reaches that endpoint. Nevertheless,
the interval cannot supply the claimed justification for pruning.

Any uniformly negligible-carry argument must use the smallest admissible
step size and also account for the state magnitude. Empirical pruning instead
needs explicit quality evidence, including long-context retrieval and state
tracking. The old PPL `22.307→22.333` artifact does not bind the current
payload's degrees, head mask and coefficients; it is not that certificate.
This is a reason to test a bounded composite over all heads instead of
attributing a speedup to mathematically free removal of memory.

### Direct composite approximation

Mamba-2 has a public A per head, so

```text
exp(A softplus(z)) = (1 + exp(z))^A.
```

Approximate this composite directly, per head, in parallel with the softplus
needed by the write path. The current serial softplus-square → exponential
polynomial → repeated-square chain is not mathematically compulsory.
This can reduce the critical depth; it changes the surrogate and therefore
requires a new quality certificate. Head-specific coefficients require
plaintext coefficient vectors, not the current scalar-coefficient evaluator.
Compare weighted state/readout error, coefficient magnitudes, ct-pt products
and levels, not degree alone. This simplification does not carry over directly
to input-dependent A in the official Mamba-3 module.

A bounded decay approximation is particularly valuable. On `x ∈ [0,1]`, a
Bernstein polynomial with all coefficients in `[0,ρ]` lies in `[0,ρ]`, since
its basis is a nonnegative partition of unity. This gives a range certificate
*conditional on the input interval*. It does not make an unbounded input safe.
Direct Bernstein approximation may converge too slowly; constrained fitting
and conversion to a depth-efficient evaluation basis should be compared.
No nonconstant polynomial can be bounded on the entire real line.

An implemented [composition screen](../../fhemamba/experiments/probe_decay_composition.py)
fits every layer/head on its exported softplus domain and evaluates on 4,097
independent uniform points. Among 529 existing active heads, degree 64 reaches
worst sampled error `1.100e-4`, but 126 heads leave `[0,1]` by more than
`1e-12`; the overall sampled range is approximately
`[-2.353e-5, 1.000006603]`. With all 576 heads, worst error is `1.894e-3`.
These are [negative screening results](../../fhemamba/results/decay_composition_20260921.json),
not a certified maximum or a reason to silently clip encrypted values. The
screen approximates the exact composite; it does not claim equality to the
existing squared/head-clipped surrogate. Constrained fitting or narrower
certified per-head domains are candidates for controlling these violations;
they must be followed by a new model-quality gate.

A tiny range violation is not itself a proof of unusable finite-horizon error.
For an isolated recurrence with fixed inputs, if `|a_t| <= 1+η`, local errors
are at most ε, and initial error is zero, the bound is
`ε[(1+η)^T-1]/η`, tending to `T ε` as η tends to zero. Allowing slight expansion
may be a legitimate measured tradeoff for a declared horizon. This bound does
not certify the input-dependent gates and stacked nonlinearities of the full
model; report them separately rather than accepting or rejecting a polynomial
solely from its degree or a sampled overshoot count.

## 6. CKKS error, refresh and normalization

CKKS encodes scaled approximate complex arithmetic in a polynomial ring.
Multiplication adds message-dependent error, rescaling consumes modulus, and
rotations require automorphisms/key switching. The original
[CKKS construction](https://eprint.iacr.org/2016/421) and
[full-RNS implementation](https://eprint.iacr.org/2018/931) explain why bit-width
or plaintext FLOP reductions do not directly predict ciphertext latency.

At the ring level, take `R_Q = Z_Q[X]/(X^N+1)` and an encoding map E from
complex slots to polynomial coefficients. A two-component ciphertext obeys
the schematic invariant

```text
c₀ + c₁ s = round(Δ E(z)) + e  (mod Q).
```

The secret key s is not needed to add or multiply these expressions.
Multiplication initially introduces an `s²` component; relinearization uses
evaluation keys to return to two components. A slot rotation is a ring
automorphism followed by key switching, not a cheap array index change.
Rescaling divides by a modulus prime and rounds, changing both the remaining
modulus and the message scale. Merely dropping a modulus limb does not perform
that division. Thus `(level, scale degree, magnitude, error)` is a more useful
planning state than level alone.

For this project, the resulting design inference is concrete: an algebraic
rewrite is useful when it reduces work on wide, high-modulus ciphertexts or
allows precise cancellation before expensive transforms. Moving a multiply
onto fewer packed slots helps only if the new layout actually occupies fewer
ciphertexts; zeros do not shrink a ciphertext. Public weights remove a
ciphertext operand, but a general real-valued plaintext product still changes
scale and can require rescaling.

For message errors εx, εy, one useful local bound is

```text
|(x+εx)(y+εy)-xy| <= |x||εy| + |y||εx| + |εx εy|.
```

A refresh of an already perturbed message behaves schematically as
`B(x+e) = x+e+η_B`. It renews computational capacity; it does not know the
unperturbed model activation x. Repeated refresh cannot undo a polynomial
extrapolation or an earlier wrong state.

For inverse-square-root Newton refinement, let `q=1-v r²`. In exact arithmetic,

```text
r' = r (3-v r²)/2
q' = q²(3+q)/4.
```

For positive r and `0 < v r² < 3`, the first refinement keeps the positive
root branch and subsequent residuals converge. Outside it, a sign flip or
divergence is possible. The derivative with respect to variance is
`∂r'/∂v = -r³/2`; small variance can amplify variance error sharply.
More Newton steps are therefore not an unconditional repair. The residual
identity and an out-of-basin counterexample have executable tests.

Track approximation-domain margins and `v r₀²`, not only input/output maxima.
Treat range scaling, seed damping, refresh precision and layer sensitivity
jointly. The existing kernel already folds norm weights into public linear
maps and uses normalized coordinates; these are foundations to extend.

The initial Spark 24-layer failure first becomes large around layer 6 on token zero.
The native `runtime.has_state` branch skips decay evaluation for that token.
Consequently this failure cannot be attributed to long-horizon recurrence or
the decay polynomial. A shorter instrumented chain can change refresh
placement and uses a different randomized key, so passing that chain alone
does not identify or repair the full-depth cause.

### A concrete scale-alignment defect found during this review

In the pinned FIDESlib backend, `Ciphertext::addPt` adjusts a plaintext when
its modulus level differs from the ciphertext's, but skips that adjustment
when levels are equal. The following assertion requires equal noise-scale
degrees; release builds omit the assertion. Our consumption-level cache-miss
encoding supplies degree 1, while a preceding product often has degree 2.

Schematically, adding `Δ b` to `Δ² x` and decoding at `Δ²` gives
`x + b/Δ`, not `x+b`. This is a scale mismatch, not ordinary approximation
error. It affects the vector constants in convolution and step-size branches.
Changing the number of loaded layers changes which constants fit the cache,
explaining why a small gate can pass while a deeper gate fails at the first
uncached constants.

Disabling consumption-level encoding restores the 24-layer, one-token
zero-debug gate at error `0.02175`, but encodes all misses at the expensive
full level. The targeted [addition helper](../../native/fideslib_stage0/src/fideslib_plaintext_ops.hpp)
instead re-encodes an incompatible additive constant at the ciphertext's
actual level **and degree**. It preserves consumption-level multiplication.
The [encrypted regression probe](../../native/fideslib_stage0/src/stage1_plaintext_add_probe.cpp)
compares `Enc(x)^2 + b` with the explicit expected value at several levels.
Its measured gate and subsequent full-chain runs are recorded in the
[evidence registry](../evidence.md), separately from the older failure artifacts.
All four corrected 24-layer/one-token ABBA runs pass, with maximum polynomial
errors 0.02188–0.02857 and 36 targeted re-encodes per run. This closes the
observed first-token defect; multi-token and held-out coverage remain distinct.
The follow-up five-step baseline matches generated IDs but misses the
precision threshold at its second output (`0.08236 > 0.05`). Later errors
fall below 0.05; the remaining issue is not identified merely by calling it
recurrence instability. Its complete failed artifact remains in the registry.

### A separate plaintext failure: polynomial extrapolation

The payload-bound WikiText-2 screen establishes a problem before encryption.
On the first 1,024 test tokens, layer 5's gate input reaches `28.5305`, outside
the fitted SiLU interval `[-22.6745, 25.4530]`. Its degree-64 polynomial produces
about `4.0213e9`, causing a gated variance of `1.0851e14`. The following
inverse-square-root polynomial becomes non-finite at token index 756. The
[operator diagnostic](../../fhemamba/results/payload_domain_20260921.json)
records each input/output domain without clipping. This is distinct from
CKKS refresh noise and from the earlier plaintext-addition backend bug.

Original and exact-with-pruning circuits remain finite on these inputs.
For two 1,024-token windows, PPL is 18.39695 and 18.39309 respectively, with
99.902% next-token argmax agreement. These small-sample results do not establish
general harmlessness of pruning: 27 of the 47 pruned heads have an observed
exact decay above 0.5, and the largest is 0.994969. A calibrated interval's
invalid proof and a measured quality loss are different questions.

### Prove normalization first, then derive public gate ranges

There is a useful stronger statement than Newton convergence. Let
`z = v y0^2`, with `v>0`, `y0>=0`, and `0<=z<=3`. After one exact refinement,

```text
y1 = y0 (3 - v y0^2) / 2
sqrt(v) y1 = sqrt(z) (3-z) / 2 in [0,1].
```

The upper bound follows from maximizing the last expression at z=1.
Further refinements stay within [0,1]. Thus the approximate normalization is
**non-expansive**, even when the initializer is imperfect. This proves a range
property; an initializer of zero is non-expansive but inaccurate.

For RMSNorm followed by a public projection row w and norm weights gamma,
with `v=||x||_2^2/d + eps`, the consequence is

```text
||x y||_2 <= sqrt(d)
|w^T diag(gamma) x y| <= sqrt(d) ||w ⊙ gamma||_2.
```

The right-hand side depends only on public weights. It can replace the
observed gate maximum as the fitting domain, conditional on the normalization
contract. In particular, the observed gate outlier does not imply an escape
from this weight-derived envelope. Current per-layer gate maxima from this
formula are approximately 35.46–122.05. Per-channel envelopes are often much
smaller and could support packed coefficient vectors, at added planner/cache
complexity. A convolution envelope follows by summing these row bounds with
the absolute public convolution weights and adding the absolute bias.

The new [rational Bernstein verifier](../../fhemamba/src/fhemamba/polynomial_certificate.py)
interprets stored binary64 coefficients as exact rational numbers, transforms
the Chebyshev polynomial to Bernstein form, and subdivides by exact midpoint
de Casteljau steps. The convex hull of each segment's Bernstein coefficients
certifies its range. Unresolved depth/node limits fail closed; endpoint
violations are exact counterexamples. It verifies `y0>=0` and `v y0^2<=3` for
all **48** existing block/gated initializers on their declared intervals in
the [certificate report](../../fhemamba/results/payload_range_certificate_20260921.json).
The final RMS reuses the last block's polynomial and interval.

This does **not** establish that private variances stay in those intervals,
or certify CKKS rounding, approximate division, or FIDESlib's rounded affine
coefficients. Those need explicit numerical slack. The gate-envelope values
in the report are floating-point estimates of the analytic expression,
separate from the exact-rational initializer certificates.

The [public-domain gate screen](../../fhemamba/results/public_gate_polynomials_20260921.json)
uses a 10% numerical margin and 8,193 independent uniform validation points.
Worst sampled SiLU errors across 24 layers are 0.4720 / 0.08956 / 0.02185 /
0.004756 / 0.0002425 for degrees 64 / 128 / 192 / 256 / 384. These candidates
trade more polynomial work for a defensible domain; sampling is not a uniform
error certificate or a language-quality gate. They are not native defaults.

Changing only the gate fits is insufficient. The
[degree-384 public-gate diagnostic](../../fhemamba/results/payload_public_gate_domain_20260921.json)
gets past layer 5 but encounters a second domain failure: layer 11's gated
variance reaches 8,296.32, outside its interval ending at 7,403.74, and the
initializer becomes non-finite at token 249. This confirms that certificates
on declared intervals must be paired with actual interval coverage.

### A positive-series initializer for broader norm domains

There is a way to broaden the Newton basin without assuming a positive lower
fit endpoint. For public H and `0<v<=H`, set `t=1-v/H` and truncate

```text
v^(-alpha) = H^(-alpha) sum_(k>=0) (alpha)_k / k! * t^k,
alpha = 1/2 for a direct seed, or 1/4 for a seed that will be squared.
```

All coefficients and omitted terms are nonnegative. The truncated series is
positive and below the exact inverse power on this entire interval. With
positive damping at most one, both seed constructions start within
`0<=v*y0^2<=1`; Newton is monotone and non-expansive. This provides an analytic
range property on `[0,H]`, rather than an extrapolation claim for a least-squares
polynomial fitted on `[lo,H]`. At v=0 the initializer is finite; actual RMS
variance includes positive epsilon.

The tradeoff is slower convergence for small v/H. For a direct undamped seed
of degree K, `sqrt(q) sum_(k=0)^K c_k(1-q)^k`, q=v/H, is increasing in q:
its derivative is `(K+1/2)c_K(1-q)^K/sqrt(q)`. Thus a chosen positive q_min
bounds the initial residual, and `e_(j+1)=e_j^2(3+e_j)/4 <= e_j^2` for
`e_j=1-v*y_j^2` in [0,1]. Degree and iteration count can be chosen jointly
against a specified accuracy interval. Non-expansion alone is not accuracy.

`positive_binomial_seed` constructs this candidate and converts it to the
existing Chebyshev representation. The conversion is floating-point; the
actual coefficients can be checked with the rational Bernstein verifier.
Unit tests certify both seed forms on `[0,100]`. The diagnostic option uses
degree 63, eight Newton steps and H equal to four times the old upper bound,
with all overrides recorded. This is an unpromoted development experiment;
H still needs a defensible activation bound and CKKS slack, and extra Newton
steps add depth and refresh cost. No native default changes with this probe.

The first development prefix becomes finite, but the
[window at offset 16,384](../../fhemamba/results/payload_positive_seed_quality_20260921.json)
still fails: layer 13's convolution SiLU receives 20.399 above its upper domain
15.1154 and extrapolates to approximately -7.8359e24. Extending the public-weight
envelope to convolution channels is therefore part of the same design,
not evidence that a norm initializer alone solves the quality problem.

Adding degree-768 convolution SiLU fits over the public depthwise-convolution
envelopes makes this development window finite. The
[resulting candidate](../../fhemamba/results/payload_public_activations_quality_20260921.json)
has PPL 31.00269 versus exact 30.84142 (+0.5229%) on 1,023 predicted tokens;
all 313 observed stages stay finite, with no non-decay domain escape.
This is a small plaintext development
result with explicitly different polynomials. Its larger degrees and eight
Newton refinements must be included in any encrypted depth/refresh budget.

The coefficients were then kept fixed for an
[unused 1,024-token window at offset 32,768](../../fhemamba/results/payload_public_activations_holdout_20260921.json).
It stays finite with exact/candidate PPL 19.67562/19.65756 and 99.022% argmax
agreement. All 97 override descriptors and coefficient hashes match the
development run. These two short windows support continued evaluation of
the candidate; they do not certify long-context quality or encrypted cost.

### Broader evaluation exposes the time-step/decay composition

The fixed 97-override candidate fails on a new 4,096-token window at offset
65,536. Layer-0 time-step input reaches -5.01163 outside its lower endpoint
-4.69550; the first non-finite checkpoint is `y`. Exact PPL is 17.56673 and
candidate PPL is undefined. The two earlier finite 1,024-token windows remain
valid observations, but do not support a general stability claim. The
[raw report](../../fhemamba/results/payload_public_activations_long_20260921.json)
retains all operator ranges and coefficient hashes.

A further calibration attempt uses four disjoint training windows of 256
tokens, resetting state between windows. Exact states are finite; the frozen
polynomial states become non-finite in layers 9–23, so no new payload is
written. Expanding empirical calibration alone cannot repair this circuit.

One alternative is to approximate the gates **jointly** on a common public
domain. Write `t=(z-lo)/(hi-lo)` and use degree-n Bernstein basis functions
`B_i(t)=binom(n,i)t^i(1-t)^(n-i)`. For coefficients satisfying

```text
0 <= a_i <= 1,
|b_i| <= K (1-a_i),
a(t) = sum_i a_i B_i(t),  b(t) = sum_i b_i B_i(t),
```

nonnegativity and partition of unity imply the same inequalities for `a(t)`
and `b(t)` throughout `[0,1]`. Thus `S'=a(t)S+b(t)V` preserves
`|S|<=K*V_max`, including the perfect-memory endpoint `a=1,b=0`. This is a
finite set of linear coefficient constraints, suitable for constrained fitting
or training. It removes the need to feed a possibly inaccurate softplus
result into a second polynomial before obtaining decay. It does not remove
the encrypted write/readout products.

The implementation checks these inequalities on the exact dyadic values of
stored coefficients. A first construction samples the exact gate pair at
Bernstein knots, with `K=D+1/c`, and rounds write coefficients inward when
necessary. All 576 unpruned heads certify at degrees 32/64/128. Yet at degree
128 the worst sampled decay error is **0.39630** and write error **2.85355**:
the simple Bernstein approximation excessively smooths sharp transitions on
wide public envelopes. See the [probe](../../fhemamba/results/bounded_selective_gates_20260921.json).
Stability is necessary, but this approximation is rejected for checkpoint
replacement. Subsequent Chebyshev conversion and CKKS rounding would require
new certificates; this certificate applies to the Bernstein representation.

### Shared dissipation factors retain accuracy and certify the recurrence

The [official Mamba-2 step](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba2.py#L295-L300)
uses exponential decay and an Euler write coefficient. With public `c=-A>0`
and `d=softplus(z)`, the following factorization preserves both exact targets:

```text
delta = 1 - exp(-c*d)
p = sqrt(delta),  q = sqrt(d/delta)
a = 1 - p^2,     b = p^2*q^2
S' = (1-p^2) S + p^2 (q^2 V).
```

As `d -> 0`, `q -> 1/sqrt(c)`: the apparent singularity is removable.
Fit p and q directly as public per-head Chebyshev series in raw z.
Evaluation of the fitted circuit needs only additions and products; it does
not evaluate square roots, exponentials or the private division above.
The exact identity preserves the checkpoint's discretization. Polynomial
fitting and inward scaling still introduce a new surrogate, whose quality
must be measured.

For actual rounded coefficients, prove `|p(t)|<=1` on `[-1,1]` and set
`K=(sum_k |q_k|)^2`. Since `|T_k(t)|<=1`, `q(t)^2<=K`. Then

```text
0 <= a <= 1,  0 <= b <= K(1-a),
|S| <= K V_max and |V| <= V_max  =>  |S'| <= K V_max.
```

The proof includes `p=0`, where memory is retained and the write is zero.
Shrinking p by a constant changes dissipation proportionally; it does not
impose a uniform forgetting floor on heads whose decay is near one.
Moreover a root-evaluation perturbation changes `p^2` by `2p*e+e^2`.
This is a local sensitivity observation, not a CKKS noise proof: ciphertext
multiplication, independent gate rounding and refresh add other errors.
The equivalent residual update `S'=S+p^2(q^2 V-S)` is a lowering candidate;
the current oracle still uses the existing affine scan.

Coupling retention and update by complementary coefficients is established
in GRUs and minGRUs; see [Sections 2.2 and 3.1 of Were RNNs All We Needed?](https://arxiv.org/html/2410.01201v3).
The work here is a checkpoint-specific polynomial construction and certificate,
not a claim to have invented that recurrence or established research novelty.

`certify_chebyshev_bounds` uses exact integers with common denominators for
the Chebyshev-to-Bernstein conversion and midpoint subdivisions. It avoids
repeated rational reduction at every arithmetic operation. The original
Fraction verifier independently checks its results on small polynomials,
including subdivision limits and binary-epsilon counterexamples. No roots
or sampled extrema are trusted to certify the range.

The [fit report](../../fhemamba/results/dissipative_selective_gates_20260921.json)
certifies **all 576 heads**, without pruning. Fitting reads public weights,
not text or calibration activations. The policy starts at degrees 1,024 / 512
for p / q, trims trailing coefficient l1 mass up to `1e-11` per head, and
scales p inward by `1/(1+2e-6)`. Certification applies after both operations.
Effective p degrees range from 31 to 1,024 and q from 30 to 512.
On 8,193 independent uniform points per head, worst sampled errors are
**4.0170e-6** for decay and **3.7885e-4** for write; maximum certified K is
2,559.956. These accuracy numbers are samples, not uniform error certificates.
The degree-128 Bernstein probe used a different arithmetic budget, so this
comparison does not establish a cost improvement.

The [first 4,096-token evaluation](../../fhemamba/results/payload_dissipative_long_20260921.json)
revisits the development failure at offset 65,536. All observed stages are
finite, with no recorded domain or decay-range escape. Exact PPL is 17.56702,
candidate PPL 17.65695 (+0.5119%), and top-1 agreement 97.9976%.
Maximum logit error is still 82.6964, so PPL proximity is not a uniform
output-error guarantee. This run uses CUDA float32 with TF32 disabled and
float64 gate-root evaluation; the prior CPU reference PPL was 17.56673.
The other 97 wider-activation/norm overrides retain their fitting policy.
Twenty-two convolution coefficient hashes differ from the earlier CPU
artifact, so this is not a bit-identical copy of that earlier candidate.
A separate CUDA control binds the comparison to matching coefficients.
The 24 joint gates also restore the previously pruned heads.
The [matched control](../../fhemamba/results/payload_public_activations_cuda_control_20260921.json)
has identical source/dependency hashes and all 97 other coefficients; it
still fails at layer-0 `y` on the same input.

The [previously unused 4,096-token window at offset 98,304](../../fhemamba/results/payload_dissipative_holdout_20260921.json)
uses the same bundle and all 121 override descriptors/coefficients as the
development run. Exact PPL is 15.22679, candidate PPL 15.25058 (+0.1563%),
top-1 agreement 98.4860%, and maximum logit error 53.8167. All observed stages
are finite, with zero recorded domain or decay-range escapes. This is a
fixed-candidate holdout screen within WikiText-2, not a full-dataset result.

An [ablation with only the gates approximated](../../fhemamba/results/dissipative_gate_only_ablation_20260921.json)
revisits offset 65,536 and leaves the other nonlinearities exact. It uses the
same gate bundle, obtains PPL 17.56700 versus 17.56702, preserves all 4,095
next-token argmax choices, and has maximum logit error 0.025864. Thus the
gate-only approximation is faithful on this window; the large combined
error is a reason to investigate the other polynomial stages and their
interactions. This diagnostic is not itself a fully polynomial circuit.

The invariant assumes z lies in its public interval, bounded `V=x B`, and
an initially bounded state. Projection envelopes still depend on the
non-expansive normalization contract; the added 10% margin does not prove
float/CKKS coverage. Whole-model domain closure, encrypted precision and cost,
and long-context task quality remain separate obligations. Native lowering
needs per-head coefficient vectors and can potentially share the Chebyshev
basis between p and q. No such kernel, speedup or full-model stability proof
is claimed by these plaintext results.

### Interval-certified normalization schedules

A single-site [exact-normalization ablation](../../fhemamba/results/payload_exact_first_norm_ablation_20260921.json)
identifies the next major defect. At offset 65,536, replacing only layer 0's
gated inverse square root reduces maximum logit error **82.6964 → 4.50343**
and the PPL gap **+0.5119% → +0.00999%**. This diagnostic uses an exact
nonlinearity, so it is not an FHE candidate. The previous degree-63 positive
seed with eight Newton steps attenuates the smallest observed layer-0 input
by a normalization factor of about **0.4634**. Non-expansion alone did not
ensure accurate normalization.

The replacement is a public schedule of scaled cubic Newton/Goldschmidt
updates. This family is established: [aSOR, Section 4.3 and Algorithm 4](https://eprint.iacr.org/2024/1366.pdf)
derives accelerated inverse square roots; [THOR, Section 5.2 and Appendix C](https://eprint.iacr.org/2024/1881.pdf)
uses them for encrypted LayerNorm. [Polar Express, Sections 3.2–3.4](https://arxiv.org/html/2505.16932v5)
places scaled polynomial compositions in a broader approximation framework
and discusses finite-precision cushioning. This implementation adapts that
prior art to the checkpoint and certifies rounded coefficients; it does not
claim a new iterative method or superiority to these systems.

For variance `v∈[L,H]`, initialize a public `y0≈1/sqrt(H)` rounded inward
so `H*y0²<=1`. Track `u=v*y²`. Each public coefficient pair `(a,b)` gives

```text
f = a - b*u
y_next = y*f
u_next = u*f².
```

For `sqrt(u)∈[l,r]`, set `S=l²+l*r+r²`,
`b=3*sqrt(3)/(2*S^(3/2))`, and `a=b*S`. The cubic `a*t-b*t³` has equal
endpoint values and interior maximum one. Use `l=max(l,r/16)` for coefficient
design to limit early upper-end cancellation, and multiply both coefficients
by `1-1e-12` before certification. This cushion sacrifices minimax optimality;
no optimum is claimed. All decisions depend on the public interval only.

The certifier does not trust the floating-point design calculation. It treats
every stored binary64 coefficient as an exact rational. The range of
`F(u)=u*(a-b*u)²`, with positive `a-b*u`, is determined by its endpoints and
the possible critical point `u=a/(3b)`, whose value is `4*a³/(27*b)`. Exact
rational extrema are enclosed outward on a 128-bit dyadic grid after every
step. This avoids exponential denominator growth while retaining a proof.
The loader recomputes the certificate from the recipe and rejects a failed
certificate, duplicate/missing sites or a mismatched input payload.

Coupled updates alone can conceal consistency error: `v*y²/u` is invariant
under an exact update, so a rounding-induced discrepancy need not disappear
when `u` reaches one. The last step therefore recomputes the residual from
the original encrypted variance:

```text
y_final = y * (1.5 - 0.5*v*y²).
```

If `sqrt(v)*y=1+e`, this step gives `e_final=-(3*e²+e³)/2` in exact arithmetic.
It corrects an existing small error quadratically; it does not remove new
CKKS errors in the final products. It requires three ct-ct products and no
decryption, input-dependent branch, division or lookup.

The [49-site report](../../fhemamba/results/normalization_schedules_20260921.json)
certifies `1-1e-7 <= sqrt(v)*y_final <= 1` on every declared interval. Each
lower endpoint includes the model epsilon rounded to float32; upper endpoints
remain `4*old_hi`, exactly the previous widened-domain policy. These are
conditional domains, not a global variance bound. No text enters planning.
Schedules need 9–15 accelerated steps plus the final recomputation. The
float64 oracle's worst error on 8,193 geometric points per site is `8.754e-8`;
the rational enclosure, rather than this sample, supplies the uniform claim.

For `n>=2` accelerated steps, literal DAG counts are `3n` ct-ct products and
`2n+2` ct-ct critical depth, including final recomputation. The same coefficient
sequence in that factored evaluation order has depth `3n`. A symbolic tracer
tests these counts against the executed oracle. **Follow-up correction:**
balancing the cubic products reduces recomputed depth to `2n`. For layer 0,
the accelerated sequence has 36 products / balanced depth 24;
constant-seed, fixed-coefficient Newton on the same interval and tolerance
needs 75 products / balanced depth 50. The earlier 75-depth figure counted
the unbalanced DAG, not the best native implementation. The old eight-step binomial candidate
does not meet this accuracy target and is not an equal-accuracy baseline.

These counts exclude scalar rescaling, level alignment, bootstrapping, SIMD
replication and memory. In particular, the aSOR paper's Section 5 explores
scale management and reports different practical tradeoffs between Newton
and Goldschmidt. It cannot be replaced by a ct-ct-only depth comparison.
The frozen bundle's smallest initial `u` is around `1.7e-10`: absolute CKKS noise and
refresh error must be budgeted before native adoption. A float64 certificate
does not demonstrate an executable CKKS precision schedule.

The [fully polynomial development run](../../fhemamba/results/payload_scheduled_norm_long_20260921.json)
reproduces the ablation's improvement without an exact-operator substitution:
PPL **17.56702 → 17.56877 (+0.00994%)**, top-1 agreement **99.8779%**, maximum
logit error **4.5000**, versus 82.6964 before this normalization replacement.
All 72 other override descriptors/coefficients, gate bundle and input tokens
match the preceding joint-gate candidate.

The [fresh window at offset 131,072](../../fhemamba/results/payload_scheduled_norm_holdout_20260921.json)
uses the identical 121 override descriptors and source hashes: PPL
**14.15001 → 14.15097 (+0.00682%)**, top-1 agreement **99.7314%**, maximum logit
error **9.1330**. Both 4,096-token runs are finite with no observed domain
or decay escapes. This is a fixed-candidate WikiText-2 screen, not a full
quality benchmark. Passive local diagnostics show the largest remaining
development operator error in layer 10's convolution SiLU (`0.003152`);
local errors alone do not prove responsibility for every output discrepancy.
The [native follow-up](2026-09-21-normalization-native.md) now measures the
arithmetic schedules on Spark and isolates the convolution-SiLU error.
It distinguishes standalone inverse-square-root decryption from computing
the complete scalar normalization before decryption. Packed vector reductions,
refresh and full-model integration remain open.

### Conditioning and polynomial schedules without model refitting

An existing row calibration can be regularized by
`M'=max(M, f*M_group, 1e-6)`. For `f=1/8`, relative amplification is at most
eight, with unchanged state algebra and mask operation count. This rule
uses public scales, not evaluation states. The observed fixed/autoregressive
coverage improves from 2,096 to 344 violations of the 1.1 margin, but the
largest ratio is still 5.67923. It is not a uniform state bound. The separate
GPU experiment evaluates the numerical tradeoff; a covered calibration set
would still not certify all future text.

The native Chebyshev evaluator now has an optional coefficient-aware
Paterson–Stockmeyer planner. For each power-of-two baby width, it counts the
cached Chebyshev basis DAG and recursive splits. It chooses fewer ct-ct
products while preserving the baseline scalar-product and depth ceilings.
This exploits actual coefficient sparsity, including symmetric activation
fits, without assuming that halving a formal degree halves encrypted work.
The original degree-only depth ledger remains a conservative refresh budget.
Public coefficient plans are cached; no decision depends on encrypted values.

The existing `1e-12` coefficient cutoff remains. The planner bounds the sum
of discarded leaf coefficients by `max(1e-10, baseline_drop)`. Since products
of Chebyshev factors have magnitude at most one on `[-1,1]`, this bounds the
discard-only contribution in exact arithmetic. It excludes coefficient-split
rounding, ciphertext error and out-of-domain inputs. Native host tests check
both sparse and asymmetric series through degree 768; normal encrypted
execution must separately establish accuracy and cost.

The first matched encrypted 24-layer/one-token comparison passes for both
schedules: maximum errors 0.01536 / 0.02299 with zero intermediate decryptions.
Actual ct-ct products fall from 3,363 to 3,195 and ct-pt products from 13,950
to 13,404, exactly matching the planner's 168 / 546 savings. Rotations 7,575
and physical bootstraps 98 are unchanged; direct level drops increase from
5,867 to 6,620. Evaluation is 129.594 / 128.353 s, one run per arm, so no
latency claim follows. See the [campaign](../../fhemamba/results/dgx/2026-09-21/spark-chebyshev-ab-20260921.json).
The candidate remains opt-in pending carried-state and additional-prompt gates.

## 7. Cryptographic mechanisms worth testing

| Mechanism and primary source | Relevant mathematical change | Gate for this repository |
|---|---|---|
| [Double hoisting / non-sparse-key bootstrap](https://eprint.iacr.org/2020/1203) | Reuse key-switch decomposition and organize linear maps | Actual FIDESlib API, limb-indexed time, memory and precision |
| [Generalized composites and integrated rescaling](https://eprint.iacr.org/2025/429) | Co-design modular approximation and matrix/rescale scheduling | Compare consumed modulus and error at matched parameters |
| [Meta-BTS](https://eprint.iacr.org/2022/1167) | Correct refresh error through additional bootstrap computation | Apply selectively at high-sensitivity sites; preserve input-error accounting |
| [EvalRound+](https://eprint.iacr.org/2024/1379) | Cancel/restructure rounding terms to reduce transform modulus consumption | Backend port and precision/latency curves before changing depth |
| [Error-variance-minimizing modular approximation](https://eprint.iacr.org/2020/1549) | Fit the bootstrap function and schedule lazy polynomial evaluation together | Average precision plus tail/failure measurements |
| [RBOOT](https://www.usenix.org/conference/usenixsecurity26/presentation/yang-zhaomin) | Fuse an activation into refresh | Its ReLU construction does not automatically implement RMSNorm or SiLU |
| [Sparse roots-of-unity bootstrap](https://arxiv.org/abs/2607.27401) | Alternative recent CKKS modular-reduction construction | Research candidate; no compatible implementation measured here |

### Error cancellation, not simply repeating a refresh

The useful mechanism in [Meta-BTS, Section 3.2](https://eprint.iacr.org/2022/1167)
is visible in the existing native implementation. Ignoring alignment error,
write a normalized message and its refresh as `u` and `v=u+e₁`. Then

```text
r = 2^α (u-v) = -2^α e₁
w = Bootstrap(r) = -2^α e₁ + e₂
v + 2^(-α) w = u + 2^(-α) e₂.
```

The original ciphertext supplies an encrypted error reference. The residual
must fit the bootstrap interval, and its construction needs reserved levels.
Increasing α can violate that interval; scale/level alignment adds further
error. Pre-existing model error inside u remains. Our design inference is to
allocate α and extra refreshes by *downstream sensitivity*, using public
calibration bounds for each site, rather than repeatedly refreshing every state.

[EvalRound+, Section 3](https://eprint.iacr.org/2024/1379) instead restructures
the inside of a bootstrap. Modulus raising introduces an unwanted `qI` term.
A cheaper coefficient-to-slot branch estimates that term; skip connections
cancel it before the final slot-to-coefficient transform. That final transform
then handles the smaller message, avoiding the precision cost of transforming
the large `qI` correction. A more accurate branch and extra live storage are
still required. The cheap branch has an error threshold: the cancellation is
not valid at arbitrary precision. For this repository, compare saved modulus
against added transform time and peak memory at identical output precision.
This requires a backend implementation, not changing the current Meta-BTS flag.

Functional refresh is attractive only when its input representation matches
the desired function. A reciprocal-square-root of a **sum of squared slots**
still needs the reduction; ordinary slotwise functional bootstrapping cannot
make that dependency disappear. Likewise `W B(x)` cannot generally be
replaced by `B(Wx)` at identical precision/range, even though the exact
linear map commutes with an ideal identity operation.

Use a graph objective such as
`sum latency(op, level) + refresh costs + memory-pressure costs`, subject to
live-ciphertext levels, output error, and validated security parameters.
Increasing modulus to avoid a refresh can increase every rotation's cost and
force a larger ring for the same security. A lower bootstrap count is not the
objective by itself.

## 8. Exact arithmetic, selection and the full encrypted loop

[BFV](https://eprint.iacr.org/2012/144) and
[BGV](https://eprint.iacr.org/2011/277) provide modular arithmetic; this does not
provide exact real arithmetic for a neural network. Fixed-point scaling,
rounding, overflow and nonlinear functions still require a design. Their
natural role here is an explicitly quantized or finite-state subsystem.
[TFHE](https://eprint.iacr.org/2018/421) supports bootstrapped logical/lookup
computation. Switching every dense activation into scalar logical operations
would forfeit much of the current SIMD advantage; a small discrete subsystem
or token selection is a more bounded experiment.

There are two explicit completion targets:

1. **Interactive private inference:** the server never decrypts state or
   activations; the client handles vocabulary selection and the next embedding.
   This is the current protocol direction.
2. **Autonomous encrypted generation:** vocabulary projection, token selection,
   embedding lookup, state carry and a fixed/public generation schedule all
   remain encrypted until final delivery. This is the stronger endpoint.

For greedy decoding, `argmax(softmax(logits)) = argmax(logits)`, so softmax
can be removed exactly. If every logit has error at most ε, the winner is
unchanged when the true top-two gap exceeds `2ε`. This is a conditional
certificate, not a global promise for nearly tied logits. Sampling is a
different computation and needs its own randomness/protocol specification.

OpenFHE has [CKKS↔FHEW comparison and argmax examples](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/scheme-switching.cpp).
Use these to build a separate small-vocabulary gate, then scale to the actual
vocabulary. Its output index or one-hot representation must feed encrypted
embedding lookup; plaintext-indexed array access would break the protocol.
Do not infer an entire encrypted loop from the existing client/server key
separation probe.

Public fixed decay is a training option, supported as a design direction by
[Public-Decay HSSM](https://arxiv.org/abs/2605.16647), whose evaluated workload
is classification on bounded projected features, not a generated language
model. Public ct-pt decay removes the ct-ct carry product, but a general
scaled plaintext multiply can still consume a level. Special fixed sign or
permutation carries avoid some scaling costs while changing model capacity;
encrypted selection among them restores encrypted work. Additions and
rotations also accumulate noise. There is no free indefinitely recurrent
ciphertext from declaring A public.

An FHE-oriented training track should compare: public multiscale decay,
bounded learned selective decay, and bounded complex SISO transitions. Include
polynomial and refresh perturbations during training. Score perplexity,
long-context retrieval and actual reasoning/state-tracking tasks alongside
encrypted cost. Encryption correctness and the model's reasoning ability are
separate requirements.

## 9. Security and precision share a budget

Keep public weights and a client-only secret key as the initial threat model.
RLWE parameter validation is distinct from output/decryption security.
[Li–Micciancio](https://eprint.iacr.org/2020/1533) analyze the danger of exposing
approximate decryptions beyond ordinary IND-CPA guarantees. The precise
adversary and information returned by the protocol matter; this is not a
claim that any use of CKKS automatically leaks its key.

The [OpenFHE noise-flooding notes](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/CKKS_NOISE_FLOODING.md)
make the extra noise depend on an error estimate, a query budget and a
statistical-security target. Its described imaginary-part error estimator
assumes real messages. A design using both complex components for Mamba-3
must not reuse that estimator while treating the imaginary signal as noise.
This is an additional integration gate for complex packing, not evidence
that complex CKKS itself is invalid.

Output precision should include any required flooding error and the selection
margin above. Hiding prompts, hiding model weights, circuit privacy and
correctness against a malicious server are distinct properties. The current
`security=not-set` Spark runs do not certify them.

## 10. Concrete next experiments

| Order | Experiment | Required evidence before promotion |
|---|---|---|
| 0 | Extend the repaired full-depth Spark baseline | First-token ABBA gate passes; next require held-out payloads and carried-state horizons with normal execution |
| 1 | Deferred state, k=2/4/8 | Dense/factor state parity, exact slot simulator, encrypted long-horizon error and complete cost |
| 2 | Composite decay fitting and coordinate balancing | New surrogate quality gate or exact-transform proof; input-domain and sensitivity evidence |
| 3 | Direct complex SISO versus rotating frame | Complex refresh/estimation compatibility, radial/phase error, retrieval and state-tracking quality |
| 4 | Small encrypted vocabulary loop | Encrypted argmax and embedding, secret-key-free server, measured switching and output security |
| 5 | Train FHE-oriented SSM variants | Matched quality/context/precision and end-to-end encrypted latency |

Reproduce the implemented algebra probe:

```bash
.venv/bin/python fhemamba/experiments/probe_ssm_algebra.py \
  --checkpoint checkpoints/mamba2-130m-hf --tokens 64 \
  --output runs/ssm-algebra.json
.venv/bin/pytest fhemamba/tests/test_ssm_algebra.py
.venv/bin/python fhemamba/experiments/probe_decay_composition.py \
  --payload fhemamba/results/m2_chain_payload_headclip \
  --output runs/decay-composition.json
.venv/bin/python fhemamba/experiments/probe_phase_schedules.py \
  --output runs/phase-schedules.json
```

The native runner does not yet execute deferred state or Mamba-3. The new
oracles make these candidates falsifiable before spending time on ciphertext
layouts or retraining.
