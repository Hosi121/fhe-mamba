# First-principles review after the 761-second Mamba-3 run

Research scope: inspect the current circuit and primary sources, derive costs,
and identify discriminating experiments. No new GPU timing, backend port,
checkpoint change, or deployment is part of this study. The preceding
[three-candidate campaign](2026-09-25-packed-frontiers.md) is complete: its
19.21% improvement did not reach the 20% target.

The strongest next directions are **changing the bootstrap circuit** and
**changing the arithmetic representation used by the GPU**. A third, more
ambitious direction separates the ring used for ordinary computation from the
ring used for bootstrapping. More packing is not automatically the answer:
the failed polynomial-batching experiment already demonstrated that fewer
ciphertext products can produce a slower complete run.

## What the computation actually requires

The current contract is the trained SISO-187M checkpoint, encrypted backbone
and carried state, public weights, and client vocabulary selection only at
token boundaries. Both exact-reference and frozen-polynomial error gates are
0.001. The recorded workload is 12 layers and five evaluations, yielding four
tokens. It is not a long-context qualification or a universal accuracy bound.
The current cryptographic result is explicitly `security=not-set`, not a
validated 128-bit parameter set.

These requirements do not mathematically mandate a 65,536-degree ring for every
operation, 59-bit primes, the order of the bootstrap transforms, or a leading
contiguous slot layout. Changing those choices preserves the intended model
but still needs fresh numerical and cryptographic validation. Changing the
nonlinear approximations preserves the checkpoint target, but invalidates
equality to the old surrogate. MPC interaction or retraining changes further
parts of the contract and must be evaluated as a separate alternative.

The objective should be complete latency under those constraints:

```text
T = ordinary arithmetic and preparation
  + packing and representation conversion
  + refresh count × refresh cost
  + client feedback and other serial work.
```

Minimizing a single count—rotations, products, refreshes, or live slots—can
increase another term. Future autoregressive embeddings remain unavailable
until token selection; parallel prompt scan does not remove that dependency.

## New attribution of the frozen circuit

The [reproducible analysis](../../results/cpu/2026-09-25/first-principles/README.md)
hashes the full 1.25 GB payload, reads all 7,910 nodes, and reconciles 7,814 live
nodes and 12,005 ciphertext products with the final native result. It maps the
previous evaluator inventory back to the exact exported function coefficients.

| Nonlinearity | Live evaluations | Logical values per evaluation | Polynomial-core ct×ct calls |
| --- | ---: | ---: | ---: |
| Inverse square root | 245 | 1 | 2,720 |
| Negative transition rate | 48 | 24 | 2,036 |
| Softplus, sigmoid, tanh, exponential | 228 | 24 or 32 | 2,270 |
| Sine and cosine | 120 | 768 | 1,330 |
| SiLU | 120 | 1,536 | 1,635 |
| Total | 761 | — | 9,991 |

**7,026 / 9,991 = 70.32% of these products evaluate at most 32 useful values.**
There are 521 such polynomial nodes. The first step's decay is dead because
the initial state is zero, explaining 48 rather than 60 rate/exp evaluations.
These counts exclude the bootstrap's own internal products. They are not
shares of runtime or of GPU instructions.

All logical graph values have at most 3,432 elements, while each ciphertext
uses 32,768 slots. This does **not** prove that a 4,096-slot executor can run
unchanged: replicated BSGS scratch and packed refresh groups use additional
space. It does show that the ring choice is serving the cryptographic and
execution machinery more than the logical tensor sizes.

The non-overlapping measured categories are:

| Category | Seconds | Whole-run share |
| --- | ---: | ---: |
| Refresh, including its packing/correction wrapper | 363.586 | 47.78% |
| Gather, scatter, repeat, sum, outside refresh | 161.492 | 21.22% |
| Public linear maps, outside refresh | 123.939 | 16.29% |
| Polynomial evaluation, outside refresh | 99.029 | 13.01% |
| Remaining work and timer gaps | 12.955 | 1.70% |

Host encoding's 151.814 seconds is nested in those timers; it must not be
added again. Likewise, the 255.019 seconds of refresh charged to `cheb` nodes
does not identify which particular nonlinearity caused each grouped refresh.

There is no established 761-second theoretical lower bound. For a fixed
polynomial of degree d, binary multiplication needs depth at least
`ceil(log2(d))`, so degree 1023 requires at least ten multiplication stages.
That bound does not require this evaluator, these masks, or these refreshes,
and does not apply unchanged to a different approximation/protocol. A hardware
lower bound also needs mandatory traffic and achievable modular throughput;
we have no trace supporting a numerical global bound. Conversely, even making
all refresh work free leaves 397.416 seconds if other work stays fixed: a
1.915× speed ceiling for that **one-category** scenario. Larger gains require
improving ordinary execution too.

## 1. Change the bootstrap's mathematical circuit

The pinned FIDESlib path calls `ModRaise`, CoeffsToSlots, modular reduction,
then SlotsToCoeffs. With full packing, `Bootstrap.cu:270` splits into two
components, and `ApproxModEval.cu:24` calls the Chebyshev evaluator separately
for both (`COMPLEX=true`). The wrapper's two-pass precision correction means
238 refresh events, 476 backend bootstrap calls, and **952 internal modular
polynomial evaluations** on this path. These are structural counts, not
952 separately timed GPU jobs. Merely disabling the imaginary branch would
discard information and is incorrect.

[OpenFHE's v1.5.0 documentation](https://github.com/openfheorg/openfhe-development/blob/v1.5.0/src/pke/examples/CKKS_BOOTSTRAPPING.md)
describes SlotsToCoeffs-first bootstrapping: the first transform runs before
modulus raising, and full real packing can use one internal component. It
reserves input levels for that transform. The local FIDESlib GPU API does not
expose this mode; using it requires the corresponding precomputation and
backend implementation, not toggling an existing wrapper argument.

Our inference: this targets repeated work inside the largest measured
category without changing model polynomials. It is the highest-priority
structural experiment. However, saving one internal branch does not halve
the entire refresh wrapper, and reserving extra input levels can increase
refresh frequency. Measure the complete two-pass operation, its output
precision, and usable levels, including near-boundary and nonzero-offset
messages. Then recompute graph scheduling before any full-model claim.

[LCR+AKS (PKC 2026)](https://eprint.iacr.org/2025/1403) provides another
structural change: integrate plaintext multiplication/rescaling with
key-switching, particularly for sparse transform matrices. The authors'
[implementation discussion](https://ckks.org/blog/2026/less-mod-ckks/)
reports 20–35% **throughput** improvement, which includes usable output
levels; its sample bootstrap latencies improve roughly 10.7–18.2% on a CPU.
Those percentages are neither DGX predictions nor whole-model speedups.
Special evaluation keys, the initial post-ModRaise coefficient bounds, and
the paper's sparse-secret encapsulation setting prevent copying the idea
as a generic rescale deletion in our uniform-ternary backend.

Both mechanisms deserve comparison with the earlier generalized-composite
and EvalRound+ proposals. They are alternatives with interacting level and
key requirements; their reported gains must not be multiplied together.

## 2. Use GPU-native integer widths without lowering model precision

CKKS plaintext precision and the size of an RNS limb are different quantities.
A large modulus can be represented with more small primes instead of fewer
large primes. Thus 32-bit integer arithmetic need not mean 32-bit CKKS scale,
let alone FP16 model arithmetic. It requires an appropriate rescaling scheme.

[Cheddar, ASPLOS 2026](https://arxiv.org/html/2407.13055v2), implements 32-bit
RNS with a 25/30-bit prime construction and rational rescaling, plus fused
GPU operations. Its paper tests Blackwell RTX 5090 as well as earlier GPUs,
making this more relevant than an A100-only design. Its published speedups
are against other workloads/backends, not this patched FIDESlib on GB10.
The described scale choices and public parameter examples do not establish
compatibility with our 59-bit-scale, two-pass circuit.

Our first-principles check also rules out a misleading memory claim. Ignoring
auxiliary primes and padding, a two-component N=65,536 ciphertext with twenty
59-bit limbs occupies 20 MiB. Forty 30-bit limbs occupy the **same 20 MiB**.
The opportunity is fewer instructions for wide modular arithmetic and better
fusion; halving the machine word does not automatically halve memory traffic.

The smallest useful comparison is a separate backend probe implementing
multiply/rescale, a representative BSGS transform, and two-pass refresh at
matched output error and usable depth. Include basis conversion, key memory,
and encoding in costs. Passing arithmetic tests at a lower scale is not
enough to promote a full-model port. Cheddar's official README also limits
its [client helpers to testing](https://github.com/scale-snu/cheddar-fhe#security);
a production client path needs separate
integration rather than inheriting that helper code unchanged.

Tensor-core digit decomposition remains possible, as in
[TensorFHE](https://arxiv.org/abs/2212.14191), but accumulation/carry bounds and
conversion work are part of exact modular computation. Start with 32-bit RNS
before assuming advertised FP4 throughput helps. The newer
[FHECore](https://arxiv.org/abs/2602.22229v2) proposes a modified GPU functional
unit and reports simulations; it is not a kernel we can install on GB10.

## 3. Separate the computation ring from the refresh ring

The scalar/headwise inventory gives a concrete reason to consider a smaller
ring between refreshes, then pack/switch to a larger ring for refresh.
[Lattigo's v6.2.0 implementation](https://github.com/tuneinsight/lattigo/blob/v6.2.0/circuits/ckks/bootstrapping/evaluator.go)
supports bootstrapping smaller-ring ciphertexts with automatic packing and
unpacking; its [parameter example](https://github.com/tuneinsight/lattigo/blob/v6.2.0/examples/singleparty/ckks_bootstrapping/basics/main.go)
separates residual and bootstrap parameters. This establishes an available
mechanism, not GPU performance or compatibility with FIDESlib.

The candidate cost is

```text
T' = A_small_ring + pack/switch/unpack + B_new_schedule.
```

Smaller N usually constrains the admissible modulus at a fixed security
target, so the available depth can decrease. In an optimistic scenario where
**all** 397.416 non-refresh seconds halve, unchanged refresh would give
562.294 seconds. But increasing refresh work by 50% gives **744.087 seconds**
even before conversion costs. A 54.65% increase consumes the entire saving.
These are break-even calculations, not speed forecasts. Restricting the
smaller ring to scalar paths gives less saving and additional boundaries.

For this reason the next step is a parameter-and-boundary cost model, not
changing `SetRingDim(65536)` to a smaller value. Include QP, the secret/error
distributions, key-switch noise, all transition keys, output precision, and
the new refresh schedule. Both representations need security analysis;
current `security=not-set` is no certificate to preserve by assumption.

Sparse packing is a related but different option. A scalar followed by zeros
is not the periodic sparse encoding required by the fast sparse transforms.
[The 2026 sparse-transform paper](https://eprint.iacr.org/2026/1023) exploits
that repetition, with depth/precision and sparse-secret conditions. Its
headline gains over depth-one transforms are much larger than its comparison
with deeper baselines. Establish periodic layouts and count conversion and
lost cross-value batching before using its timings as motivation.

## Model algebra and compiler representation

**Negative rate.** On every exported interval, the 1e-4 floor is inactive;
the smallest possible unfloored value is above 0.08018. The exact positive
rate magnitude is therefore

```text
g(x) = 1+x       for x >= 0
     = 1/(1-x)  for x < 0;       A(x) = -g(x).
```

The function is C1 but not C2 at zero. The expensive degrees 255/511/1023
are approximating a real loss of smoothness, not a complicated elementary
function. The secret sign cannot be used for a free public branch.
Rational/piecewise evaluation must pay for reciprocal/selection; directly
approximating `exp(-g(x)*dt)` must handle two private inputs and still inherits
the kink. Its sensitivity to g is `dt*exp(-g*dt)` for dt>=0, suggesting a
downstream-weighted error objective rather than uniform gate error, but this
requires a new approximation certificate. Removing the floor alone cannot
speed up an already compiled polynomial. This remains a research candidate,
not a demonstrated replacement for 2,036 products or a 20% runtime saving.

**Mamba-3 complex arithmetic.** For the frozen polynomials C,S, define
`Rhat(theta)=[[C,-S],[S,C]]`. Multiplying `(C+iS)*(b0+i*b1)` reproduces the
same real arithmetic, so complex packing can preserve this surrogate.
The often tempting simplification `Rhat^T Rhat = I` is false: it equals
`(C^2+S^2) I`. Our 65,537-point-per-layer screen finds a unitarity defect up
to 3.86e-7. The Gram identity including that factor holds to 3.56e-15 in the
float64 oracle. Thus eliminating current-token rotation, or replacing
accumulated angles with a local phase recurrence, needs explicit treatment
of the surrogate. Direct complex state and phase drift were already
[identified in September's algebra study](2026-09-21-ssm-cryptographic-design.md#4-complex-ckks-slots-and-mamba-3);
the new point here is the check against the actual frozen coefficients.
Half as many logical coordinates also need not halve ciphertext work when
slots were mostly empty. Conjugation, packing, and complex refresh remain.

**Semantic layouts.** All 245 inverse-square-root nodes feed scalar-to-vector
repeat operations. Reductions contract headwise values and later broadcast
them again. A compiler can carry replicated/strided layouts through pointwise
operations and choose conversion sites jointly with the consuming linear
map. [HEIR's layout design](https://heir.dev/docs/design/layout/) provides a
concrete representation for such analysis. This is broader than the rejected
contiguous-window prototype. It is not a proof that 245 repeats can simply
be deleted: creating a repeated reduction can require extra rotations, and
the wider live representation competes with refresh batching. Start with a
single RMSNorm→projection and a factored readout, preserving masks, padding,
frozen coefficients, and error gates. The measured routing category is an
upper bound on directly removable routing time, not a savings estimate.

**Precision allocation.** Centering and scaling refresh inputs with public
calibration, and assigning correction passes by downstream sensitivity,
could reduce both needed precision and refresh work. The relevant bound is
output perturbation, e.g. `sum_i sensitivity_i * refresh_error_i`, including
nonlinear remainder and recurrent accumulation. One-pass global replacement
already failed; it is not a new candidate. Calibration maxima and Jacobians
are not certified global bounds. This is a coupled conditioning/refresh
problem and overlaps with every bootstrap proposal above.

**State horizon.** Exact finite-history factorization is already implemented;
it must not be counted as a new optimization. It carries roughly
`H*T*(N+P+1)` factor scalars and contracts T factors per step, giving quadratic
total history work over T steps. For the checkpoint's H=24, P=64, N=128, those
factor scalars exceed the dense state's `H*P*N` after about 42 steps, before
angle/lag caches and ciphertext-layout overheads. This is a logical storage
crossover, not a measured runtime threshold. A dense checkpoint plus a bounded
exact factor window, as discussed in the prior algebra study, is the relevant
long-horizon representation to screen. The five-evaluation result cannot be
linearly extrapolated to hundreds of generated tokens.

## Alternatives screened down

| Alternative | Finding and consequence |
| --- | --- |
| SPRU roots-of-unity bootstrap | [The paper's Tables 2–3](https://arxiv.org/html/2607.27401v1) use 14-bit precision and only one usable output level. On its CPU, one slot improves from 1.7 to 0.3 s, but 1,024 slots worsen from 3.6 to 23.6 s. Its block-sparse secret differs from ours. Keep only as a separate few-value probe; “5×” does not justify replacing our batched refresh. |
| Functional bootstrap / LUT | [RBOOT](https://www.usenix.org/conference/usenixsecurity26/presentation/yang-zhaomin) targets ReLU fusion; it does not directly supply our RMSNorm or smooth gates. A slotwise LUT cannot remove the sum of squares before normalization, and discrete LUTs introduce a quantization contract. |
| Hybrid HE–MPC | [FESC](https://arxiv.org/html/2608.17442v2) uses leveled HE and interactive MPC nonlinearities, with a distilled/fine-tuned long-document model. It is evidence for a different protocol, not a faster implementation of this five-step generative checkpoint. Client participation moves from token boundaries into the backbone; communication and preprocessing become explicit costs. |
| FHE-native training | Smooth bounded gates, coupled decay/write certificates, and approximation-aware training may remove expensive nonlinear circuits. They change model weights/quality obligations and require their own baseline. They are a plausible route to a larger step than local kernel changes, without evidence of preserved checkpoint behavior. |
| More GPUs / larger batches | Independent requests can improve throughput and share public preparation. They do not make future autoregressive tokens available. Report latency and throughput separately, including inter-device ciphertext/key transfer. |

## Recommended next decision

The next bounded implementation campaign should choose mechanisms after
three small, separate screens: **SlotsToCoeffs-first two-pass refresh**;
**32-bit RNS arithmetic at matched precision/depth**; and a **dual-ring
cost/parameter model with actual conversion measurements**. The first two
target expensive backend work shared by Mamba-2 and Mamba-3. The third decides
whether a larger compiler/backend redesign has enough headroom to justify it.

Stop a screen when its benefit disappears after usable-depth, precision,
memory, and conversion costs are included. Promote survivors to the complete
unchanged model, comparing against 761.002 seconds; use held-out prompts and
longer carried-state horizons before making a broader quality claim. A new
20% reduction from this baseline means at most **608.802 seconds**. Halving
the complete refresh category alone would give 579.209 seconds; this is an
explicit conditional target, not a prediction that the proposed bootstrap
achieves it.

The previous cache, event-lifetime, and rotation-hoisting plans remain useful
engineering work, but do not settle the larger question. The evidence now
supports investigating **the cost of representing a small computation as a
large encrypted computation**, rather than assuming the present ciphertext
shape, bootstrap order, and machine-word width are unavoidable.
