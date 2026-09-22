# Design: encrypted recurrent language models

Updated 2026-09-21. This is the active design; `fhemamba/DESIGN.md` remains a
historical log. The immediate reference is the public Mamba-2-130M checkpoint.
The larger objective is a usable language model whose server-side inference
never decrypts client data.

The [SSM/cryptographic design survey](research/2026-09-21-ssm-cryptographic-design.md)
extends this design with derivations and executable algebra gates. Its primary
next candidate is exact deferred-state evaluation; direct complex Mamba-3 and
an autonomous encrypted vocabulary loop are separate research tracks.

## Protocol and correctness contract

The client retains the secret key, plaintext prompt/history, embedding and
public vocabulary head. The server receives public/evaluation keys and
ciphertexts, evaluates all blocks and final normalization, and returns a
ciphertext hidden vector. State and the convolution FIFO persist encrypted.
Client token selection is an allowed protocol boundary; intermediate
activation decryption or re-encryption must never become an accuracy repair.

Three distinct gates prevent one kind of success from masking another:

1. **Model fidelity:** exact checkpoint versus polynomial surrogate, with
   closed-loop perplexity and generated sequences on held-out prompts.
2. **Circuit fidelity:** CKKS versus the identical polynomial reference,
   with per-token errors, decryptability, and carried-state diagnostics.
3. **Protocol security:** a separate secret-key-free server, parameter
   validation, output/decryption security analysis, and measured transfers.

The first two gates can run in one process. They cannot establish the third.
CKKS is approximate; equal selected tokens do not prove hidden-state parity.
Diagnostic runs are labelled separately, and decrypted values never influence
subsequent ciphertext computation. Timing includes setup, cold/warm tokens,
state refresh, memory and transfers where applicable.

The stronger endpoint performs vocabulary projection, token selection and
embedding lookup homomorphically and decrypts only at final delivery. The
current interactive protocol is a milestone toward that endpoint. Greedy
selection can omit softmax exactly, but still requires encrypted comparisons,
a logit-error/margin contract and encrypted embedding access.

## Why Mamba, and what is still hard

For Mamba-2, each head has scalar decay `a_t` and matrix state:

```text
S_t = a_t S_(t-1) + u_t B_t^T
output_t = S_t C_t + D x_t
u_t = dt_t x_t
```

The recurrence keeps a fixed-size decode state rather than a growing KV cache.
Affine transitions compose associatively:

```text
(a₂, U₂) ∘ (a₁, U₁) = (a₂ a₁, U₂ + a₂ U₁)
```

That allows scans when prompt tokens are known. Future autoregressive tokens
are not known in advance and cannot be evaluated by the same parallel scan.
This distinction follows the [Mamba-2/SSD formulation](https://arxiv.org/abs/2405.21060).

For this checkpoint, each layer's state contains `24 × 64 × 128 = 196,608`
real values. At 32,768 slots this is six ciphertexts per layer, before FIFO,
keys, temporary values or bootstrap workspace. Fixed size is useful, but not
small. The expensive work includes state refresh, public linear projections,
and nonlinear normalization. Spark's CPU and GPU share memory, so a higher
plaintext-cache hit rate can make total inference slower.

Even with `|a_t| <= 1`, a bound such as
`e_t <= |a_t| e_(t-1) + local_error_t` applies only to an isolated recurrence
with fixed inputs. Input-dependent gates and 24 stacked layers can amplify
errors. Polynomial extrapolation can also violate the decay bound. Long
horizons therefore require measurements and range audits, not a stability
claim based solely on the exact-model exponential.

## Immediate algorithm work

**Public linear maps.** Preserve the checkpoint weights. Binary rotate-add
sums implement `sum(rot(x, j*stride), j=0..r-1)` in
`floor(log2(r)) + popcount(r) - 1` rotations for any positive `r`.
They serve input extension, replica fill and replica folding, including the
seven- and twenty-replica layouts. Masks, multiplication count and depth remain
unchanged. The key inventory and frequency planner use the same schedule as
the native evaluator; an independent slot simulator checks the full matmul.
CKKS addition/rotation noise still requires an encrypted A/B gate.

**Prompt processing.** `_affine_scan` avoids full shifted identity/state tensors
and skips prefix elements that already have their answer. The current Mamba-2
reference streams 64-token chunks and carries the last state serially between
chunks. For 512 tokens that carry gives depth 14 in the current arithmetic
schedule, not the former estimate of 9. `prefill_budget.py` now accounts for
this, partial chunks, and explicit hypothetical packing assumptions. There is
no native encrypted scan-prefill implementation yet.

The next prefill design should compare a two-level scan of chunk summaries
with streaming serial carry. A parallel summary scan reduces carry depth but
needs additional live state and rotations. A work-efficient tree can reduce
arithmetic while increasing critical depth; under FHE that trade may add
bootstraps. Compare packed rotations, ct-ct products, live ciphertexts and
refreshes together before choosing a schedule.

**Refresh placement.** Use the actual dataflow DAG with fan-out, live-out
levels and normalization bounds. A list of observed bootstrap events alone is
not enough to delete refreshes safely. Paired real/imaginary refresh and
normalization are existing candidates; first establish a multi-token Spark
baseline, then change one dimension at a time. Interval-2 refresh already has
negative full-depth evidence and is not a default.

**Scale correctness.** Modulus level alone does not specify a CKKS value's
scale. Vector-constant additions must also match the ciphertext's noise-scale
degree. The Spark backend helper handles same-level degree-2 additions
explicitly; otherwise a degree-1 constant can effectively disappear. This
contract must hold for both cache hits and consumption-level cache misses.

**State coordinates.** The optional `ROW_NORMALIZED_STATE=1` uses a public
scale per `(head, channel)` instead of one per four-head ciphertext. Input
division and readout restoration occupy the existing masks; operation count
and depth are unchanged. Bounds come from independent exact and polynomial
calibration trajectories. The group and row modes must be compared with the
same recalibrated payload. See the [derivation and limits](research/2026-09-21-ssm-cryptographic-design.md#implemented-variant-scale-the-channel-rows).

**Approximation domains.** The current frozen payload produces non-finite
values on held-out 1,024-token text. A SiLU gate exceeds its calibrated input
range before the following norm diverges. More accurate CKKS cannot repair
that plaintext circuit. An exact-rational certificate now proves the
non-expansive Newton condition on all 48 declared normalization intervals.
This supports public-weight projection envelopes for wider gate fits;
variance-domain membership and CKKS error slack are separate obligations.
Both end-to-end quality and encrypted parity remain required before adoption.
The wider SiLU/positive-seed candidate also fails an independent 4,096-token
window: the time-step fit is still empirical, and its extrapolation destabilizes
decay in layer 0. Domain coverage must include the entire gate composition.

The new plaintext candidate replaces that composition with joint factors:
`a=1-p(z)^2`, `b=p(z)^2*q(z)^2`, fitted to the exact checkpoint's decay and
Euler write. A certificate of `|p|<=1` plus a coefficient bound on `q` proves
the recurrence invariant on its declared domain. All 576 heads certify, and
two 4,096-token screens are finite with PPL gaps +0.512%/+0.156% and no observed
domain escapes. The matched old composition still fails in layer 0. These
are high-degree, unpruned plaintext candidates: native coefficient-vector
evaluation, encrypted cost/precision and whole-model domain closure remain
open. See the [derivation and limits](research/2026-09-21-ssm-cryptographic-design.md#shared-dissipation-factors-retain-accuracy-and-certify-the-recurrence).

**Normalization accuracy.** Non-expansion did not ensure convergence: the
first gated norm attenuated small inputs despite eight Newton steps. Public
scaled Goldschmidt schedules now certify relative error ≤ `1e-7` for all 49
norms on `[epsilon,4*old_hi]`, with a final Newton step recomputing the residual
from the original variance. Development/fresh 4,096-token plaintext PPL gaps
are +0.00994%/+0.00682%; no observed domain escapes. All other 72 overrides
match the previous candidate. This applies established aSOR/THOR methods,
with rounded-coefficient certificates, and does not claim a novel iteration.
The [native follow-up](research/2026-09-21-normalization-native.md) now passes
synthetic scalar RMS normalization on all 49 domains, with maximum sampled
output error `9.00e-8` at OpenFHE-accepted 128-bit parameters. It evaluates
`v=x²+epsilon` and multiplies by encrypted x before decryption; standalone
inverse decryption has a separate failing wide-domain gate. Real-decoder
noise stays enabled. Balanced cubics minimize depth in this comparison;
publicly weighted residuals reduce scalar products while adding two levels.
The [vector follow-up](research/2026-09-21-vector-rms.md) also passes all 49
sites with actual feature reduction and learned gamma: 784 fresh-encrypted
vectors, worst error `6.96e-5` including stress cases. Moving gamma onto the
numerator branch saves one level. Ordinary output refresh fails its accuracy
gate; Meta-BTS with public component coordinates passes four selected sites.
This changes refresh coordinates, with default decoder noise retained.
A refresh returns less depth than a complete next norm consumes,
so internal refresh scheduling, prior-layer errors, domain closure and
full-model layout integration remain obligations. See the [derivation](research/2026-09-21-ssm-cryptographic-design.md#interval-certified-normalization-schedules).

**Polynomial evaluation.** `COEFFICIENT_AWARE_PS=1` chooses the public
Paterson–Stockmeyer split from actual coefficients, including cached basis
reuse and the existing small-coefficient cutoff. It retains baseline depth
and scalar-product ceilings; coefficients are not refitted. The bootstrap
ledger stays conservative. Ciphertext error and end-to-end latency still
require matched measurements for the selected schedule.

## Architecture research alongside checkpoint completion

The checkpoint is a correctness anchor, not a constraint on all future models.
A separately trained FHE-oriented SSM can investigate bounded decay/gates,
smaller or structured state, cheaper normalization, and fusion of adjacent
public linear maps. Each proposal must state which weights require retraining,
its approximation domain, and its encrypted operation/refresh budget.

Training objectives should include language loss and sensitivity to measured
CKKS/bootstrap perturbations. Distillation and range regularization need
held-out perplexity, long-context retrieval and free generation checks.
Evaluate recurrence horizons of 2, 5, 16, 64 and 256 steps, then longer prompts;
report failures and avoid calibrating on the evaluation prompt. A low-rank
projection approximation or removal of gated normalization cannot be assumed
to preserve the existing checkpoint's quality.

The current head-pruning surrogate needs that same discipline. Its
`A * dt_max` threshold tests the smallest decay; it cannot establish uniformly
negligible memory. The current exported intervals permit large decays in all
47 omitted heads. Treat this as an empirical approximation, require a new
payload-bound quality gate, and compare unpruned/composite alternatives before
claiming preservation of the original model's long-context advantage.

Client re-prefill from its own history is an explicit possible protocol, not
a hidden decryption repair. It resets state age at a cost that grows with
history; measure the amortized cost and privacy boundary before adopting it.

## Security boundary

`security=not-set` at ring 65536 is a feasibility setting. OpenFHE acceptance
of 128-bit parameters is a necessary separate gate and does not by itself
provide a complete protocol-security result. Interactive decryption/output
security and circuit privacy require a threat model and an appropriate
construction. The upstream [OpenFHE noise-flooding documentation](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/CKKS_NOISE_FLOODING.md)
describes additional requirements; ordinary refresh or encrypting zero must
not be presented as satisfying them.

The first deployable milestone requires a secret-key-free server process,
validated full-chain parameters, bounded numerical behavior across held-out
sessions, and a stated output-security construction. Neither missing B300
artifacts nor speculative speed estimates substitute for these gates.
