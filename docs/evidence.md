# Evidence registry

This registry distinguishes tracked execution artifacts from measurements that
currently exist only in research notes. It is the provenance source for
headline README claims.

## Evidence classes

- **Raw/tracked**: repository contains the backend output JSON.
- **Raw/tracked, legacy schema**: output is present but predates current commit
  or binary provenance requirements.
- **Derived/tracked**: report generated from named raw artifacts.
- **Documented only**: value is recorded in prose, but the raw success artifact
  is not in the repository. It must not be represented as raw evidence.

## Current headline evidence

| Claim | Value | Class | Source / action |
|---|---|---|---|
| Bounded cache integration | Prefix ABBA 1.35% reduction; full pair `1004.622 → 976.107 s`; cache adoption: True | Raw and derived runs on the preceding immutable binary | [Study](research/2026-09-24-packed-cache-integration.md), [artifacts](../results/dgx/2026-09-24/packed-cache-integration/); full baseline precedes intervening Mamba-2 validation; gates, counts and selected IDs unchanged |
| Borrowed plaintext upload and routing | Mamba-3 same-binary full `1041.348 → 1004.622 s` (3.53% reduction); model rotations 59,900→50,780; Mamba-2 ABBA 2.73% reduction and 32.94-minute full parity | Raw and derived controls, exact-RNS/CPU contracts, source/binary/header hashes | [Study](research/2026-09-24-borrowed-plaintext.md), [artifacts](../results/dgx/2026-09-24/borrowed-plaintext/); unchanged architecture-specific gates, one full Mamba-3 run per mode, Mamba-2 full is parity only |
| Plaintext coefficient ownership | Mamba-3 prefix ABBA `15.4315 → 15.1540 s` (−1.80%); full candidate `1038.777 s` with matching IDs, max exact error `0.0001713`; Mamba-2 ABBA +0.48%, full follow-up skipped | Raw and derived runs, exact-RNS probes, frozen sources and retained validation failures | [Study](research/2026-09-24-coefficient-ownership.md), [artifacts](../results/dgx/2026-09-24/coefficient-ownership/); the single full run establishes parity, not incremental speed; Mamba-2 recommendation keeps moves off |
| Packed resources and direct upload | Mamba-3 same-binary 12×5 `1114.93 → 1054.86 s` (−5.39%), matching IDs, maximum exact error `0.0001266`; matrices 673.3→168.3 MiB; Mamba-2 direct-upload ABBA −5.56% and 33.83-minute full parity | Raw and derived runs, mirrored individual controls, exact-RNS probes, static inventory, CPU state parity and frozen source | [Study](research/2026-09-24-packed-resources.md), [artifacts](../results/dgx/2026-09-24/packed-resources/); one full Mamba-3 run per mode, unchanged gates, opt-in; persistent packing is inventory-only |
| Shared plaintext preparation | Mamba-2 same-binary 24×5 `2318.10 → 2142.11 s` (−7.59%), all four IDs and operation counts/levels match, max polynomial error `0.005044`; Mamba-3 12×5 regression passes | Raw and derived runs, two exact-RNS policy probes and frozen source | [Study](research/2026-09-24-shared-plaintext-preparation.md), [artifacts](../results/dgx/2026-09-24/shared-plaintext-preparation/); mirrored short controls, one full run per mode, unchanged architecture-specific gates, opt-in |
| Mamba-3 GPU plaintext encoding | Same-binary full 12×5 run `1312.20 → 1115.51 s` (−15.0%, 1.176×), all four IDs match, max exact error `8.83e-5`; prefix ABBA −16.7% | Raw and derived runs, exact-RNS probes, frozen source and dependency hashes | [Study](research/2026-09-24-mamba3-gpu-encoding.md), [artifacts](../results/dgx/2026-09-24/mamba3-gpu-encoding/); one full run per mode, unchanged 0.001 gates, opt-in, inline client, security not-set |
| Mamba-3 plaintext cache | Same-binary prefix ABBA `19.59 → 19.32 s` (−1.4%); preparations `888 → 825`, unchanged 0.001 gates; four-step state regression passes | Raw runs, comparison script and frozen source | [Study](research/2026-09-24-mamba3-plaintext-cache.md), [artifacts](../results/dgx/2026-09-24/mamba3-plaintext-cache/); opt-in component result, no new full-generation timing |
| Mamba-3 microkernels | ARM mask permutation: 12.4× locally, bitwise equal; GPU scratch reuse: matched prefix ABBA `22.64 → 22.23 s` (−1.8%), one-layer generation ABBA `67.66 → 66.14 s` (−2.2%) | Raw samples, native runs, assembly and Nsight summaries | [Study](research/2026-09-24-mamba3-microkernels.md), [artifacts](../results/dgx/2026-09-24/mamba3-microkernels/); local/component results, no full-session microkernel speed claim |
| Mamba-3 planned depth and batched refresh | Same 12-layer × 5 payload: `2771.84 → 1496.95 s` (−46.0%, 1.85×), all four tokens match, max exact error `1.55e-4` | Raw full runs and hash-bound comparison | [Study](research/2026-09-24-mamba3-depth-batching.md), [artifacts](../results/dgx/2026-09-24/mamba3-depth-batching/); one run per variant, inline client, security not-set |
| Trained Mamba-3 SISO generation and routing | Complete 12-layer × 5 evaluation run: `2771.84 s`, four matching tokens, max exact hidden error `2.17e-4`; matched one-layer routing: `130.17 → 100.83 s` (−22.5%) | Raw and derived local evidence, source/input hashes recorded | [Study](research/2026-09-24-mamba3-trained-generation.md), [artifacts](../results/dgx/2026-09-24/mamba3-trained-generation/); one full run, speedup measured only on one layer, inline client, security not-set |
| Trained Mamba-3 SISO preflight | Full 187M CPU/reference generation agrees; one-layer encrypted client loop: `129.67 s`, CKKS-to-exact error `3.52e-5` | Historical preflight evidence | [Study](research/2026-09-24-mamba3-trained-preflight.md), [artifacts](../results/dgx/2026-09-24/mamba3-trained-preflight/); superseded for full-generation status by the later study |
| Mamba-3 SISO mixer | Random one-mixer fixture, 4/8 encrypted steps: `102.68 / 195.71 s`; maximum CKKS-to-exact error `7.72e-8 / 1.58e-7` including carried states | Raw and derived/tracked | [Study](research/2026-09-24-mamba3-siso.md), [raw artifacts](../results/dgx/2026-09-24/mamba3-siso/); independent upstream reference parity, inline feasibility context, security not-set; no full-checkpoint claim |
| Stabilized prompt-to-text generation | 24 layers × 5 evaluations; four IDs match exact/poly references; maximum CKKS-to-poly error `0.009192`; evaluation `3,038.12 s`; peak RSS `37.05 GiB` | Raw and derived/tracked | [Study](research/2026-09-22-client-generation.md), [raw result and reports](../results/dgx/2026-09-22/client-generation/); one-process client loop, security not-set |
| Periodic joint-gate coefficients | Same 24×5 request and generated IDs; evaluation `3,038.12 -> 2,573.37 s` (−15.3%), joint gates −34.2%; maximum CKKS-to-poly error `0.011767 <= 0.05`; peak `37.09 GiB`; unchanged depth and 2187 bootstraps | Raw and derived/tracked | [Study](research/2026-09-22-periodic-gate-coefficients.md), [raw runs and comparison](../results/dgx/2026-09-22/periodic-gates/); opt-in, one full run per variant plus repeated interleaved smoke; existing client-loop/security scope |
| Subring coefficient encoding | Same 24×5 request; `2,573.37 -> 2,310.80 s` (−10.2%), joint gates −29.2%; polynomial error `0.016255 <= 0.05`; `121,975` subring encodes; identical generated IDs, operation counts and levels | Raw and derived/tracked | [Study](research/2026-09-22-subring-gate-encoding.md), [raw runs and comparison](../results/dgx/2026-09-22/subring-gates/); exact RNS parity, two isolated probes and four interleaved smokes; one full run per variant, opt-in |
| Generation and normalization bounds | Frozen-payload Amdahl/NTT model; widest variance interval requires polynomial degree at least `307,582`, hence at least 19 binary-product stages at relative error `1e-7`; current balanced recipe uses 30 | Derived analysis/tracked | [Cost model](research/2026-09-22-generation-cost-bounds.md), [normalization proof and report](research/2026-09-22-normalization-bounds.md); necessary bounds and conditional scenarios, not achievable latency or a benchmark |
| Mamba-2 polynomial quality | PPL `22.307 -> 22.333`, 280 windows | Raw/tracked, legacy schema | `results/ppl_ladder_mamba2_frozen_cert.json` |
| Decode lowering parity | max error `3.0518e-5` over five verified tokens | Raw/tracked, legacy schema | `results/decode_budget_mamba2.json` |
| Official Transformers parity | matching next token; logits max difference `4.158e-4` | Raw/tracked, legacy schema | `results/parity_mamba2-130m-hf.json` |
| 24-layer, three-token B300 chain | errors `0.01295 / 0.01173 / 0.03475`; eval `145.75 s`; 469 physical bootstraps; 120.24 GiB peak RSS | Documented only | `docs/research/2026-07-13-fhe-mamba-bottleneck-survey.md`; recover/rerun under PBI-M4-001 |
| 128-bit layer-0, two-token probe | errors `0.01182 / 0.03124`; eval `393.23 s` | Raw/tracked, legacy schema | `results/dgx/m1_decode_128bit_r131072_d43_s59_t2.json` |
| Three-process key separation probe | max error `1.7867e-12` | Raw/tracked | `results/dgx/client_server_probe.json` |
| Existing tracked B300 raw results | three 24-layer/one-token failures at package `0.4.4` | Raw/tracked | `results/b300/`; negative evidence only |

The historical PPL report does not identify the current Spark payload's
coefficients or head mask. Its reported degrees also differ from the current
exported surrogate. Do not use that value as a quality certificate for the
new GPU results; PBI-QUALITY-001 requires a payload-bound quality evaluation.

## DGX Spark review, 2026-09-21

The new results are under [`results/dgx/2026-09-21/`](../results/dgx/2026-09-21/).
They are feasibility runs with ring 65536, depth 44, scale 59 and
`security=not-set`, not a 128-bit or complete protocol claim.

| Measurement | Linear replication | Binary replication | Scope |
|---|---:|---:|---|
| Maximum polynomial error | 0.005011 | 0.003994 | 1 layer, 2 tokens; both pass 0.05 |
| Physical application rotations | 802 | 732 | Same 1-layer/two-token circuit |
| Loaded application keys | 137 | 103 | Same balanced key budget, reduced required inventory |
| Peak RSS | 38.369 GiB | 32.638 GiB | Process RSS, not total unified-memory use |
| Setup | 29.390 s | 24.795 s | One run per variant |
| Evaluation | 12.1165 s | 12.1172 s | No demonstrated evaluation speedup |

Raw sources:
[`linear smoke`](../results/dgx/2026-09-21/m2_chain_spark-smoke-linear_l1_t2.json),
[`binary smoke`](../results/dgx/2026-09-21/m2_chain_spark-smoke-binary_l1_t2.json).
Both include binary/build provenance and zero intermediate decrypts. These
smokes preceded the new payload-digest enforcement; do not attribute a payload
hash to them retrospectively. Their input path is recorded in the
[`smoke campaign`](../results/dgx/2026-09-21/spark-smoke-20260921.json).

The initial full 24-layer, one-token **linear baseline fails** with finite
but invalid output (maximum error `2.0787e91`). Its
[`raw artifact`](../results/dgx/2026-09-21/m2_chain_spark-linear-r1_l24_t1.json)
includes payload SHA-256 and build/library identity. The
[`campaign report`](../results/dgx/2026-09-21/spark-replication-fused-ab-20260921.json)
therefore skips the dependent optimized runs. This is negative evidence;
neither a 24-layer speedup nor multi-token promotion was established.

The [24-layer diagnostic](../results/dgx/2026-09-21/m2_chain_spark-depth-diagnostic_l24_t1.json)
has layer-5 boundary error `0.003384`, followed by layer-6 error `22.7573`.
Earlier boundaries are all below `0.00484`. This is the first token, so the
native decay-of-past-state branch is not evaluated. These diagnostics localize
an observation; they do not identify its cause.

The subsequent [seven-layer diagnostic campaign](../results/dgx/2026-09-21/spark-numerical-diagnostic-20260921.json)
passes with errors `0.004763` (instrumented baseline), `0.009710` (unity
alignment), and `0.018520` (level-0 cache-miss encoding). Each uses a fresh key
and phase-level debug decryption. A shorter chain changes final-layer handling
and cache pressure; these runs do not establish that either setting fixes the
full-chain failure, and their timings are not normal-execution benchmarks.

The [full-depth phase diagnostic](../results/dgx/2026-09-21/m2_chain_spark-full-values_l24_t1.json)
reproduces the failure with debug decryption. Keeping 24 layers and disabling
consumption-level plaintext encoding then
[passes without debug decrypts](../results/dgx/2026-09-21/m2_chain_spark-full-encode0_l24_t1.json)
at maximum error `0.021748`. Inspection found a same-level plaintext-addition
scale mismatch: FIDESlib skips alignment when modulus levels match, even when
the ciphertext and plaintext have different noise-scale degrees. Cache misses
encode convolution/step-size constants at degree 1, while the preceding
product can have degree 2. The small-cache Spark setting exposes this defect.

The targeted fix re-encodes those additive constants at the ciphertext's
actual degree and level, preserving consumption-level encoding for products.
The [encrypted regression probe](../results/dgx/2026-09-21/spark-plaintext-add-probe-20260921.json)
tests fresh ciphertexts and ciphertext squares at consumed levels 0/1/2:
uncorrected maximum error `0.2500001`, corrected error `2.170e-7`, and exactly
three required re-encodes. This probe uses ring 16384 / depth 8 / scale 40,
`security=not-set`; it is a backend correctness test, not a model gate.

After the targeted fix, the
[24-layer/one-token ABBA campaign](../results/dgx/2026-09-21/spark-scale-fixed-ab-20260921.json)
passes all four fresh-key runs, with normal execution and zero intermediate
decrypts. The [raw files](../results/dgx/2026-09-21/scale-fixed/) have
matching binary, native-source, library and payload hashes. Each reports 36
additive scale corrections; the consumption-level encoding optimization stays
enabled.

| Corrected 24-layer / 1-token measurement | Linear (two runs) | Binary (two runs) |
|---|---:|---:|
| Maximum polynomial error, individual runs | 0.028573 / 0.024756 | 0.021877 / 0.026309 |
| Physical application rotations | 8,415 | 7,575 |
| Application keys | 137 | 103 |
| Mean peak RSS | 41.855 GiB | 36.169 GiB |
| Mean setup | 28.911 s | 24.141 s |
| Mean evaluation | 131.154 s | 130.396 s |
| ct-pt / ct-ct products / physical bootstraps | 13,950 / 3,363 / 98 | 13,950 / 3,363 / 98 |

The memory reduction is repeatable in this small sample; evaluation differences
are comparable to run-to-run variation. Do not claim a demonstrated latency
speedup from these two measurements per variant. The maximum errors against
the **exact model**, rather than its polynomial circuit, are 0.09513–0.14499.
This gate therefore does not replace surrogate-quality evaluation. It also
does not exercise previous-token state, or certify a new held-out prompt set.

The subsequent [five-step campaign](../results/dgx/2026-09-21/spark-scale-fixed-five-step-20260921.json)
uses the same corrected binary and payload, with a two-token prompt and four
greedy selections in the client simulation. The
[linear baseline](../results/dgx/2026-09-21/scale-fixed-five-step/m2_chain_spark-five-step-linear_l24_t5.json)
decrypts all five outputs with zero intermediate debug decrypts, and generated
IDs `[273,253,4687,273]` match the reference. Per-step polynomial errors are
`[0.009847,0.082360,0.040823,0.045558,0.028547]`. **The gate fails** because the
second value exceeds 0.05; the dependent binary run is skipped. Error does not
grow monotonically in this run, and its source has not yet been localized.

Evaluation takes 828.462 s, with mean carried-state step time 174.197 s,
790 physical bootstraps, 42.285 GiB peak RSS, and 180 additive scale corrections.
Exact-model errors range from 0.11796 to 0.32822. This is a one-process,
`security=not-set` experiment: matching generated IDs is neither a substitute
for its failed precision criterion nor a complete encrypted protocol claim.

The [CPU checkpoint parity report](../results/parity_mamba2_cpu_20260921.json)
separately checks 138 tokens: official-versus-reference logits error
`2.136e-4`, chunked-versus-loop error `2.670e-4`, matching next-token ID. It
does not certify a new perplexity result or encrypted execution.

The [five-step layer diagnostic](../results/dgx/2026-09-21/state-diagnostic/m2_chain_spark-five-step-layer-diagnostic_l24_t5.json)
uses the same 24-layer geometry, payload and corrected binary. Its operation
counts match the earlier zero-debug run, but intermediate diagnostic decrypts
are enabled. Errors are `[0.012710,0.022998,0.053582,0.041218,0.089760]`;
all outputs decrypt and generated IDs match. The largest-error token changes
between these fresh-key runs. Layerwise growth is visible, but this does not
isolate one faulty recurrence operation. Its 942.354 s evaluation includes
diagnostic overhead and is not a normal-execution performance measurement.

The [row-state A/B campaign](../results/dgx/2026-09-21/spark-row-state-ab-20260921.json)
uses independently recalibrated state bounds (`e788c496…`) and the same new
binary (`578f8080…`) for both variants. Binary replication is enabled in both.
The [group-scaled control](../results/dgx/2026-09-21/row-state-ab/m2_chain_spark-group-state_l24_t5.json)
fails at its last output, while the
[row-scaled candidate](../results/dgx/2026-09-21/row-state-ab/m2_chain_spark-row-state_l24_t5.json)
passes every output. Both decrypt all outputs, match all four generated IDs,
and use zero intermediate debug decrypts.

| 24-layer / 5-step measurement | Group scale | Row scale |
|---|---:|---:|
| Per-step maximum polynomial errors | 0.02259 / 0.03961 / 0.01835 / 0.02176 / 0.18949 | 0.01503 / 0.02090 / 0.01146 / 0.01444 / 0.02314 |
| Evaluation | 808.690 s | 809.369 s |
| Peak RSS | 36.6255 GiB | 36.6260 GiB |
| Rotations / ct-pt / ct-ct products | 47,483 / 75,442 / 18,559 | same |
| Physical bootstraps | 790 | 790 |

This is an improvement in observed precision with unchanged operation count,
not a latency or memory improvement. A single pair uses different random keys
and is not a distributional precision guarantee. Calibration remains an open
issue: the [coverage audit](../results/state_scale_coverage_20260921.json)
finds row-scale escapes up to 9.07 times the measured bound, even though this
encrypted candidate passes. A successful CKKS run does not certify its assumed
calibration domain. Row scaling stays opt-in and does not resolve the plaintext
surrogate's long-context failures or the `security=not-set` scope.

A [fresh-key confirmation](../results/dgx/2026-09-21/row-state-confirm/m2_chain_spark-row-state-confirm_l24_t5.json)
uses the identical row-scale payload, binary and parameters and also passes.
Its per-step errors are `[0.024741,0.019736,0.013423,0.016571,0.044053]`, with
all outputs decrypted, matching IDs, zero debug decrypts and identical operation
counts. Evaluation takes 812.831 s and peak RSS is 36.6436 GiB. The two candidate
passes are evidence across fresh keys on one short prompt, not calibration
coverage or a long-horizon guarantee. The last error is still close to 0.05.

The [coefficient-aware Chebyshev comparison](../results/dgx/2026-09-21/spark-chebyshev-ab-20260921.json)
uses the same new binary and original row-calibrated payload for both arms,
with row normalization and binary replication enabled, 24 layers and one
fixed input token. Only the polynomial split planner flag differs.

| Measurement | Degree-only split | Coefficient-aware split |
|---|---:|---:|
| Maximum polynomial error | 0.0153648 | 0.0229899 |
| ct-ct products | 3,363 | 3,195 |
| ct-pt products | 13,950 | 13,404 |
| Additions | 21,848 | 20,852 |
| Physical rotations | 7,575 | 7,575 |
| Physical bootstraps | 98 | 98 |
| Direct level drops | 5,867 | 6,620 |
| Evaluation seconds | 129.594 | 128.353 |
| Peak RSS GiB | 36.1688 | 36.1713 |

Both [baseline](../results/dgx/2026-09-21/chebyshev-ab/m2_chain_spark-ps-default_l24_t1.json)
and [candidate](../results/dgx/2026-09-21/chebyshev-ab/m2_chain_spark-ps-planned_l24_t1.json)
pass 0.05 with zero intermediate decryptions. The planner changes 72 evaluation
calls, predicting 168 fewer ct-ct and 546 fewer scalar products; these exactly
match the actual differences. Its largest discarded-coefficient L1 sum is
3.91245e-12, excluding arithmetic/CKKS error. The source/binary and payload
hashes are identical across arms. More direct level drops reflect the changed
addition tree. One run per arm establishes deterministic operation-count
reduction, not a latency or precision-distribution improvement. `COEFFICIENT_AWARE_PS`
remains off by default; previous-token state and new prompts need separate gates.

## Payload quality and domain evidence, 2026-09-21

The [frozen-payload quality screen](../results/payload_quality_20260921.json)
binds coefficients/head masks, checkpoint and token-file hashes. On two
1,024-token WikiText-2 test windows, exact PPL is 18.39695 and exact-with-mask
PPL is 18.39309, with 99.902% argmax agreement. The exported polynomial circuit
produces non-finite values. This is a small plaintext screen, not encrypted
execution or full-dataset quality. Removing the invalid pruning justification
does not itself prove a PPL loss from the mask.

The same screen over two 4,096-token windows predicts 8,190 tokens: exact PPL
20.66927, exact-with-mask PPL 20.66836, argmax agreement 99.719%, and a non-finite
frozen polynomial circuit. The 1,024- and 4,096-token configurations share the
beginning of the test corpus and must not be counted as disjoint examples.
In the longer screen, 28 pruned heads have observed exact decay above 0.5;
the largest is 0.995671. Retrieval/state-memory quality remains unmeasured.

The [operator diagnostic](../results/payload_domain_20260921.json)
finds layer 5's gate input 28.5305 above its SiLU domain maximum 25.4530.
Polynomial extrapolation yields about 4.0213e9, followed by a non-finite gated
norm at token 756. This failure occurs before encryption.

The [rational Bernstein certificate](../results/payload_range_certificate_20260921.json)
proves nonnegative initialization and `v*y0^2<=3` for all 48 declared block and
gated-normalization intervals. It certifies a real-arithmetic polynomial range,
not private variance-domain membership, inverse-sqrt accuracy, or CKKS errors.
The [public-weight gate screen](../results/public_gate_polynomials_20260921.json)
and [first candidate diagnostic](../results/payload_public_gate_domain_20260921.json)
show why both range obligations matter: wider gate fits bypass the first failure
but reach a second gated-norm domain escape at layer 11, token 249.

The [positive-series seed diagnostic](../results/payload_positive_seed_domain_20260921.json)
combines wider degree-384 gate fits with degree-63 binomial Newton seeds,
eight refinements and four times the previous upper variance bounds. All
313 recorded stages over 24 layers/final norm stay finite on the first 1,024
tokens, with no non-decay domain escape. This changes the polynomial circuit;
the explicit 73 overrides include final-norm reuse. It is an unpromoted
development probe with added depth, not a repaired native quality certificate.

The [separate-window probe](../results/payload_positive_seed_quality_20260921.json)
starts at test-token offset 16,384. The exact model has PPL 30.84142, while the
same candidate is non-finite. Layer 13 convolution input 20.399 exceeds its
SiLU domain maximum 15.1154 and produces about -7.8359e24; the subsequent norm
fails. The window is now development evidence. Success on the earlier prefix
does not establish domain coverage elsewhere, and changing only gates/norm
seeds is insufficient.

The [public-activation candidate](../results/payload_public_activations_quality_20260921.json)
also uses degree-768 convolution SiLU fits on public projection/convolution
envelopes with a 10% margin. On the same offset-16,384 development window,
all 313 stages remain finite with no non-decay domain escape, exact PPL is 30.84142,
and candidate PPL is 31.00269 (+0.5229%), with 98.143% argmax agreement.
All 97 operator overrides are recorded. This is a promising plaintext
candidate, not the payload used in Spark gates; its larger degrees and eight
Newton steps have not been evaluated for encrypted cost or precision.

With these coefficients fixed, a
[previously unused window at offset 32,768](../results/payload_public_activations_holdout_20260921.json)
also stays finite: exact PPL 19.67562, candidate PPL 19.65756 (-0.0918%), and
99.022% argmax agreement on 1,023 predicted tokens. The payload hash and all
97 operator override descriptors/hashes match the development run. Neither
window has a non-decay operator-domain escape; raw decay violation counts
include inputs to masked heads. The slightly lower PPL on this small window
is not evidence of a general model-quality improvement. Longer contexts,
additional data and encrypted execution remain open.

The subsequent [4,096-token screen at offset 65,536](../results/payload_public_activations_long_20260921.json)
uses the same 97 override descriptors and coefficient hashes on a previously
unused interval. It **fails in layer 0**, before encryption. The time-step
input reaches -5.01163 below its fitted lower endpoint -4.69550. The first
non-finite checkpoint is the gated readout `y`; exact-model PPL is 17.56673,
and candidate PPL is undefined. The earlier two finite windows therefore do
not establish long-context reliability for this candidate.

The [independent calibration attempt](../results/state_coordinate_multicalibration_20260921.json)
uses four disjoint 256-token training windows at offsets 4,096 / 65,536 /
262,144 / 1,048,576, each with zero initial state. Exact states remain finite,
but the frozen polynomial circuit has non-finite states in layers 9–23.
Calibration fails closed and writes no payload. No failed windows are dropped
to obtain apparently successful bounds.

Separately, [public scale regularization](../results/state_coordinate_floor_20260921.json)
clones the earlier row-calibrated payload with `M'=max(M,M_group/8,1e-6)`.
It consumes no new text or evaluation reference. The
[read-only coverage audit](../results/state_scale_floor_coverage_20260921.json)
counts **344** values above the 1.1 margin versus **2,096** previously, across
33,030,144 fixed and autoregressive state scalars. Maximum reference/scale
ratio falls from 9.07164 to 5.67923, so coverage is still insufficient.
This combined count includes both reference sets; it differs from the earlier
autoregressive-only count. Mean row/group scale rises from 0.09597 to 0.15651,
trading some precision advantage for less amplification.

The [regularized five-step encrypted gate](../results/dgx/2026-09-21/row-state-floor/m2_chain_spark-row-state-floor_l24_t5.json)
passes with errors `[0.041854,0.032306,0.035398,0.010313,0.032676]`. All outputs
decrypt, generated IDs match, and intermediate diagnostic decryptions remain
zero. The binary is the same as the earlier row runs; the payload differs
only in public scale conditioning and its metadata. Rotations 47,483,
ct-pt products 75,442, ct-ct products 18,559 and physical bootstraps 790 are
unchanged. Evaluation takes 807.529 s and peak RSS is 36.6438 GiB. This is one
fresh-key numerical pass, not evidence of better precision/latency than the
unregularized row runs, nor a resolution of the remaining coverage failures.

The [joint bounded-gate probe](../results/bounded_selective_gates_20260921.json)
checks all 576 heads without pruning, at Bernstein degrees 32 / 64 / 128.
Exact-rational coefficient inequalities certify a conditional state invariant
for every head. However, worst sampled decay errors remain 0.42573 / 0.42032 /
0.39630, and write errors 6.31185 / 4.27376 / 2.85355. **This construction is
not an acceptable checkpoint replacement.** It provides a stable constrained
family for further fitting/training, not a quality result, a Chebyshev
conversion certificate, or an encrypted implementation.

The [shared-dissipation fit](../results/dissipative_selective_gates_20260921.json)
instead constructs `a=1-p²`, `b=p²q²` from public weights. Exact integer
Bernstein subdivision certifies `|p|<=1`; a Chebyshev coefficient bound on q
then certifies the joint invariant for all **576 heads**. Fitted maximum
degrees are 1,024/512 before trailing-coefficient trimming. On 8,193 grid
points per head, worst decay/write errors are **4.0170e-6 / 3.7885e-4**.
These are sampled approximation errors; the invariant is the certified part.
This larger-degree construction is not a matched-cost improvement over the
earlier Bernstein probe.

Two fixed-candidate **plaintext** evaluations combine those gates with the
wider SiLU/norm policy and retain all heads:

| 4,096-token WikiText-2 window | Exact PPL | Candidate PPL | Relative gap | Top-1 agreement | Maximum logit error |
|---|---:|---:|---:|---:|---:|
| [65,536, development](../results/payload_dissipative_long_20260921.json) | 17.56702 | 17.65695 | +0.5119% | 97.9976% | 82.6964 |
| [98,304, previously unused](../results/payload_dissipative_holdout_20260921.json) | 15.22679 | 15.25058 | +0.1563% | 98.4860% | 53.8167 |

Both have 4,095 predicted tokens, finite observed stages, no recorded input
domain escapes and no decay outside `[0,1]`. All 121 override descriptors,
including coefficient hashes and gate bundle SHA, match between the two runs.
Fitting uses no text. The original gate bundle SHA is
`c15a58e757b928881d11602f162ba095ac597c9de1cc23cb94a09d669cdaf743`.

The [matched CUDA control](../results/payload_public_activations_cuda_control_20260921.json)
uses the same source/dependency hashes, 97 other override descriptors and
coefficients, input tokens and TF32-disabled float32 execution. With the old
time-step/decay composition it still fails at layer-0 `y`; candidate PPL is
undefined. The earlier CPU artifact had 22 different convolution coefficient
hashes, so it is retained as historical evidence rather than the matched
control. The two successful runs do not establish uniform output closeness,
whole-model domain closure, retrieval/reasoning quality, encrypted precision
or a runtime improvement. The frozen native Spark payload is unchanged.

The [gate-only ablation](../results/dissipative_gate_only_ablation_20260921.json)
keeps every other nonlinearity exact on the development window. With the
same gate bundle, PPL is 17.56700 versus 17.56702, all 4,095 next-token argmax
choices agree, and maximum logit error is 0.025864. This isolates a faithful
gate approximation on that window; the much larger combined-candidate error
requires work on the other approximations and their interactions. The ablation
uses exact nonlinearities and is not a fully polynomial/FHE model.

## Normalization accuracy follow-up, 2026-09-21

The [single-site ablation](../results/payload_exact_first_norm_ablation_20260921.json)
replaces only layer 0's gated inverse square root with the exact operation.
On the same development tokens, maximum logit error falls from 82.6964 to
4.50343 and PPL gap from +0.5119% to +0.00999%. It is a diagnostic using an
exact nonlinearity, not an encrypted-compatible candidate.

The [public schedule report](../results/normalization_schedules_20260921.json)
then certifies relative inverse-square-root error ≤ `1e-7` and non-expansion
for all 49 norms, conditional on `[min(eps64,eps32),4*old_hi]`. Actual binary64
coefficients are checked with exact rational extrema and outward enclosures.
No text is used in planning. Scaled Goldschmidt plus a final recomputed Newton
step replaces the old degree-63 seeds/eight fixed Newton steps. These are
established iterative methods, adapted and verified here.

| 4,096-token WikiText-2 window | Exact PPL | Candidate PPL | Relative gap | Top-1 agreement | Maximum logit error |
|---|---:|---:|---:|---:|---:|
| [65,536, development](../results/payload_scheduled_norm_long_20260921.json) | 17.56702 | 17.56877 | +0.00994% | 99.8779% | 4.5000 |
| [131,072, previously unused](../results/payload_scheduled_norm_holdout_20260921.json) | 14.15001 | 14.15097 | +0.00682% | 99.7314% | 9.1330 |

Both runs are fully polynomial **plaintext** evaluations with 4,095 predicted
tokens, finite stages, zero observed domain/decay escapes, CUDA float32 and
TF32 disabled. The normalization and joint-gate oracles use float64 workspace.
All 121 overrides and implementation hashes match between these two runs.
The 72 non-normalization overrides also match the preceding development
candidate, with identical token hash and exact PPL; only the 49 normalizers
are replaced. Source changes add this schedule and passive diagnostics.
The lower PPL gap does not establish uniform logit accuracy or reasoning
quality. The fresh window has no matched old-candidate comparison.

Bundle SHA-256:
`89801adcc149d3183bf758fb8d022f4949b9b048ae5c665ff90a6892632c1c15`.
Schedules use 9–15 accelerated steps and one final repair. Layer 0's gated
norm has 36 ct-ct products and coupled critical depth 26. **Correction:** a
balanced cubic evaluation has depth 24; fixed Newton on the same domain and
accuracy needs 75 products / balanced depth 50. The previously reported 75
depth described an unbalanced DAG, retained in the historical JSON. This ledger
excludes scalar rescaling, level alignment, bootstrapping, layout and CKKS
error. The [native follow-up](research/2026-09-21-normalization-native.md)
measures isolated normalization performance; it does not compare with THOR.
The frozen full-model Spark payload and native executable are unchanged.

Validation at the plaintext-study stage: **207 tests passed**, including the
native C++ contracts; active-package coverage was **86.17%**. All **64** dated
artifacts passed commit-aware schema validation without errors or warnings.
Reproduction commands and ablation semantics are in [testing](testing.md).

## Encrypted normalization core, 2026-09-21

The [all-site Spark campaign](../results/dgx/2026-09-21/normalization/all-sites/campaign.json)
passes **49/49** frozen variance domains, with 8,192 synthetic signed inputs
per site and a fresh key per run. The circuit computes `v=x²+epsilon`, the
scheduled inverse square root, then encrypted `x*y` before decryption.
Worst sampled absolute output error is **9.00146e-8** against exact scalar
normalization. All runs use ring 131072, depth 40, scale 59, uniform ternary
secrets, OpenFHE-accepted **128-bit classic** parameters and the default real
decoder's noise. Consumed levels range from 25 to 37, with no bootstraps,
rotations or evaluation decryptions. This is a scalar core; vector reductions,
gamma, prior-layer errors, refresh and whole-model integration are unmeasured.

The [matched widest-domain ABBA campaign](../results/dgx/2026-09-21/normalization/abba/campaign.json)
compares balanced cubics and public weighting of the coupled residual.
Both use 47 ct-ct products including variance formation/output. Weighted
reduces scalar products **34→19**, consumes levels **35→37**, and has errors
`2.457e-8 / 2.790e-8` versus balanced `2.066e-9 / 1.945e-9`. Timings are
`0.66160 / 0.66116 s` versus `0.75539 / 0.74997 s`, two samples per variant.
This measures a small-kernel tradeoff; it does not establish a full-model
speedup. The [rounded-ratio certificates](../results/normalization_probe_inputs_20260921.json)
check the effective weighted coefficients, not just the original recipe.

Standalone inverse-square-root decryption remains a separate negative gate
on the widest interval. Other retained failures cover an excessive limb budget
and the pinned backend's reverse-scalar-subtraction path. The isolated runner
records missing output as failure even if the native process exits zero.
See the [derivation, decoder-noise discussion and failure analysis](research/2026-09-21-normalization-native.md).
No failure is reclassified using the normalized-output criterion.

The [convolution-SiLU ablation](../results/payload_exact_conv_ablation_20260921.json)
localizes the next model approximation issue: with only convolution SiLUs
made exact, development maximum logit error is **0.17742** instead of 4.5000,
and all 4,095 next-token argmax choices agree. This diagnostic is not a fully
polynomial model and does not change the measured deployable-candidate PPL.

Validation at the scalar-native stage: **218 tests**, including **10 native C++ contracts**;
active-package coverage **86.32%**. All **129** dated artifacts validate with
zero errors/warnings. The full-model Spark binary remains unchanged; the new
probe has its own build directory and archived source/binary provenance.

## Packed vector RMSNorm and refresh, 2026-09-21

The [vector campaign](../results/dgx/2026-09-21/vector-rms/all-sites/campaign.json)
passes **49/49 sites**, including feature reduction and original learned gamma,
on 784 freshly encrypted vectors / 897,024 active components. Each site has
eight exact-checkpoint inputs from a new 128-token window at offset 180,224
and eight public stress vectors. No coefficients are fitted on these inputs.
Worst output error is **6.95634e-5**, or **1.17633e-6** on checkpoint inputs
alone, at ring 131072 / depth 40 / scale 59 / 128-bit classic parameters.
The worst site's encrypted-versus-polynomial error is only `1.38776e-7`;
gamma amplifies the existing polynomial error. Padding outputs are checked.

Gamma on the numerator branch saves one level: a widest-domain balanced
comparison consumes **36 instead of 37**, with unchanged operation counts.
Weighted vector runs consume levels 26–38 and at most 11.67 GiB RSS. The
packing uses independent strided vectors, not consecutive autoregressive
steps or the current full-kernel layout.

Ordinary output refresh fails (`0.0123022`). Explicit Meta-BTS at alpha 12
passes the widest block and layer-11 gated domains, but initially fails the
largest-gamma site (`1.81119e-4`). All failures remain archived. Public
component scaling around Meta-BTS then passes a matched **global / component /
component / global** comparison at that site: global errors
**1.59564e-4 / 1.57742e-4**, component errors **5.06395e-5 / 5.32236e-5**.
Both use two physical bootstraps per batch and finish at level 23; component
scaling replaces two scalar products with two diagonal plaintext products.
It has no demonstrated runtime speedup. Fresh-key confirmations pass at
four selected sites; a full 49-site refresh campaign is unmeasured.

Default real-decoder noise stays enabled. This tests output refresh from
fresh ciphertexts, not error from preceding layers. A refreshed result still
has insufficient depth for another complete normalization, requiring internal
refresh scheduling before full-model integration. See the
[derivation, raw comparisons and limits](research/2026-09-21-vector-rms.md).

Validation at the vector-study stage: **220 tests**, **11 native C++ contracts**, **86.32%**
coverage; **196** dated artifacts validate without errors or warnings. The
frozen full-model executable and plaintext model-quality measurements remain
unchanged.

## Scheduled RMSNorm full-kernel integration, 2026-09-21

The [new opt-in payload and native path](research/2026-09-21-normalization-integration.md)
integrates all 49 frozen normalization recipes into the existing contiguous
full-model layout. A dedicated final recipe replaces reuse of the last block
fit. Internal inverse refresh uses public stage bounds, and transient Meta-BTS
protects the incoming normalization precision. The depth planner reserves
headroom for the immutable variance branch across inverse refreshes.

The [24-layer/two-fixed-token result](../results/dgx/2026-09-21/normalization-integration/m2_chain_scheduled-norm-full-chain-live-variance-r1_l24_t2.json)
and [campaign](../results/dgx/2026-09-21/normalization-integration/full-chain-live-variance-campaign.json)
pass the unchanged **0.05** polynomial-circuit error gate. Per-token errors
are **0.0000921 / 0.007136**; exact-model errors are **0.047809 / 0.045393**.
Both final outputs are finite. State and convolution FIFO remain encrypted
between tokens, and evaluation uses zero intermediate decryptions. This is
fixed-input execution, not generated-token selection.

Evaluation takes **580.95 s** after **23.28 s** of setup, with peak RSS
**36.15 GiB**, **796** physical bootstraps, **17,556** rotations,
**28,538** ct-pt products and **8,467** ct-ct products. This is one run at
ring 65536 / depth 44 / scale 59 / `security=not-set`, in one process.
It establishes neither a 128-bit full-chain result nor a performance gain.
Raw results retain source, binary, dependency and payload hashes.

The [comparison artifacts](../results/dgx/2026-09-21/normalization-integration/)
preserve failures: inverse-only Meta-BTS reaches error **0.172350** on the
one-layer/two-token gate; unity alignment alone still reaches **0.171923**.
With transient Meta-BTS it passes at **0.006343**, versus a matched legacy
control at **0.003886**. The initial full-chain attempt exhausts depth after
layer 3; accounting for the live variance branch enables the completed run.

This payload retains legacy SiLU/time-step/decay fits and head pruning. Its
[matching 1,024-token plaintext quality screen](../results/payload_native_scheduled_norm_20260921.json)
**fails with non-finite polynomial outputs**; exact / exact-with-mask PPL is
**13.25128 / 13.24844**. It is a different surrogate from the 4,096-token
quality-passing joint-gate candidate above. The joint-gate/wider-activation
follow-up below evaluates a separate matching payload.

Validation: **228 tests**, including **11 native C++ contracts**, and **87.04%**
coverage. All **13** new measurement/report JSON files validate with zero
errors. The failed partial depth run has two warnings for unavailable operation
counts; successful results have no validation warnings. Missing counters are
not reconstructed from later runs.

## Joint gates and public activations in the native kernel, 2026-09-21

The [complete candidate integration](research/2026-09-21-stabilized-native.md)
exports all 49 norm recipes, 48 public-envelope SiLU fits and 24 joint gate
recipes, retaining all 576 heads. Each exported coefficient set matches the
previous plaintext candidate. The new payload format requires the matching
native vector-coefficient evaluator and regenerates its own references.

The [actual exported-payload quality report](../results/payload_stabilized_native_quality_20260921.json)
is finite on the first 1,024 and 4,096 WikiText-2 test tokens, with zero observed
operator-domain escapes. Exact/poly PPL is **13.25146/13.24670** and
**17.40732/17.40441**; differences are **−0.03596% / −0.01669%**. Top-1
agreement is **100% / 99.9023%**, with maximum logit errors **4.59 / 5.63**.
These are overlapping plaintext windows, evaluated on CUDA with TF32 disabled;
the small PPL changes are not a general quality-improvement claim.

The [one-layer/two-token gate](../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-smoke-r3_l1_t2.json)
passes at errors **0.0001124 / 0.0003140**, both outputs finite and zero
intermediate decryptions. It uses **22** physical bootstraps, **732** rotations,
**3,302** ct-pt products and **466** ct-ct products, with **26.83 s** evaluation
and **32.69 GiB** peak RSS. Ring 65536 / depth 44 / scale 59 /
`security=not-set` and one process remain the numerical experiment boundary.
The encrypted and plaintext artifacts bind the same payload hash.

Retained failures show two encoder limits: cancellation in folded domain
offsets during preparation, then tiny sparse coefficient rows at layer 8 of
the initial full-chain attempt. Separate bias additions and exact power-of-two
PS block scaling address these without dropping frozen coefficients. See the
[raw comparisons](../results/dgx/2026-09-21/stabilized-integration/).
The first completed [24-layer/two-token run](../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-full-chain-r2_l24_t2.json)
still fails accuracy at **4.58e144 / 4.09e144**, despite finite outputs and
zero intermediate decryptions. Its **1,166.50 s** evaluation and **787**
bootstraps are failed-run measurements, not a successful performance result.

The [phase diagnostic](../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-diagnostic-r1_l24_t1.json)
and [log excerpt](../results/dgx/2026-09-21/stabilized-integration/stabilized-diagnostic-r1-excerpt.txt)
locate the first corruption in layer 8's Meta-BTS correction at exhausted
level 44. The first eight boundaries remain within **0.000221**. Accounting
for all six products from joint write to gated readout, including the readout
mask, refreshes that write earlier. The [nine-layer confirmation](../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-tail-r1_l9_t1.json)
then passes at **0.0002689**, with zero intermediate decryptions, **226.68 s**
evaluation, **34.89 GiB** peak RSS and **138** physical bootstraps. Its same
readout refresh now begins at level **38**, rather than 40. This partial
confirmation excludes final model normalization and token-state carry.

The corrected [24-layer/two-fixed-token result](../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-full-chain-r3_l24_t2.json)
and [campaign](../results/dgx/2026-09-21/stabilized-integration/stabilized-full-chain-r3-campaign.json)
**pass** at polynomial-circuit errors **0.0004716 / 0.0033107** and exact-model
hidden-output errors **0.0116666 / 0.0213601**. Both final normalized outputs
are finite, all 49 normalization sites execute, state/FIFO remain encrypted
between tokens, and evaluation uses zero intermediate decryptions. The
highest input level for Meta-BTS preparation is now **38**.

Evaluation is **1,192.95 s** after **24.13 s** setup, with **36.09 GiB** peak
RSS, **831** physical bootstraps, **17,598** rotations, **108,690** ct-pt and
**12,994** ct-ct products. Joint-gate evaluation accounts for **552.24 s**
(about **46.3%**); separately accumulated bootstrap time is **410.52 s**.
Source and binary identities match the archived build; the payload hash
matches the plaintext quality screen. This is one fresh-key run,
`security=not-set`, in one process,
with fixed embeddings and no native lm-head token selection. Longer encrypted
horizons, repeated keys/prompts, process separation and 128-bit full-chain
execution remain separate gates; measured gate/refresh cost is the next
performance bottleneck.

Local validation: **232 tests**, including **12 native C++ contracts**, with
**87.03%** coverage. The shared-basis evaluator is compared with independent
Clenshaw values through degree 1,024, including tiny rows, separate heads,
multiple lanes and padding.
All **17** new measurement/report JSON files validate with zero errors.
The two aborted encoder runs have four warnings for unavailable operation
counts; completed and successful results have no validation warnings.

## Stabilized prompt-to-text baseline — 2026-09-22

The [end-to-end study](research/2026-09-22-client-generation.md) completes
`The capital` → `The capital of the Republic of` with the frozen stabilized
operators and state bounds. All five native evaluations decrypt, all four
selected IDs match both exact and polynomial references, and maximum
CKKS-to-polynomial error is **0.009192**. Exact-model hidden-output error
reaches **0.072572** and is reported separately. No intermediate diagnostic
decryptions occur; four output decryptions are the client-selection boundary.

Evaluation is **3,038.12 s** after **23.44 s** setup, peak RSS **37.05 GiB**,
with **2,187** physical bootstraps. The new local command consumes the entire
prompt, transfers a fresh payload, invokes the validated binary and decodes
measured IDs. The [native output](../results/dgx/2026-09-22/client-generation/m2_chain_client-generation-20260922_l24_t5.json),
[campaign](../results/dgx/2026-09-22/client-generation/client-generation-campaign.json)
and [derived text report](../results/dgx/2026-09-22/client-generation/generation.json)
validate with no errors or warnings. Source archives and hashes preserve the
launch code and the metadata-only report reconstruction.

This is one prompt with fresh keys, a one-process client loop and
`security=not-set`. Client/server key separation, encrypted token selection,
long sessions and a 128-bit full-chain parameter gate remain open. Runtime
optimization now has a complete functional baseline to preserve.

## Algebra and approximation research, 2026-09-21

The [SSM/cryptographic design survey](research/2026-09-21-ssm-cryptographic-design.md)
links the derivations to primary sources and executable probes.

- [Deferred-state report](../results/ssm_algebra_20260921.json): actual
  checkpoint factors from 24 layers and 64 tokens, windows 2/4/8/16. All 96
  float64 reassociation comparisons pass; worst readout error `2.843e-13` and
  final-state error `1.137e-13`. No encrypted execution or performance measured.
- [Composite-decay screen](../results/decay_composition_20260921.json):
  degree-64 direct approximation has sampled error `1.100e-4` on existing active
  heads but 126 range-violating heads. It remains unpromoted. Sampling is not
  an interval certificate, and this is not the existing polynomial circuit.
- The same report audits all 47 pruned heads. Their maximum exact decays on
  the exported input intervals are 0.97701–0.999999745; none satisfies the
  `exp(-32)` negligible-decay criterion throughout the interval. These extrema
  do not show actual endpoint occurrence or measured language-quality loss.
- Complex-trapezoid parity is a synthetic real/complex algebra test. The cubic
  phase negative control loses most magnitude over 1,024 steps at angle 0.5.
  Neither result certifies a trained Mamba-3 model.
- [Shear-phase screen](../results/phase_schedules_20260921.json): three
  polynomial shears preserve a modified energy for fixed angle 0.5, with norm
  1.0000–1.0328 over 1,024 steps, but accumulate 5.489 radians of frequency
  error. A changing-angle period has spectral radius 1.018696 despite unit
  determinant. This candidate is unpromoted and requires architecture/quality
  work, not a silent phase substitution.

## B300 `0.4.5` recovery contract

The recovered or regenerated success artifact must satisfy all of the
following before the README evidence class changes to **Raw/tracked**:

- `status=passed` and `passed=true`;
- `repo_commit` identifies the evaluated source;
- `binary_sha256` identifies the executable;
- 24 loaded layers, final RMSNorm, and three sequential tokens;
- zero intermediate decrypts and real ciphertext state/FIFO carry;
- `security=not-set`, ring `65536`, scale `59`;
- fully synchronized FIDESlib profile;
- fused replicated transform enabled for `out-proj` only;
- complex state pairing enabled;
- all token errors `<= 0.05`;
- per-token error, timing, bootstrap, rotation, cache, and RSS telemetry.

If the original JSON cannot be recovered, rerun the exact promoted baseline.
Do not infer missing fields from the prose measurement.

## Promotion policy

A new headline claim requires:

1. a tracked raw artifact or a tracked derived report whose raw inputs are
   named and available;
2. validation with `fhemamba validate-artifacts --require-commit`;
3. matching README, evidence registry, backlog, and package version text;
4. explicit security, model, token-horizon, process-separation, and hardware
   scope.

Failed and negative artifacts remain valuable. They must be labeled as such
and must never be used as substitutes for a missing success artifact.
