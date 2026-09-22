# Shared-factor gates and public activations in the native kernel

This follow-up exports and lowers the complete polynomial candidate from the
plaintext normalization study: 49 scheduled normalizations, 48 public-envelope
SiLUs, and 24 joint write/decay gates retaining all 576 heads. It addresses the
1,024-token non-finite failure of the RMSNorm-only integration payload.

## Frozen arithmetic and payload contract

Each joint gate evaluates per-head Chebyshev roots on its public input domain:
`d = p(z)^2`, `write = d*q(z)^2`, `decay = 1-d`. The native implementation
uses a shared encrypted Chebyshev basis and vector plaintext coefficients with
a Paterson-Stockmeyer decomposition. Its host contract compares against an
independent Clenshaw evaluator through degree 1,024, including distinct heads,
multiple lanes, endpoints and padding. Coefficients are not thresholded in
this evaluator. Persistent basis values are cloned before level alignment.

The native input map adds the checkpoint time-step bias before multiplying
by the per-head domain scale, then adds the domain offset. An initial version
folded the two offsets on the host. Their cancellation produced values around
`1e-17`, which the pinned OpenFHE encoder rejected at scale degree one. Keeping
the additions separate preserves the coefficients and avoids discarding that
residue. The failed preparation run is retained as negative evidence.

The first full-chain attempt then reaches layer 8 (the ninth layer) but fails
when encoding sparse coefficient rows as small as `7.87e-15`. The fixed input
bias path is no longer the problem: inverse-FFT dilution across 32,768 slots
also affects coefficient masks that populate only the head slots. The updated
evaluator scales each PS leaf block by a public power of two, accumulates it,
then undoes the scale once. Coefficients are preserved; the extra scalar
product consumes the one level reserved in the joint-gate depth estimate.
The supported shift is at most 50 bits; larger requests fail explicitly.
The host Clenshaw comparisons also exercise extremely small coefficient rows.

`export_m1_payload.py --stabilized-gate-bundle` requires the normalization
bundle. Before writing, it recertifies each head's actual coefficients and
checks the public domains and rates against the checkpoint. It regenerates
fixed-vector and optional autoregressive references using the same joint-gate
hook. Fixed references must be finite and inside all declared operator domains.
The format names `fhemamba-m1-joint-v1` / `fhemamba-m2-chain-joint-v1` make older
kernels reject this payload instead of silently evaluating independent gates.
Legacy scalar gate fits remain inactive provenance in the metadata.

The selective-gate bundle SHA-256 is
`c15a58e757b928881d11602f162ba095ac597c9de1cc23cb94a09d669cdaf743`.
The normalization bundle SHA-256 remains
`89801adcc149d3183bf758fb8d022f4949b9b048ae5c665ff90a6892632c1c15`.
All **48 activation coefficient hashes** match the previous 4,096-token
plaintext candidate exactly. Their degrees are 768 for convolution SiLU and
384 for gate SiLU. Joint roots retain the previously frozen matrices, with
maximum degrees 1,024 / 512; no quality text enters coefficient fitting.

## Quality of the actual exported payload

The [payload-bound quality result](../../results/payload_stabilized_native_quality_20260921.json)
uses the first 1,024 and 4,096 cached WikiText-2 test tokens. These are
overlapping prefix windows, not independent samples. Each window resets state,
uses teacher forcing, and compares exact, exact-with-mask and exported-poly
circuits. All heads are retained, so the two exact controls agree.

| Input tokens | Exact PPL | Exported polynomial PPL | Relative PPL difference | Next-token top-1 agreement | Maximum logit error |
|---|---:|---:|---:|---:|---:|
| 1,024 | 13.25146 | 13.24670 | -0.03596% | 100% | 4.59465 |
| 4,096 | 17.40732 | 17.40441 | -0.01669% | 99.9023% | 5.63278 |

Both evaluations are finite, with zero observed escapes across activation,
normalization and joint-gate input domains. They run on the local RTX PRO 500
GPU in float32 with TF32 disabled; joint gates and normalization use float64
workspace. This is plaintext quality, not an encrypted 4,096-token run. The
small negative PPL differences do not establish a quality improvement, and
the remaining logit errors preclude a uniform approximation-accuracy claim.

## Reproduction

The gate bundle is the frozen output of `fit_dissipative_gates.py` documented
in [testing](../testing.md). A byte-identical copy is now included under
`config/`; see [the reproduction guide](../reproducing.md). Export into a new directory:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
  experiments/export_m1_payload.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --normalization-bundle config/mamba2-130m-normalization-20260921.json \
  --stabilized-gate-bundle config/mamba2-130m-gates-20260921.npz \
  --output runs/stabilized-payload --tokens 2 \
  --cal-tokens 128 --bound-cal-tokens 128 --device cpu
```

The built-in calibration text contains 34 tokens; the 128-token argument is
only a cap. Calibration bounds are separate from the public polynomial
domains. They do not establish carried-state coverage on long sessions.

The matching plaintext screen uses the cached token tensor from the preceding
quality study:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
  experiments/run_payload_quality.py \
  --payload runs/stabilized-payload \
  --tokens-pt runs/state-study-20260921/wikitext2.test.pt \
  --windows 1024 4096 --max-windows 1 --threads 2 --device cuda \
  --output results/payload_stabilized_native_quality_rerun.json
```

Build and transfer with the [Spark runbook](../dgx-spark.md), using a separate
build root. `dgx_spark_stabilized_smoke.json` runs one layer/two fixed tokens;
`dgx_spark_stabilized_chain.json` runs 24 layers/two fixed tokens. Both retain
the 0.05 polynomial-circuit error threshold, full synchronization, zero
evaluation debug decryptions and a 5 GiB plaintext cache. Their ring 65536 /
depth 44 / scale 59 / `security=not-set` settings are numerical experiments,
not a 128-bit full-chain security result or a separated server protocol.

## Encrypted integration measurements

The corrected [one-layer/two-token probe](../../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-smoke-r2_l1_t2.json)
passes with polynomial-circuit errors **0.0001111 / 0.0002910** and exact-model
errors **0.002256 / 0.009276**. Both outputs are finite, with zero intermediate
decryptions. Evaluation is **26.97 seconds** after **23.34 seconds** of setup,
peak RSS **32.63 GiB**, with **22** physical bootstraps, **732** rotations,
**3,278** ct-pt products and **466** ct-ct products. This is one fresh-key run.

That run's binary SHA-256 is
`e86258cf0e634a54c59fcfb87263ca460b47a263c510d0eba70611ca4d149f49`.
The input payload hash is
`b40d33b30e018ef91e894aa9e2bd26adb6be6d5e1ae49873e356d78795d5c745`,
also recorded by the plaintext quality screen. Full-chain results must be
assessed separately from this one-layer probe.

The subsequent [probe with PS block scaling](../../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-smoke-r3_l1_t2.json)
also passes: errors **0.0001124 / 0.0003140**, **26.83 seconds** evaluation,
**32.69 GiB** peak RSS, and **3,302** ct-pt products. Rotations, ct-ct products
and physical bootstraps remain **732 / 466 / 22**. The additional 24 scalar
products restore the scaled leaf blocks over both tokens. Its fresh key differs
from the preceding run; these observations are not a latency or precision
improvement claim. This version's binary SHA-256 is
`38daf92cd7cccc62d3af0072787a162f52cf33fc254bd7499e349a2774036d12`.

The first [completed full-chain run](../../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-full-chain-r2_l24_t2.json)
with this binary **fails numerical accuracy** despite completing all 24 layers
and both final normalizations. Its finite decrypted outputs have errors
**4.58e144 / 4.09e144** against the polynomial reference. Evaluation takes
**1,166.50 s**, with **36.08 GiB** peak RSS, **787** physical bootstraps,
**17,598** rotations, **108,602** ct-pt products and **12,994** ct-ct products.
Zero intermediate decryptions and successful control flow do not make this
a passing encrypted integration. This failed result is retained separately
from the one-layer probes and plaintext quality reports.

The [phase diagnostic](../../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-diagnostic-r1_l24_t1.json)
and its [verbatim log excerpt](../../results/dgx/2026-09-21/stabilized-integration/stabilized-diagnostic-r1-excerpt.txt)
locate the first corruption at `t0.L08.y_scaled`.
The first eight layer boundaries differ from their references by at most
`0.000221`. The ninth readout reaches level 40 with maximum real magnitude
`0.992739` and imaginary magnitude about `1e-6`. Normalization and Meta-BTS
preparation consume the remaining four levels. The amplified correction is
still small (`0.028403`), but its refresh at level 44 introduces imaginary
magnitude `2.893985` into the restored output. The following normalization
then diverges. The depth-44 correction is the first observed failing operation.

The joint write checkpoint had budgeted five levels for a six-level path:
head placement, multiplication by x and B, readout multiplication by C,
readout masking, and gating. The corrected estimate includes all six, plus
one more when shared head extraction is enabled. This refreshes a level-34
write before that path rather than sending its readout into the exhausted
Meta-BTS correction. A host regression replays these levels.

The [nine-layer/one-token confirmation](../../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-tail-r1_l9_t1.json)
passes at polynomial-circuit error **0.0002689** and exact-model error
**0.0128842**, using the unchanged coefficients and **0.05** threshold with
zero intermediate decryptions. At layer 8, the joint write now refreshes from
level **34 to 21**, and the readout reaches its refresh at level **38**, instead
of 40. Evaluation takes **226.68 s** with **34.89 GiB** peak RSS and **138**
physical bootstraps. This truncated chain has no final model normalization
or carried second token. The corrected binary SHA-256 is
`57636814d5792c3bc87e1c6721a8aaaaaf3aa79946e54ea9e4c956fcc95ce085`.

The corrected [24-layer/two-token result](../../results/dgx/2026-09-21/stabilized-integration/m2_chain_stabilized-full-chain-r3_l24_t2.json)
and [campaign](../../results/dgx/2026-09-21/stabilized-integration/stabilized-full-chain-r3-campaign.json)
**pass** with the same payload, binary and **0.05** error threshold:

| Measurement | Result |
|---|---:|
| Polynomial-circuit error, tokens 0 / 1 | 0.0004716 / 0.0033107 |
| Exact-model hidden-output error, tokens 0 / 1 | 0.0116666 / 0.0213601 |
| Setup / evaluation / final decryption | 24.13 / 1,192.95 / 0.073 s |
| First / carried token evaluation | 577.38 / 615.57 s |
| Peak RSS | 36.09 GiB |
| Physical bootstraps | 831 |
| Rotations / ct-pt products / ct-ct products | 17,598 / 108,690 / 12,994 |
| Highest level entering Meta-BTS preparation | 38 |

Both final normalized hidden outputs are finite. All 49 normalization sites,
all joint gates and public activation fits execute with encrypted state and
convolution FIFO carried between fixed inputs, and zero intermediate
decryptions. This single fresh-key run is an opt-in numerical integration
gate at `security=not-set`, without native lm-head token selection, long-session
generation, a separated server or a 128-bit full-chain result.

The next measured cost is joint-gate evaluation: **552.24 s**, about **46.3%**
of evaluation. Bootstrap telemetry separately totals **410.52 s**; it is also
nested within some phase timers, so phase totals must not be summed blindly.
Repeat keys/prompts and extend the carried horizon while assessing these
costs. The plaintext 4,096-token result remains a separate quality screen.

The local gate passes **232 tests**, including **12 native C++ contracts**,
with **87.03%** active-package coverage. All **17** measurement/report JSON
files validate with zero errors. Four warnings belong to the two interrupted
encoder runs whose operation counters were unavailable; successful artifacts
have no warnings. Source/build archives and failed artifacts are retained under
`~/fhemamba/gate-integration-20260921/` on Spark; local payloads and logs are
under `runs/gate-integration-20260921/`.
