# Testing strategy

The repository has three verification tiers: local Python/native-contract
tests, local coverage, and hardware-backed encrypted integration gates.

## Fast local gate

```bash
scripts/run_fast_checks.sh
```

This runs:

- `ruff format --check`;
- `ruff check`;
- the complete configured pytest suite without coverage;
- slow-test duration reporting.

`pyproject.toml` includes both `tests/` and `fhemamba/tests/`. The root suite
covers repository/native integration contracts; `fhemamba/tests/` covers the
active Mamba-2 reference, lowering, CLI, and artifact validation.

For focused iteration:

```bash
scripts/run_fast_checks.sh fhemamba/tests/test_state_layout.py
scripts/run_fast_checks.sh tests/test_native_fideslib_stage0.py
```

When using `uv`, install with `uv sync --extra dev`. The development tools are
an optional project extra, not a dependency group.

## Full local gate

```bash
scripts/run_checks.sh
```

This runs formatting, lint, pytest, and the `fhemamba` coverage gate. Retired
compatibility code is not included in package discovery or coverage.

Parallel execution is available when `pytest-xdist` is installed:

```bash
CHECK_JOBS=auto scripts/run_checks.sh
```

To exercise the installed pre-commit hook explicitly:

```bash
RUN_PRECOMMIT=1 scripts/run_checks.sh
```

## Native C++ contract tests

The FIDESlib-free C++ tests cover payload parsing, layout, planning, depth,
process-role restrictions, and artifact emission:

```bash
cmake -S native/fideslib_stage0 -B build/stage0-layout-tests \
  -DFHE_STAGE0_BUILD_KERNEL=OFF \
  -DFHE_STAGE0_BUILD_TESTS=ON
cmake --build build/stage0-layout-tests
ctest --test-dir build/stage0-layout-tests --output-on-failure
```

Pytest invokes this path through `tests/test_native_layout_cpp.py`.
The depth contract also checks coefficient-aware Chebyshev splits through
degree 768, both sparse and asymmetric coefficients, unchanged polynomial
values, and baseline ceilings for depth and scalar products. The encrypted
A/B campaign must check actual counts and output error independently.

These tests do not execute CKKS on a GPU. They are intended to catch contract
breakage before a GPU experiment, including DGX Spark.

## Algebra research gates

`fhemamba/tests/test_ssm_algebra.py` compares deferred state with an independent
dense recurrence, including zero decays and nonzero initial states; compares
complex trapezoidal recurrence with real two-by-two rotations; and checks
long-horizon radial drift and the Newton residual identity.
`test_phase_algebra.py` additionally checks a three-shear phase, its modified
energy invariant, and a switching counterexample to a local-stability claim.

`probe_ssm_algebra.py` uses factors captured from the actual checkpoint at
multiple buffer lengths. `probe_decay_composition.py` separately reports
unconstrained polynomial range violations and checks the maximum exact decay
of pruned heads over their exported input intervals. See the
[research note](research/2026-09-21-ssm-cryptographic-design.md) for commands and
limits. Algebra parity and sampled approximation error are not encrypted
precision, language-model quality or performance evidence.

`test_polynomial_certificate.py` checks exact rational change of basis,
Bernstein subdivision, an unsafe Newton seed, and positive-series seed
certificates. It also rejects a write gate that accumulates at unit decay,
and accepts reset/perfect-memory cases satisfying the coupled state invariant.
`certify_payload_ranges.py` proves the normalization basin on
the declared intervals; it does not establish private-input domain membership
or CKKS rounding bounds.
The Bernstein gate certificate checks the *shared-basis* coefficient
inequalities directly, including rejection of nonzero writing at unit decay.
`test_state_coordinates.py` checks conditioning limits, observed-bound
coverage, and update/readout parity after regularization.

```bash
.venv/bin/python fhemamba/experiments/probe_bounded_selective_gates.py \
  --payload fhemamba/results/m2_chain_payload_rows \
  --degrees 32 64 128 --output runs/bounded-selective-gates.json
.venv/bin/python fhemamba/experiments/regularize_payload_state.py \
  --payload fhemamba/results/m2_chain_payload_rows --group-scale-floor 0.125 \
  --output-chain runs/payload-row-floor --output runs/row-floor.json
```

The joint Bernstein gate probe certifies conditional stability but its first
construction has excessive model-approximation error. Do not export it as a
replacement based on the invariant alone. State-scale regularization leaves
polynomials and reference binaries unchanged and never reads evaluation data.

`test_selective_gates.py` checks the shared-factor construction, including
perfect memory, reset, public-function approximation and adversarial state
updates. Its integer certificate is compared with the independent Fraction
verifier, including failed depth/node budgets and exact binary-epsilon
counterexamples. The recurrence recorder also has a regression check for
preserving joint-gate dispatch without duplicating checkpoints.

To fit and certify the new gates without reading text:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python fhemamba/experiments/fit_dissipative_gates.py \
  --payload fhemamba/results/m2_chain_payload_headclip \
  --bundle runs/dissipative-gates.npz --output runs/dissipative-gates.json
```

The fitter refuses to overwrite existing outputs and emits a bundle only
when every head certifies. The NPZ contains the actual coefficients, domains,
rates, public fitting policy and provenance; the JSON records per-head
certificates and independently sampled errors. Keep both outputs together.
This bundle is a plaintext research oracle, not an exported native payload.

For payload-bound quality, prepare int64 `[1, tokens]` files using the same
checkpoint tokenizer and record the data split. The current dataset is
[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
configuration `wikitext-2-raw-v1`, joined with two newlines between rows.

```bash
.venv/bin/python fhemamba/experiments/run_payload_quality.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --payload fhemamba/results/m2_chain_payload_headclip \
  --tokens-pt /path/to/wikitext2.test.pt \
  --data-description 'WikiText-2 raw test; two-newline join' \
  --windows 1024 4096 --max-windows 2 --output runs/payload-quality.json
.venv/bin/python fhemamba/experiments/diagnose_payload_domain.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --payload fhemamba/results/m2_chain_payload_headclip \
  --tokens-pt /path/to/wikitext2.test.pt --tokens 1024 \
  --output runs/payload-domain.json
.venv/bin/python fhemamba/experiments/diagnose_payload_domain.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --payload fhemamba/results/m2_chain_payload_headclip \
  --tokens-pt /path/to/wikitext2.test.pt --tokens 4096 --offset 65536 \
  --public-gate-degree 384 --public-conv-degree 768 \
  --positive-newton-iterations 8 --quality-reference --threads 2 \
  --output runs/public-activations-long.json
```

The last command intentionally reproduces the newly discovered time-step/
decay extrapolation failure. A diagnostic JSON with null candidate PPL is
negative evidence even if the diagnostic process itself exits successfully.

For the fixed shared-factor candidate, use the same BLAS settings for both
the control and candidate; coefficient hashes must match for other overrides:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python fhemamba/experiments/diagnose_payload_domain.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --payload fhemamba/results/m2_chain_payload_headclip \
  --tokens-pt /path/to/wikitext2.test.pt --tokens 4096 --offset 65536 \
  --device cuda --public-gate-degree 384 --public-conv-degree 768 \
  --positive-newton-iterations 8 --quality-reference --threads 2 \
  --dissipative-gate-bundle runs/dissipative-gates.npz \
  --output runs/dissipative-long.json
```

Omit `--dissipative-gate-bundle` for the matched old-composition control.
Offset 98,304 was the second, previously unused window; both offsets are now
evaluated development evidence. Pick new data before further tuning. CUDA
evaluation disables TF32 and records its device; it still executes plaintext
polynomials. `--device cpu` also works but requires its own matched baseline.
`probe_dissipative_ablation.py` evaluates only the gate approximation while
keeping other nonlinearities exact; its result cannot be counted as a fully
polynomial model or encrypted inference.

The normalization follow-up replaces all 49 inverse square roots with public
scaled Goldschmidt schedules. Generate the frozen recipe and rational
certificates first, then evaluate it without changing other coefficients:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python fhemamba/experiments/plan_normalization_schedules.py \
  --payload fhemamba/results/m2_chain_payload_headclip \
  --bundle runs/normalization-schedules.json \
  --output runs/normalization-certificates.json
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python fhemamba/experiments/diagnose_payload_domain.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --payload fhemamba/results/m2_chain_payload_headclip \
  --tokens-pt /path/to/wikitext2.test.pt --tokens 4096 --offset 131072 \
  --device cuda --public-gate-degree 384 --public-conv-degree 768 \
  --norm-schedule-bundle runs/normalization-schedules.json \
  --dissipative-gate-bundle runs/dissipative-gates.npz \
  --quality-reference --local-errors --threads 2 \
  --output runs/scheduled-norm-holdout.json
```

Do not combine `--norm-schedule-bundle` with `--positive-newton-iterations`.
The loader rechecks every certificate from its coefficients and requires the
complete site set and matching payload SHA. `test_normalization.py` covers
the epsilon-to-upper-bound domain, interior extrema, tampered recipes,
roundoff consistency repair and a symbolic trace of actual arithmetic counts.
Its cost ledger excludes scalar rescaling, bootstrap and layout costs.

`--exact-sites 0:gated_rms_invsqrt` with the previous positive-seed policy
reproduces the single-site diagnostic. This changes the trajectory and marks
the model as not fully polynomial. In contrast, `--local-errors` evaluates
exact operators only as passive diagnostics at candidate inputs. It does not
change the candidate trajectory. Both remain plaintext tests; float64 workspace
and final float32 casts are not CKKS error bounds. Offset 131,072 is now used
evaluation data and must not be reused as a fresh holdout after further tuning.

For independent multi-window calibration, `calibrate_state_coordinates.py`
accepts `--calibration-tokens-pt` and disjoint `--offsets`; non-finite states
produce a failure report, nonzero exit status, and no output payload.

The three quality circuits use identical teacher-forced tokens and state
resets: exact, exact with the exported head mask, and the unchanged polynomial
payload. Reports bind payload, token file, checkpoint and implementation hashes.
Non-finite outputs must remain failures, not clipped values or a finite PPL.
Two windows are a diagnostic screen, not a full dataset quality certificate.

## Isolated encrypted normalization

The scalar probe evaluates a full polynomial normalization core on geometric
synthetic inputs. It uses the existing Spark dependencies in a separate build
directory, with binary/library hashes and an archived source snapshot.
Stop any probe using that build directory before rebuilding it.

First export the certified recipes locally:

```bash
.venv/bin/python fhemamba/experiments/export_normalization_probe.py \
  --bundle runs/norm-study-20260921/normalization-schedules.json \
  --output-dir runs/normalization-probe-recipes
rsync -az runs/normalization-probe-recipes/ \
  dgx:fhemamba/cipher/runs/normalization-probe-recipes/
```

On Spark, after provisioning the dependencies described in the runbook:

```bash
python3 fhemamba/experiments/build_normalization_probe.py \
  --build-dir "$HOME/fhemamba/spark/normalization-probe" \
  --fideslib-prefix "$HOME/fhemamba/spark/install-2a70798e869944af" \
  --openfhe-prefix "$HOME/fhe-deps/openfhe-fides"
python3 fhemamba/experiments/run_normalization_campaign.py \
  --binary "$HOME/fhemamba/spark/normalization-probe/stage1_normalization_probe" \
  --recipes runs/normalization-probe-recipes \
  --output-dir runs/normalization-core-all-sites
```

The default campaign uses weighted residuals, depth 40, scale 59 and
`128-classic`; OpenFHE chooses the ring dimension. It refuses to overwrite a
campaign. For a widest-domain ABBA comparison, add
`--sites l23_rms_invsqrt.txt --modes balanced weighted weighted balanced` and
use a new output directory. Each process generates a new key.

`run_normalization_probe.py --input-mode normalize` encrypts x and computes
`x*poly_invsqrt(x²+epsilon)`. Its gate is absolute normalized-output error.
`--input-mode variance` instead encrypts the variance directly and gates
relative inverse-square-root error. These are different tests; retain their
metric names and failures. Both use normal real CKKS decoding, including its
noise. Neither tests packed feature reduction, learned gamma, prior-layer
errors, refresh, full-model language quality or separate server processes.

The pinned HYBRID-3 probe caps depth at 44 because combined Q/P limb storage
must fit the backend's fixed arrays. Runtime failures must produce a failed
wrapper artifact even when the native backend exits with status zero. Local
tests cover this failure mode, tampered recipe hashes and the depth preflight.

The packed vector follow-up exports actual checkpoint inputs and learned gamma
locally, without changing the frozen inverse-square-root recipes:

```bash
.venv/bin/python fhemamba/experiments/export_vector_rms_probe.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --token-file runs/state-study-20260921/wikitext2.test.pt \
  --recipes runs/norm-native-20260921/recipes-weighted \
  --output-dir runs/vector-rms-fixtures
```

Transfer the fixtures to Spark along with the recipes. Build with
`build_normalization_probe.py --target stage1_vector_rms_probe`, using a new
build directory and the same dependency arguments above. Then run
`run_normalization_campaign.py` with that binary and
`--fixtures runs/vector-rms-fixtures`. The default tests fresh ciphertexts
without refresh; `--gamma-placement after` is the matched depth baseline.
The two widths use separate lanes with zero-padded features. The host C++
contract checks exact sums and lane isolation. Native output comparisons use
the true feature width as the mean divisor.

For a single vector probe, pass `--input-mode vector --fixture <file>` to
`run_normalization_probe.py`. `--refresh output` measures ordinary output
refresh, and `--refresh output-meta --meta-alpha 12` explicitly evaluates the
two-bootstrap residual correction. The latter requires a live level for
amplification; a failed refresh remains a failed gate. Existing scalar modes
retain their original arguments. See the
[vector study](research/2026-09-21-vector-rms.md) for scope and level-budget
limits: even a successful output refresh does not establish that another
complete normalization fits after it.

`--refresh-coordinates channel` applies public per-component scaling instead
of one global bound; it requires a vector refresh. Campaigns accept this flag
and `--refresh output-meta --meta-alpha 12` as well. Every campaign gets a new
output directory, and numerical failures stay in its report.

## Scheduled RMSNorm in the full kernel

The exporter opt-in uses the frozen 49-site bundle and regenerates references
for the same scheduled circuit, including a dedicated final RMSNorm:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  fhemamba/experiments/export_m1_payload.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --normalization-bundle config/mamba2-130m-normalization-20260921.json \
  --output runs/scheduled-norm-payload --tokens 2 --cal-tokens 128 --device cpu
```

The output directory must be new. Invalid site coverage, uncertified recipes,
epsilon mismatch, non-finite fixed-vector references and escaped norm domains
are rejected. `test_scheduled_payload.py` covers the export/load contract,
including autoregressive references. Native host contracts check parsing,
the independent final recipe, the live-variance input depth budget and the
scheduled transient Meta-BTS policy.

Build the updated kernel and transfer this payload using the
[Spark runbook](dgx-spark.md). From the Spark checkout, after placing the
payload at the path below:

```bash
python3 fhemamba/experiments/run_dgx_campaign.py \
  --manifest fhemamba/experiments/dgx_spark_scheduled_norm_smoke.json \
  --runner scripts/run_dgx_spark.sh \
  --env INPUT_CHAIN="$HOME/fhemamba/payloads/scheduled-norm" \
  --output-json "$HOME/fhemamba/results/scheduled-norm-smoke.json"

python3 fhemamba/experiments/run_dgx_campaign.py \
  --manifest fhemamba/experiments/dgx_spark_scheduled_norm_chain.json \
  --runner scripts/run_dgx_spark.sh \
  --env INPUT_CHAIN="$HOME/fhemamba/payloads/scheduled-norm" \
  --output-json "$HOME/fhemamba/results/scheduled-norm-chain.json"
```

Both manifests require finite decryption of two tokens, zero intermediate
decryptions and error ≤ 0.05; the second also requires all 24 layers and final
RMSNorm. These numerical probes use `security=not-set` and do not sample
generated tokens. For an isolated build root, also override
`FHEMAMBA_REMOTE_ROOT` and `RESULTS_DIR` as described in the runbook.

The [integration study](research/2026-09-21-normalization-integration.md)
records a passing full-chain run and all failed controls. Run the matching
plaintext quality screen separately:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  fhemamba/experiments/run_payload_quality.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --payload runs/scheduled-norm-payload \
  --tokens-pt runs/state-study-20260921/wikitext2.test.pt \
  --data-description 'WikiText-2 test, first 1024 tokens' \
  --windows 1024 --max-windows 1 --threads 4 --device cpu \
  --output runs/scheduled-norm-quality.json
```

This exact short-calibration payload currently fails the quality screen with
non-finite outputs. It retains the old non-normalization fits and is distinct
from the joint-gate/wider-activation plaintext candidate. The frozen norm
certificates do not imply closure of the whole model's activation domains.

The complete candidate can now be exported with
`--stabilized-gate-bundle runs/joint-gates-20260921/dissipative-gates.npz`
alongside `--normalization-bundle`. This replaces both scalar gate operations
with the joint hook, retains all heads, and fits the public-envelope
convolution/gate SiLUs at degrees 768/384. All 48 activation coefficient hashes
match the prior quality candidate. See the
[stabilized native study](research/2026-09-21-stabilized-native.md) for commands
and measured scope. Use `dgx_spark_stabilized_smoke.json` and
`dgx_spark_stabilized_chain.json` after rebuilding in an isolated prefix.
The corrected full-chain gate passes at maximum error **0.003311** over
24 layers/two fixed tokens, with zero intermediate decryptions. Its host
regression budgets the six products from joint write through gated readout;
the previous five-level estimate corrupted Meta-BTS at the last modulus.
The study retains the failing run, phase diagnostic and nine-layer confirmation.

`run_payload_quality.py --offset <tokens>` selects a recorded token offset
without changing coefficients. CUDA quality runs disable TF32 and record both
gate and normalization bundle identities. The first 1,024/4,096-token screens
of the new exported payload are finite with no observed domain escapes; these
are overlapping windows. The C++ shared-basis evaluator is checked against
independent Clenshaw values through degree 1,024, with distinct head
coefficients and lane/padding checks. These contracts are part of the normal
local suite; encrypted precision is measured separately.

## Artifact validation

Curated benchmark JSON should be checked with:

```bash
fhemamba validate-artifacts \
  --require-commit \
  path/to/result.json
```

Direct encrypted backend artifacts must include, where applicable:

- package/artifact version and repository commit;
- backend, hardware/configuration, and input mode;
- status and numerical gate;
- operation and rotation counts;
- bootstrap count and CKKS level telemetry;
- setup/evaluation/decrypt timing and peak RSS;
- a human-readable claim and explicit non-claims.

Do not manufacture a summary artifact from prose and present it as a raw
backend result. Summary/collection artifacts must identify themselves as such.

## GPU integration gates

FIDESlib GPU execution is not part of ordinary CI. Relevant gates are:

1. bootstrap and complex-pair micro-probes;
2. one-layer/full-width encrypted smokes;
3. full 24-layer multi-token execution;
4. process-separated execution;
5. 128-bit full-chain execution.

`stage1_plaintext_add_probe` checks vector-constant addition to degree-1 and
degree-2 ciphertexts at levels 0/1/2, including the consumption-level encoding
case. The Spark build compiles it alongside the model kernel. Its explicit
`x² + b` oracle detects a scale mismatch that a small-cache/full-depth model
run exposed; plaintext layout tests cannot detect this backend behavior.
See the tracked [probe and model evidence](evidence.md).

A micro-probe cannot promote a synchronization or bootstrap change. The B300
reduced-barrier build demonstrated why: its bootstrap probe passed while the
full chain silently produced corrupt finite values.

The current Spark build resolves `config/dgx-spark.env`, uses an isolated
native CUDA 13 / SM121 prefix, and validates source, executable and shared-
library hashes before execution. See [dgx-spark.md](dgx-spark.md). Its ABBA
replication campaign compares both schedules on the same payload and binary;
the smaller key inventory is an intended part of the candidate's memory cost.

Historical B300 entry points resolve `config/b300-platform.env`. Profile builds clone
the pinned FIDESlib commit into patch-set-addressed source snapshots; they do
not apply patches to the shared checkout. Before an expensive run, the runner
validates `<binary>.build.json` against the local container image ID, configured
CUDA/SM/profile, FIDESlib commit, patch-set hash, compiler record, and current
binary SHA-256. The validated metadata is attached to the native result as
`build_provenance`.

Replicated-BSGS cache A/B runs must keep the payload, rotation keys, circuit
configuration, and synchronization profile fixed. The `pt_cache` telemetry
separates replicated cache hits, misses, and level bypasses from host mask
build counts, bytes, and seconds; fully cached warm projection evaluation
should report zero `replicated_eval_mask_builds`. Promotion still requires
unchanged polynomial-circuit error/generated IDs plus measured improvement in
cold setup, first/warm token, total evaluation, CPU utilization, and peak RSS.

The current five-step Spark candidate campaign is:

```bash
python fhemamba/experiments/run_dgx_campaign.py \
  --manifest fhemamba/experiments/dgx_spark_autoregressive.json \
  --runner scripts/run_dgx_spark.sh \
  --env BINARY="$HOME/fhemamba/spark/kernel/stage1_mamba2_decode_fideslib" \
  --env INPUT_CHAIN="$HOME/fhemamba/payloads/mamba2-130m" \
  --output-json "$HOME/fhemamba/results/spark-five-step.json" \
  --resume
```

The candidate manifest has a fail-closed `acceptance` gate. The campaign exits
nonzero if the complete artifact set misses its layer/token geometry, error
tolerance, per-token decrypt, autoregressive-token, zero-intermediate-decrypt,
or full-synchronization requirements. Missing and malformed artifacts remain
infrastructure failures; a valid artifact that misses a promotion criterion is
a candidate failure.

`--resume` verifies artifact schema and provenance before reuse, including the
artifact version, full repository commit (plus a dirty-tree content
fingerprint), binary SHA-256, layer/token geometry, synchronization profile,
and the effective environment recorded in the prior campaign report. A missing
prior report or any mismatch causes the stale artifact to be rerun.
Autoregressive client decryption of completed `final_norm` output is recorded
as a protocol-boundary token-selection operation, not an intermediate
diagnostic decrypt.

## Prompt-to-text generation

The stabilized prompt-to-text runner is
`fhemamba/experiments/run_dgx_generation.py`; see the
[complete command](dgx-spark.md#complete-prompt-to-text-generation).
Its local tests check whole-prompt consumption, unchanged source payloads,
checkpoint identity and finite-but-out-of-domain final normalization. Report
tests reject stale payloads and incomplete horizons, and ensure failed token
selection is displayed as measured instead of replaced by reference text.
These tests do not replace the native five-evaluation numerical gate.

The [2026-09-22 native run](research/2026-09-22-client-generation.md) passes all
five evaluations and four greedy token choices, with maximum polynomial-
circuit error 0.009192. The complete local suite has 246 passing tests; the
native result, campaign and text report validate without errors or warnings.

## Known gaps

- The documented `0.4.5` three-token B300 success JSON is not currently
  tracked; recovery or exact rerun is PBI-M4-001, deferred while B300 is unavailable.
- GPU CKKS execution cannot be reproduced by GitHub-hosted CI.
- A new backend build must pass the micro-probes and full 24-layer
  multi-token gate before its values can replace the promoted configuration.
- A full process-separated Mamba run and a 24-layer 128-bit run remain open.
