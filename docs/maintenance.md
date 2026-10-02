# Maintenance boundary and debt register

Active implementations are the Mamba-2 and Mamba-3 SISO references, shared
arithmetic and native GPU backend. See the [repository map](repository.md)
for directories and generated files.

## Canonical ownership

| Concern | Location |
| --- | --- |
| Model/reference arithmetic, payloads and reusable tools | `src/fhemamba/` |
| GPU evaluators and independent CPU contracts | `native/fideslib_stage0/` |
| Specialized workflows and manifests | Grouped `experiments/` directories |
| Job lifecycle, comparison, build identity and publication | `src/fhemamba/benchmarks/` |
| Current Spark platform identity | `config/dgx-spark.env` |
| Historical CUDA 12 / SM100 B300 template | `config/b300-platform.env`; current B300 studies record their own build |

The distribution, import package and CLI are all `fhemamba`.
Version 0.5.0 retired the previous names. If an older editable installation
still advertises the removed command, remove only its generated metadata:

```bash
rm -rf fhe_native_mamba3.egg-info fhemamba/src/fhe_native_mamba3.egg-info
uv sync --locked --extra dev
```

## Guardrails

- Put new Python entry points in the installed package. Import shared modules
  directly; keep help usable without native GPU libraries.
- Reuse `fhemamba.benchmarks.io` for file identities, JSON and checkout lookup,
  and `fhemamba.benchmarks.process` for process-group cleanup.
- Native options require parser tests and artifact fields. Prefer versioned
  manifest settings to additional shell variables.
- The Mamba-2 native executable defaults to `128-classic`; the packed executor
  and its Python runner default to `not-set`. Select and record security explicitly.
- Preserve model arithmetic, payloads, security and acceptance gates during
  maintenance. Keep instrumented timing separate from ordinary inference.
- Publish claim-bearing measurements with provenance. Keep originals locally
  and identify public derivatives and prose-only observations.

## Archive boundary

The retired stack is on `archive/pre-compat-retirement-20260811`, outside the
active wheel and coverage. Port only necessary behavior behind current tests;
do not restore the old package. Historical command locations are documented
in [legacy implementation](archive/legacy-implementation.md).

Keep measured sources, numerical results and failed controls inspectable.
Delete superseded operational scripts. Host access details and session notes
belong outside public documentation.

## Research code retirement

Use these maintained replacements; removed source is identifiable at Git
revision `5d1e520`. Historical measurements remain evidence.

| Previous entry point | Replacement |
| --- | --- |
| Dual-ring `campaign.py` / `run_probe.py` | [Job recipe](../experiments/gpu_dual_ring/probe-job.json) with the shared runner |
| `analysis/run_budget.py` / `fhemamba.lowering` | Measured native telemetry, Mamba-2 generation-cost analysis or `fhemamba diagnose levels` |
| Standalone gate/state calibration | `fhemamba calibrate`: ranges, gate-screen, gate-fit, state, state-regularize |
| Normalization planning/export | `fhemamba calibrate normalization` / `normalization-export` |
| Native normalization probe/campaign scripts | `fhemamba benchmark normalization probe\|campaign` |

The retired Phase 0 estimator used exact nonlinearities and historical timing
assumptions; its estimates do not qualify the current encrypted circuit.
Existing Mamba-2 campaign resume, identity and acceptance gates remain active.

## Shared Python contracts

- The packed runner's signature owns option defaults and native forwarding.
  Keep preflight and native-result validation explicit.
- DGX generation uses one mapping for requested modes, environment switches
  and required evidence.
- Mamba-2 fixed-input and autoregressive exports share trace collection and
  layer-reference writing. Preserve native float32 bytes and legacy metadata.
- Calibration uses shared payload/provenance helpers. `QualityStudy` owns
  train-window recording, closed-loop refitting and held-out evaluation;
  candidate selection and quality gates remain in each study.
- Noise probes keep independent state copies. Upstream parity loads independent
  reference functions rather than using local model arithmetic as the oracle.

## Test scope

Test observable contracts: reference arithmetic, state ownership, frozen data,
security/precision, provenance and process lifecycle. Share setup and
parameterize distinct failures; never generate expected results with the
production validator.

Check native-result defects once through the packed runner. Budget tests
separately cover option forwarding, result preservation, deadlines and charging,
including real subprocess success, missing output and timeout cases.
Keep executable platform/CLI checks and publication/private-file boundaries.

CPU tests include native CPU contracts. GPU build correctness, dependency
patch semantics and encrypted accuracy require separate hardware qualification.
See [testing](testing.md) and [research validation](validation.md).

## Native decomposition queue

The large evaluators combine configuration, execution plans, caches,
cryptographic operators, client/server roles and telemetry. When adding
behavior, extract the touched responsibility behind a testable component.
Keep ownership restructuring separate from cryptographic schedule changes so
correctness and performance differences remain attributable.
