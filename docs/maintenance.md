# Maintenance boundary and debt register

This repository ships Mamba-2 and Mamba-3 SISO references with shared arithmetic
and an encrypted GPU backend. The pre-rebuild research
stack is preserved on `archive/pre-compat-retirement-20260811` and is not part
of `main`, the wheel, coverage, or the supported command line.

## Canonical ownership

The [repository map](repository.md) documents the directory layout, generated
files and migration from older paths.

| Concern | Canonical location | Policy |
|---|---|---|
| Mamba-2 reference and polynomial operators | `src/fhemamba/` | Active; new model math belongs here. |
| Payload and client-reference export | `src/fhemamba/m1_payload.py` | Active; the historical filename is debt, not a second implementation. |
| Encrypted GPU execution | `native/fideslib_stage0/` | Active; CPU-only units must remain buildable without FIDESlib. |
| Current experiments | Grouped `experiments/` directories | Active only when backed by a manifest or documented command; reusable code is imported from the package. |
| Artifact schema validation | `src/fhemamba/artifacts.py` | Active shared provenance contract. |
| Current Spark platform identity | `config/dgx-spark.env` | Versioned CUDA/SM/FIDESlib build with isolated dependencies and source/binary/library validation. |
| Historical B300 build template | `config/b300-platform.env` | CUDA 12 / SM100 reproduction configuration; current B300 studies document CUDA 13 / SM103 separately. |
| Root `scripts/` | Local checks and current DGX/B300 helpers only | Keep wrappers narrow; experiments belong under `experiments/`. |
| Job lifecycle and evidence publication | `src/fhemamba/benchmarks/` | Reuse shared utilities; machine settings live in ignored `config/local/`. |

The installed command is `fhemamba`. The historical `fhe-mamba3` command is
available only from the archive branch.

The distribution, import package, and installed command are all named
`fhemamba`. Version `0.5.0` records the compatibility-breaking retirement of
the old distribution metadata, import package, and command.

An editable checkout used before `0.5.0` can retain ignored build metadata that
advertises the removed command. Remove only those generated directories once,
then resync the environment:

```bash
rm -rf fhe_native_mamba3.egg-info fhemamba/src/fhe_native_mamba3.egg-info
uv sync --locked --extra dev
```

## Guardrails

- CI measures coverage for the installed `fhemamba` package. Repository and
  native contract tests supplement that package-level coverage.
- Native direct invocations default to `128-classic`. Feasibility campaigns may
  use `not-set` only by setting it explicitly and recording it in their result.
- Calibration and telemetry must not be included in an inference timing unless
  the metric is explicitly named as instrumented timing.
- New Python entry points must be importable from the installed package. Do not
  add another `sys.path.insert` workaround.
- Shared file identities, chain hashes, JSON I/O and checkout discovery live in
  `fhemamba.benchmarks.io`. Common diagnostics and profiling are installed
  modules, not sibling script imports. Keep command help lazy and usable without
  native GPU libraries. See [the tool map](package.md#research-tools).
- New native options require a parser test and an artifact field. Prefer a
  versioned manifest field over another shell environment variable.
- Headline results require a published measurement and provenance manifest.
  Retain raw originals locally, distinguish public derivatives, and label
  prose-only measurements as such.

## Archive boundary

- Do not copy the old package back into `main` to recover a helper. Port the
  smallest behavior behind an active test and current naming.
- Retired commands and Slurm files remain reproducible from the dedicated
  archive branch pinned in [legacy implementation](archive/legacy-implementation.md).
  The active-model cluster launchers are under `experiments/slurm/`.
- Claim-bearing measurements and technical reports stay publicly inspectable.
  Keep necessary measured sources and failure records; delete superseded operational
  scripts instead of archiving them. Reuse the installed benchmark utilities.
  Never add host access details, personal absolute paths or session diaries.

## Research code retirement

Keep one implementation of orchestration and provenance. Build identity fields
are declared once for writing, CLI arguments and validation; campaigns and
comparison contracts share dotted-field access. Mixer and complete-backbone
parity share an adapter that loads the original upstream CPU functions. It
does not call local model arithmetic to construct the expected output.

| Retired code | Maintained path and reason |
| --- | --- |
| Dual-ring `campaign.py` / `run_probe.py` | [Job recipe](../experiments/gpu_dual_ring/probe-job.json), packed runner and explicit comparison contracts. The old controller hard-coded a completed study's checkout and machine layout. |
| `analysis/run_budget.py`, `fhemamba.lowering`, `test_lowering.py` | The Phase 0 exact-value cost simulator had no caller beyond that retired estimator and its own tests. Use measured native telemetry, the Mamba-2 generation-cost analyzer or packed-program level analysis. The current model/reference and native execution paths are unchanged. |
| Six `calibration/*.py` commands | `fhemamba calibrate`: `ranges`, `gate-screen`, `gate-fit`, `normalization`, `state`, `state-regularize`. Shared projection loading, provenance, report writing and derivative-payload updates replace script-local copies. |
| `export/export_normalization_probe.py` | `fhemamba calibrate normalization-export`, alongside the schedule planner; coefficients and certificate gates are unchanged. |
| `normalization/run_normalization_probe.py` / `run_normalization_campaign.py` | `fhemamba benchmark normalization probe|campaign`. Both use the same typed configuration and preflight/execution function, keeping native process and fresh-key isolation per sample. |

The removed source remains identifiable at Git revision `5d1e520`. Historical
measurements, including `results/decode_budget_mamba2.json`, are retained; the
current tree has no new archive copies. Existing Mamba-2 campaign manifests
remain supported with their resume, identity and acceptance gates.

The PPL ladder now selects only the six supported nonlinear sites. Intermediate
checkpoint observations such as `conv_silu_out` remain in calibration diagnostics
but are not fitted as operators; previously they caused a `KeyError` before the
ladder completed. Train/test separation, polynomial arithmetic and candidate gates
are unchanged. The ladder and gated-norm sweep share `QualityStudy` recording,
closed-loop fitting and evaluation; state coverage and gate ablations reuse the
payload/provenance helpers as well.

## Shared Python contracts

The packed runner's function signature owns option defaults, CLI argument types
and native flag order. Keep preflight and result validation explicit; deriving
arguments must not bypass either gate. DGX generation uses one mode mapping for
request metadata, environment switches and the corresponding native evidence.

Fixed-input and autoregressive Mamba-2 exports share a stateful trace collector
and layer-reference writer in `m1_payload.py`. The writer preserves the native
float32 format, tensor names and legacy metadata representation. Noise probes
share a prefill/decode boundary, with private state copies for each perturbation.

Test model/tokenizer factories live in `tests/conftest.py`. Reference tests cover
both architectures with their original tolerances; native-evidence cases retain
separate parameterized failures for missing fields, booleans, invalid counts and
security mismatches. Expected results must remain independent of production
validation predicates. Preserve these distinctions when reducing duplication.

## Test scope

Keep tests for independently observable failures: reference arithmetic, state
ownership, frozen payloads, security and precision gates, provenance, and process
lifecycle. A test count is not a quality target. Parameterize distinct inputs;
share setup without importing the production validator as the expected answer.

Native-result defects are checked once through the packed runner. Budget tests
separately check option forwarding, unchanged success/failure results, deadlines
and charging, with real subprocess cases for success, missing output and timeout.
Do not cross every evidence defect with both budget modes. Campaign acceptance
and GPU preflight cases share setup while retaining each distinct failure.

Source-fragment assertions and one-time retired-name checks have been removed.
Keep executable platform/CLI contracts, published-evidence availability and
private-file boundaries. CPU checks compile and run native CPU contracts; they
do not establish GPU build correctness, patch semantics or encrypted accuracy.
Those still require native build and GPU qualification.

## Native decomposition queue

`stage1_mamba2_decode_fideslib.cpp` currently combines orchestration, cache
construction, cryptographic operators, protocol roles, debug decrypts, and JSON
reporting. New behavior should first move the touched responsibility behind a
testable component. The intended split is:

1. typed runtime configuration and manifest loading;
2. immutable model/execution plan;
3. plaintext cache construction;
4. encrypted block operators;
5. client/server protocol orchestration;
6. telemetry and artifact serialization.

This queue is deliberately separate from kernel optimization: changing the
cryptographic schedule and restructuring ownership in the same patch makes both
correctness and performance regressions difficult to attribute.
