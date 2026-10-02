# Python package

`fhemamba` is installed from [`src/fhemamba/`](../src/fhemamba/). Its import name,
distribution name and command name are all `fhemamba`. Run the
[CPU example](../examples/cpu_smoke.py) after following the
[installation guide](reproducing.md#1-install-the-python-environment).

For generation, use `fhemamba.load_model`, `model.generate`, `model.prepare`
and `fhemamba.load_prepared`. The matching commands are `fhemamba generate`
and `fhemamba prepare`. Generation detects Mamba-1, Mamba-2 or Mamba-3 SISO;
preparation is available for Mamba-2/3.
See the [generation guide](generation.md) for inputs,
backend selection and the request-specific preparation contract.
`fhemamba.inspect_model` / `fhemamba inspect-model` describe capabilities without
loading weights. To add a model, follow the [integration guide](model-integration.md).

## Modules

Paths below are relative to `src/fhemamba/`.

| Area | Modules | Responsibility |
| --- | --- | --- |
| Public generation | `inference.py`, `inference_cli.py`, `inputs.py`, `checkpoints.py` | Token requests, model discovery, identity checks and common results |
| Model integrations | `models/contracts.py`, `models/registry.py`, `models/mamba1.py`, `models/mamba2.py`, `models/mamba3.py` | Model-owned loading, arithmetic adapters, typed preparation options and FHE profiles |
| Model and decode | `reference.py`, `ops.py`, `tensor_ops.py`, `packed_program.py` | Model formulas, exact/polynomial operations and packed circuit export |
| Architecture and Mamba-3 | `architectures.py`, `mamba3.py`, `mamba3_lm.py`, `tensor_ops.py`, `packed_program.py` | Architecture-specific state, trained SISO model, shared tensor operations and packed FHE programs |
| Payload and generation | `m1_payload.py`, `generation.py`, `generate.py`, `ppl.py`, `payload_surrogate.py` | Checkpoint export, prompt preparation, text reports and plaintext quality |
| Polynomial contracts | `normalization.py`, `selective_gates.py`, `polynomial_certificate.py`, `gated_norm_sweep.py` | Public schedules, joint gates and conditional interval certificates |
| Packing and keys | `bsgs_layout.py`, `readout_layout.py`, `rotation_keys.py`, `state_layout.py`, `state_coordinates.py` | Slot layouts, key plans, state scaling and refresh schedules |
| Research models | `ssm_algebra.py`, `phase_algebra.py`, `prefill_budget.py`, `noise_flow.py` | Algebra oracles, hypothetical budgets and error sensitivity |
| Provenance and CLI | `artifacts.py`, `bootstrap_telemetry.py`, `cli.py`, `_version.py` | Artifact validation, telemetry, commands and package version |

`reference.py` dispatches Mamba-1/2/3 formulas and allocates their own state
types through `architectures.py`. Exact, recording and polynomial operations
are injected through `Ops`; Mamba-3's full tensor algebra is also lowered to
the shared packed program. Measure checkpoint-to-polynomial
quality separately from CKKS-to-polynomial execution error. The historical
filename `m1_payload.py` also serves the active Mamba-2 export path.

## Running experiments

The installed package provides model/layout utilities and `fhemamba --help`.
Native builds, specialized study scripts and recorded results are used from a
repository checkout; they are not bundled into the Python wheel. The reusable
commands described below are included in the wheel.

- [Reproduction guide](reproducing.md): checkpoint, coefficients, parity and generation.
- [Experiment index](../experiments/README.md): choose a runner or campaign.
- [Design](design.md): mathematical and protocol invariants.
- [Testing](testing.md): local checks and research acceptance gates.

For development, install with `uv sync --locked --extra dev`. Python code is
loaded through the installed package; the root directory needs no import shim.

## Research tools

Use `python -m fhemamba --help` (or the installed `fhemamba` command).
Subcommand help loads only the selected implementation. The job, build,
publication, packed-runner and profiling tools use the standard library;
model export and reference propagation additionally use NumPy/PyTorch.

| Command group | Importable implementation | Responsibility |
| --- | --- | --- |
| `benchmark` | `fhemamba.benchmarks` | Jobs, completion events, comparisons, packed qualification, generation reports, build identity and evidence publication |
| `calibrate` | `fhemamba.calibration` | Frozen gate/state calibration, public-domain certificates, normalization schedules and native recipe export |
| `diagnose` | `fhemamba.diagnostics` | Frozen references, prefix verification, observed-input analysis, replay cases, CPU error propagation and circuit summaries |
| `profile` | `fhemamba.profiling` | Nsight Systems SQLite and Nsight Compute CSV analysis |
| `recurrent` | `fhemamba.recurrent` | Capture/export/audit component fixtures and compare storage/scheduling results |
| `workload` | `fhemamba.workloads` | Mamba-3 model/mixer payloads and upstream CPU parity |

Call functions directly when composing work, for example
`fhemamba.profiling.nsys.summarize(connection)` or
`fhemamba.benchmarks.packed.run(binary, payload, output, security="128-classic")`.
The packed runner has one option-forwarding path for ordinary and budgeted jobs.
Generic jobs, packed runs and campaigns share subprocess execution and process-group
cleanup in `fhemamba.benchmarks.process`; their acceptance criteria remain local
to each workload. Mamba-2 generation and campaigns reuse the same gates in
`fhemamba.benchmarks.acceptance`.

Calibration operations return report dictionaries and can be composed directly,
for example `fhemamba.calibration.gates.certify_ranges(payload)` and
`fhemamba.calibration.schedules.plan_normalization(payload, bundle=destination)`.
The CLI writes strict JSON reports; derivative payloads preserve their legacy
metadata representation, including an unbounded `time_step_limit`. Weight and
reference files are copied unchanged. New reports identify both the operation
module and shared payload helpers in `source_sha256`; historical reports are intact.

`fhemamba.calibration.quality.QualityStudy` shares train-window range recording,
one closed-loop refit and held-out PPL evaluation between substitution ladders
and gated-normalization sweeps. Candidate choice and quality gates stay in each
study. `fhemamba benchmark normalization probe|campaign` shares one native
preflight/execution API, `run(Probe(...))`. Each sample still launches a fresh
native process and keys, with timeout cleanup and preserved failure artifacts.

Use `fhemamba.benchmarks.io.file_sha256`, `payload_sha256`, `read_object`, `field` and
`write_json` instead of copying file helpers. `repository_root` resolves an
explicit source checkout from the working directory, not from `site-packages`.
`field` rejects absent nested JSON fields unless a default is explicitly supplied.
Repository-specific build/export commands should run from a checkout; the Spark
build command also accepts `--repo`.

`fhemamba.ckks_probes` shares the Chebyshev circuit and local CPU CKKS setup.
`fhemamba diagnose ckks --recipe ... --output ...` runs
[CPU primitive and error-growth recipes](../experiments/local_ckks/README.md)
through common curve evaluators and recurrence loops. Degree, interval, input
and arm changes belong in the recipe instead of another Python script.
OpenFHE is imported only when constructing that optional feasibility context;
its small-ring `not-set` configuration is not a classical-128 claim.
Specialized experiment configurations remain in the grouped
[experiment directories](../experiments/README.md).
