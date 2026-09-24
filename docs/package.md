# Python package

`fhemamba` is installed from [`src/fhemamba/`](../src/fhemamba/). Its import name,
distribution name and command name are all `fhemamba`. Run the
[CPU example](../examples/cpu_smoke.py) after following the
[installation guide](reproducing.md#1-install-the-python-environment).

## Modules

Paths below are relative to `src/fhemamba/`.

| Area | Modules | Responsibility |
| --- | --- | --- |
| Model and decode | `reference.py`, `ops.py`, `lowering.py` | Shared model formulas, exact/polynomial operations and decode schedules |
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
Native builds, experiment runners and recorded results are used from a
repository checkout; they are not bundled into the Python wheel.

- [Reproduction guide](reproducing.md): checkpoint, coefficients, parity and generation.
- [Experiment index](../experiments/README.md): choose a runner or campaign.
- [Design](design.md): mathematical and protocol invariants.
- [Testing](testing.md): local checks and research acceptance gates.

For development, install with `uv sync --locked --extra dev`. Python code is
loaded through the installed package; the root directory needs no import shim.
