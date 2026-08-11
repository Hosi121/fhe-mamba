# Maintenance boundary and debt register

This repository contains a current Mamba-2 implementation and an older research
stack. Keeping both installable without an explicit boundary made it too easy to
add features to the wrong path and too hard for an external contributor to know
which result a test protected.

## Canonical ownership

| Concern | Canonical location | Policy |
|---|---|---|
| Mamba-2 reference and polynomial operators | `fhemamba/src/fhemamba/` | Active; new model math belongs here. |
| Payload and client-reference export | `fhemamba/src/fhemamba/m1_payload.py` | Active; the historical filename is debt, not a second implementation. |
| Encrypted GPU execution | `native/fideslib_stage0/` | Active; CPU-only units must remain buildable without FIDESlib. |
| Current experiments | `fhemamba/experiments/` | Active only when backed by a manifest or documented command. |
| Pre-rebuild Python stack | `src/fhe_native_mamba3/` | Compatibility-only; no new Mamba-2 math or backend features. |
| Root `scripts/` and `slurm/` | Historical orchestration and compatibility | Do not copy a script to create a new experiment. |

The installed `fhe-mamba3` command still targets the compatibility package. It
must not be presented as the entry point for the current native Mamba-2 result.
The active `fhemamba/src`, `fhemamba/experiments`, and `fhemamba/slurm` trees do
not import `fhe_native_mamba3`; CI enforces this one-way boundary. The
compatibility package imports the shared release version from `fhemamba`.

## Guardrails

- CI measures coverage for both installed packages. A green compatibility suite
  is not a substitute for exercising `fhemamba`.
- Native direct invocations default to `128-classic`. Feasibility campaigns may
  use `not-set` only by setting it explicitly and recording it in their result.
- Calibration and telemetry must not be included in an inference timing unless
  the metric is explicitly named as instrumented timing.
- New Python entry points must be importable from the installed package. Do not
  add another `sys.path.insert` workaround.
- New native options require a parser test and an artifact field. Prefer a
  versioned manifest field over another shell environment variable.
- Headline results require a tracked raw artifact. Prose-only measurements stay
  visibly labeled as such.

## Retirement queue

1. Inventory imports of `fhe_native_mamba3` from root scripts and classify each
   as migrate, archive, or delete. The active `fhemamba` tree is already
   dependency-free; the remaining import surface belongs to root compatibility
   orchestration and the legacy `fhe-mamba3` entry point.
2. Move reusable artifact/provenance helpers into a small package shared by the
   active path; do not retain the old model implementation for those helpers.
3. Replace the compatibility CLI with a current `fhemamba` CLI, keeping only
   narrow deprecated aliases for commands with real users.
4. Archive historical run scripts and non-golden result JSON outside the source
   tree. Keep a small validator-tested golden corpus in the repository.
5. Remove the compatibility package and its tests once no supported command or
   documented workflow imports it.

Deletion is gated by import/use evidence, not by raw test count. Compatibility
code that is exercised only by compatibility tests is still a removal candidate.

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
