# Backlog

This is the canonical backlog for the active `0.4.x` Mamba-2 line. Historical
Stage 0/1/2 PBIs remain available in Git history and `docs/artifact_ledger.md`,
but they are not the current work queue.

Status meanings:

- **Done**: implementation and accepted evidence are tracked.
- **In Progress**: the repository contains part of the acceptance path.
- **Open**: executable and not blocked by missing upstream evidence.
- **Blocked**: an explicit dependency must close first.

## Current PBIs

| ID | Priority | Status | Depends on | Acceptance |
|---|---:|---|---|---|
| PBI-M4-001 | P0 | In Progress | none | Recover the `0.4.5` 24-layer/three-token B300 raw success JSON and validate the documented values, or rerun the exact promoted baseline and replace the headline values with that measured artifact; require `repo_commit`, `binary_sha256`, full-sync profile, configuration, per-token errors, timing, bootstraps, and peak RSS; track the curated artifact and update `docs/evidence.md`. |
| PBI-M4-002 | P0 | Blocked | PBI-M4-001 | Run the promoted 24-layer five-step B300 campaign. All steps decrypt, each polynomial-circuit error is `<= 0.05`, generated IDs match, no intermediate decrypt is used, and a validator-clean campaign plus raw result JSON is tracked. |
| PBI-M4-003 | P0 | Blocked | PBI-M4-002 | Execute the full Mamba kernel as separate `client-init`, secret-key-free `server-eval`, and `client-decrypt` processes. Record serialized ciphertext sizes, server RSS/runtime, round-trip error, and a server-directory secret-key audit. |
| PBI-M4-004 | P0 | Blocked | PBI-M4-002 | Run all 24 layers at OpenFHE-accepted 128-bit parameters. Record error, runtime, key memory, peak RSS, bootstrap count, and non-claims about return-path noise flooding. |
| PBI-OPT-001 | P1 | Blocked | PBI-M4-002 | Replace B300 key-switch device barriers with explicit stream dependencies. Bootstrap micro-probes and the complete 24-layer five-step gate must both pass before promotion. |
| PBI-OPT-002 | P1 | Blocked | PBI-M4-002 | Compare `out-proj`-only and all-scope fused transforms on the same five-step payload and binary family. Promote only on passing accuracy plus lower end-to-end evaluation, not projection time alone. |
| PBI-OPT-003 | P1 | Blocked | PBI-M4-001 | Build an offline bootstrap-placement planner over recorded CKKS level/event traces. It must preserve live-out requirements and emit a replayable candidate schedule before native execution. |
| PBI-OPT-004 | P2 | Blocked | PBI-M4-002 | Re-test shared dt/decay head expansion in a session long enough to amortize setup and key generation. Track total runtime and RSS, not only warm phase timing. |
| PBI-OPS-101 | P1 | Open | none | Enforce coverage for the active `fhemamba` package in addition to the compatibility package, with a documented threshold that passes the full suite. |
| PBI-OPS-102 | P1 | Blocked | PBI-M4-001 | Create the `v0.4.5` Git tag only after the raw three-token success artifact is tracked or regenerated and full checks pass. |
| PBI-OPS-103 | P2 | Open | none | Define the retirement boundary for `src/fhe_native_mamba3`: list commands still depending on it, move active utilities where appropriate, and remove it from package discovery only in a planned compatibility-breaking release. |

## Completed current capabilities

| Capability | State | Evidence |
|---|---|---|
| Polynomial Mamba-2 quality | Done | WikiText-2 PPL `22.307 -> 22.333`, tracked result JSON |
| Reference/lowering parity | Done | `fhemamba/results/decode_budget_mamba2.json` |
| Full-width native encrypted layer | Done | FIDESlib native kernel and historical Stage 1 artifacts |
| 24-layer ciphertext residual handoff | Implemented; evidence closure open | Documented B300/DGX measurements; raw current success artifact still PBI-M4-001 |
| Three-token ciphertext state/FIFO carry | Implemented; evidence closure open | Documented errors `0.01295 / 0.01173 / 0.03475` |
| Complex-paired state refresh | Done | Native implementation, unit contracts, and B300 measurement note |
| Three-process key separation probe | Done | `fhemamba/results/dgx/client_server_probe.json` |
| Layer-0 128-bit two-token probe | Done | `fhemamba/results/dgx/m1_decode_128bit_r131072_d43_s59_t2.json` |

## Execution order

```text
PBI-M4-001 evidence closure
          |
          v
PBI-M4-002 five-step B300 gate
      |                 |
      v                 v
PBI-M4-003 process      PBI-M4-004 128-bit full chain
separation
```

Optimization PBIs may be implemented in parallel, but no candidate becomes a
default until its required full-chain gate passes.

## Archived backlog policy

Do not append new work to the pre-rebuild Stage 0/1/2 list. When old scripts or
artifacts are still relevant, reference their existing PBI and artifact-ledger
row from a current PBI. Git history is the source for the full historical table.
