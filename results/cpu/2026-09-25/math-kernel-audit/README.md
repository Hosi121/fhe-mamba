# Static arithmetic and kernel study

This directory supports the [research note](../../../../docs/research/2026-09-25-math-kernel-audit.md).
It contains static counts, backend source identities and conditional timing
calculations. It contains no new encrypted evaluation or measured speedup.

- `analyze.cpp` uses the existing parser, liveness planner, NAF decomposition,
  routing planner and replicated-linear shapes. `inventory.json` retains the
  counted rotation groups and individual polynomial records.
- `build.json`, `build.log` and `run.json` identify the final analyzer execution.
- `derive.py` independently checks prefix counts, matches the frozen runtime's
  rotation/product counts, builds equal-degree nonlinear antichains, and
  computes conditional Amdahl scenarios. Run `python3 derive.py` from any
  directory to reproduce `summary.json` and `frontier-groups.json`.
- `source-identities.json` binds the full input program, relevant application
  sources, profiles, compiler and temporary analyzer binary.
- `backend-sources.json` binds the inspected DGX FIDESlib files. The API and
  ciphertext source hashes match the preceding preparation study.
- `provenance.json` hashes the evidence files. The temporary analyzer binary
  was removed after checking its recorded hash; model inputs remain unchanged.

To regenerate the inventory, use application commit `88870f9` and the exact
program hash in `source-identities.json`. Compile `analyze.cpp` as recorded in
`build.json` (create its output directory first), then execute the command in
`run.json`, redirecting stdout to a new `inventory.json`. The large model payload
is not duplicated into this evidence directory.

The antichains show mathematical independence and slot capacity. They do not
prove runtime level compatibility, net savings after packing, or acceptable
FHE error. No speedup should be inferred from the count of groups alone.
