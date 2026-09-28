# Reusing identical public initial ciphertexts

Adopted as an opt-in initialization improvement: mean full 16-token evaluation
falls from **1,157.98896 to 1,078.48210 seconds (6.87%)**, or **67.40513
seconds per generated token**. Both within-device pairs improve (7.02% and
6.71%) and all four full requests pass unchanged accuracy and security gates.

The optional `--reuse-public-ciphertexts` path encrypts two distinct public
vectors instead of 120 separate public sources in this frozen 16-token request.
Each source retains its own mutable handle; private inputs and inline client
feedback still use fresh encryption. Reuse does not cross requests or keys.

- `decision.json`, `comparisons/`: qualification, both within-device pairs,
  pooled descriptive timing and the separate-device prefix screen.
- `measurements/`, `qualification/`: native results, parameter audits, exact
  job records and successful token/all-output validation for each process.
- `inventory/`: maintained static inventory, source counts and input identities.
- `profiling/`: additional counters extracted from the previous published raw
  Nsight CSVs, register-residency limits and transform duration attribution.
  These are derived analyses of existing captures, not fresh profiler runs.
- `checks/`: source/archive agreement, matched pair configurations, test records
  and identities of the separate analysis source snapshot.
- `jobs/`, `build/`, `settings.json`, `contract.json`, `study.json`: portable
  commands and the predeclared accuracy, security and adoption conditions.

Measured source snapshots and original process/build/test logs are retained in
`provenance.tar.gz`. Disposable controllers are excluded. The build controller
initially collided with an existing output directory before compilation;
`build/orchestration-correction.json` records the corrected destination.

The native evaluator and backend library are identical in both modes. GPU/CPU/
NUMA placement is fixed within each pair, with opposite order on two devices.
Independent placements run concurrently; the result is descriptive, without
a statistical-significance claim. Keys and evaluator caches are fresh in each
process. The binary profiler is not attached to these timing runs.

Classical-128 parameters, both 0.001 error gates, encrypted arithmetic and
refresh counts remain fixed. The 16 selected tokens must match both frozen
references. All timings identify their scope; initialization savings do not
establish a steady-state token speedup. This study does not qualify the rejected
64-token request, arbitrary prompts, process-separated deployment or full-model
Mamba-2 performance. The option remains disabled by default.

See the [study](../../../../docs/research/2026-09-28-public-state-reuse.md)
for results and interpretation. `publication.json` distinguishes original
identities from redacted public derivatives. Local settings must be supplied
when reproducing the job specifications.
