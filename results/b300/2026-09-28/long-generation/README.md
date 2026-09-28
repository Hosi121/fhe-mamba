# Full-model long generation and GPU diagnosis

The trained Mamba-3 SISO 187M model completes **16 actual client-selected tokens**
under the classical-128 profile and both unchanged 0.001 hidden-error gates.
The 64-token candidate is rejected: selection 24 has error about 0.0196 even
though the observed 26-token prefix still agrees. These are generation tests,
not the preceding offline recurrence tests, and they do not establish a speedup.

- `tiled16-generation.json`: actual decoded continuation, counts, timing scope,
  all-output error checks, and original manifest/native identities.
- `measurements/`: successful 16-token run, failed 128-value schedule and exact
  prefix, exact-history OOM, parser-limit rejection, rejected 64-token run,
  and the unprofiled frozen two-token control.
- `payload/`: original calibration manifests and numerical fixtures. Programs,
  public vocabulary weights and checkpoints are reproduced with the maintained
  exporter instead of committing multi-gigabyte copies.
- `profiling/`: graph and graph-node Nsight Systems summaries, selected Nsight
  Compute counters, raw CSV exports, numerical results and export/analysis jobs.
  Binary profiler reports remain on the measurement host; their hashes are in
  the corresponding export jobs and summary inputs.
- `qualification/`: positive and negative gates, including the early 64-token
  rejection and client-error guard checks. An expected-rejection test is not a
  successful generation run.
- `cpu-qualification.json`, `tiled16-polynomial-profile.json`: phase-identity
  parity, frozen polynomial checks, excluded prompts and operation attribution.
- `jobs/`, `build/`, `settings.json`: portable argv records, build results and
  redacted settings. Logs and measured source snapshots are in provenance.

See the [study](../../../../docs/research/2026-09-28-long-generation.md) for
configuration, limits and interpretation. The 16-token evaluation is 1160.78589 s,
72.54912 s/generated token, excluding setup, final client selection, validation,
teardown and external transport. It uses 17 server evaluations. The successful
run has maximum exact error 6.62754e-5 and a sampled device peak of 179.416 GiB.

State tiling and phasor rotation preserve the full state algebra. Calibration
widens positive normalization domains but still does not cover arbitrary prompts.
The inline client decrypts the final hidden vector for vocabulary selection;
this is not a process-separated deployment. The finite DAG and validation
outputs still grow with generation length.

The successful 16-token run uses the qualified memory executor. Revision r1 adds
client diagnostics, r2 admits larger finite programs, and r3 stops on a known
client-reference failure. These changes do not alter CKKS arithmetic. Source
archives and each binary/library hash distinguish the actual measured versions.
Profiling times are separate from ordinary timings; different CPU/device
placements and concurrent independent jobs preclude a controlled overhead claim.

Publish/verify with the common benchmark tools. `publication.json` distinguishes
original hashes from redacted public derivatives; disposable controllers are
excluded. No access credentials or host connection instructions are required.

The client-reference guard passes the frozen two-token request at 0.001 and
rejects an intentionally strict 1e-12 test before any token selection. The
exception exit is 2. Python validation passes 424 tests; native validation
passes 22 tests. These checks do not qualify the rejected 64-token candidate.
