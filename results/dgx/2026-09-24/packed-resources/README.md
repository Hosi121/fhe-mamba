# Packed resource costs and shared transfer improvements

Same-binary full Mamba-3 generation measures **18.58→17.58 minutes
(5.39% less)**, with four matching tokens and unchanged 0.001 gates.
Evaluator rotations fall **86,063→67,052** and 3,898 outer binary-input
Clone calls are avoided. Lossless public matrix storage falls **673.3→168.3
MiB**; process peak RSS falls **27.9648→27.3143 GiB**.

See the [study](../../../../docs/research/2026-09-24-packed-resources.md).

- `full-base/`, `full-all/`: byte-preserved native results, measured launch
  records and logs. Derived `generation.json` reports actual selected IDs/text.
- `prefix-*/`: each individual optimization and their combination, in mirrored
  order. NAF and direct upload reduce mean time; reuse and compact storage alone
  do not demonstrate an isolated latency improvement. Combined mean is
  16.3368→15.4745 s (−5.28%).
- `m2-1-base/` through `m2-4-base/`: shared direct-upload ABBA on Mamba-2,
  two layers and two evaluations, **59.0315→55.7476 s (−5.56%)**. Parameters,
  levels and all operation counts match. `m2-full-direct/` is the full
  completion/parity follow-up: 33.83 minutes, maximum polynomial error
  0.006424 and four matching IDs. The ABBA supports the incremental speed claim.
- `cache-all/`: combined options with bounded plaintext caching and four-step
  carried state. All checks pass, with 236 hits and 210 avoided input clones.
- `probe-mamba2/`, `probe-mamba3/`: 160 exact-RNS inputs with both upload paths,
  four NTT widths, 90 shared-policy cases and mirrored preparation timings.
- `inventory.json`, `lm-full-inventory.json`, other `*-inventory.json`: CPU
  inventory from shared planners. Full model rotations excluding runtime
  refreshes are 78,740→59,900; runtime confirms these counts exactly. The
  4.37% peak logical slot occupancy is not a demonstrated persistent-packing
  speedup or a process-RAM estimate.
- `cpu-state.json`, `check_cpu_state.py`: owning SSM/conv slices reduce retained
  backing storage, with bit-identical prefill and decode. The raw config uses
  Python JSON's `Infinity` for an unbounded model limit; finite measured storage
  and parity fields are also in the standard-JSON `comparison.json`.
  `cpu-state-invalid-harness.json` retains the explicitly invalid first harness
  result; it is not supporting evidence.
- `compiled-sources.{json,tar.gz}`: 111 frozen native/Python source and test
  files. `build.log`, `build-provenance.json`, `target-provenance.json` bind the
  target executable, source, backend, compiler, input and hardware identities.
  The predecessor's source/build remain separate and unchanged.
- `compare.py`, `comparison.json`: regenerate all comparisons after validating
  raw hashes, source archive, target/predecessor identity, execution counts,
  modes, generated IDs, numerical gates and campaign completion. Run
  `python3 compare.py` from any directory. Adjacent predecessor evidence is
  required for its provenance binding.
- `finish_reports.py`: decode the recorded tokens with pinned local tokenizers.
  `run_campaign.py`, `runlib.py`, `build.sh`, `launch_m2.sh`, `environment.json`
  retain measured commands; paths and CPU IDs are target-specific. Use fresh
  output directories for a new measurement.
- `plan.json`, `budget.json`, `completion*.json`, `notification.json`: finite
  serial GPU sequence, review ledger and collection/desktop completion hook.
- `pytest.log`, `cpu-ctest.log`, `cpu-rebuild.log`: 286 Python tests and 17 C++
  CPU contracts. `provenance.json` hashes the retained artifact set.

The unchanged input identities refer to the preceding
[shared-preparation evidence](../shared-plaintext-preparation/),
[full Mamba-3 payload](../mamba3-trained-generation/full-payload/) and
[prefix payload](../mamba3-microkernels/prefix-payload/). Large checkpoints,
programs, client heads, executables and build trees are not duplicated.

All new options remain opt-in. Persistent state co-location is inventory-only
in this cycle; its access/depth costs need a separate schedule. These measurements
cover one frozen prompt on DGX Spark, with one fresh-key full run per Mamba-3
mode. Mamba-2 retains its separate 0.05 polynomial gate. The models are not an
architecture speed ranking. Inline client boundaries and `security=not-set`
remain unchanged.
