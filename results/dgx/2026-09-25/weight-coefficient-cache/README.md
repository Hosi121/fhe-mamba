# Rejected compact public-weight coefficient cache

The candidate preserves exact RNS coefficients/metadata and GPU NTT results,
but its cold one-layer three-step mean is **47.117472 → 47.711634 seconds**
(**1.261% slower**). It fails the predeclared prefix qualification, so the
full pair is skipped and production source is restored. `passed: true` in the
completion record means the validation campaign completed; it does not mean
performance adoption.

See the [study](../../../../docs/research/2026-09-25-weight-coefficient-cache.md).

Recompute the decision with Python's standard library:

```bash
python3 results/dgx/2026-09-25/weight-coefficient-cache/compare.py
```

- `comparison.json` and `adoption.json` give the decision and restored paths.
- `lm-layer1-prefix-*` and `lm-layer1-*` preserve every ABBA sample unchanged.
- `probe-*-revision-3` contains the final 120 exact RNS/GPU cases. The earlier
  `probe-mamba3`/`probe-mamba2` runs remain as superseded evidence.
- `baseline/` and `revision-{1,2,3}/compiled-sources.tar.gz` bind all source
  snapshots. Revision 3 is the final candidate. `source-complete.diff` includes
  the complete rejected implementation and test changes.
- `revision-1/build.log`, `checks.log` and `checks-revision2.log` retain the
  include-directory failure, stale test expectations and corrected validation.
- `revision-3/target-provenance.json` and `postflight.json` bind the preserved
  baseline executable, candidate executables, compiler, libraries and payloads.
- `plan.json`, `campaign.py`, `runlib.py` and `budget.json` preserve the protocol,
  finite review interval, original commands and all native process charges.
- `cleanup-remote.json` records verified executable archives on the DGX and
  removal of temporary source/build directories. Shared dependencies and model
  payloads were left in place. Binary archives are not included in Git.

To rerun on compatible hardware, extract the candidate source archive, use the
build flags in `revision-3/build.sh` with your local paths, and reproduce the
input manifests in `payload-manifests/` using the project's pinned checkpoint
and exporter. Run the original commands from each `run.json`, adjusting paths
only. The rejected `--weight-cache-mib` option exists in that archived source;
it is deliberately absent from the active application. Models and large input
payloads are downloaded separately and are not part of this evidence bundle.

The partial trained workload has one layer and three evaluations, selecting
IDs `[6864, 6864]`. It is not full-backbone generation. The smaller encoding
benchmark repeats 16 real weight diagonals five times at a fixed level inside
one process; its 59–62% reduction is not an inference speedup claim.
