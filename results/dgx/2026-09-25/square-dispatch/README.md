# Square dispatch: bounded DGX Spark trial

The baseline is the adopted implementation at repository commit `88870f9`.
The candidate changes six native source/test files. Each variant has a
separate immutable source archive and executable; installed backend libraries,
compiler, input payloads and architecture-specific thread policies are held
constant. This is a prefix experiment, not a new full-model benchmark.

## Evidence

- `preregistration.json`: fixed hypothesis, gates and ABBA sample order.
- `baseline-sources.*`, `compiled-sources.*`, `candidate.patch`: exact source
  snapshots and the implementation change.
- `target-provenance.json`: verified baseline/candidate executable hashes,
  dependency identities, compiler, CUDA, hardware and compilation commands.
- `build-commands.json`, `build-cache.txt`: build commands and resolved CMake configuration.
- `probe-rns-mamba*/`: 72 new square cases plus 192 existing binary cases.
- `m3-*/`, `m2-*/`: complete native output, stdout/stderr and run records.
- `*-short-comparison.json`, `verified-summary.json`: all samples and means.
- `local-validation.json`, `pytest.log`: local formatting and test results.
- `completion*.json`, `notification.json`: durable completion and desktop
  notification delivery status, when the local desktop interface is present.
- `cleanup.json`: verified removal of expanded source and build intermediates;
  the three measured DGX executables remain available for later qualification.

The GPU probe compares every stored RNS word, ciphertext metadata and live
input with three previous multiplication entry points. Model gates preserve
operation counts, levels, refresh settings, output tolerances and the
no-intermediate-decryption contract. Keys are fresh per process. Decoded
error variation between runs is not evidence of improved accuracy.

## Recompute and reproduce

From the repository root, recheck archived identities and timing arithmetic:

```bash
python3 results/dgx/2026-09-25/square-dispatch/summarize.py
```

`campaign.py`, `runlib.py`, `preflight.py`, `launch_m2.sh` and `environment.json`
preserve the exact controller and target commands. Paths refer to the
recorded DGX installation; they are evidence, not portable setup instructions.
Use the repository's [reproduction guide](../../../../docs/reproducing.md)
to install dependencies and export the required payloads elsewhere. New
measurements need a fresh output root: the controller refuses to reuse a run
directory. Restore the archives into separate source roots, build with the
recorded compiler/options, update controller paths, then run the preflight
before measuring. `watch_completion.py` collects completed output and submits
the requested local desktop notification.

The Mamba-3 control uses one layer and three evaluations; the Mamba-2 control
uses two layers and two evaluations. Neither result establishes full-depth
latency, arbitrary-prompt quality, statistical significance, or a change to
the existing `security=not-set` feasibility scope. See the
[study](../../../../docs/research/2026-09-25-square-dispatch.md).
