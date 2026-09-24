# Static preparation and lifetime design

This evidence describes an unchanged measured model and a static access trace.
It does not contain a new cache, synchronization implementation or speed test.
The [design study](../../../../docs/research/2026-09-25-preparation-design.md)
records the integration choices and remaining validation.

- `cache_access_trace.cpp` reads the frozen program with the existing parser,
  liveness planner and replicated-linear geometry. Its complete ordered output
  is `cache-access-trace.json`; `cache-access-trace-run.json` records execution.
- `cache_policy.py` recomputes all LRU/fixed-subset counts and verifies the five
  identical cycles. Run `python3 cache_policy.py` in this directory.
- `m3-native.json` and `m3-run.json` are unchanged copies of the completed
  ownership candidate. `profile_analysis.py` checks their relationship and
  reproduces the profile calculations; run `python3 profile_analysis.py`.
- `source-identities.json` identifies the program and application sources.
  The full frozen application archive is in the
  [ownership campaign](../../../dgx/2026-09-24/owned-arithmetic/).
  `build-command.json` records the local analyzer build command.
- `backend-observations.json` binds inspected FIDESlib sources and the existing
  patch set. It records CPU metadata, registry concurrency and destructor
  lifetime constraints without applying another backend patch.

To regenerate the trace, extract the ownership campaign's final application
source archive into an unused directory. Compile this analyzer against those
`native/fideslib_stage0/include` and `src` directories, linking its
`src/stage1_mamba2_plan.cpp`, as shown in `build-command.json`. Supply the full
program whose hash is in `source-identities.json`; program export and model
inputs are covered by the repository's reproduction guide. Write new output
to a separate directory and copy the two analysis scripts there to compare.

The temporary analyzer executable was removed after its hash and outputs were
verified. No GPU experiment or source change was made for this static study.
