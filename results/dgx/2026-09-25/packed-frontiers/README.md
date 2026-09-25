# Packed refresh, polynomial and layout trials

The bounded trial tests three mechanisms against a 20% reduction in full
Mamba-3 evaluation time, preserving the frozen `0.001` gates and generated IDs.
It selects ready-node refresh scheduling. Joint polynomial evaluation and
persistent slot windows remain reproducible prototypes; their flags and
shared-evaluator refactor are removed from the released code.

The baseline is `18496e4e6a4e42c367fc356e99a7c4ab8e935eb0` and the selected
implementation is `1eda1e2fe8364ea5273dd4b6a08d46710048271f`.
The final full pair is **941.95 → 761.00 seconds
(19.21% shorter)**, below the 20% target.
See the [study](../../../../docs/research/2026-09-25-packed-frontiers.md) for
numerical results, the timer boundary and the adoption decisions.

## Evidence map

- `preregistration.json`: objective, unchanged gates, three-mechanism limit and
  stopping rule. `budget.json` is an internal review ledger, not an inferred
  user spending limit. Its extension covers final-source qualification only.
- `baseline-compiled-sources.*`, `baseline-git-verification.json` and
  `baseline-target-provenance.json`: baseline source, commit verification,
  executable, backend/library hashes and original build configuration.
- `batch-r1/`, `layout-r1/`, `frontier-r1/`, `layout-r2/`: immutable source
  archives, file manifests, build records and prefix comparisons for each
  revision. The second layout revision shares grouped refresh outputs.
- `final-r1/`, `final-source-verification.json`: frozen committed release
  sources, build commands and binary identities. The Mamba-2 source files
  match the baseline after the unused shared polynomial extraction is removed.
- `full-baseline/`, `frontier-r1-full/`, `layout-r2-full/`,
  `frontier-only-full/`: full-depth candidate selection. Prefix rankings alone
  do not decide adoption.
- `final-synthetic/`, `final-candidate-full/`, `final-baseline-full/`,
  `final-completion.json`: small gate and final-source full comparison, with
  the candidate followed by a closing fresh-process baseline.
- `*-prefix-*/`, `*-synthetic/`, `*-layout-fixture/`,
  `*-combined-fixture/`: every encrypted trial's native JSON, log and validated
  run record. All failed or slower results are retained.
- `m2-regression-*/`, `m2-completion.json`: interleaved baseline/candidate
  Mamba-2 regression for the prototype's shared polynomial extraction. It
  checks the unchanged `0.05` gate plus operation/CKKS-level parity; no full
  Mamba-2 speed claim follows.
- `layout-fixture/`, `make_layout_fixture.py`: dedicated encrypted alias,
  layout, polynomial and refresh fixture, using independent CPU expected values.
- `simulate_*.cpp`, `*-model-*.json`, `cpu-model-provenance.json`: approximate
  level models used to reject further variants. These are not hardware lower
  bounds or measured latency savings.
- `payload-manifests/`: frozen input identities. Model payloads/checkpoint
  weights are exported separately and are not duplicated in this directory.
- `release-checks.log`: 288 Python tests, including the native C++ suite, and
  both Ruff checks for the selected implementation. `final-checks.log` records
  the larger prototype suite before unused features were removed.
- `cleanup.json`: archive/source/binary verification before removing expanded
  trial sources and disposable builds; selected executables are retained.
- `*-completion.json`, `*-notification.json`, `watch*.py`: durable completion
  records, local notification submission status and collection hooks.

`run.json` binds each sample to its source manifest and executable hash.
`native.json` contains error gates, every output's errors, token IDs, timings
and counters. `summary.json` rechecks these identities before deriving totals.
Fresh keys vary between processes; error differences are not improvements in
accuracy. Counts inside bootstrap kernels are outside the model operation
counters. The experimental `security=not-set` boundary is unchanged.

## Recheck and reproduce

From the repository root:

```bash
python3 results/dgx/2026-09-25/packed-frontiers/summarize.py
```

For portable setup and payload export, use the
[Mamba-3 guide](../../../../docs/mamba3.md). `runlib.py`, `run_*.py`,
`measure_only.py`, `final_qualification.py` and the build records retain exact
commands for the recorded DGX installation. Their absolute paths are historical
evidence, not portable defaults. Use a new output root, restore the desired
source archive into an isolated directory, build with the recorded dependency
versions and run the same flags/environment. Controllers refuse to overwrite
existing trial directories.

The submitted executor accepts `--frontier-refresh` with
`--planned-refresh --batch-refresh`. To reproduce rejected polynomial/layout
modes, use their own frozen source archive and recorded commands; those flags
are deliberately absent from the current executable.
