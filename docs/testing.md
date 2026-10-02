# Testing strategy

Install development tools with `uv sync --locked --extra dev`.
Local checks cover Python behavior and native CPU contracts. Encrypted
accuracy and GPU performance require separate hardware runs.

## Fast local gate

```bash
CHECK_JOBS=2 scripts/run_fast_checks.sh
```

Runs Ruff formatting/lint and the configured pytest suite without coverage.
For a focused check:

```bash
scripts/run_fast_checks.sh tests/test_state_layout.py
scripts/run_fast_checks.sh tests/test_native_layout_cpp.py
```

Tests cover model/reference parity, state ownership, frozen payloads,
calibration, CLI/provenance contracts and process cleanup. Expected results
must be independent of production validators. See
[maintenance test scope](maintenance.md#test-scope) when changing coverage.

`tests/test_inference.py` exercises the public generation lifecycle with a small
local model, real polynomial export/recomputation and synthetic native reports.
Those subprocess tests validate result handling, not encrypted arithmetic.

## Full local gate

```bash
CHECK_JOBS=2 scripts/run_checks.sh
```

Adds coverage for the installed `fhemamba` package, with the threshold in
`pyproject.toml`. Parallelism requires pytest-xdist; otherwise the script
runs serially. Use `RUN_PRECOMMIT=1` to also execute the installed hooks.

CI is configured to run the local gate, build distributions, exercise both CPU
examples, validate published evidence and check the installed wheel outside the
checkout. See [ci.yml](../.github/workflows/ci.yml).

## Native C++ contract tests

Requires CMake >= 3.25.2 and a C++20 compiler; CUDA and FIDESlib are unnecessary:

```bash
cmake -S native/fideslib_stage0 -B build/stage0-layout-tests \
  -DFHE_STAGE0_BUILD_KERNEL=OFF -DFHE_STAGE0_BUILD_TESTS=ON
cmake --build build/stage0-layout-tests
ctest --test-dir build/stage0-layout-tests --output-on-failure
```

Pytest runs this path through `tests/test_native_layout_cpp.py`. Contracts
cover payload parsing, layout/routing, depth and refresh planning, cache and
operand ownership, protocol roles, and artifact emission.

## Artifact validation

```bash
fhemamba validate-artifacts --require-commit path/to/result.json
```

This checks the recorded schema and applicable success predicates; it does
not rerun inference. Direct backend artifacts identify source/binary, inputs,
hardware, CKKS parameters, numerical gates, operation counts, timing and memory.
Derived summaries must identify their source rather than pose as raw results.

For publication hashes and archived source verification, use
`fhemamba benchmark verify`; see the [experiment workflow](experiments.md).

## Research and GPU validation

Follow [research validation](validation.md) for approximation quality and
encrypted qualification, and the [Spark guide](dgx-spark.md) for builds.

| Change | Additional hardware checks |
| --- | --- |
| Plaintext encoding/upload | `packed_plaintext_probe`: exact RNS, scale/level handling, encrypted multiplication; test both model contexts |
| Rotations or ownership | Native exact-RNS/input-preservation probes and a matching model run |
| Refresh, state layout or model arithmetic | Every output/error/token gate over the intended horizon |
| Performance | Matched inputs, security, precision, cache state and placement; retain failed controls |

The [native guide](../native/README.md) and
[qualification suites](optimizations.md#reusable-qualification-suites) describe
individual probes. Passing CPU contracts or an isolated GPU probe does not
qualify full-model encrypted generation.
