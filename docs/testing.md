# Testing strategy

The repository has three verification tiers: local Python/native-contract
tests, local coverage, and hardware-backed encrypted integration gates.

## Fast local gate

```bash
scripts/run_fast_checks.sh
```

This runs:

- `ruff format --check`;
- `ruff check`;
- the complete configured pytest suite without coverage;
- slow-test duration reporting.

All Python tests live in `tests/`: model/approximation unit tests, CLI and
artifact validation, and repository/native integration contracts. Native C++
contracts live in `native/fideslib_stage0/tests/` and run through pytest.

For focused iteration:

```bash
scripts/run_fast_checks.sh tests/test_state_layout.py
scripts/run_fast_checks.sh tests/test_native_fideslib_stage0.py
```

When using `uv`, install with `uv sync --locked --extra dev`. The development tools are
an optional project extra, not a dependency group.

## Full local gate

```bash
scripts/run_checks.sh
```

This runs formatting, lint, pytest, and the `fhemamba` coverage gate. Retired
compatibility code is not included in package discovery or coverage.
Coverage targets the installed package explicitly, so leftover local folders
from the previous layout do not change what is measured.

Parallel execution is available when `pytest-xdist` is installed:

```bash
CHECK_JOBS=auto scripts/run_checks.sh
```

To exercise the installed pre-commit hook explicitly:

```bash
RUN_PRECOMMIT=1 scripts/run_checks.sh
```

## Native C++ contract tests

The FIDESlib-free C++ tests cover payload parsing, layout, planning, depth,
process-role restrictions, and artifact emission:

```bash
cmake -S native/fideslib_stage0 -B build/stage0-layout-tests \
  -DFHE_STAGE0_BUILD_KERNEL=OFF \
  -DFHE_STAGE0_BUILD_TESTS=ON
cmake --build build/stage0-layout-tests
ctest --test-dir build/stage0-layout-tests --output-on-failure
```

Pytest invokes this path through `tests/test_native_layout_cpp.py`.
The depth contract also checks coefficient-aware Chebyshev splits through
degree 768, both sparse and asymmetric coefficients, unchanged polynomial
values, and baseline ceilings for depth and scalar products. The encrypted
A/B campaign must check actual counts and output error independently.

These tests do not execute CKKS on a GPU. They are intended to catch contract
breakage before a GPU experiment, including DGX Spark.

## Artifact validation

Curated benchmark JSON should be checked with:

```bash
fhemamba validate-artifacts \
  --require-commit \
  path/to/result.json
```

Direct encrypted backend artifacts must include, where applicable:

- package/artifact version and repository commit;
- backend, hardware/configuration, and input mode;
- status and numerical gate;
- operation and rotation counts;
- bootstrap count and CKKS level telemetry;
- setup/evaluation/decrypt timing and peak RSS;
- a human-readable claim and explicit non-claims.

Do not manufacture a summary artifact from prose and present it as a raw
backend result. Summary/collection artifacts must identify themselves as such.

## Research and GPU validation

GPU benchmarks are separate from CPU CI. Follow [research validation](validation.md)
for algebra, approximation quality, isolated normalization, full-model encrypted
acceptance and the known gaps. The [Spark runbook](dgx-spark.md) describes builds
and execution.
