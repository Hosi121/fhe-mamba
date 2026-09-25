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
The packed contracts include depth-planner invariants, dense routing oracles
and bitwise coefficient copies across wraparound, short tails and special
IEEE-754 values. Native microkernel performance is measured separately on
[DGX Spark](research/2026-09-24-mamba3-microkernels.md); local CI does not
establish CUDA speed or encrypted accuracy.

The plaintext-cache contract checks deliberate hash collisions, coefficient
bits (including signed zero), CKKS levels, LRU eviction and backend ownership.
It also verifies that dense/non-finite vectors bypass the cache. Encrypted
accuracy and timing still require a matched run on the target GPU.

The hardware-only `packed_plaintext_probe` checks the plaintext-upload and GPU
NTT bridge. Its 160 input cases cover 1/32/1024/32768 slots, levels 0/21/34/44,
scale degrees 1/2 and five coefficient patterns. All RNS residues must equal
OpenFHE's standard encoding exactly, including NTT batches of 1/4/16/64 limbs.
It also checks encrypted multiplication at three working levels with a `1e-6`
error limit. These checks require the pinned OpenFHE/FIDESlib build and a CUDA
GPU. They supplement the unchanged `0.001` exact/polynomial model gates and
actual token comparison; CPU CI cannot validate this path. The
[encoding study](research/2026-09-24-mamba3-gpu-encoding.md) retains raw reports,
the failed initial allocation probe and the corrected implementation.

The shared preparation policy checks 162 cases for periodic coefficients and
new/cached degree-2 addends. Run `packed_plaintext_probe OUTPUT.json --mamba2`
for sparse-ternary keys with complex slots, or omit `--mamba2` for the Mamba-3
uniform-ternary/real configuration. These probes verify both evaluators' common
dispatch and keep the existing exact RNS and `1e-6` arithmetic gates.
The direct-upload path is checked in both evaluation and coefficient formats,
including all four NTT batch sizes. Earlier archived studies used 54 policy
cases, before adding the direct-upload modes; the direct-only study used 90
cases and the coefficient-ownership study used 126.

Borrowed upload checks both plaintext formats for all 160 inputs (320
input/format cases), all four coefficient NTT widths, unchanged CPU residues
after upload and the moved-coefficient path. It must actually dispatch the
borrowed loader on the pinned target. Mirrored preparation samples separate
ownership transfer from borrowed upload. The shared policy also exercises
their implicit flags and preserves the periodic CPU encoder.

Native CPU contracts exhaust packed NAF offsets and finite BF16 bit patterns,
compare compact and double BSGS masks, and protect retained/aliased operands
from destructive reuse. Stateful CPU tests check the actual backing-storage
bytes after prefill and verify continuation against full-sequence inference.
Routing contracts compare destination masks with each original routing stage,
including dirty padding, gather/scatter and all supported radices. Existing
BSGS tests separately compare masked transforms with direct indexing.
The [resource study](research/2026-09-24-packed-resources.md) retains both
configuration probes, complete model regressions and the corrected storage
harness alongside its invalid first result.

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
process-role restrictions, artifact emission and packed slot routing. Routing
tests compare random and model-shaped gather/scatter maps with direct indexing,
including inactive slots and all supported routing radices:

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

The ready-schedule contract covers out-of-order independent work, duplicate
input edges, pinned outputs, dead nodes and two client-feedback barriers.
Runner tests reject frontier refresh without grouped refresh and verify flag
propagation through both budgeted and unbudgeted execution.

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
