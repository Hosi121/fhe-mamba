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

`pyproject.toml` includes both `tests/` and `fhemamba/tests/`. The root suite
covers repository/native integration contracts; `fhemamba/tests/` covers the
active Mamba-2 reference, lowering, CLI, and artifact validation.

For focused iteration:

```bash
scripts/run_fast_checks.sh fhemamba/tests/test_state_layout.py
scripts/run_fast_checks.sh tests/test_native_fideslib_stage0.py
```

When using `uv`, install with `uv sync --extra dev`. The development tools are
an optional project extra, not a dependency group.

## Full local gate

```bash
scripts/run_checks.sh
```

This runs formatting, lint, pytest, and the `fhemamba` coverage gate. Retired
compatibility code is not included in package discovery or coverage.

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

These tests do not execute CKKS on a GPU. They are intended to catch contract
breakage before an expensive B200/B300 allocation.

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

## GPU integration gates

FIDESlib GPU execution is not part of ordinary CI. Relevant gates are:

1. bootstrap and complex-pair micro-probes;
2. one-layer/full-width encrypted smokes;
3. full 24-layer multi-token execution;
4. process-separated execution;
5. 128-bit full-chain execution.

A micro-probe cannot promote a synchronization or bootstrap change. The B300
reduced-barrier build demonstrated why: its bootstrap probe passed while the
full chain silently produced corrupt finite values.

The current five-step campaign is:

```bash
python fhemamba/experiments/run_dgx_campaign.py \
  --manifest fhemamba/experiments/b300_autoregressive_prompt2_generate4.json \
  --runner scripts/run_b300_mamba2.sh \
  --output-json /home/kataiwa/fhemamba-b300/results/b300-p2-g4-campaign.json \
  --resume
```

## Known gaps

- The documented `0.4.5` three-token B300 success JSON is not currently
  tracked; recovery or exact rerun is PBI-M4-001.
- GPU CKKS execution cannot be reproduced by GitHub-hosted CI.
- A full process-separated Mamba run and a 24-layer 128-bit run remain open.
