# Contributing

Start with the [current state](docs/status.md) and [backlog](docs/backlog.md).
Useful changes include CPU examples, reference/layout tests, diagnostics and
GPU work with reproducible evidence.

## Development setup

Requires Python 3.10+, uv, CMake >= 3.25.2 and a C++20 compiler:

```bash
uv sync --locked --extra dev
uv run --no-sync pre-commit install
CHECK_JOBS=2 scripts/run_fast_checks.sh
```

Before a release, tag or claim-changing merge:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  CHECK_JOBS=2 scripts/run_checks.sh
```

See [testing](docs/testing.md) for focused checks and coverage.
GPU validation requires a separate [native build](docs/dgx-spark.md#build)
and [qualification](docs/validation.md). The historical B300 helpers target
CUDA 12 / SM100; current B300 studies record CUDA 13 / SM103 separately.

## Active code

- Put reusable Python in `src/fhemamba/`, native code in `native/fideslib_stage0/`,
  and specialized studies in grouped `experiments/` directories.
- Use installed package imports and shared job/provenance helpers.
- Keep machine settings in `config/local/`, fresh outputs in `runs/`,
  and private handoff notes in `.local/`.

See [repository ownership](docs/repository.md) and
[maintenance contracts](docs/maintenance.md) for implementation boundaries.

## Definition of done

- State the problem, change and supported scope.
- Pass the relevant local checks. For hardware-dependent behavior, include
  measured evidence or explicitly identify pending qualification.
- Preserve model arithmetic, frozen data, security and acceptance gates during
  maintenance. Numerical changes need their own comparison.
- Record source, binary and input identities for claim-bearing results.
- Update the affected guide and current status when behavior or claims change.

Review model approximation error separately from CKKS execution error.
Check slot layouts, levels, refresh placement, rotation keys and operand
ownership; diagnostic decryptions must never influence encrypted execution.
A component probe does not qualify the full model.

## Benchmark artifacts

Use the [experiment workflow](docs/experiments.md) for execution, comparison
and publication. Keep original bytes locally, publish reviewed derivatives
with distinct original/public hashes, and retain failures and measured sources.
Delete superseded operational scripts rather than creating another archive.

Artifacts identify configuration/security, numerical gates, per-output errors,
token IDs, operation counts, timing scope and memory. Validate them with:

```bash
fhemamba validate-artifacts --require-commit path/to/result.json
```

Label derived reports and prose-only observations. Do not reconstruct a raw
measurement from notes.

## Versioning and tags

Use SemVer. Patch versions cover fixes and changes within a capability;
minor versions mark new runnable capabilities or breaking changes before 1.0.
Version 0.5.0 retired the old distribution, import package and command.
Version 1.0.0 is reserved for reproducible interactive generation at 128-bit
parameters with an explicit protocol-security statement.

Create a release tag only when versions agree, the full local gate passes,
claim-bearing artifacts validate and the [evidence registry](docs/evidence.md)
links every headline result. Historical missing-evidence work stays in the
[backlog](docs/backlog.md).

## Review priorities

Explain the numerical or user-visible consequence of a change, how it was
checked, and any remaining limits. Link the relevant evidence rather than
repeating the experiment history in each guide.
