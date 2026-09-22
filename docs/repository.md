# Repository layout

The repository contains one installable Python package and a separate native
backend. Numerical implementations live in `src/` and `native/`; experiment
orchestration and recorded evidence live outside the package.

| Location | Ownership |
| --- | --- |
| `src/fhemamba/` | Python package, including the CLI and version source |
| `native/fideslib_stage0/` | C++/FIDESlib backend, patches, probes and native contracts |
| `tests/` | Python unit tests and repository/native integration tests |
| `examples/` | Small runnable examples for new users |
| `config/` | Platform pins, checkpoint hashes and frozen coefficient bundles |
| `experiments/` | Research runners and shared runner helpers |
| `experiments/manifests/` | Declarative campaign configurations |
| `experiments/slurm/` | Historical cluster launchers for the active model code |
| `scripts/` | Local checks, checkpoint setup and platform build/run helpers |
| `results/` | Curated artifacts; see the [result index](../results/README.md) |
| `results/archive/` | Artifacts from the retired implementation |
| `docs/research/` | Dated studies and their evidence links |
| `docs/archive/` | Historical plans and ledgers |
| `docker/` | Historical B300 build image |

## Generated files

Use `runs/<experiment>/` for fresh local output, payloads and temporary logs.
Download checkpoints to `checkpoints/`; native builds go in `build/` or an
isolated platform prefix. These directories, `.venv/`, `dist/` and cache files
are ignored by Git. Existing local files under the previous paths need not be
moved to follow the current source layout.

Promote only the artifacts needed to substantiate a reviewed result into
`results/`. Keep raw result bytes and failing statuses intact. Checkpoints,
private/evaluation keys, large exported payloads and build products are not
repository content. Public numerical coefficient bundles belong in `config/`.

## Previous layout

The file cleanup follows the immutable
[`research-2026-09-22` snapshot](https://github.com/Hosi121/fhe-native-mamba3/tree/research-2026-09-22).
Use that tag when reproducing commands and source hashes from the original
measurements. Current guides and launchers use the paths below.

| Original location | Current location |
| --- | --- |
| `fhemamba/src/fhemamba/` | `src/fhemamba/` |
| `fhemamba/tests/` | `tests/` |
| `fhemamba/experiments/*.py`, `*.sh` | `experiments/` |
| `fhemamba/experiments/*.json` | `experiments/manifests/` |
| `fhemamba/slurm/` | `experiments/slurm/` |
| `fhemamba/results/` | `results/` |
| Previously tracked `runs/` files | `results/archive/pre-compat-retirement/` |
| `fhemamba/README.md` | [Python package guide](package.md) |
| `fhemamba/DESIGN.md` | [Historical Phase 0 design](archive/phase0-design.md) |
| `docs/artifact_ledger.md` | [Historical artifact ledger](archive/artifact-ledger.md) |
| `docs/legacy-archive.md` | [Legacy implementation archive](archive/legacy-implementation.md) |
| `docs/probes/2026-05-10-b200-fideslib.md` | [Historical B200 probe log](archive/2026-05-10-b200-fideslib.md) |

Recorded JSON, logs and coefficient bundles were preserved byte for byte.
Paths embedded inside those artifacts identify the original environment and
were not rewritten. Updated source/build paths produce new provenance hashes;
rebuild the native backend after upgrading a checkout across this change.

The installed `fhemamba` API and CLI name are unchanged. For an existing editable
environment, run `uv sync --locked --extra dev` after updating the checkout.
