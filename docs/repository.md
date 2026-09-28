# Repository layout

The repository contains one installable Python package and a separate native
backend. Numerical implementations and shared experiment utilities live in
`src/` and `native/`; workload specifications and evidence live outside the package.

| Location | Ownership |
| --- | --- |
| `src/fhemamba/` | Python package, including the CLI and version source |
| `native/fideslib_stage0/` | C++/FIDESlib backend, patches, probes and native contracts |
| `tests/` | Python unit tests and repository/native integration tests |
| `examples/` | Small runnable examples for new users |
| `config/` | Public platform pins, checkpoint hashes, coefficient bundles and configuration templates |
| `experiments/` | Specialized research workflows grouped by purpose; common code is in the package |
| `experiments/manifests/` | Declarative campaign configurations |
| `experiments/local_ckks/` | CPU feasibility recipes for `fhemamba diagnose ckks` |
| `experiments/slurm/` | Historical cluster launchers for the active model code |
| `scripts/` | Local checks, checkpoint setup and platform build/run helpers |
| `results/` | Public measurements and bundled provenance; see the [result index](../results/README.md) |
| `results/archive/` | Artifacts from the retired implementation |
| `docs/research/` | Dated studies and their evidence links |
| `docs/archive/` | Historical plans and ledgers |
| `docker/` | Historical B300 build image |
| `src/fhemamba/benchmarks/` | Shared job execution, completion events, publication and verification |
| `src/fhemamba/diagnostics/`, `profiling/` | Frozen-circuit diagnostics and offline Nsight analysis |
| `src/fhemamba/recurrent/`, `workloads/` | Reusable fixtures, recurrence analysis and upstream parity |
| `src/fhemamba/ckks_probes.py` | Shared optional CPU CKKS probe arithmetic and context construction |
| `config/local/`, `.local/`, `runs/` | Ignored machine settings, private notes, original evidence and working output |

## Generated files

Use `runs/<experiment>/` for fresh local output, payloads and temporary logs.
Download checkpoints to `checkpoints/`; native builds go in `build/` or an
isolated platform prefix. These directories, `.venv/`, `dist/` and cache files
are ignored by Git. Existing local files under the previous paths need not be
moved to follow the current source layout.

Publish reviewed derivatives into `results/` through the
[experiment workflow](experiments.md). Keep original bytes locally and record
their hashes separately from public hashes. Preserve failing statuses. Checkpoints,
private/evaluation keys, large exported payloads and build products are not
repository content. Public numerical coefficient bundles belong in `config/`.

## Previous layout

The current Python tools are listed by `python -m fhemamba --help`.
Standalone packed/diagnostic/profiling/recurrent/Mamba-3 scripts have been
removed in favor of these installed commands. Their numerical options remain
unchanged. Other scripts moved into purpose-specific experiment directories;
[the experiment index](../experiments/README.md) is the current entry point.
Source snapshots in published evidence retain the original measured paths.

The file cleanup follows the immutable
[`research-2026-09-22` snapshot](https://github.com/Hosi121/fhe-mamba/tree/research-2026-09-22).
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

The September 22 migration preserved recorded bytes. The subsequent publication
cleanup normalizes environment identifiers and deletes superseded operational
scripts. Necessary measured sources and logs are bundled. `results/publication.json` distinguishes original
hashes from public derivatives; numerical values and failure statuses are retained.
Existing Git history is not rewritten. Updated source/build paths produce new
provenance hashes; rebuild the native backend after changing source layout.

The installed `fhemamba` API and CLI name are unchanged. For an existing editable
environment, run `uv sync --locked --extra dev` after updating the checkout.

## Project name

The public repository is `Hosi121/fhe-mamba`, and the project is **FHE Mamba**.
Implemented and measured checkpoint workloads include **Mamba-2-130M** and
**Mamba-3 SISO 187M**, with distinct security and precision scopes documented in
the [evidence registry](evidence.md).

The repository was renamed from `fhe-native-mamba3` to align its public name
with that scope. Historical artifact paths, retired package names and recorded
host directories retain their original names. For an existing clone:

```bash
git remote set-url origin git@github.com:Hosi121/fhe-mamba.git
```
