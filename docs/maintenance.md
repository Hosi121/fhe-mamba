# Maintenance boundary and debt register

This repository ships one Mamba-2 implementation. The pre-rebuild research
stack is preserved on `archive/pre-compat-retirement-20260811` and is not part
of `main`, the wheel, coverage, or the supported command line.

## Canonical ownership

| Concern | Canonical location | Policy |
|---|---|---|
| Mamba-2 reference and polynomial operators | `fhemamba/src/fhemamba/` | Active; new model math belongs here. |
| Payload and client-reference export | `fhemamba/src/fhemamba/m1_payload.py` | Active; the historical filename is debt, not a second implementation. |
| Encrypted GPU execution | `native/fideslib_stage0/` | Active; CPU-only units must remain buildable without FIDESlib. |
| Current experiments | `fhemamba/experiments/` | Active only when backed by a manifest or documented command. |
| Artifact schema validation | `fhemamba/src/fhemamba/artifacts.py` | Active shared provenance contract. |
| Promoted B300 platform identity | `config/b300-platform.env` | Versioned authority for image, CUDA, SM, FIDESlib commit/profile, and binary path. |
| Root `scripts/` | Local checks and current DGX/B300 helpers only | Keep wrappers narrow; experiments belong under `fhemamba/experiments/`. |

The installed command is `fhemamba`. The historical `fhe-mamba3` command is
available only from the archive branch.

The distribution, import package, and installed command are all named
`fhemamba`. Version `0.5.0` records the compatibility-breaking retirement of
the old distribution metadata, import package, and command.

An editable checkout used before `0.5.0` can retain ignored build metadata that
advertises the removed command. Remove only those generated directories once,
then resync the environment:

```bash
rm -rf fhe_native_mamba3.egg-info fhemamba/src/fhe_native_mamba3.egg-info
uv sync --locked --extra dev
```

## Guardrails

- CI measures coverage for the installed `fhemamba` package. Repository and
  native contract tests supplement that package-level coverage.
- Native direct invocations default to `128-classic`. Feasibility campaigns may
  use `not-set` only by setting it explicitly and recording it in their result.
- Calibration and telemetry must not be included in an inference timing unless
  the metric is explicitly named as instrumented timing.
- New Python entry points must be importable from the installed package. Do not
  add another `sys.path.insert` workaround.
- New native options require a parser test and an artifact field. Prefer a
  versioned manifest field over another shell environment variable.
- Headline results require a tracked raw artifact. Prose-only measurements stay
  visibly labeled as such.

## Archive boundary

- Do not copy the old package back into `main` to recover a helper. Port the
  smallest behavior behind an active test and current naming.
- Historical commands and Slurm files remain reproducible from the dedicated
  archive branch pinned to an exact commit in `docs/legacy-archive.md`; they are
  not supported commands on `main`.
- Claim-bearing raw artifacts and research notes stay in the main history even
  when their generating implementation is archived.

## Native decomposition queue

`stage1_mamba2_decode_fideslib.cpp` currently combines orchestration, cache
construction, cryptographic operators, protocol roles, debug decrypts, and JSON
reporting. New behavior should first move the touched responsibility behind a
testable component. The intended split is:

1. typed runtime configuration and manifest loading;
2. immutable model/execution plan;
3. plaintext cache construction;
4. encrypted block operators;
5. client/server protocol orchestration;
6. telemetry and artifact serialization.

This queue is deliberately separate from kernel optimization: changing the
cryptographic schedule and restructuring ownership in the same patch makes both
correctness and performance regressions difficult to attribute.
