# Working on FHE Mamba

Read [docs/status.md](docs/status.md) first. It records the current qualified
configuration, unresolved numerical failure and next useful investigation.
Use [docs/repository.md](docs/repository.md) for ownership and
[docs/experiments.md](docs/experiments.md) for jobs and evidence. Read individual
research reports only when their evidence is needed.

- Reusable Python belongs in `src/fhemamba/`. Public tools start at
  `python -m fhemamba --help`; specialized studies live in grouped `experiments/`.
- Import package modules directly. Do not add sibling-script imports,
  `sys.path` modifications or another copy of hashing/job/CKKS helpers.
- Keep model arithmetic, frozen payloads, security parameters and acceptance
  gates unchanged during maintenance. GPU qualification is separate from CPU tests.
- Published result bytes and hashes are evidence. Add a new result for a new
  measurement; do not rewrite failures. Delete obsolete operational scripts.
- Preserve staged and unstaged work already present. Do not commit or push
  without an applicable user instruction. Local handoff details, when present,
  are in `.local/HANDOFF.md`; machine settings stay outside public documentation.
- Run `CHECK_JOBS=2 scripts/run_checks.sh` after code changes. Update current
  status and relevant guides when behavior or supported command paths change.
