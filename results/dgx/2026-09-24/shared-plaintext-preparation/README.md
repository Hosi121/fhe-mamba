# Shared plaintext preparation

Mamba-2 full generation uses the same executable and frozen 24-layer,
five-evaluation payload in both modes: **2318.10 → 2142.11 seconds**
(**38.64 → 35.70 minutes, 7.59% less**). The generated IDs and all operation
counts/levels agree. The Mamba-3 prefix comparison checks extraction of the
common policy; its 0.29% timing difference is effectively neutral.

See the [study](../../../../docs/research/2026-09-24-shared-plaintext-preparation.md).

- `full-a/`, `full-c/`: raw Mamba-2 baseline/candidate output and launch logs.
  `generation.json` decodes actual selected tokens; `native-with-provenance.json`
  is a derived copy with the independently verified input hash. The adjacent
  `native.json` is unchanged.
- `smoke-*/`: mirrored A/B/C/C/B/A, one layer and two evaluations. Ordinary
  masks are cached, so these runs cannot establish a GPU NTT benefit.
- `pressure-*/`: two-layer B/C/C/B, with ordinary cache misses and nonzero GPU
  NTT dispatch; `selection.json` records the candidate-selection rule.
- `mamba3-prefix-*/`, `mamba3-synthetic/`, `mamba3-full/`: common-policy
  regression with old/new/new/old prefix, cached carried state and generation.
- `probe-mamba2/`, `probe-mamba3/`: 160 exact-RNS cases at four batch widths
  and 54 shared-policy cases for both secret/data-type configurations.
- `smoke-1-a/run.json` retains the initial controller rejection caused by an
  incorrect JSON-key lookup. `run-corrected.json` validates the same successful
  raw output; it does not hide or replace the original controller record.
- `compiled-sources.{json,tar.gz}`, `build.log`, `target-provenance.json`:
  immutable measured source, compiler commands, backend and executable hashes.
  Later working-tree changes belong to a separate optimization cycle.
- `compare.py`, `comparison.json`: source-archive, raw-hash, token, numerical
  gate, parameter, operation-count and level checks. Regenerate with
  `python3 compare.py` from any directory.
- `finish_reports.py`: regenerate decoded reports from the pinned local
  tokenizers. `run_study.py`, `run_remaining.py`, `launch_m2.sh` and
  `environment.json` retain measured machine-specific commands.
- `campaign-plan.json`, `budget.json`: bounded serialized GPU sequence under
  the existing hours-of-GPU authorization. `completion*.json`,
  `watch_completion.py` and `notification.json` record collection and the
  desktop completion hook.
- `checks-*.log`, `provenance.json`: local gates and artifact hashes.

Payloads are referenced from the prior
[Mamba-2 evidence](../../2026-09-22/subring-gates/),
[Mamba-3 generation](../mamba3-trained-generation/) and
[Mamba-3 prefix](../mamba3-microkernels/prefix-payload/).
Large model/program files, tokenizers, binaries and build trees are not copied.
Mamba-2 uses its unchanged 0.05 CKKS-to-polynomial gate; Mamba-3 retains 0.001
exact/polynomial gates. These different models are not a speed ranking.
There is one fresh-key complete run per mode, one frozen prompt, and the same
inline-client / `security=not-set` scope as the predecessor experiments.
