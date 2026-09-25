# First-principles investigation

This is a static and float64 algebra study, **not** a new encrypted benchmark.
The [research note](../../../../docs/research/2026-09-25-first-principles.md)
contains interpretation, primary references, falsification criteria, and the
distinction between unchanged-model and changed-protocol alternatives.

Run from the repository root:

```sh
.venv/bin/python results/cpu/2026-09-25/first-principles/analyze.py
```

The script needs NumPy and the existing full exported payload at
`runs/mamba3-lm-full-20260924/program.txt`. `--program` accepts another path,
but the SHA-256 must match the recorded manifest. It skips the public weight
arrays while parsing, retaining bounded per-line memory. No model download,
GPU, encrypted computation, or production code modification is involved.

`summary.json` records input hashes, all 761 live polynomial-node attributions,
counts reconciled with the final native run, nested-timer-aware cost scenarios,
and float64 algebra checks on 65,537 points for each of 12 layers. The grid
is a screening test, not an interval proof or a model-quality certificate.
The rate-floor result additionally follows analytically from each interval's
lower endpoint. RNG seed 20260925 controls the independent vector identities.

The polynomial-core call counts are reused from the prior source-level
inventory for the identical frozen program, with node size/degree checks and
total reconciliation. They do not count internal bootstrap multiplications.
Backend source identities and literature versions are in `sources.json`.
The analyzer and summary hashes are recorded in `validation.json`.
