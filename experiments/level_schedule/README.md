# Observed level-schedule screen

This CPU tool reconstructs value versions, operation order and refresh batches
from `packed_fideslib --trace-levels`. It propagates remaining-depth constraints
backwards through the observed DAG. Client feedback and refresh start new
versions. It does not change the executor or its refresh placement.

```bash
python -m fhemamba diagnose levels program.txt run.log --output slack.json
```

The input program must use the v2 line-oriented export. Large linear-weight
arrays are hash-checked as part of the whole file but omitted from metadata.
Missing outputs, an unexpected level change or an internal refresh not modeled
by the trace cause an error. The current screen models the qualified S2C-first
profile: consumed-level ceiling 35, returned level 18, and 45 Q towers.

`actions` and `versions` retain the inferred transitions; `summary` counts
input modulus towers before and after the hypothetical drops. These are
optimistic structural opportunities, not a measured latency prediction, an
encrypted-accuracy qualification or an optimal level scheduler. In particular,
later refresh inputs can increase two-pass residual-alignment cost, and public
FFT preparation does not become cheaper just because Q is smaller.

The September 27 secure trace contains 7,814 live nodes and 484 physical
bootstrap calls. Typical input-tower reductions are 7.64% for polynomials,
8.1–8.5% for linear projections and 14.34% for scatter. Multiplying all measured
nonrefresh operation times by these ratios gives an intentionally optimistic
12.56-second ranking proxy (5.27% of that trace), before adding scheduling/drop
cost or changed refresh cost. It is not a time bound. Correction merging was
chosen for implementation first because it removes known duplicate operations.

## Polynomial attribution

`polynomials.py` joins live polynomial nodes to the frozen calibration manifest
by their exact interval and coefficients. It reports function counts, degrees
and calibration intervals, including upper/lower interval ratios for inverse
square roots. It rejects a manifest whose program hash differs.

```bash
python -m fhemamba diagnose polynomials --program program.txt \
  --manifest manifest.json --native native.json --output polynomial-profile.json
```

Omit `--native` for a static inventory. Supply `--metadata` instead of
`--program` to reuse `program_metadata()` output from `analyze.py`. A current
native run with `--profile-evaluation` emits `polynomial_stats`; these timings
must cover every live polynomial exactly once and sum to the aggregate `cheb`
record. Keep the benchmark job's input hashes with the profile: workload sizes
alone cannot establish that a native result belongs to the same program.

This attributes polynomial-node work, not an entire normalization module.
Reductions, masks and products remain separate operations. A refresh batch is
charged to the node that starts it, even when it refreshes other live values.
Calibration intervals describe the export; they do not certify unseen inputs.
