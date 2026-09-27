# Observed level-schedule screen

This CPU tool reconstructs value versions, operation order and refresh batches
from `packed_fideslib --trace-levels`. It propagates remaining-depth constraints
backwards through the observed DAG. Client feedback and refresh start new
versions. It does not change the executor or its refresh placement.

```bash
python experiments/level_schedule/analyze.py program.txt run.log --output slack.json
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
