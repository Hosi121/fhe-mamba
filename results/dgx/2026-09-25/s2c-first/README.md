# S2C-first refresh evidence

One GPU bootstrap-circuit candidate, following the
[first-principles review](../../../cpu/2026-09-25/first-principles/README.md).
The [study](../../../../docs/research/2026-09-25-s2c-first.md) explains the
contract and circuit. The candidate is retained as an opt-in experimental
path: **759.7043 → 747.6291 s (1.59%)**, unchanged numerical/token gates,
and **27.3141 → 26.2067 GiB** peak process RSS.

- `contract.json`: workload, unchanged numerical gates, parameters and stop rule.
- `trial-source-manifest.json`: initial standalone and prefix source hashes.
- `final-source-manifest.json`: frozen sources for the full comparison.
- `validation.json`, `post-validation.json`: local tests and both native API builds.
- `initial-*`, `s2c-level*`: explicit two-pass probes and the rejected input boundary.
- `prefix-*`: four alternating prototype prefixes; `final-*-prefix` checks final sources.
- `final-*-full`: complete runs, commands, numerical gates, token and payload parity.
- `summary.json`, `summarize.py`: source/archive verification and disjoint cost categories.
- `selection.json`: adoption scope and reason; `source-verification.json` distinguishes
  compiled source identity from a delivery-patch formatting correction.
- `dependency-verification.json`: unchanged baseline library/source identities.
- `trial-sources.tar.gz`, `final-sources.tar.gz`: source overlays and probe code;
  `baseline-backend.patch` reconstructs the pre-existing patched dependency.
- `cleanup.json`: temporary local sources and DGX build trees removed after
  archive verification; selected executables and raw records remain on DGX.

Large checkpoint/payload files are not redistributed. Failed build
attempts and the intentionally rejected level boundary remain distinguishable
from passing numerical runs.

Recheck the collection without a GPU:

```bash
python3 results/dgx/2026-09-25/s2c-first/summarize.py
```
