# Two-pass refresh correction evidence

The [report](../../../../docs/research/2026-09-27-b300-refresh-correction.md)
explains the 235.839 → 210.326 second classical-128 comparison and its limits.

- `full-comparison.json` and `prefix-comparison.json` validate the fixed
  cryptographic/workload/accuracy contract and the 3% improvement screen.
- `full-correction-*` and `prefix-correction-*` contain native results,
  parameters and process records. Both arms use the same measured executable;
  only `--merge-refresh-correction` differs.
- `probe-correction-real-r2` is the successful complete refresh qualification.
  `probe-correction-complex-r3` covers the integer primitive only. The earlier
  failed attempts remain distinguishable by their process status.
- `verified-input-identities.json` binds the program, fixture and client head.

`publication.json` records original and published hashes. Logs, source
snapshots and the measurement affinity patch are in `provenance.tar.gz`.
One-off orchestration scripts are excluded. To inspect this material:

```bash
python -m fhemamba.benchmarks verify results/b300/2026-09-27/refresh-correction
python -m fhemamba.benchmarks extract \
  results/b300/2026-09-27/refresh-correction runs/refresh-correction-inspect
```
