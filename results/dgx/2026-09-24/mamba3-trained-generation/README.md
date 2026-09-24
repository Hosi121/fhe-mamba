# Trained Mamba-3 generation and slot routing

See the [study](../../../../docs/research/2026-09-24-mamba3-trained-generation.md)
for conditions, commands and limitations.

- `layer1-legacy/` and `layer1-radix8/`: matched, single-layer, three-evaluation
  runs from one executable and payload. `layer1-comparison.json` binds their hashes.
- `full-radix8/`: complete 12-layer, five-evaluation run and the actual client
  generation report. Four generated IDs match the exact model.
- `synthetic-radix8/`: four-step mixer regression, including carried-state errors.
- `*-payload/`: input manifests and CPU fixtures. Large public-checkpoint
  programs and vocabulary matrices are regenerated with the documented exporter.
- `budget.json`: all four runs charged against a shared 7200-second limit.
- `provenance.json`: executable, compiled source, Python source and artifact hashes.
- `checks.log` and `native-tests.log`: local regression results.

All encrypted runs use `security=not-set` and an inline client. Source changes
were uncommitted when measured; the base commit and exact source hashes are
recorded. The 22.54% speedup applies to the one-layer comparison, not to a
matched full-model experiment or a comparison with Mamba-2.
