# Coefficient ownership transfer

Mamba-3's same-binary prefix ABBA measures **15.4315→15.1540 s (−1.80%)**.
Its full candidate passes in 17.31 minutes; this is a completion/parity check,
not a new paired full-runtime comparison. Mamba-2's short mean is 0.48% higher,
so that model's full follow-up is explicitly skipped.

See the [study](../../../../docs/research/2026-09-24-coefficient-ownership.md).

- `m3-*`, `m2-*`, `cache-move/`: raw native results and launch records.
- `m3-full-move/generation.json`: actual selected tokens, decoded with the pinned tokenizer.
- `probe-*`: both configurations, 160 original plus 160 moved RNS inputs and 126 policy cases.
- `prototype/`: separate initial exactness/phase prototype and immutable source archive.
- `compiled-sources.*`, `target-provenance.json`: model source, actual binaries, dependencies and compiler identity.
- `validation-sources.json`, `validation/`: the runner assertion repair after the native snapshot, with the initial failure and omitted CTest option retained.
- `compare.py`: recompute comparisons and check source archives, binaries, payloads, counts, gates and predecessor identity. Run `python3 compare.py` with the adjacent `packed-resources/` evidence present.
- `finish_reports.py`: regenerate decoded reports with local pinned tokenizer files.
- `plan.json`, `budget.json`, `completion*.json`, `notification.json`, `postprocess-completion.json`: finite jobs, collection/notification and successful postprocessing.

Measured controllers retain their exact bytes and target paths. Use new output
directories for reproduction. Build trees, large payloads and checkpoints are omitted.
