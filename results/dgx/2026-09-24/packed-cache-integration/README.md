# Plaintext cache integration

This campaign reuses the immutable executable from `../borrowed-plaintext/`.
It checks the existing 64-entry mask cache with the preceding selected upload
and routing options. No native rebuild or source change occurs between these
campaigns. Prefix controls run uncached/cached/cached/uncached, each in a fresh
process. A cached full run is attempted only if that prefix comparison improves.

`baseline-prefix/` and `baseline-full/` are byte-preserved artifacts from the
preceding campaign. Its full candidate is the uncached full baseline. The
cached full run follows an intervening Mamba-2 validation, so the single full
pair is reported separately from the interleaved prefix controls.

Recompute source/binary/input bindings, actual cache dispatch, unchanged gates,
operation counts, selected IDs and timing comparisons with `python3 compare.py`.
`comparison.json` includes the final adoption decision. The copied Mamba-2
launcher is an unused provenance dependency of the common run helper; every
native command in this campaign runs the packed Mamba-3 evaluator.

See the [study](../../../../docs/research/2026-09-24-packed-cache-integration.md).
This is a DGX Spark feasibility workload, one prompt, inline client and
`security=not-set`; it does not establish a model architecture ranking.
