# Borrowed plaintext upload and routing stages

The shared upload bridge borrows validated OpenFHE coefficient arrays through
the synchronized CUDA transfer, avoiding its remaining host staging vector.
Packed routing stages reuse the existing baby-step/giant-step evaluator when
its measured-key rotation count is lower. No error gate, key set or mask depth
is changed. All paths remain explicit options.

`comparison.json` is derived by `python3 compare.py` from mirrored separate
and combined controls, both exact-RNS probes, a same-binary full Mamba-3 pair
and Mamba-2 ABBA plus full parity. The latter's full run is a completion check;
its incremental speed claim uses ABBA. `compiled-sources.tar.gz` is the exact
compiled snapshot; `target-provenance.json` binds binaries, dependencies and
the OpenFHE layout headers. `validation-sources.json` separately binds the
later lint-exclusion configuration; all measured native sources are unchanged.

`superseded-borrow-only/` is an explicitly unbuilt waiting-queue revision.
Syntax-check failures from missing local include paths and the initial
immutable-artifact formatting failure are retained alongside corrected checks.
No borrowed GPU run failed those local harness checks. `layout.json` is a
local 66,593-word byte-layout check, separate from target GPU evidence.

The completion and postprocess hooks collect outputs, decode actual selected
IDs, verify source/binary/input hashes and save durable success/failure records.
`notification.json` records submission of the local desktop notification.

See the [study](../../../../docs/research/2026-09-24-borrowed-plaintext.md)
for results and scope: DGX Spark, one frozen prompt, inline client and
`security=not-set`, unchanged architecture-specific numerical gates.
