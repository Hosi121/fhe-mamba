# Mamba-3 depth and batch refresh measurements

See the [study](../../../../docs/research/2026-09-24-mamba3-depth-batching.md)
for the measured full comparison and reproduction.

This folder records the common packed executor's depth-saving rewrites,
small-diagonal BSGS and packing of simultaneous logical refreshes. The fixed
contract is the trained SISO 187M checkpoint, frozen polynomials, the original
0.001 exact/polynomial error gates and actual client-selected tokens.

- `prefix-trace/`: instrumented baseline for a short trained prefix.
- `layer1-planned/`: initial depth-saving implementation, three evaluations.
- `layer1-single/`: rejected one-pass refresh; its 0.179 error fails despite
  matching greedy token IDs.
- `layer1-batch/`: depth-saving, BSGS and batch-refresh component measurement.
- `full-prefix-planned/`, `full-prefix-batch/`: all 12 layers for the first
  evaluation, without the autoregressive client loop. The later prefix uses
  the same frozen executable as the full comparison. It measured 268.38 s
  versus 306.74 s for the earlier unbatched prefix; those evolving binaries
  do not isolate a single mechanism.
- `full-prefix-manifest.json`, `prefix-derivation.json`: exact input identity and
  derivation of that prefix. The inherited manifest summary refers to the
  full export; the derivation records the actual 1414-node prefix.
- `full-batch/`: complete 12-layer × five-evaluation encrypted generation.
- `full-comparison.json`: matched full payload/context and result hashes.
- `budget-after-full.json`: cumulative charges through the full run; later
  microkernel probes continue the same 7200-second campaign.
- `provenance.json`: source archive, executable and runner identities.
- `compiled-sources.tar.gz`: exact native sources of the frozen full executable,
  before subsequent microkernel instrumentation.
- `runner-source.tar.gz`: the exact runner used for the full comparison.
- `checks.log`, `native-tests.log`: local Python and C++ regression checks.

Early probes use evolving binaries; do not present all of them as independent
ablations of a single final binary. The experiment uses an inline client and
`security=not-set`. It does not establish arbitrary-prompt or long-context
accuracy, or process-separated production FHE inference.
