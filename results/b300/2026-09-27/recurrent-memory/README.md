# Recurrent memory admission evidence

The [study](../../../../docs/research/2026-09-27-recurrent-memory.md) qualifies
one optional change to the shared ready scheduler. Both classical-128 64-step
recurrence representations now complete under the unchanged 0.001 error gate.
The fixed-state run observes 138.416 GiB peak device usage and 129 live DAG
values; its unlimited control fails near 267.414 GiB with 772 live values.
These are sampled values from a completed candidate and an incomplete control,
not two successful full-run peaks. There is no 64-step speedup against the failed
control, nor a full-model encrypted long-generation claim.

The separate 40-step pair measures latency on the same GPU sequentially after
the diagnostic jobs finish. See `comparison.json` for that pair; cross-device
64-step timings are diagnostic. All measured processes share a native binary,
dependency libraries and trace. Each control/candidate pair uses the same
payload, device and CPU placement, changing only the admission threshold.

| Artifact | Purpose |
| --- | --- |
| [`decision.json`](decision.json) | Adoption scope, gates, default compatibility and remaining limits |
| [`comparison.json`](comparison.json), [`study.json`](study.json) | Case-level qualification, failed controls and the explicitly matched timing pair |
| [`measurements/`](measurements/) | Native numerical/security results, actual commands and input/source hashes |
| [`payload/`](payload/), [`input-reference.json`](input-reference.json) | Unchanged recurrence manifests; original trace and payload-source provenance |
| [`control-memory-growth.json`](control-memory-growth.json) | Increasing live values and device usage after cache entry counts plateau |
| [`broadcast-followup.json`](broadcast-followup.json) | Static repeat-chain inventory; no implementation or speed claim |
| [`qualification.json`](qualification.json) | CPU checks and native source/binary/dependency identities |
| [`visualization.json`](visualization.json) | Figure identity and reproduction command |
| [`publication.json`](publication.json), [`provenance.tar.gz`](provenance.tar.gz) | Original/published hashes, logs, hardware/build records and measured sources |

The archive contains `native-source.tar.gz` for the compiled implementation and
`analysis-source.tar.gz` for the maintained validators/plotter. The original
trace and recurrence exporter are shared with the
[preceding study](../recurrent-state/README.md). Disposable operational scripts
are excluded. Machine identities and local paths are publication substitutions;
numerical measurements are unchanged.

`all_runs_qualified: false` includes the intentionally failing unlimited controls.
A process exit of zero is insufficient: missing native results remain failures.
Every completed candidate independently passes the cryptographic/error contract,
including the cases with failed controls. Input preparation remains in evaluation
time; state arithmetic excludes the `input` and `public` operation-profile times.
Job wall time, lock wait and process time are recorded separately.

## Verify and reproduce the analysis

From the repository root:

```bash
uv run --no-sync python -m fhemamba.benchmarks verify \
  results/b300/2026-09-27/recurrent-memory
uv run --no-sync python -m fhemamba.benchmarks extract \
  results/b300/2026-09-27/recurrent-memory runs/recurrence-memory-review
cp -R results/b300/2026-09-27/recurrent-memory/measurements/. \
  runs/recurrence-memory-review/measurements/
uv run --no-sync python -m fhemamba recurrent analyze \
  --payload results/b300/2026-09-27/recurrent-memory/payload \
  --results runs/recurrence-memory-review/measurements \
  --schedule-study results/b300/2026-09-27/recurrent-memory/study.json \
  --output runs/recurrence-memory-review/comparison.json
```

The extracted logs supply the progress samples and CUDA failures. The public
JSON supplies the successful results and run bindings. To render the figure,
use the optional Matplotlib command documented in the
[maintained probe](../../../../experiments/recurrent_state/README.md#qualify-scheduler-memory-admission).
This rechecks recorded evidence; new FHE runs require rebuilding the qualified
native implementation and exporting the same trace/payloads.
