# Fixed-size Mamba-3 recurrence evidence

See the [study](../../../../docs/research/2026-09-27-recurrent-state.md) and
[maintained probe](../../../../experiments/recurrent_state/README.md).

One B300, the same qualified native binary and the classical-128 parameter
profile are used throughout. The new state carries eight ciphertexts and retains
every coordinate. It is slower at 4/16 steps, then improves state arithmetic
126.229 → 111.084 s (12.0%) at 40 steps. Total recurrence-subcircuit evaluation,
including offline input encryption, improves 291.114 → 277.288 s (4.75%).
The 40-step fixed-state run includes 46 physical bootstraps and maximum absolute
error 1.22e-5 under the unchanged 0.001 gate.

Both 64-step runs fail with CUDA OOM. The result is an opt-in recurrence
prototype, not qualified encrypted long-text generation. The full-model CPU
domain audit finds 74 sites outside the frozen short-generation intervals,
beginning at step 8.

| Artifact | Purpose |
| --- | --- |
| [`decision.json`](decision.json) | Retain the prototype; preserve the short-generation factored default |
| [`comparison.json`](comparison.json) | Matched timings, errors, carry sizes and failed arms; `passed: false` reflects the 64-step failures |
| [`measurements/`](measurements/) | Native results, actual crypto parameters and source/input-bound run records |
| [`payload/`](payload/) | Export manifests, per-step operation counts and retained outputs |
| [`trace/`](trace/) | Identical captured recurrence inputs for both representations; plaintext generation provenance |
| [`domain-audit/audit.json`](domain-audit/audit.json) | Exact full-model replay against frozen nonlinear domains |
| [`short-generation-manifest.json`](short-generation-manifest.json) | Frozen polynomial definitions used by that audit |
| [`static-memory.json`](static-memory.json) | Source-order DAG inventory; excludes runtime frontier ordering and GPU scratch |
| [`allocation-symbolization.json`](allocation-symbolization.json) | Failed allocation stacks, without attributing all retained memory to that allocation |
| [`excluded-measurements.json`](excluded-measurements.json), [`excluded/`](excluded/) | First instrumentation attempts; readout pruning makes their timings invalid |
| [`qualification.json`](qualification.json) | CPU test/lint commands and measured source identities |
| [`publication.json`](publication.json), [`provenance.tar.gz`](provenance.tar.gz) | Original/published hashes; logs and necessary measured-source snapshots |

The dependency exits zero on these CUDA errors. Missing native results are
failures regardless of process exit status. Failed pairs have no speedup value;
successful samples must independently pass the shared security/accuracy contract.
There is one process per representation and length, not a statistical comparison.
GPU inference was measured on one card; multi-GPU execution is not qualified here.

## Verify and recompute the report

From the repository root:

```bash
uv run --no-sync python -m fhemamba.benchmarks verify \
  results/b300/2026-09-27/recurrent-state

uv run --no-sync python -m fhemamba.benchmarks extract \
  results/b300/2026-09-27/recurrent-state runs/recurrence-review
cp -R results/b300/2026-09-27/recurrent-state/measurements/. \
  runs/recurrence-review/measurements/
uv run --no-sync python -m fhemamba recurrent analyze \
  --payload results/b300/2026-09-27/recurrent-state/payload \
  --results runs/recurrence-review/measurements --lengths 4 16 40 64 \
  --output runs/recurrence-review/comparison.json
```

The extracted logs let the analyzer identify the CUDA OOM failures while the
public JSON supplies their input bindings and successful measurements. This
rechecks recorded evidence; it does not execute FHE. For new encrypted runs,
export payloads from the provided trace and follow the maintained probe's
classical-128 runner requirements. Native build provenance is shared with the
[GPU plaintext FFT study](../gpu-plaintext-fft/README.md).

Machine identities and local paths are publication substitutions. Numerical
arrays are unchanged. Both original and published hashes are retained; loose
operational scripts are excluded rather than archived as supported interfaces.
