# Fixed-size recurrent state qualification

This probe compares Mamba-3's exact history factorization with a fixed-size state
split over ciphertexts along the head axis. Neither path drops history, lowers
the state rank, or changes the mathematical recurrence. The shared implementation
is `mamba3_state_step`; its tiled update is also used by the full mixer.

The plaintext producer runs the pinned trained model autoregressively and captures
one layer immediately before its state update. Both FHE arms encrypt exactly those
same keys, queries, values and recurrence coefficients. This isolates the state
representation from projections, polynomial nonlinearities and token feedback.
**It is a recurrence benchmark, not encrypted long-text generation.**

The [first classical-128 study](../../docs/research/2026-09-27-recurrent-state.md)
records successful short comparisons, resource failures and the full-model domain
audit. Fixed recurrent storage alone does not bound the executor's working set.
The [scheduler follow-up](../../docs/research/2026-09-27-recurrent-memory.md)
qualifies both representations at 64 steps by limiting ready-node lookahead.

## Shared implementation

`PackedProgram.recurrent_state` splits a `(batch, heads, channels, state)` tensor
into complete-head blocks. One head must fit in the program's logical slot
capacity; a final partial head group is supported. `HeadTiledState` and
`ssm_update_readout` also operate on Torch tensors for reference checks.

To lower the full mixer with fixed state, construct its carry as follows, then
call the existing `mamba3_step` with that carry and the packed algebra:

```python
reference = init_mamba3_state(mixer)
carry = Mamba3State(
    program.constant(reference.angle),
    program.recurrent_state(reference.ssm),
    program.constant(reference.k),
    program.constant(reference.v),
)
```

The Python tests cover complete mixer execution with this storage, including a
state larger than one ciphertext. The FHE probe below qualifies the recurrent
subcircuit separately; full-model encrypted qualification remains a later gate.

## Capture and export

Install the experiment dependencies and download the pinned checkpoint/tokenizer
as described in [the Mamba-3 guide](../../docs/mamba3.md). Paths below are examples;
the script verifies the file hashes in `config/mamba3-reproduction.json`.

```bash
python -m fhemamba recurrent probe capture \
  --checkpoint /path/to/checkpoint --tokenizer /path/to/tokenizer \
  --output runs/recurrence/trace --steps 64 --layer 0 --prompt 'The capital'

for steps in 4 16 40 64; do
  for representation in factored tiled; do
    python -m fhemamba recurrent probe export \
      --trace runs/recurrence/trace --steps "$steps" \
      --representation "$representation" \
      --output "runs/recurrence/$representation-$steps"
  done
done
```

The export checks both representations against an independent dense NumPy
recurrence. It records per-step operation counts, state sizes and output node IDs.
Every readout is retained for validation, preventing dead-node elimination of
intermediate readout work. The tiled arm also checks every coordinate of its final
state. Every output length uses a prefix of the same captured trace.

Run the payloads with the qualified `packed_fideslib` binary, preserving the
classical-128 context, two-pass refresh, S2C-first schedule and accuracy thresholds
of 0.001. Use the generic `python -m fhemamba.benchmarks run` job runner to record
deadlines, dependency/input hashes and completion events. The comparison expects
one directory per arm and length, containing `run.json` and, on success,
`native.json`. Preserve failed runs and their logs too.
Job `inputs` must bind `binary`, `fideslib`, `openfhe_core`, `source`, `native_source`,
`trace`, `program`, `fixture` and `manifest`; the first six must match across runs.
`source` is the probe source snapshot; `native_source` identifies the qualified
native implementation. See [experiment tooling](../../docs/experiments.md) for
the portable runner interface.

```bash
python -m fhemamba recurrent analyze \
  --payload runs/recurrence --results /path/to/measurements \
  --lengths 4 16 40 64 --output runs/recurrence/comparison.json
```

The analyzer checks correctness, security, runtime settings and provenance before
reporting timing differences. Slower short sequences remain in the comparison.
Setup and full process time are reported separately from evaluation time. Input
encryption is substantial at the secure geometry. The analyzer therefore also
reports state arithmetic as the sum of operation-profile times excluding `input`
and `public`; their materialization times remain visible separately. Model
projections would normally supply these inputs as existing ciphertexts.
An unsuccessful process or missing native result makes that arm unqualified,
even when the process exits zero. Incomplete pairs have no speedup value. The
report's `passed` field is false if any requested arm failed; completed arms
must still satisfy every accuracy and cryptographic check. At least one completed
run of each representation is required to validate their shared context.

## Audit the existing polynomial domains

An independently frozen short-generation manifest can be screened against the
same plaintext trace before attempting a longer full-model export:

```bash
python -m fhemamba recurrent probe audit-domains \
  --checkpoint /path/to/checkpoint --trace runs/recurrence/trace \
  --polynomial-manifest /path/to/short-generation/manifest.json \
  --output runs/recurrence/domain-audit.json
```

This checks every nonlinear site without evaluating or refitting its polynomial.
Failure prevents reuse of those domains; passing is only a necessary condition
and does not establish accuracy of the approximate or encrypted recurrence.

## Qualify scheduler memory admission

With `--frontier-refresh`, the native executor can skip a branch that needs
refresh and prepare later independent work. `--frontier-live-limit N` limits
that lookahead: when at least N DAG values are retained, it selects the oldest
ready branch and performs its normal refresh. Zero (the default) preserves the
existing scheduler. The option does not change the ciphertext parameters,
refresh implementation, state rank or retained validation outputs.

This is a **soft admission threshold**, not a memory allocator limit. Required
source-order work and pinned outputs may exceed it. The native result records
`maximum_live_dag_values` and `frontier_limit_selections`. With
`--profile-evaluation`, every ten completed nodes also report the current live
values, plaintext-cache entries and device memory; `sampled_peak_device_bytes`
is the largest device usage observed at those points. It includes keys, caches,
allocator pools and other device allocations. It excludes unsampled transients
inside operations and is not a breakdown of memory ownership.

Use the same binary, payload, device, CPU placement and flags in each control /
candidate pair, including explicit `--frontier-live-limit 0` for the control.
A study JSON selects the cases and marks which pairs may be compared for speed:

```json
{
  "schema_version": 1,
  "cases": [
    {"name": "control", "mode": "tiled", "steps": 64, "limit": 0},
    {"name": "candidate", "mode": "tiled", "steps": 64, "limit": 128}
  ],
  "pairs": [
    {"baseline": "control", "candidate": "candidate", "compare_timing": true}
  ]
}
```

Each results directory contains `run.json`, `run.log` and, if completed,
`native.json`. The analyzer binds the original payload manifests and accepts
different container names/output paths, while rejecting other command or device
changes besides the admission threshold:

```bash
python -m fhemamba recurrent analyze \
  --payload runs/recurrence --results /path/to/measurements \
  --schedule-study /path/to/study.json --output runs/recurrence/memory.json
```

Every completed sample must pass the same classical-128 and 0.001 error gates,
including candidates whose controls failed. Missing native results stay in the
report as failures, even after process exit zero; incomplete pairs have no
speedup. Use `compare_timing: false` for concurrent diagnostic runs whose host
contention is not controlled. Admission may trade refresh batching for memory,
so successful completion alone is not evidence of lower latency.

With the optional `matplotlib` package installed, plot the completed-node samples
from the validated report:

```bash
python -m fhemamba recurrent plot \
  --report runs/recurrence/memory.json --results /path/to/measurements \
  --steps 64 --output runs/recurrence/memory.png
```

## Scope of the result

- A fixed number of carried ciphertexts bounds the *recurrent state*. The current
  executor still unrolls a finite DAG and retains validation outputs; this probe
  does not establish constant total host or device memory.
- Inputs are prepared offline. The scheduler can work ahead across steps, so the
  average recurrence cost is not a measured autoregressive token latency.
- Refresh bounds come from this trace with eightfold headroom and a minimum of
  one. They are not certified bounds for arbitrary text.
- Keeping state size fixed does not bound accumulated CKKS error by itself.
  The final-state and intermediate readout checks cover only the measured trace.
- The qualified native binary selects one GPU. Its compact RNS/GPU FFT encoder
  is also qualified for one device; exposing more devices alone does not
  distribute this recurrence. Separate jobs can use different GPUs, with CPU
  placement and contention controlled before comparing timings.
- Full generation additionally needs bounded rotary-phase evaluation, validation
  of all nonlinear domains, actual token feedback and long-horizon quality tests.
