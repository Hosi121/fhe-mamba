# Roadmap

This roadmap describes the active `0.4.x` Mamba-2 line. The canonical,
executable work items live in [backlog.md](backlog.md); evidence provenance is
tracked separately in [evidence.md](evidence.md).

## Objective

Demonstrate privacy-preserving autoregressive inference for the public
`mamba2-130m` checkpoint with:

- client-side embedding and token selection;
- server-side 24-layer Mamba-2 evaluation entirely under CKKS;
- encrypted recurrent state and convolution FIFO carried between token steps;
- reproducible correctness, runtime, memory, and security-parameter artifacts.

Mamba-2 is the active model, not merely a profiling source. Mamba-3-lite is a
separate architecture experiment and must not replace completion of the
Mamba-2 protocol path.

## Milestone 0: evidence closure

The code and documentation record a passing 24-layer, three-token B300 run,
but its raw success JSON is not present in the tracked result set. Before a
release tag or new performance claim:

1. recover the original JSON, or rerun the exact promoted baseline;
2. validate commit, binary SHA-256, configuration, timing, errors, bootstrap
   counts, and peak RSS;
3. store a small curated artifact under `fhemamba/results/b300/`;
4. make the README and evidence registry point to that artifact.

This is provenance work, not a request to reinterpret failed `0.4.4` B300
artifacts as successful runs.

## Milestone 1: five-step autoregressive gate

Run prompt-2/generate-4 assets through five sequential encrypted decode steps
using the promoted B300 configuration:

- 24 layers and final RMSNorm;
- `out_proj`-only fused linear transform;
- complex-paired recurrent-state refresh;
- refresh interval 1;
- shared head expansion disabled;
- fully synchronized FIDESlib build;
- ring `2^16`, scale 59, and `security=not-set`.

Promotion requires every token to decrypt, polynomial-circuit error `<= 0.05`
for every step, matching generated IDs, zero intermediate decrypts, and a
validator-clean artifact with per-token timings and operation counts.

This milestone establishes a longer systems/correctness horizon. It still does
not establish a secure deployment protocol.

## Milestone 2: process-separated full kernel

Promote the existing key-separation probe into the real Mamba kernel:

```text
client-init -> server-eval -> client-decrypt
```

The server role must have no secret-key input or decrypt/debug path. The first
gate may use a fixed encrypted input and serialized recurrent state; a
persistent interactive server can follow once correctness and transfer costs
are measured.

## Milestone 3: 128-bit full chain

Move from the current layer-0/two-token 128-bit result to a 24-layer result with
OpenFHE-accepted 128-bit parameters. Record key memory, setup, evaluation,
transfer size, and numerical error. Return-path noise flooding remains a
separate protocol-security requirement and must not be implied by parameter
selection alone.

## Optimization line

Correctness gates precede promotion. The current order is:

1. replace B300 device-wide key-switch barriers with explicit stream
   dependencies, gated by a full 24-layer multi-token run;
2. re-evaluate all-scope fused projections over the longer session;
3. build an offline global bootstrap-placement planner for residual/projection
   coordination;
4. re-test shared dt/decay head expansion only where setup is amortized;
5. investigate a faster or more accurate bootstrap backend.

Reduced synchronization, all-scope fusion, refresh interval 2, and shared head
expansion remain experiments until the complete gate passes.

## Non-goals before 1.0

- encrypted full-vocabulary argmax;
- support for private model weights;
- claims for arbitrary Mamba checkpoints;
- Mamba-3 replacement of the current checkpoint path;
- performance extrapolation presented as measured runtime.

## Version boundary

- `0.4.x`: real Mamba-2 weights under encrypted execution and bounded
  autoregressive gates.
- `0.5.x`: process-separated full-kernel execution and 128-bit full-chain
  promotion work.
- `1.0.0`: reproducible interactive encrypted generation at 128-bit parameters
  with an explicit protocol-security statement.
