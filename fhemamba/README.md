# FHE Mamba implementation

This is the active Python/PyTorch reference, polynomial substitution, packing,
payload and experiment code for **Mamba-2-130M**. The native encrypted kernel
lives in [`../native/fideslib_stage0/`](../native/fideslib_stage0/).

Current status and claim boundaries are maintained in the
[root README](../README.md) and [evidence registry](../docs/evidence.md).
Use the [design](../docs/design.md), [roadmap](../docs/roadmap.md), and
[DGX Spark runbook](../docs/dgx-spark.md) for new work. `DESIGN.md` is historical.
B300 is currently unavailable; its old manifests are retained for reproduction.

## Design rules

1. `reference.py` is the single Mamba formula; exact, recording and polynomial
   operations are injected through `Ops`.
2. Measure checkpoint-to-polynomial quality separately from CKKS-to-polynomial
   error. Do not clamp live CKKS values or feed diagnostics back into execution.
3. Validate state/FIFO bounds on calibration text separate from test prompts.
4. Slot layouts, operation counts, key selection and native evaluation must
   agree. Candidate promotion requires multi-token checks across keys and prompts.
5. A serial carry between prompt chunks is not logarithmic in prompt length.
   The prefill budget is a structural model; native encrypted prefill is open.
6. The client owns the secret key. One-process benchmarks do not prove a
   secret-key-free server or a complete secure protocol.

## Layout

| Module | Responsibility |
|---|---|
| `src/fhemamba/reference.py` | Stateful Mamba-1/Mamba-2 formula and compact affine scan |
| `src/fhemamba/ops.py` | Exact operations, ranges, polynomial/Newton approximations |
| `src/fhemamba/lowering.py` | Decode operation and level schedule |
| `src/fhemamba/m1_payload.py` | Calibrated checkpoint payload and reference export |
| `src/fhemamba/generation.py` | Full-prompt preparation and measured-token text reports |
| `src/fhemamba/bsgs_layout.py` | Replicated/interleaved projections and binary rotate-add oracle |
| `src/fhemamba/state_layout.py` | Recurrent-state layout and refresh plans |
| `src/fhemamba/rotation_keys.py` | Direct/composite rotation-key planning |
| `src/fhemamba/prefill_budget.py` | Explicitly hypothetical prompt-processing budget |
| `src/fhemamba/ssm_algebra.py` | Deferred-state and complex-trapezoid research oracles |
| `src/fhemamba/phase_algebra.py` | Three-shear phase candidate with stability counterexamples |
| `src/fhemamba/polynomial_certificate.py` | Rational Bernstein certificates for Newton basins and coupled state gates |
| `experiments/` | Reproducible commands/manifests and build provenance |
| `results/` | Small curated artifacts; large payloads are ignored |

From the repository root, run `scripts/run_fast_checks.sh` during development
and `scripts/run_checks.sh` for coverage. Python checkpoint experiments require
`uv sync --extra dev`; the native Spark campaign uses Python's standard library.

## Campaigns

`run_dgx_generation.py` runs the stabilized prompt-to-text path from the Python
host: prepare a frozen payload copy, invoke Spark, collect measured IDs and
decode the completion. It consumes the whole prompt and rejects domain escapes
before launch. See the [command and output files](../docs/dgx-spark.md#complete-prompt-to-text-generation).
The [first complete run](../docs/research/2026-09-22-client-generation.md) passes
five evaluations and four generated tokens at maximum polynomial-circuit error
0.009192. Preserve this functional baseline while optimizing its runtime.
The [subring encoding follow-up](../docs/research/2026-09-22-subring-gate-encoding.md)
passes the same request in **38.5 minutes**, with unchanged generated IDs,
operation counts and levels. Enable both `--joint-periodic-coefficients` and
`--joint-subring-encoding` against its matching isolated build.

`run_dgx_campaign.py` records per-experiment environments, accepts portable
`--env KEY=VALUE` path overrides, checks resource availability, and supports
`--resume` when commit, binary, geometry, configuration and prior report match.
Spark manifests also resolve the binary and bind dependencies and payload
contents to hashes before resume. A manifest's
`acceptance` object makes numerical failures fail the campaign as well.

Use `dgx_spark_replication_ab.json` for matched 24-layer projection experiments
and `dgx_spark_autoregressive.json` for the five-step candidate gate. Read the
[Spark runbook](../docs/dgx-spark.md) before using old payloads on the host.
`dgx_spark_row_state_ab.json` compares group/row coordinate scales on the same
independently recalibrated payload. `run_payload_quality.py` and
`diagnose_payload_domain.py` check plaintext quality separately; a native
polynomial-parity pass cannot repair a non-finite polynomial circuit.

Client decryption of completed final-normalization output for token selection
is a protocol boundary. Debug layer decryptions and diagnostic client
re-encryption are counted separately and cannot satisfy the zero-intermediate-
decrypt gate.
