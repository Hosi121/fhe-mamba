# Recorded results

Curated execution artifacts and derived reports live here. Read the
[evidence registry](../docs/evidence.md) for the claims each result supports,
its security/execution scope, and failed controls. Raw measurements retain
their original bytes, input paths, commits and hashes.

## Complete prompt-to-text workload

These directories record the same 24-layer, five-evaluation request with
four generated tokens. Each contains raw backend output, campaign records and
text/comparison reports.

| Run | Recorded files | Study |
| --- | --- | --- |
| Initial pipeline | [`client-generation/`](dgx/2026-09-22/client-generation/) | [Baseline](../docs/research/2026-09-22-client-generation.md) |
| Periodic coefficients | [`periodic-gates/`](dgx/2026-09-22/periodic-gates/) | [Periodic encoding](../docs/research/2026-09-22-periodic-gate-coefficients.md) |
| Subring encoding | [`subring-gates/`](dgx/2026-09-22/subring-gates/) | [Subring encoding](../docs/research/2026-09-22-subring-gate-encoding.md) |
| Shared plaintext preparation | [`shared-plaintext-preparation/`](dgx/2026-09-24/shared-plaintext-preparation/) | [Mamba-2 port and Mamba-3 regression](../docs/research/2026-09-24-shared-plaintext-preparation.md) |
| Direct plaintext upload | [`packed-resources/m2-full-direct/`](dgx/2026-09-24/packed-resources/m2-full-direct/) | [Shared direct upload and packed resources](../docs/research/2026-09-24-packed-resources.md) |
| Borrowed plaintext upload | [`borrowed-plaintext/m2-full-borrow/`](dgx/2026-09-24/borrowed-plaintext/m2-full-borrow/) | [Borrowed upload and routing](../docs/research/2026-09-24-borrowed-plaintext.md) |
| Shared ciphertext ownership | [`owned-arithmetic/m2-full-candidate/`](dgx/2026-09-24/owned-arithmetic/m2-full-candidate/) | [Ownership comparison](../docs/research/2026-09-24-owned-arithmetic.md) |

Inspect the latest recorded completion without a GPU:

```bash
uv run --no-sync fhemamba validate-artifacts --require-commit \
  results/dgx/2026-09-24/owned-arithmetic/m2-full-candidate/generation.json
```

This validates a recorded artifact; it does not rerun encrypted inference.

## Other evidence

| Location | Contents |
| --- | --- |
| [`dgx/2026-09-25/packed-frontiers/`](dgx/2026-09-25/packed-frontiers/) | Three mechanisms, interleaved prefixes, full-depth selection and final-source qualification; rejected prototypes and Mamba-2 regression retained |
| [`dgx/2026-09-25/square-dispatch/`](dgx/2026-09-25/square-dispatch/) | Shared square dispatch: exact RNS cases and separate-binary prefix controls for both models |
| [`dgx/2026-09-25/weight-coefficient-cache/`](dgx/2026-09-25/weight-coefficient-cache/) | Rejected compact coefficient cache: 120 final exact GPU/RNS cases, two ABBA workloads, source archives and failed attempts; 1.26% slower three-step probe |
| [`dgx/2026-09-24/owned-arithmetic/`](dgx/2026-09-24/owned-arithmetic/) | Shared scratch ownership: 192 exact-RNS cases, both prefix ABBA controls and qualified fresh full pairs; failed probe revisions retained |
| [`cpu/2026-09-25/preparation-design/`](cpu/2026-09-25/preparation-design/) | Static public-weight access trace, LRU/fixed-subset calculations, existing profile analysis and source-bound integration constraints; no new speed test |
| [`cpu/2026-09-25/encoding-range/`](cpu/2026-09-25/encoding-range/) | Actual local OpenFHE encoder comparison: exact coefficients/metadata, two frozen library variants, all eight process samples and portable reproduction recipe; no DGX/model claim |
| [`dgx/2026-09-24/packed-cache-integration/`](dgx/2026-09-24/packed-cache-integration/) | Existing bounded cache composed with optimized upload/routing; unchanged executable, prefix ABBA and qualified full follow-up |
| [`dgx/2026-09-24/borrowed-plaintext/`](dgx/2026-09-24/borrowed-plaintext/) | Both exact-RNS configurations, separate/combined controls, full Mamba-3 pair and Mamba-2 parity |
| [`dgx/2026-09-24/static-waste-audit/`](dgx/2026-09-24/static-waste-audit/) | Static call inventory, remaining copy/encoding/synchronization candidates and unchanged prior measurements; no new speedup claim |
| [`dgx/2026-09-24/coefficient-ownership/`](dgx/2026-09-24/coefficient-ownership/) | Shared coefficient moves: Mamba-3 prefix −1.80% and full parity in 17.31 minutes; Mamba-2 non-improvement retained |
| [`dgx/2026-09-24/packed-resources/`](dgx/2026-09-24/packed-resources/) | Mamba-3 full 18.58→17.58 minutes; NAF/lifetime/upload/storage controls, Mamba-2 ABBA/full parity and CPU state backing-storage check |
| [`dgx/2026-09-24/mamba3-gpu-encoding/`](dgx/2026-09-24/mamba3-gpu-encoding/) | Same-binary full generation: 21.87 → 18.59 minutes; exact GPU RNS probes, matched prefix, failed allocation and frozen sources |
| [`dgx/2026-09-24/mamba3-trained-generation/`](dgx/2026-09-24/mamba3-trained-generation/) | Complete trained 187M encrypted generation and matched one-layer routing comparison |
| [`dgx/2026-09-24/mamba3-trained-preflight/`](dgx/2026-09-24/mamba3-trained-preflight/) | Historical preflight: trained 187M CPU parity, one-layer encrypted client loop and matrix-operation comparison |
| [`dgx/2026-09-24/mamba3-plaintext-cache/`](dgx/2026-09-24/mamba3-plaintext-cache/) | Encoded-mask cache ABBA, carried-state regression, source archive and budget |
| [`dgx/2026-09-24/mamba3-microkernels/`](dgx/2026-09-24/mamba3-microkernels/) | ARM SIMD copy samples, CUDA profile, GPU scratch reuse ABBA and accuracy gates |
| [`dgx/2026-09-24/mamba3-depth-batching/`](dgx/2026-09-24/mamba3-depth-batching/) | Matched full 187M generation speedup, depth/batching probes and rejected one-pass refresh |
| [`dgx/2026-09-24/mamba3-siso/`](dgx/2026-09-24/mamba3-siso/) | Complete synthetic Mamba-3 mixer, 4/8-step encrypted probes, upstream parity and refresh arithmetic |
| [`dgx/2026-09-21/`](dgx/2026-09-21/) | Spark integration, normalization, vector RMS, state and diagnostic experiments |
| [`dgx/`](dgx/) | Earlier encrypted probes and feasibility runs |
| [`b300/`](b300/) | Historical B300 artifacts, including failed full-chain controls |
| Top-level JSON files | Plaintext parity, polynomial quality, certificates, layouts and derived budgets |
| [`archive/`](archive/README.md) | Outputs from the retired implementation, previously tracked under `runs/` |

The current model revision and frozen public coefficients live in
[`config/`](../config/README.md). Older quality reports may use different
coefficients; they do not automatically validate the latest native payload.
Some legacy JSON predates the current provenance schema.

## Recording new work

Write new local output under ignored `runs/<experiment>/`. Promote only reviewed,
claim-bearing evidence here and link it from the corresponding study. Do not
edit a recorded result to match a new source or payload hash, and do not drop
failed controls. See [contribution requirements](../CONTRIBUTING.md#benchmark-artifacts)
and the [old-to-new path map](../docs/repository.md#previous-layout).
