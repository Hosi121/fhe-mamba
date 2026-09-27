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
| [`b300/2026-09-27/security128/`](b300/2026-09-27/security128/) | Classical-128 parameter audit and frozen full Mamba-3 qualification; GPU addends / larger public-mask cache, 327.51→235.75 s (28.02%); exact probes, parameter rejections, digit/cache screens and Nsight evidence |
| [`cpu/2026-09-27/sylph-cachemir-screen/`](cpu/2026-09-27/sylph-cachemir-screen/) | Source-paper identities and reproducible frozen-program applicability screen; 785 polynomial nodes and a float64 parity decomposition check, with no encrypted performance claim |
| [`b300/2026-09-27/rotation-stream-chain/`](b300/2026-09-27/rotation-stream-chain/) | Optional three-barrier rotation patch; exact rotation/refresh qualification, full ABBA 146.36→141.77 s (3.14%), fresh Nsight pair, source/archive audits and remaining bottleneck analysis |
| [`b300/2026-09-27/refresh-even-polynomial/`](b300/2026-09-27/refresh-even-polynomial/) | Optional even bootstrap seed; separate libraries and encrypted qualification, full single pair 1.62% faster with refresh 8.22% faster; recorded errors/levels, existing Nsight attribution, three static follow-up candidates |
| [`b300/2026-09-27/ifft-reuse/`](b300/2026-09-27/ifft-reuse/) | Public-weight inverse FFT cache, exact probes and cold prefix/full pairs; 2.85% full improvement falls below predeclared 5% promotion gate; candidate patch retained, active sources restored |
| [`b300/2026-09-27/polynomial-degrees/`](b300/2026-09-27/polynomial-degrees/) | Frozen full Mamba-3 payload inventory: maximum Chebyshev degree 1,023 at the tenth layer's `m3_negative_a`; manifest/program hashes and reproducible read-only audit |
| [`b300/2026-09-27/fused-feed/`](b300/2026-09-27/fused-feed/) | Fused RNS/NTT plus persistent-workspace completion change; qualified full single pair 1.50% faster, not promoted; exact probes, candidate Nsight trace, failed build and final source-restoration record |
| [`b300/2026-09-27/parallel-prefetch/`](b300/2026-09-27/parallel-prefetch/) | Ordered two-worker preparation; full ABBA 173.16 → 148.80 s (14.07%), 37.20 s/generated token; fixed/observed CPU placement, exact qualification and unchanged GPU kernel counts |
| [`b300/2026-09-26/ntt-warp-tail/`](b300/2026-09-26/ntt-warp-tail/) | Forward NTT shared-memory/register experiment; 720 exact kernel cases and 960 RNS cases pass, prefix ABBA 2.31% slower; candidate not adopted, full comparison gated off |
| [`b300/2026-09-26/rns-pipeline/`](b300/2026-09-26/rns-pipeline/) | Two-stage RNS fusion / bounded CPU preparation trial; 4,080 exact and 153 encrypted primitive cases, eight prefixes and eight full controls; CPU pipeline 234.84 → 202.33 s (13.84%); negative fusion block, clock observations and three Nsight traces retained |
| [`b300/2026-09-26/nsight-ifft/`](b300/2026-09-26/nsight-ifft/) | Nsight Systems/Compute analysis, exact inverse-FFT plan, eight prefix and four full controls; B300 Mamba-3 mean 266.22 → 234.84 s (11.79%), unchanged gates; first noisy prefix block retained |
| [`dgx/2026-09-26/gpu-dual-ring/`](dgx/2026-09-26/gpu-dual-ring/) | GPU N=32,768 ordinary / N=65,536 refresh bridge; primitive and prefix gates, full ABBA (29.86% reduction), default-path regression and verified source archives |
| [`dgx/2026-09-25/structural-four/`](dgx/2026-09-25/structural-four/) | Four mechanisms, rotation/basis qualification, four full ABBA processes (0.86% reduction), and rejected 32-bit/CPU ring-switch prototypes with controls |
| [`dgx/2026-09-25/gpu-rns/`](dgx/2026-09-25/gpu-rns/) | Exact GPU plaintext RNS expansion: 720 exact cases, encrypted arithmetic, ABBA prefixes and qualified full pair; 17.52% full Mamba-3 reduction with S2C-first retained |
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
| [`b300/2026-09-26/mamba3-dual-ring/`](b300/2026-09-26/mamba3-dual-ring/) | Trained Mamba-3 on CUDA 13 / SM103: primitive qualification and a complete 265.16-second generation run; hardware comparison, not an isolated code speedup |
| [`b300/`](b300/) | B300 measurements and historical failed full-chain controls |
| Top-level JSON files | Plaintext parity, polynomial quality, certificates, layouts and derived budgets |
| [`archive/`](archive/README.md) | Outputs from the retired implementation, previously tracked under `runs/` |

The current model revision and frozen public coefficients live in
[`config/`](../config/README.md). Older quality reports may use different
coefficients; they do not automatically validate the latest native payload.
Some legacy JSON predates the current provenance schema.

## Archived provenance

This checkout contains public derivatives of the recorded evidence. Numerical values and
failure statuses are preserved; private environment identifiers are normalized.
Relevant logs, build records and measured source snapshots are in
[`provenance.tar.gz`](provenance.tar.gz), indexed by
[`publication.json`](publication.json). Original and published hashes are separate.
Disposable controllers, watchers and cleanup scripts have been removed, including
their archive copies. Shared tools live in `src/fhemamba/benchmarks/`.
Opaque profiler recordings remain local; their identities and public text summaries are retained.

```bash
python -m fhemamba.benchmarks verify results
python -m fhemamba.benchmarks extract results runs/evidence-inspection
```

See the [experiment workflow](../docs/experiments.md) for reusable execution,
completion events and publication. This cleanup does not rewrite earlier Git history.

## Recording new work

Write new local output under ignored `runs/<experiment>/`. Publish reviewed derivatives here and link them from the corresponding study.
Keep raw bytes locally, record original and public hashes, and retain failed controls. See [contribution requirements](../CONTRIBUTING.md#benchmark-artifacts)
and the [old-to-new path map](../docs/repository.md#previous-layout).
