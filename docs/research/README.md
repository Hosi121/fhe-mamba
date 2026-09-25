# Research studies

Dated notes record the assumptions and evidence available during each study.
Use the [evidence registry](../evidence.md) for current claim status and the
[reproduction guide](../reproducing.md) for portable commands.

## Complete generation and optimization

| Study | Focus |
| --- | --- |
| [2026-09-25 — Ready-node refresh scheduling](2026-09-25-packed-frontiers.md) | Three mechanisms tested; full Mamba-3 941.95 → 761.00 s (19.21%); scheduler adopted, polynomial/layout prototypes archived; 20% target unmet |
| [2026-09-25 — Shared square dispatch](2026-09-25-square-dispatch.md) | 72 new exact-RNS cases; separate-binary ABBA prefixes improve 0.84% for Mamba-3 and 0.88% for Mamba-2; isolated prefix evidence; included in the later refresh-study baseline |
| [2026-09-25 — Arithmetic and kernel opportunities](2026-09-25-math-kernel-audit.md) | Static inventory: 4,386 hidden squares, 465 shared-basis products, and an optimistic 9,991 → 5,980 polynomial-core product count from SIMD grouping; no new speed measurement |
| [2026-09-25 — Compact coefficient cache, rejected](2026-09-25-weight-coefficient-cache.md) | Exact CPU/GPU reconstruction passes, but level-dependent reuse gives 1.26% slower cold three-step inference; prototype archived, active code restored |
| [2026-09-24 — Shared ciphertext ownership](2026-09-24-owned-arithmetic.md) | Mamba-3 16.27 → 15.82 min (2.79%, adopt); Mamba-2 32.66 → 32.38 min (0.83%, adopt); exact-RNS gates and unchanged model contracts |
| [2026-09-25 — Preparation and lifetime design](2026-09-25-preparation-design.md) | Static cache admission curve: 2 GiB LRU gives zero identity hits versus 16,384 for a fixed subset; CPU/GPU ownership and refresh scheduling constraints |
| [2026-09-25 — Encode range scan](2026-09-25-encoding-range.md) | Local CPU prototype: 167,772,160 exact coefficient comparisons, 2.31–3.92% full-encode reduction; not applied to DGX or model inference |
| [2026-09-24 — Cache integration](2026-09-24-packed-cache-integration.md) | Existing 64-entry cache with the same optimized binary; prefix ABBA 1.35% reduction, full qualification and adoption decision |
| [2026-09-24 — Borrowed upload and routing](2026-09-24-borrowed-plaintext.md) | Mamba-3 matched full 17.36 → 16.74 minutes (3.53% reduction); 9,120 fewer model rotations; Mamba-2 ABBA 2.73% reduction and full parity |
| [2026-09-24 — Static waste audit](2026-09-24-static-waste-audit.md) | Remaining internal ciphertext copies, repeated weight preparation and synchronization; source-bound counts, no new speedup claim |
| [2026-09-24 — Coefficient ownership](2026-09-24-coefficient-ownership.md) | Mamba-3 same-binary prefix −1.80%, full parity in 17.31 minutes; Mamba-2 +0.48%, so its full follow-up is skipped |
| [2026-09-24 — Packed resources and direct upload](2026-09-24-packed-resources.md) | Mamba-3 matched full generation 18.58 → 17.58 minutes (−5.39%); lossless matrix storage 673.3 → 168.3 MiB; Mamba-2 ABBA −5.56% and full parity |
| [2026-09-24 — Shared plaintext preparation](2026-09-24-shared-plaintext-preparation.md) | Mamba-2 matched full generation: 38.64 → 35.70 minutes (−7.6%); Mamba-3 full regression passes |
| [2026-09-24 — Mamba-3 GPU plaintext encoding](2026-09-24-mamba3-gpu-encoding.md) | Exact RNS parity and matched full generation: 21.87 → 18.59 minutes (−15.0%) |
| [2026-09-24 — Mamba-3 plaintext cache](2026-09-24-mamba3-plaintext-cache.md) | Bounded encoded-mask reuse: matched prefix ABBA −1.4%, unchanged error gates |
| [2026-09-24 — Mamba-3 microkernels](2026-09-24-mamba3-microkernels.md) | ARM SIMD mask copies, CUDA/host profile and matched GPU scratch reuse |
| [2026-09-24 — Mamba-3 depth and batch refresh](2026-09-24-mamba3-depth-batching.md) | Matched full 187M generation: 46.20 → 24.95 minutes, unchanged accuracy gates |
| [2026-09-24 — Mamba-3 generation and routing](2026-09-24-mamba3-trained-generation.md) | Complete 187M encrypted generation and matched slot-routing optimization |
| [2026-09-22 — Client generation](2026-09-22-client-generation.md) | First complete stabilized prompt-to-text run |
| [2026-09-22 — Periodic coefficients](2026-09-22-periodic-gate-coefficients.md) | Reusing repeated joint-gate coefficient patterns |
| [2026-09-22 — Subring encoding](2026-09-22-subring-gate-encoding.md) | Historical 38.5-minute candidate and matched controls |
| [2026-09-22 — Generation cost bounds](2026-09-22-generation-cost-bounds.md) | Runtime decomposition and conditional acceleration scenarios |
| [2026-09-22 — Normalization bounds](2026-09-22-normalization-bounds.md) | Necessary polynomial degree and multiplication-depth bounds |

## Polynomial design and native integration

| Study | Focus |
| --- | --- |
| [2026-09-24 — Mamba-3 SISO](2026-09-24-mamba3-siso.md) | Shared mixer formula, upstream parity, 4/8-step encrypted execution and refresh probe |
| [2026-09-24 — Trained Mamba-3 preflight](2026-09-24-mamba3-trained-preflight.md) | Complete 187M CPU generation, exact state factorization and encrypted client-loop component |
| [2026-09-21 — SSM cryptographic design](2026-09-21-ssm-cryptographic-design.md) | State algebra, domain failures, joint gates and normalization certificates |
| [2026-09-21 — Native normalization](2026-09-21-normalization-native.md) | Isolated encrypted inverse-square-root schedules |
| [2026-09-21 — Vector RMS](2026-09-21-vector-rms.md) | Vector-level RMSNorm and refresh experiments |
| [2026-09-21 — Normalization integration](2026-09-21-normalization-integration.md) | Scheduled RMSNorm in the full model |
| [2026-09-21 — Stabilized native circuit](2026-09-21-stabilized-native.md) | Complete joint-gate/activation payload and carried-state validation |

## Earlier surveys

- [2026-07-13 — FHE Mamba bottlenecks](2026-07-13-fhe-mamba-bottleneck-survey.md)
- [2026-07-12 — Mamba and FHE landscape](2026-07-12-mamba-fhe-landscape.md)

Historical plans and the B200 probe log are in the [archive](../archive/README.md).
Numerical parity, approximation quality, cryptographic security and runtime
claims have separate acceptance criteria; see [research validation](../validation.md).
