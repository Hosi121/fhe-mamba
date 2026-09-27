# Research studies

Dated technical reports record the assumptions and evidence available during each study.
Use the [evidence registry](../evidence.md) for current claim status and the
[reproduction guide](../reproducing.md) for portable commands.

For reusable code, start with [optimization mechanisms](../optimizations.md)
and the [experiment workflow](../experiments.md). Session diaries, infrastructure
access details and working scripts are not maintained documentation. Historical
logs and measured source snapshots are in the results provenance bundle;
superseded controllers are removed. Public derivatives distinguish original and
published hashes.

## Complete generation and optimization

| Study | Focus |
| --- | --- |
| [2026-09-27 — B300 GPU plaintext FFT](2026-09-27-b300-gpu-plaintext-fft.md) | Shared GPU FFT/rounding through RNS/NTT; exact primitive gates, classical-128 full pair 211.71→165.37 s (21.89%), 41.34 s/generated token; RMSNorm timing attribution |
| [2026-09-27 — B300 refresh correction merging](2026-09-27-b300-refresh-correction.md) | Exact integer merge before extraction; classical-128 full pair 235.84→210.33 s (10.82%), unchanged 484 bootstraps and numerical/token gates |
| [2026-09-27 — B300 classical-128 prototype](2026-09-27-b300-classical128.md) | Audited uniform-ternary QP=3,376 bits at N=131,072; full Mamba-3 numerical/token gates, exact GPU addend qualification and larger mask cache, secure full pair 327.51→235.75 s (28.02%) |
| [2026-09-27 — Sylph and Cachemir review](2026-09-27-sylph-cachemir.md) | Primary-paper review mapped to existing packing and new secure-level scheduling; slim-polynomial applicability screened on all 785 payload nodes; no new speed claim |
| [2026-09-27 — B300 rotation stream chaining](2026-09-27-b300-rotation-stream-chain.md) | Three internal rotation barriers removed; both model/ring exact checks, new Nsight comparison, full ABBA 146.36→141.77 s (3.14%), 35.44 s/generated token; previous refresh gain preserved |
| [2026-09-27 — B300 even-polynomial refresh](2026-09-27-b300-refresh-even-polynomial.md) | Optional dependency patch; encrypted seed/refresh/level gates pass, full single pair 148.36→145.96 s (1.62%), refresh 37.77→34.66 s (8.22%), 36.49 s/generated token; three further candidates screened |
| [2026-09-27 — Refresh polynomial symmetry](2026-09-27-refresh-even-polynomial.md) | Static design screen: phase shift exposes an even polynomial; square + degree 44 replaces degree 88, seed difference bounded by 2.431e-14 over [-1,1], EvalMod products 24→20; static stage; encrypted follow-up above |
| [2026-09-27 — B300 public-weight inverse FFT reuse](2026-09-27-b300-ifft-reuse.md) | 80% hit rate with 2.502 GiB retained; exact gates pass, cold full single pair 148.32 → 144.10 s (2.85%); below the predeclared 5% promotion gate, prototype archived and active sources restored |
| [2026-09-27 — B300 fused RNS/NTT feed](2026-09-27-b300-fused-feed.md) | Two-worker control, 960 exact and 36 encrypted primitive cases; full single pair 147.72 → 145.51 s (1.50%); patch retained without promotion, active two-worker source restored |
| [2026-09-27 — B300 parallel plaintext preparation](2026-09-27-b300-parallel-prefetch.md) | Two ordered workers within the same two-item window; full ABBA 173.16 → 148.80 s (14.07%), 37.20 s/generated token; verified CPU placement, 960 exact and 36 encrypted primitive cases |
| [2026-09-26 — B300 NTT warp-shuffle tail](2026-09-26-b300-ntt-warp-tail.md) | One GPU candidate rejected: 720 exact kernel cases pass, but matched prefix ABBA 8.712 → 8.912 s (2.31% slower); full comparison gated off, Nsight diagnosis retained |
| [2026-09-26 — B300 RNS fusion and CPU preparation](2026-09-26-b300-rns-pipeline.md) | RNS fusion not adopted; bounded CPU pipeline full ABBA 234.84 → 202.33 s (13.84%), unchanged gates; exact probes, three Nsight prefixes and critical-path bounds with CPU variation retained |
| [2026-09-26 — B300 GPU throughput design](2026-09-26-b300-gpu-throughput-design.md) | Static follow-up: derive 1.37M expansion launches; all-modulus batching, preparation overlap and NTT shared-memory candidates; no new GPU run or speed claim |
| [2026-09-26 — B300 Nsight and inverse-FFT plan](2026-09-26-b300-nsight-ifft.md) | Systems/Compute profiling; exact public twiddle plan, 249 bitwise FFT cases and 251.7M coefficient words; full B300 ABBA 266.22 → 234.84 s (11.79%), optional dependency patch |
| [2026-09-26 — GPU ordinary/refresh ring switching](2026-09-26-gpu-dual-ring.md) | One GPU dual-ring mechanism qualified and adopted opt-in; full Mamba-3 ABBA 613.83 → 430.51 s (29.86%), same accuracy/token gates and large-ring refresh circuit |
| [2026-09-25 — Four structural trials](2026-09-25-structural-four.md) | Four mechanisms implemented; rotation/basis sharing adopted opt-in, full ABBA mean 619.41 → 614.11 s (0.86%); nominal 59-bit 32-bit profile fails, CPU ring-switch route is uncompetitive |
| [2026-09-25 — GPU plaintext RNS expansion](2026-09-25-gpu-rns.md) | One shared preparation mechanism; full Mamba-3 750.41 → 618.95 s (17.52%), ordinary evaluation −23.96%; S2C-first retained, 720 exact-RNS cases, opt-in adoption |
| [2026-09-25 — S2C-first refresh](2026-09-25-s2c-first.md) | One GPU circuit implemented; full Mamba-3 759.70 → 747.63 s (1.59%); refresh saves 115.00 s but ordinary work grows 102.92 s; opt-in, same accuracy/token gates |
| [2026-09-25 — First-principles review](2026-09-25-first-principles.md) | 70.32% of polynomial products process at most 32 values; bootstrap ordering, 32-bit RNS and dual-ring candidates; exact-coefficient algebra checks and rejection criteria; no new GPU speed claim |
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
