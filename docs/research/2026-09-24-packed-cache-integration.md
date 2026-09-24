# Integrating the bounded plaintext cache

The existing 64-entry encoded-mask cache is tested together with the selected
borrowed-upload/routing options. Its prefix ABBA mean is
**14.689982 → 14.491925 seconds**
(1.35% reduction). This is the existing cache under
the latest composition of optimizations; no native code or binary changes
between the [preceding campaign](2026-09-24-borrowed-plaintext.md) and this one.

[Raw controls, frozen source/binary identities and reproducible comparison](../../results/dgx/2026-09-24/packed-cache-integration/).

## Result and adoption

The full pair completes in **1004.622 →
976.107 seconds** (16.74 →
16.27 minutes, 2.84% reduction).
The final recommendation enables
the cache for this workload. Both runs generate IDs `[315,279,1614,315]`,
`The capital of the state of`, under the same 0.001 exact/polynomial gates.

| Full-model property | Uncached | Cached |
| --- | --- | --- |
| Evaluation seconds | 1004.62162 | 976.106806 |
| Host encoding seconds | 200.271789 | 170.617747 |
| Plaintext upload seconds | 26.5929751 | 23.0988729 |
| Host encodes | 61696 | 50913 |
| Cache hits | 0 | 10783 |
| Peak RSS GiB | 27.3141365 | 27.3141365 |
| Maximum exact error | 0.000160619945 | 0.000166614586 |
| Maximum polynomial error | 0.00012925338 | 0.000156554607 |


Prefix runs use uncached/cached/cached/uncached order with a fresh process and
key each time. The cache starts empty. The samples in that order are
`[14.665308077, 14.5191603, 14.464689982, 14.714656376]` seconds. The binary, complete source archive,
payload hash, CPUs 15–19, four OpenMP threads and all other flags are fixed.
The selected preceding mode is `all`.

The uncached full baseline is the preceding campaign's completed candidate.
A Mamba-2 validation lies between that run and the cached full run, so the
single full pair is reported separately from the interleaved prefix controls.
It is not a repeated full-model statistical estimate.

## Contracts and provenance

The cache is bounded by 64 entries and keys exact coefficient bits and level
within one fixed context/slot count. Only degree-one multiplication plaintexts
are admitted; additive plaintexts keep their separate degree alignment. The
existing bypass policy avoids filling it with dense model weight masks.
Validation checks actual cache hits and entry bounds,
unaltered operation counts, rotations, depth, refresh schedule, key set,
model weights, selected IDs and both numerical gates. Each run uses zero
intermediate diagnostic decryptions. Decoded text comes from actual client
output IDs. CPU cache contracts and target cached-state tests are preserved
in the predecessor studies; this campaign adds composition and full-model
evidence without rebuilding the native evaluator.

`baseline-prefix/` and `baseline-full/` retain their original bytes. The
comparator verifies their hashes, source archive, binary, payload, environment,
commands and candidate bindings, then reproduces the selection and timings.
The copied Mamba-2 launcher is unused by these native commands; the common
measurement helper records its hash as a provenance dependency.

Native process wall totals 1194.006 seconds within
the one-hour internal review interval. Collection, comparison and token-report
hooks save durable completion records and submit a desktop notification.
Scope remains a DGX Spark feasibility run on one prompt with an inline client
and `security=not-set`. This does not validate long sessions or other prompts.
