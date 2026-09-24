# Mamba-3 depth planning and batched refresh

Full encrypted generation fell from **2771.84 to 1496.95 seconds**
(**46.20 to 24.95 minutes, 46.0% less time, 1.85× speedup**) on the same DGX
Spark GB10 and frozen trained-model payload. All four selected IDs remain
`[315, 279, 1614, 315]`, producing `The capital of the state of`.
Maximum hidden error is `0.000155461` versus exact FP64 and `0.000133383`
versus frozen polynomials; both pass the unchanged `0.001` gates.

[Raw artifacts](../../results/dgx/2026-09-24/mamba3-depth-batching/),
[hash-bound comparison](../../results/dgx/2026-09-24/mamba3-depth-batching/full-comparison.json)
and [baseline study](2026-09-24-mamba3-trained-generation.md).

| Metric | Radix-8 baseline | Planned depth + batched refresh |
| --- | ---: | ---: |
| Evaluation (s) | 2771.84 | 1496.95 |
| Setup (s) | 27.39 | 27.26 |
| Refresh time (s) | 1739.69 | 534.70 |
| Physical bootstrap calls | 2652 | 726 |
| Ciphertext products | 12625 | 12005 |
| Plaintext products | 130883 | 107738 |
| Rotations | 133400 | 86063 |
| Peak RSS (GiB) | 27.96 | 27.96 |

Setup, payload loading and client/report work outside the evaluation timer are
additional. Native process wall time was 1539.15 s for the candidate. Refresh
time includes its packing, correction and unpacking; it is not an isolated
bootstrap-kernel timing.


The comparison uses the trained 187M SISO checkpoint, all 12 layers, the same
frozen program, two prompt tokens and four generated tokens on one DGX Spark
GB10. It retains ring dimension 65536, 32768 slots, depth 44, 59-bit scaling,
the two-pass bootstrap correction and the 0.001 error gate against both the
exact FP64 model and its frozen polynomial surrogate. Each full variant was
measured once with fresh encryption randomness. This is a matched-workload
comparison, not a repeated statistical performance estimate.

## What changes

`--planned-refresh --batch-refresh` enables five related changes in the common
packed executor:

- Estimate each operation's remaining depth, skip unused nodes and hoist a
  refresh before fan-out when a bounded greedy simulation predicts fewer
  refreshes. The planner is a heuristic, not a theoretical lower bound.
- Fold refresh rescaling into the two parallel bootstrap outputs, evaluate
  `P(x) - P(0)` before adding `P(0)` only to active slots, implement negation
  without a plaintext product, and defer linear-output masking until gather.
  A deferred mask is materialized before refreshing otherwise dirty padding.
- For arithmetic progressions of at most 32 routing diagonals, reuse baby
  rotations and rotate coefficient masks before giant rotations.
- Pack multiple live, eligible tensors into disjoint slot ranges, each
  normalized by its existing public bound; bootstrap the combined ciphertext
  twice for the same correction accuracy, then unpack and restore each bound.
- Keep ciphertext temporaries alive through the existing synchronization points.

The 7910-node frozen program includes 96 dead nodes, so 7814 execute. Every
layer and client feedback remains. The full candidate performs 1108 logical
refreshes, including 260 batches with up to eight tensors each, using 726
physical bootstrap calls. Logical refresh counts can increase because nearby
live tensors are refreshed early, while physical bootstrap calls fall.

## Rejected shortcut

A one-pass bootstrap probe reduced the one-layer time to 73.49 s but reached
0.17914 maximum error, failing the 0.001 contract. Its two greedy token IDs still
matched. This demonstrates why matching tokens alone cannot promote the
implementation. The failing artifact is retained as `layer1-single/`.

## Reproduction

Build the packed target using the pinned Spark dependencies. Export the full
payload with the [Mamba-3 guide](../mamba3.md), then run:

```bash
OMP_NUM_THREADS=4 python3 experiments/run_packed_probe.py \
  --binary /path/to/packed_fideslib --payload /path/to/mamba3-lm \
  --output /path/to/depth-batch-run --planned-refresh --batch-refresh \
  --timeout 2400 --budget-file /path/to/campaign.json --budget-seconds 7200
```

The original comparison uses one aggregate 7200-second campaign ledger across
successful and failed probes. Setup and runner wall time are charged, too.
`compiled-sources.tar.gz` preserves the source of the measured full binary;
subsequent microkernel instrumentation is not part of that executable.

## Scope

A single short prompt does not establish long-context or arbitrary-prompt
accuracy. Per-node bounds have empirical headroom, not certified intervals.
The evaluator has no decryption API, but the inline client decrypts the final
hidden vector for token selection. State stays encrypted. The feasibility
context has `security=not-set`; this is not a production security claim.

## Remaining costs

Refresh accounts for 534.70 s (35.7%) of the new evaluation, down from 1739.69 s
(62.8%). Non-refresh time remains about 962 s; linear transforms, routing and
polynomial arithmetic dominate this remainder. These operation timers include
CPU preparation, encoding, GPU work and synchronization. A CUDA/host profile
is needed before attributing that time to an individual microkernel. The
subsequent mask-copy and profiling changes are recorded separately and are
not included in the 1.85× full-model result.
