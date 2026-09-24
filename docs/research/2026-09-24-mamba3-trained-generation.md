# Slot routing optimization and trained Mamba-3 generation

This experiment keeps the pinned Mamba-3 SISO 187M checkpoint, frozen
polynomials, prompt, context and `0.001` hidden-vector error gates unchanged.
It tests whether reducing slot-routing depth saves more refresh time than it
costs in additional rotations. The complete model has 12 residual/MLP blocks;
preparation and upstream CPU parity are described in the
[preflight](2026-09-24-mamba3-trained-preflight.md).

## Why routing changed

The previous trained one-layer run spent 88.51 of 129.67 evaluation seconds
in refreshes. Each refresh performs two physical bootstraps. Bitwise slot
compaction or expansion can consume a multiplication level for each
displacement bit, even though it does not change the mathematical values.

The new generic executor groups three displacement bits into each routing
stage. Each group masks the same input ciphertext, rotates its coordinates
and adds the results, so a stage consumes one multiplication level. Gather
processes groups from low to high bits; scatter reverses that order. Strictly
increasing index maps keep coordinates distinct at intermediate stages.
Non-monotone maps use the existing direct transform.

Maps with at most 32 diagonals use a direct transform. Identity scatter is
omitted when the input already has clean inactive slots; gather still masks
its output because a sliding reduction can leave nonzero inactive slots.
The legacy bitwise route remains selectable with `--legacy-routing`.

No model coefficients, polynomial degrees, refresh precision or CKKS
parameters were reduced. The ciphertext arithmetic still synchronizes before
temporary GPU buffers are released. CPU routing tests compare random and
model-shaped maps with direct indexing across four radices and both directions.

## Matched one-layer comparison

Both runs use the same executable, payload and client head on DGX Spark GB10,
with four OpenMP threads. The payload is one trained layer, three evaluations
and two generated tokens. It is a component measurement.

| Metric | Bitwise routing | Radix-8 / direct-32 |
| --- | ---: | ---: |
| Evaluation time | 130.169 s | 100.831 s |
| Time in refreshes | 88.339 s | 40.866 s |
| Physical bootstraps | 134 | 62 |
| Ciphertext × ciphertext multiplications | 732 | 732 |
| Rotations | 2298 | 5811 |
| Max error vs exact hidden vector | 3.33e-5 | 2.04e-5 |
| Generated token IDs | `[6864, 6864]` | `[6864, 6864]` |

This is a **22.54% reduction**, or **1.291x**, for this component. It is one
sample per variant, not a statistical throughput study. Refreshes fall by
53.7%, while rotations increase. The result supports the depth-reduction
hypothesis at this shape; it does not establish an architectural speed
advantage over Mamba-2 or the best routing radix for every shape.

Refresh time falls by 47.47 s; all other evaluation work rises by 18.13 s.
The net saving is 29.34 s. Multiplication count alone would miss this tradeoff:
the model's ciphertext products are unchanged, while the routing schedule
reduces the depth that triggers expensive refreshes.

[Raw reports and the hash-bound comparison](../../results/dgx/2026-09-24/mamba3-trained-generation/)
retain the executable/input hashes, operation counts, per-output errors and logs.

## Full encrypted generation

The optimized **complete 12-layer model passed** five encrypted evaluations
and generated four tokens from the two-token prompt `The capital`:

```text
The capital of the state of
```

The actual client-selected IDs `[315, 279, 1614, 315]` equal both exact and
frozen-polynomial references. Three feedback transitions use actual decrypted
final hidden vectors; the fourth token is selected after the final evaluation.
No reference next-token embedding is used by the runtime.

| Metric | Full model |
| --- | ---: |
| Encrypted evaluation | 2771.840 s (46.20 min) |
| Context/key setup | 27.388 s |
| Native process wall time | 2814.326 s |
| Max hidden error vs frozen polynomials | 2.29014e-4 |
| Max hidden error vs exact FP64 | 2.16663e-4 |
| Acceptance limit for each reference | 0.001 |
| Nonfinite outputs | 0 |
| Physical bootstraps | 2652 |
| Time in refreshes | 1739.687 s |
| Peak process RSS | 27.96 GiB |
| Evaluator decryptions | 0 |
| Client output decryptions for generation | 4 |

The error maxima cover all five final hidden vectors. This is one prompt and
one optimized full run. There is no matched full-model bitwise-routing run,
so the measured 22.54% speedup applies only to the one-layer comparison.
Mamba-2's 38.5-minute experiment uses a different checkpoint, layer count and
error contract; these measurements do not rank the two architectures.

Refreshes still account for 62.8% of full evaluation time. The operation
profile includes refresh time in the operation that requests it; its
`bootstrap_seconds` fields can be subtracted to inspect the remaining work.

## Regression checks

The same executable also passes the existing four-step synthetic mixer,
including all carried-state outputs: 37.92 s, maximum error `7.72e-8` versus
exact and `5.67e-12` versus polynomials. This is a regression check, not a
matched routing speed comparison with the older synthetic measurements.

Python checks pass **274 tests** with **87.57%** package coverage. All **14**
native CPU tests pass, including the routing oracle, through the existing
pytest/CI contract-test path. The four GPU runs charge **3165.03 s (52.75 min)**
against the approved 7200-second campaign budget; setup and failed attempts
would also be charged. No GPU job remains running.

## Reproduction and limits

Use the [trained-model commands](../mamba3.md#trained-checkpoint-and-generation).
Export once and use the same payload directory for both routing variants;
change only the runner's `--legacy-routing` option and output directory.
Every GPU attempt in this campaign shares a 7200-second budget file. The
budget charges setup and runner wall time, including failed attempts.

The parameters remain ring dimension 65536, depth 44, scale 59 and uniform
ternary secrets with **security=not-set**. The inline client decrypts final
hidden vectors for its tied vocabulary head and encrypts actual selected
embeddings. Carried state stays encrypted; evaluator decryption count is zero.
This fixture provides neither client/server process isolation nor certified
refresh bounds or polynomial domains for arbitrary prompts. The exact-history
state representation is limited to 32 evaluations.
