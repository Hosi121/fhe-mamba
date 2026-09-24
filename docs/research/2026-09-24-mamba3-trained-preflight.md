# Trained Mamba-3 SISO: implementation and encrypted preflight

The target is the official **Mamba-3 SISO 187M** checkpoint: 12 blocks, each
containing a Mamba-3 mixer and gated MLP, width 768, 24 heads, head width 64,
state width 128 and a tied vocabulary head. The [pinned assets](../../config/mamba3-reproduction.json)
and [commands](../mamba3.md#trained-checkpoint-and-generation) reproduce preparation.

**At this preflight snapshot, the complete encrypted run had not been executed.**
The results below validate implementation and a component. The later
[routing and generation study](2026-09-24-mamba3-trained-generation.md) records
the completed 12-layer run and its separate performance/accuracy results.
Neither study establishes an architectural speed advantage over Mamba-2.

## What changed

`Mamba3LM` strictly loads the released state dictionary and implements both
residual branches, all normalizations and the gated MLP. Its TensorOps boundary
is shared with exact, polynomial and packed execution. For short sessions, the
SSM is represented as a sum of rank-one writes:

\[
S_t = \sum_{j=1}^{t} w_{t,j} v_j B_j^T.
\]

At a new step, existing coefficients decay; the previous write also receives
the trapezoidal lag term. The new write is appended. Readout contracts each
stored B with the current C before multiplying by v. There is **no truncation**
of rank, history, heads or state dimensions. Storage and readout grow with
session length, so this is currently limited to 32 evaluations and is not a
constant-memory replacement for long recurrent inference.

The generic native executor reuses input-replicated BSGS matrix layouts from
the Mamba-2 kernel without clipping small weights. Monotone index maps use
masked bitwise compaction/expansion instead of one rotation per diagonal.
Extended programs support width 768 by padding reductions, reuse weight tables
between steps and attach explicit empirical refresh bounds to nodes.

Client feedback decrypts the final hidden vector, evaluates the full tied
vocabulary head, chooses its actual greedy result and encrypts that token's
embedding. A feedback node contains no precomputed embedding or token ID.
State remains encrypted. This is an inline client protocol, not process isolation.

## Validation available

The complete CPU model was checked against the original upstream preprocessing,
rotary and recurrence reference functions, plus `Block.forward` and
`GatedMLP.forward`, at commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`.
Five evaluations produce the same four tokens `[315, 279, 1614, 315]`:
`The capital of the state of`. Maximum hidden/state differences are respectively
`7.76e-8` and `1.77e-7`. The reference uses FP64 except for upstream's FP32 A
preprocessing; this is not a fused BF16 CUDA-kernel comparison.

Frozen polynomials fitted on six other prompts produce the same four token IDs.
The full-model hidden-vector difference from exact FP64 is `4.69e-5`.
Fits use 8193-point error checks, including a `1e-4` fitting tolerance for
the heavy-tail A function and `1e-6` elsewhere. Neither sampled fits nor
eightfold per-node refresh headroom certify arbitrary prompts.

A trained one-layer, three-evaluation encrypted component passed with actual
client feedback. The final build took **129.67 s** for evaluation,
with maximum errors `3.74e-5` versus polynomials and `3.52e-5` versus exact.
Its two generated token IDs are `[6864, 6864]`. These are outputs of a truncated
component and are not presented as the full model's text.

An isolated four-matrix fixture (input widths 32/64, 1024 slots) took 18.89 s
with direct diagonal multiplication and 2.12 s with replicated BSGS. Both
passed with errors below `4e-12`, using the same binary, inputs and context.
This approximately 8.9x result is a small operation comparison, not a
trained-model throughput measurement. It is one run per variant, not a
statistical throughput study.

Earlier trials are failures: a first 58-second synthetic-mixer run could not
decrypt; smaller probes exposed a one-baby-step mask convention mismatch and
nonfinite results removed by synchronizing GPU operations before temporary
buffers are released. These failed runs provide no speedup evidence.

## Next full run and limits

The complete payload has 7910 nodes, 48 distinct public matrices reused across
five evaluations, 785 polynomial nodes and three real feedback transitions.
It occupies about 1.6 GiB including the client head. A linear extrapolation of
the component timing gives roughly 44 minutes for 12 layers and five steps;
layer-dependent polynomial degrees, carry growth and client overhead make this
an estimate only. [Recorded artifacts](../../results/dgx/2026-09-24/mamba3-trained-preflight/)
include the upstream parity check, full export manifest and the trained
one-layer native run. Large checkpoint-derived payloads are regenerated from
the pinned checkpoint rather than committed.

All encrypted components use ring dimension 65536, depth 44, scale 59,
uniform ternary secrets and **security=not-set**. The hidden-vector acceptance
gate is `0.001` against both exact and polynomial references, with exact greedy
token equality required by the runner. Mamba-2's published 38.5-minute run uses
a different model and a `0.05` CKKS-to-polynomial gate, so these numbers do not
establish comparative architecture speed.
