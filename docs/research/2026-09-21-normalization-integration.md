# Scheduled RMSNorm in the Mamba-2 kernel

The 49 public normalization recipes now have an opt-in payload and native
execution path. This experiment integrates block, gated and final RMSNorm
into the existing full-model slot layout. The convolution/gate/decay fits and
head-pruning policy remain those of the exported legacy surrogate; this is
not the joint-gate model that passed the 4,096-token plaintext screens.
The [frozen bundle](../../config/mamba2-130m-normalization-20260921.json)
retains SHA-256 `89801adcc149d3183bf758fb8d022f4949b9b048ae5c665ff90a6892632c1c15`.

## Circuit and payload contract

`export_m1_payload.py --normalization-bundle` verifies site coverage,
duplicates, checkpoint epsilon and the exact-rational certificate of each
recipe. It requires a new output directory and regenerates the polynomial
references, including autoregressive references when requested. Non-finite
fixed-vector references and norm inputs outside the declared domains are
rejected. The bundle hash and fixed-vector domain observations are stored in
`chain.json`. Final RMSNorm has a dedicated `final_norm_poly`; it no longer
inherits the last block's schedule.

The native implementation retains the model's contiguous packing. Variances
are divided by the true feature width before evaluating the same balanced
polynomial DAG used in the isolated probe. Block/gated gamma stays folded
into the public projections with the matching mean-variance coordinates.
The final gamma multiply is on the numerator branch. Alignment operates on
clones: the original encrypted variance remains live and unchanged across
inverse checkpoints, so every subsequent cubic update recomputes its
relationship with the refreshed inverse. Public stage bounds scale the
inverse around explicit Meta-BTS calls.

Legacy payloads retain their previous arithmetic. Scheduled payloads use
Meta-BTS for transient checkpoints as well as their inverse checkpoints;
carried-state policy remains independently configurable. This is needed
because the longer normalization changes where downstream refreshes occur.
Improving only the inverse refresh cannot remove noise introduced earlier in
its inputs. No diagnostic decryption supplies coefficients, bounds or
refreshed inputs to the evaluated circuit.

## One-layer integration gate

The comparison uses the real 130M checkpoint, two fixed input tokens,
independent 34-token built-in calibration text (128-token cap), and identical weights/non-normalization
fits/calibration bounds. Each run gets fresh keys. Parameters are ring 65536,
depth 44, scale 59 and explicit `security=not-set`; these are numerical
experiments, not a 128-bit security result. The unchanged error threshold is
0.05 against each payload's matching polynomial reference.

| Circuit | Token 0 error | Token 1 error | Physical bootstraps | Gate |
|---|---:|---:|---:|---|
| Legacy normalization control | 0.000253 | 0.003886 | 9 | pass |
| Scheduled inverse, ordinary transient refresh | 0.068086 | 0.172350 | 17 | fail |
| Same, unity level alignment | 0.065181 | 0.171923 | 17 | fail |
| Scheduled inverse and transient Meta-BTS | 0.0000211 | 0.006343 | 23 | pass |

All eight token outputs decrypt to finite values, with zero evaluation debug
decryptions. Level alignment alone does not solve this failure. The corrected
schedule passes the existing threshold at the cost of additional physical
bootstraps; no speedup is claimed. These are individual runs, not a statistical
precision guarantee. The first failed run records `working-tree` in its commit
field; its exact native source/binary/library hashes and source archive are
preserved. Subsequent runs record the parent commit and dirty-tree digest.

The [raw artifacts](../../results/dgx/2026-09-21/normalization-integration/)
retain both failures and the matched control. Source snapshots and binaries
are under `~/fhemamba/norm-integration-20260921/history/` on Spark. Local export,
check logs and payloads are under `runs/norm-integration-20260921/`.

## Keeping the variance branch live

The initial 24-layer attempt stops after layer 3 with depth exhaustion. Its
eight-level input requirement accounted for the seed but allowed a carried
residual at consumed level 35 to enter the normalization. Refreshing the
inverse then immediately realigned subsequent products back to the deep,
unchanged variance, causing repeated refreshes without sufficient depth gain.
The failed artifact and log are retained; it is not a completed two-token run.

The corrected planner reserves 18 levels before scheduled block/gated/final
normalization, covering the live variance branch and refresh preparation.
This refreshes a level-35 input while admitting a level-21 input at depth 44.
Both the runtime plan and host ledger share the requirement, and a host
regression checks that boundary. The variance remains immutable; the fix does
not reconstruct it from a noisy affine coordinate or decrypt it.

The corrected [24-layer/two-token run](../../results/dgx/2026-09-21/normalization-integration/m2_chain_scheduled-norm-full-chain-live-variance-r1_l24_t2.json)
and [campaign gate](../../results/dgx/2026-09-21/normalization-integration/full-chain-live-variance-campaign.json)
**pass**, including the dedicated final RMSNorm on both tokens. The recurrent
state and convolution FIFO remain encrypted between the two fixed input
tokens; this does not exercise generated-token selection.

| Measurement | Token 0 | Token 1 |
|---|---:|---:|
| Maximum error against matching polynomial circuit | 0.0000921 | 0.007136 |
| Maximum error against exact-model reference | 0.047809 | 0.045393 |
| Evaluation seconds | 266.03 | 314.91 |
| Physical bootstraps | 354 | 442 |
| Final consumed level | 35 | 35 |

Both outputs are finite, with zero intermediate decryptions. Total evaluation
is 580.95 seconds after 23.28 seconds of setup; peak RSS is 36.15 GiB.
The run uses 17,556 rotations, 28,538 ct-pt products, 8,467 ct-ct products and
796 physical bootstraps. Its ring-65536, `security=not-set`, single-process
settings are the same numerical experiment boundary described above. This is
one fresh-key full-chain run, not a repeated precision or performance claim.
The extra refresh cost remains an optimization target.

The reproducible entry points are
`dgx_spark_scheduled_norm_smoke.json` for one layer/two tokens and
`dgx_spark_scheduled_norm_chain.json` for the separate 24-layer/two-token gate.
Long-session quality, joint-gate native lowering and the full 128-bit/security
protocol gates remain separate requirements.

The final binary SHA-256 is
`5eaac6d02a6895e42f3a493c5f49eb7796dfce16d216c8ca3c63ddcb6ef17c31`;
payload SHA-256 is
`90ffd42aa49c998016117dd81452faa503e92ae5b8c70b7489b25880dcf691fe`.
Raw results bind the source and shared-library hashes as well. The local gate
passes **228 tests**, including **11 native C++ contracts**, with **87.04%**
active-package coverage. New contracts check payload parsing, the dedicated
final recipe, bundle validation, transient refresh policy and the live
variance input budget.

## Matching-payload long-window check

The [1,024-token quality screen](../../results/payload_native_scheduled_norm_20260921.json)
uses this exact native payload on the first window of the cached WikiText-2
test tokens. Exact and exact-with-mask controls are finite, with PPL
13.25128 / 13.24844. The exported polynomial circuit is **non-finite**, so its
PPL is undefined and the quality gate fails. Observed domain escapes include
convolution SiLU, gate SiLU and gated normalization. Counts after non-finite
propagation cannot be read as a complete domain-closure certificate or as
proof of the first failing operator.

This short-calibration integration payload cannot support a long-context
quality claim. That failure motivated lowering the separately validated
joint gates and wider-domain activations and regenerating matching references.
Passing the short encrypted gate only establishes execution accuracy against
its polynomial circuit.

The subsequent [stabilized native study](2026-09-21-stabilized-native.md)
adds those joint gates and public-envelope SiLUs to a separate payload. Its
matching 1,024/4,096-token plaintext screens are finite with no observed
domain escapes. Its native 24-layer/two-fixed-token gate passes at error
0.003311, with zero intermediate decryptions and `security=not-set`.
