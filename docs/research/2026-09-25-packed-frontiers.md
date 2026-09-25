# Ready-node refresh scheduling

The three-mechanism trial selects **ready-node refresh scheduling**. Its
final-source 12-layer Mamba-3 run takes **761.001984 seconds**, down
from **941.951140 seconds (19.21%)**. It preserves the two `0.001` error gates
and all four selected tokens. **The 20% target is not met.** Polynomial batching and
persistent slot windows are slower at full depth, so their implementations
remain in the experiment archives instead of the active executor.

The [evidence directory](../../results/dgx/2026-09-25/packed-frontiers/)
contains every sample, frozen revision, controller and source/binary identity.
This study tests the larger opportunities from the
[arithmetic audit](2026-09-25-math-kernel-audit.md).

## Comparison contract

The baseline is `18496e4e6a4e42c367fc356e99a7c4ab8e935eb0`, including the
[shared square dispatch](2026-09-25-square-dispatch.md). A fresh frozen baseline
finishes in **941.693953 seconds**; its 20% target is **753.355162 seconds**.
The stopping rule is a qualified target or a measured decision on at most
three mechanisms. Final-source qualification follows the adoption decision;
it does not introduce a fourth mechanism.

All full runs use the same trained Mamba-3 SISO 187M checkpoint, 12 layers,
five evaluations, four generated tokens and serialized program. Weights,
frozen polynomial coefficients, public bounds and client feedback stay fixed.
Ring dimension 65,536, 32,768 slots, depth 44, 59-bit scaling and two bootstrap
passes are unchanged. This retains the existing experimental
`security=not-set` scope. Generated token IDs must equal
`[315, 279, 1614, 315]`, and every hidden output must pass both `0.001` gates.
Server arithmetic performs no intermediate decryption; the client callback
still decrypts the final hidden vector to select each next token.

Native evaluation time includes plaintext preparation/upload, routing,
polynomials, refresh and runtime planning. The polynomial prototype also
charges its preprocessing time. Setup/key generation, input parsing, final
client selection/validation and external transport are excluded. Native wall
time is recorded separately. GPU measurements and builds are serial, using
Spark GB10, the same pinned backend/libraries, `OMP_NUM_THREADS=4`, CPUs 15–19
and profiling in both modes. Each process generates fresh keys.

Synthetic encrypted fixtures precede interleaved prefix controls. The prefix
uses one layer and three evaluations; its `[6864, 6864]` generated IDs are not
the full model's text. Full-depth measurements decide adoption. These samples
cover one frozen workload, not arbitrary-prompt quality or a latency distribution.

## Three mechanisms and decisions

| Mechanism | Matched prefix evidence | Full-model evidence | Decision |
| --- | --- | --- | --- |
| Joint polynomial evaluation | 46.755 → 48.799 s; 4.37% slower alone | In the combinations below | Archive prototype |
| Persistent contiguous windows | 46.885 → 46.847 s after sharing refresh outputs; 0.08% difference | Best three-way combination: 792.399 s; 15.85% shorter | Archive prototype |
| Ready-node refresh scheduling | 46.950 → 39.129 s; 16.66% shorter alone | 761.930 s; 19.09% shorter alone | Adopt |

The first three-way combination took **816.409649 seconds (13.30% shorter)**.
Retaining shared refresh outputs improved it to **792.399205 seconds
(15.85%)**. This is a refinement of the layout mechanism, not a new candidate.
The refined combination won the prefix comparison at 38.587 seconds but lost
to the simpler scheduler at full depth. Prefix ranking did not predict the
full-depth winner.

Joint evaluation groups polynomial nodes with equal nonlinear ancestry
height, degree and client epoch. It normalizes independent inputs into separate
SIMD lanes and shares a Chebyshev basis with vector coefficients. The
761 live polynomial nodes yield 171 batches containing 524 polynomials and
237 remaining scalar polynomials. Weight references are rebound when grouping advances their first
consumer. The shared evaluator is extracted from Mamba-2's joint-gate code.

Persistent windows defer contiguous extraction, let public masks absorb
window offsets, and materialize canonical layouts when consumers need them.
Dirty nonlinear operands require masking to prevent unrelated lanes from
growing. The second layout revision also shares grouped refresh outputs,
restoring each window's public bound once. This removes eager unpacking work
but shifts some work to consumers.

These implementations are complete and pass their encrypted fixtures. Their
cost exceeds their benefit in this workload. They are removed from active
code together with their command-line flags; source archives preserve both
revisions and their tests. There is no unmeasured claim of a retained SIMD or
persistent-layout acceleration.

## Why the selected change helps

The executor normally reaches a blocked branch and refreshes it immediately.
The new `--frontier-refresh` option first advances other ready operations
that fit the actual remaining ciphertext levels. When no ready operation fits,
it uses the existing grouped, two-pass refresh. More branches then reach the
refresh boundary together, reducing physical bootstrap calls.

| Full trial counter | Baseline | Three-way, shared windows | Scheduler alone |
| --- | ---: | ---: | ---: |
| Evaluation | 941.69 s | 792.40 s | 761.93 s |
| Bootstrap calls | 726 | 586 | 476 |
| Time inside refresh | 512.41 s | 400.50 s | 363.59 s |
| Logical refreshed values | 1,108 | 1,524 | 1,331 |
| Ciphertext products | 12,005 | 7,994 | 12,005 |
| Ciphertext/plaintext products | 107,738 | 95,855 | 106,457 |
| Recorded rotations | 57,932 | 58,974 | 62,990 |
| Host encoding | 168.68 s | 164.07 s | 152.91 s |
| Peak process RSS | 27.314 GiB | 27.316 GiB | 27.314 GiB |

Rotation counts include runtime packing but exclude bootstrap internals.
Refresh timers include packing/correction work, not just the backend primitive;
host encoding is nested in these operation timers and must not be added again.
Each physical refresh, whether grouped or scalar, uses two bootstrap calls. The scheduler reduces
those calls by **34.44%**, despite refreshing more logical values. In contrast,
the polynomial combination cuts ciphertext products by a third but spends more
on coefficient preparation and packing. Arithmetic counts alone do not establish
latency savings.

The scheduler uses graph dependencies and public ciphertext-level metadata.
It disables immediate static refresh hoists in this mode and leaves arithmetic
inside each node unchanged. Client-feedback epochs remain barriers, including
independent work preceding a callback. Runtime edge-use counts replace fixed
last-use indices, count both edges of `x*x`, and protect pinned outputs.
Progress logs count completed live nodes rather than their original indices.
The default schedule remains available by omitting the new option.

The scheduler-only full trial has maximum errors `0.0001171244` against exact
FP64 and `0.0001107302` against the frozen polynomial circuit. Both are below
`0.001`; fresh-key error differences are not accuracy improvements.

## Validation and bounded follow-up

The active C++ contract covers out-of-order ready work, duplicate edges,
pinned outputs, dead nodes, two feedback barriers and runtime lifetimes. The
Python runner tests flag propagation, recorded commands and rejection of
invalid option combinations. `--frontier-refresh` requires
`--planned-refresh --batch-refresh`, preserving the existing two-pass guard.
Local release checks pass **288 Python tests, including 19 C++ contracts**,
and both Ruff checks.

Archived encrypted window fixtures use independent CPU expected values and
cover live aliases, nested slices, offset mismatch, dirty multiplication,
public terms, negation, linear maps, repetition, scattering, reduction,
noncontiguous gathers, polynomial batches, refresh and pinned outputs.
The trial's shared Mamba-2 evaluator extraction passes four encrypted controls
with exact operation/level parity and the existing `0.05` gate. Means are
**53.786 → 53.745 seconds**; this small difference has no speed claim. That
extraction is not retained after polynomial batching is rejected, so the
released Mamba-2 evaluator is unchanged.

Additional CPU level models help reject further scheduler variants without
expanding the GPU search. Deferring static refresh hoists predicts 259 rather
than 251 physical refreshes. A six-threshold/two-capacity sweep reaches 246 but
requires more logical refresh work. Alternative ready-node priorities all
predict 251. Coefficient-aware polynomial depth bounds reduce 108 conservative
bounds but increase predicted physical refreshes from 288 to 295. These are
approximate diagnostics, not measured speedups or hardware lower bounds: the
model's sequential baseline predicts 378 physical refreshes while the real
baseline uses 363.

The selected source is committed as `1eda1e2fe8364ea5273dd4b6a08d46710048271f`.
Its 117 archived files match that commit; only six files differ from the
baseline, including the new scheduler and its test. Every trial has immutable
source, executable and input hashes. The source
archive for the baseline is independently checked against all 115 corresponding
files at `18496e4`. Completion hooks collect outputs and submit local desktop
notifications. Final-source qualification and verified cleanup are recorded in the evidence
directory.

## Final-source qualification

The selected implementation was committed before its source archive was made.
A new isolated build passes the synthetic encrypted fixture, then the full
model. The final candidate runs before a closing baseline; both use the same
frozen input, dependency hashes, CPU affinity, thread count and native flags,
with only the candidate enabling `--frontier-refresh`.

| Final comparison | Baseline | Committed candidate |
| --- | ---: | ---: |
| Native evaluation | 941.951140 s | 761.001984 s |
| Complete native process | 983.691466 s | 802.296327 s |
| Bootstrap calls | 726 | 476 |
| Maximum exact error | 0.0001284168 | 0.0000809575 |
| Maximum polynomial error | 0.0001160658 | 0.0000607799 |

The final evaluation reduction is **19.21%**
(**180.95 seconds**). It misses the 20% target by
**7.44 seconds**. Complete native-process time falls
**18.44%**; transport and payload export remain outside that interval.
The closing baseline differs from the initial baseline by only
0.257 seconds. The earlier scheduler trial was
761.929789 seconds; the two selected-configuration full runs support a
repeatable benefit on this fixture, not a statistical significance claim.

The committed candidate matches the trial's evaluated-node count, ciphertext
and plaintext products, rotations, logical refreshes, bootstrap calls and
scheduler counters exactly. It preserves all selected token IDs and all five
output gates. The released source contains neither rejected prototype.
The comparison stops after decisions on all three mechanisms; the 20% target
is reported as unmet rather than extending the search without a new boundary.
