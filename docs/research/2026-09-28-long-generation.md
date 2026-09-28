# Full-model Mamba-3 generation beyond four tokens — 2026-09-28

This study extends the trained 12-layer SISO 187M generation path. Each feedback
embedding comes from the token actually selected by the client after decrypting
the preceding final hidden vector. The preceding 64-step recurrence studies
used one layer with precomputed inputs and did not establish this result.
The [curated evidence](../../results/b300/2026-09-28/long-generation/README.md)
retains successful and rejected runs, payload identities and measured sources.

## Mathematical changes

The optional tiled state stores all 196,608 SSM coordinates per layer, previous
key/value vectors and the rotary phase. No rank truncation or history dropping
is introduced. The phase representation now supports `(cos(theta), sin(theta))`
as an alternative to an accumulated angle. For one step's increment `delta`,

```text
c_next = c * cos(delta) - s * sin(delta)
s_next = s * cos(delta) + c * sin(delta)
```

Only `sin(delta)` and `cos(delta)` need polynomial evaluation. The rotary
polynomial input range no longer grows solely because many increments are
accumulated. This is an angle-addition identity, not a clamp or an approximation
to the model's state update. Polynomial approximation and CKKS arithmetic still
introduce drift and require their own checks. Six state ciphertexts, previous
key/value ciphertexts and two phase ciphertexts make ten carried ciphertexts
per layer, excluding intermediates, keys, caches and retained validation outputs.

The exact float64 phasor reference matches the original accumulated-angle
reference through 64 generated tokens to `1.96e-14` maximum hidden error, with
identical greedy IDs. Tests also cover repeated rotations over many turns,
both rotary fractions, both recurrence representations and packed tiled state.

## Calibration and plaintext gates

Six existing calibration prompts each generate 65 tokens. The evaluation prompt
`The capital` is excluded from those fit inputs. The general domain margin is
0.2; positive inverse-square-root domains use margin 1.0 and a lower endpoint
of 0.25 times the smallest observed variance. All original normalizations remain.
The fitter retains `1e-6` sampled operator error, except the existing `1e-4`
negative-A target. The maximum overall degree remains 1,023; rotary degrees are
at most 15, versus 511 in a 64-token accumulated-angle calibration.

| CPU generation | Server evaluations | Matching generated IDs | Maximum hidden error vs exact |
| --- | ---: | ---: | ---: |
| 16 tokens | 17 | 16 / 16 | 0.00006531 |
| 64 tokens | 65 | 64 / 64 | 0.00018603 |

Before fitting the positive domains separately, merely extending the existing
calibration fails domain checks at `2:mlp_rms` for 16 tokens and `10:m3_c_rms`
for 64. Widening every operator's margin to 0.5 or 1.0 instead fails the
negative-A fit within degree 1,023. These rejected attempts do not relax the
numerical gate. The final coefficients remain sampled fits, not interval
certificates. Additional prompts `Machine learning` and `A scientist` still
fail frozen-domain checks; the result does not qualify arbitrary prompts.

## Encrypted qualification

The profile is unchanged: classical-128, N=131,072, QP=3,376 bits within the
3,523-bit bound, uniform ternary secrets, HYBRID digits four and a single ring.
Two-pass S2C-first refresh, merged correction, GPU plaintext preparation and
the exact/polynomial `0.001` gates are retained. Every generated ID must agree
with both references. Intermediate evaluator decryptions remain zero.

The first 16-token tiled run with a live-value threshold of 128 fails client
decryption while selecting token four, after three matching IDs. It has no
native result and is a failed run, regardless of its partial progress. The
128-value threshold from the one-layer recurrence qualification is therefore
not qualified for this complete generation circuit. The complete model already
retains about 140 values during early evaluation, reducing ready-node lookahead
under that threshold. An exact four-token prefix reproduces the decode failure;
the three preceding client outputs have errors below `3.5e-5`. This does not
yet identify the numerical root cause of the failing schedule.

With threshold 256, the complete model **generates 16 actual tokens** through
17 server evaluations. All token IDs match both references, the maximum error
against the original model is `6.62755e-5`, and the maximum error against the
polynomial reference is `4.02188e-5`. Evaluation takes **1,160.78589 seconds**,
or **72.54912 seconds per generated token**. Setup, final client selection,
output validation, teardown and external transport are outside this timer.
The full benchmark process takes 1,329.48296 seconds. This is a single successful
request, not a steady-state latency measurement or a speed comparison against
the differently calibrated four-token study.

The actual continuation is ` of the state of New York is New York City. The city
is the largest`. Matching this text establishes execution parity, not factual
quality. The model's answer itself is not factually correct.

The tiled run peaks at 229 live DAG values and a sampled 179.416 GiB of device
memory, with 4,426 physical bootstraps. The threshold is never selected because
the full ready frontier fits under 256. The exact-history comparison, using the
same calibration and phase representation, fails with CUDA out-of-memory after
14 successful client selections. It has no complete native result. Its last
memory sample is 287,166,889,984 bytes; exit code zero does not make this a pass.

The 64-token encrypted candidate **fails the unchanged numerical gate**. The
client-hidden polynomial error jumps from `3.61221e-6` at selection 23 to
`0.0196122` at selection 24 (exact error `0.0196165`). It remains about `0.009`
for selections 25 and 26. Those 26 IDs still match the reference, demonstrating
why token agreement alone is insufficient. The run is stopped after the failure
is observed; no successful 64-token timing or native result is claimed. The
root cause is not yet isolated. CPU parity and a successful 16-token request
do not establish 64-token encrypted stability.
The sampled peak remains about 180.416 GiB, with peak live count 229 below the
256 threshold through the rejected prefix. This failure is not an observed
out-of-memory event or an activation of the live-value threshold. Raising that
threshold alone therefore has no evidence-based justification as a repair.

The evaluator now checks retained hidden references during the existing client
decryption and stops before feedback when either error exceeds its configured
tolerance. The final selected token uses the same check. This shares the error
calculation with final output validation, rejects nonfinite values, and adds
neither a decryption nor a reference-token substitution. It prevents wasting
the rest of a long run after a known failure; it does not repair that failure.
The updated executor passes the frozen two-token prefix at the original 0.001
gates. A separate test deliberately sets both tolerances to `1e-12` and confirms
rejection before the first token selection, with the native exception exit code
2 and no successful result. This negative test does not change the qualification
tolerances. The local suite passes 424 Python tests and 22 native tests.

## Cost attribution

The successful 16-token request spends 430.12751 seconds (37.05% of evaluation)
in refresh. The largest operation totals are polynomial evaluation (309.27412 s),
encrypted multiplication (231.46215 s) and broadcast/repeat (228.18127 s).
These operation totals already include their own refresh time; adding them to
the refresh total would count work twice. Of multiplication's total,
217.88440 seconds is refresh.

Normalization polynomials account for 96.42437 seconds including their refresh,
or 8.31% of evaluation. All phase sine/cosine nodes together take 4.26050 seconds,
or 0.37%. The highest polynomial degree alone therefore does not identify the
largest optimization opportunity. Nsight diagnosis of the frozen full-model
prefix is performed separately from the unprofiled timing record.

## Nsight diagnosis

The two-token prefix is cut from the frozen 64-token candidate without refitting
coefficients or bounds. It executes all 12 layers three times, including actual
client feedback. Nsight Systems 2025.5.1 captures the CUDA-profiler evaluation
range. Separate graph-level and graph-node captures both pass the original
token, numerical and classical-128 gates. Nsight's collection/report processing
finishes after the GPU computation; those instrumented process times are not
inference performance results.

The graph-node capture contains **5,849,594 kernel instances** and 220.87392
seconds of summed kernel duration. Overlapping streams reduce the union of
kernel/copy/memset activities to **154.98929 seconds** in a 299.55299-second
capture (51.74%). The first 84.77 seconds in the graph-level native record is
initial public-state encryption, with CPU NTT work in the samples. After the
first 90 seconds, the graph-node activity union covers **73.19%** of the selected
209.55299-second window. This is recorded activity coverage, not SM occupancy,
HBM utilization or a full-generation throughput estimate.

In that later window, 28.18222 seconds has no GPU activity while at least one
CUDA runtime API is active, and 27.99186 seconds has neither. The dominant
unresolved CPU leaves belong to the CUDA driver; mask-cache lookup is a resolved
application hotspot. These observations locate gaps but do not prove all of
them removable. In particular, the 824,308 device synchronizations in the full
graph-level trace include waiting for useful GPU work; their summed 140.08812
seconds cannot be treated as an independent saving.

The leading graph-node kernel totals include NTT mode 1 (25.12489 and 18.73965 s
for its two variants), INTT (23.70755 s for one variant), RNS decomposition/base
extension (21.44970 s), modulus reduction (15.83247 s) and hoisted key switching
(15.61183 s). These totals overlap across streams. Graph-level capture alone
hides some of the RNS work inside graph envelopes; both traces are retained.

An independent unprofiled run of this same prefix passes in 273.024 seconds.
The approximately 299.6-second Systems captures are diagnostics, not faster
candidates. Separate devices/CPU placements and concurrent independent work
also rule out treating this as a controlled overhead benchmark.

Nsight Compute 2025.3.1 then samples four matching launches for each of three
kernel families, using kernel replay, unchanged driver clock control and the
SpeedOfLight, LaunchStats, Occupancy and MemoryWorkloadAnalysis sections.
The selected NTT launches skip the first 100 matching calls; the two other
families skip 16. All three instrumented requests still pass the original
numerical/token/security gates.

| Selected family | Duration (microseconds) | SM throughput | DRAM throughput | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: |
| NTT mode 1, both variants | 68.35–96.80 | 62.32–63.11% | 0.33–16.80% | 55.43–66.57% |
| RNS decomposition/base extension | 119.97–127.74 | 60.50–64.29% | 1.07–1.34% | 59.32–59.72% |
| Hoisted key switching | 192.90–195.17 | 54.41–54.93% | 37.85–38.32% | 57.50–58.04% |

These selected launches do not saturate HBM. The base-extension geometry has
0.86 waves per SM, whereas the selected key-switch geometry has 32.52. This
motivates examining integer instruction dependencies, on-chip traffic and
batch geometry rather than assuming an off-chip bandwidth bottleneck. It does
not establish the best kernel modification or a speedup. Replay/cache effects,
small sample counts and different levels/launch geometries limit extrapolation.
The raw counters retain their units and the selection commands.

The reusable [Nsight analyzer](../../experiments/profiling/README.md) computes
interval unions/intersections, supports later-window selection and hashes its
input. Raw reports stay on the measurement host because they contain machine
details; reviewed summaries, commands and input identities are published.

## Reproduction and scope

The maintained [exporter](../../src/fhemamba/workloads/mamba3_export.py) exposes tiled
state, phasor rotary representation and calibration controls. See the
[generation guide](../mamba3.md#trained-checkpoint-and-generation). Its
`--prefix-source` option derives an exact prefix without refitting coefficients
or changing node bounds, for shorter failure reproduction.

The [reporter](../../src/fhemamba/benchmarks/generation.py) validates actual
token counts, every hidden-output error and source/input bindings. Its
`--validate-only --security 128-classic` mode is suitable for a completion hook;
the longer job starts only after the shorter one passes. Profiled native client
diagnostics reuse the existing protocol decryption to print hidden-vector error.
They introduce no extra evaluator decryptions and do not replace selected tokens
with reference IDs.

The finite program retains all validation outputs and is unrolled in advance.
Fixed recurrent storage does not imply constant total executor memory. This is
an inline client prototype with public weights and fixture-derived refresh
bounds, not a process-separated service or arbitrary-context quality claim.
