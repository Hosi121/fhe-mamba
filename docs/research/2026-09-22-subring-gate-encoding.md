# Subring NTT for periodic gate coefficients

Status: exact plaintext comparison, encrypted coefficient probes, four
interleaved smoke runs, and the complete generation gate all pass.

## Remove the remaining redundant transform

The [periodic-coefficient implementation](2026-09-22-periodic-gate-coefficients.md)
completed the frozen 24-layer/five-evaluation request in 2573.366 seconds. Its
joint gates still took 908.037 seconds. Each coefficient row has period 32,
but the stock encoder performs a 65536-point NTT in every remaining RNS tower.
The coefficient preparation share inside that phase is not separately measured
in the model. Eliminating the entire gate phase would save at most 35.3% of this
baseline; that is a ceiling for this phase, not a prediction.
The separate [cost model](2026-09-22-generation-cost-bounds.md) derives transform
work, representation size, polynomial depth bounds, and conditional whole-run
limits directly from this payload and artifact.

For ring dimension `N`, period `s`, and `r = N/(2s)`, the coefficient polynomial
has the form `f(X) = g(X^r)`, where `g` has degree below `2s`. If `psi` is a
primitive `2N`-th root modulo one RNS prime, then

```text
f(psi^(2j+1)) = g((psi^r)^(2j+1)).
```

Only `2s` distinct modular evaluations are needed. With OpenFHE's bit-reversed
NTT layout, each small evaluation repeats in a consecutive run of `r` entries.
For this payload, this replaces a 65536-point transform with a 64-point
transform and repetition. The final plaintext still has the original full
ring and RNS moduli. Ciphertexts, keys, security parameters, and arithmetic
order are unchanged.

## Implementation boundary

`PeriodicPlaintextEncoder` calls the existing OpenFHE encoder with small
temporary polynomial parameters, the original modulus chain and level-dependent
scaling factor, and identity roots. Identity roots make OpenFHE's format switch
skip its NTT, leaving rounded coefficients in the temporary. The adapter then
uses OpenFHE's small NTT kernel with private root/preconditioner tables and
expands into a polynomial with the original full-ring parameters before giving
it to FIDESlib. Scaling, rounding, and RNS conversion are reused from OpenFHE.
The temporary representation is never encrypted or passed to GPU arithmetic.

Private NTT tables matter: this installed OpenFHE implementation keys its
shared NTT tables by modulus, without the transform size. Performing ordinary
small-ring encodes would replace the full-ring tables and force reconstruction
or interfere with concurrent encoders. Initialization of the small slot FFT
occurs before worker threads start; encoding reads immutable adapter tables.

The temporary encoder's identity-root format switches have no transform work
to parallelize. A scoped OpenMP task setting suppresses those teams and restores
the caller's maximum active levels immediately after the temporary encoding,
including on exceptions. It does not modify OpenFHE's global thread controls.
This is material on the measured host: an initial adapter that kept the empty
teams passed all checks but reduced the smoke by only 3.9% (gate phase 18.3%).
Its raw results remain under `initial-parallel-temporary/` in the evidence.

This removes modular transform work. Full plaintext allocation, expansion,
transfer, and GPU multiplication remain. It does not cache all coefficient
plaintexts or change the masked basis, polynomial coefficients, approximation
domains, PS decomposition, level schedule, or error threshold.

## Isolated validation

Two independent processes use ABBA and BAAB order, with two warmups and eight
measured samples per block. Both modes use period 32; A is the previous
full-ring NTT and B is the new subring NTT. Each process first checks 440
plaintext encodings for exact equality of all RNS entries and scaling/level/
slot metadata. Cases cover periods 1, 2, 8, 32, 64, 128; levels 0, 1, 21, 26,
34, 44; noise-scale degrees 1 and 2; zero and signed nonzero rows at scales
`1e-14`, `1e-8`, `1e-4`, 1, 128; plus eight concurrent encoder workers.
All cases match exactly.
Separate checks verify restoration of the caller's OpenMP setting after both
successful encoding and an encoder exception.

The timed part encodes changing signed coefficients, multiplies an encrypted
24-head-supported input, and checks every one of the 32768 decrypted slots
outside the timers. The preselected error threshold is `1e-7`; observed maxima
are `1.024e-10` on active lanes and `1.973e-10` outside them.

Pooled medians over 32 measured samples per mode and level:

| Consumption level | Previous encoding, ms | Subring encoding, ms | Reduction |
| --- | ---: | ---: | ---: |
| 21 | 3.550 | 0.458 | 87.1% |
| 26 | 1.839 | 0.342 | 81.4% |
| 34 | 1.608 | 0.205 | 87.3% |

Upload/multiply medians are about 6.2 ms at level 21, 2.5--2.6 ms at level 26,
and 1.3--1.4 ms at level 34. No GPU multiplication speedup is claimed. These
measurements alone do not establish a model speedup.

The probe uses the same ring 65536, depth 44, scale 59, first modulus 60,
sparse-ternary keys, and `security=not-set`. The final probe inherits the same
environment as the model campaign, with no OMP/MKL/OpenBLAS thread override.
The initial probe used a two-thread limit; its absolute times must not be used
to predict the model's encoding cost. Final probe binary SHA-256:
`19ea59aa97d813b1d87a0846429fda5dd5f25446e5ccc3792df72bbb8e69b95c`.

The [raw evidence](../../results/dgx/2026-09-22/subring-gates/) includes
both probes, their comparison, and the build manifest. Exact source archives
and logs are retained under `runs/gate-subring-serial-20260922/` and, for the
initial adapter, `runs/gate-subring-20260922/`. The initial probe
compile needed a missing FFT header and the global `Format` namespace fixed;
the successful source is archived separately as `source-probe-r1.tar.gz`.

## Integration contract

The opt-in native flag is `--joint-subring-encoding 1`, requiring
`--joint-periodic-coefficients 1`. Campaigns set `JOINT_SUBRING_ENCODING=1` and
`JOINT_PERIODIC_COEFFICIENTS=1`; the prompt-to-text launcher accepts the same
two options without values. Native artifacts record both mode selection and
the number of subring encodes. The isolated root is
`/home/kataiwa/fhemamba/gate-subring-serial-20260922`; the 42.9-minute baseline root
remains intact.

The smoke campaign compares off/on/on/off in independent processes using the
same binary, one layer, two evaluations, and the frozen generation payload.
All four final runs pass. Mean evaluation falls from **24.116 to 22.933 seconds
(4.9%)**, and the gate phase from **4.947 to 3.660 seconds (26.0%)**. Every
recorded ciphertext level and operation counter agrees. Each on-run performs
786 subring encodes; all off-runs perform zero. Maximum output errors are
`0.000288` (off) and `0.000304` (on), with peak RSS about 32.695 GiB in both.

The full gate uses the same 24 layers, five evaluations, prompt `The capital`,
expected generated IDs `[273, 253, 4687, 273]`, and maximum polynomial-circuit
error `0.05`. The payload SHA-256 is
`9869571e4253ceb36a853aecf99ce44a0fa502d41181ef6bdcf12bee80c54375`.

## Complete generation result

The full gate passes and produces `The capital of the Republic of` from the
same prompt. All five outputs decrypt, all four generated IDs match both
references, and the maximum polynomial-circuit error is **0.016255**, below
the unchanged **0.05** threshold. There are zero intermediate diagnostic
decryptions; generated-token selection uses the four client output boundaries.

| Measurement | Periodic full-ring encoder | Subring encoder |
| --- | ---: | ---: |
| Evaluation seconds | 2573.366 | **2310.796** |
| Evaluation minutes | 42.89 | **38.51** |
| Joint-gate seconds | 908.037 | **643.186** |
| Bootstrap seconds | 1082.114 | 1081.162 |
| Mean carried-state evaluation seconds | 522.035 | 469.105 |
| Setup seconds | 24.541 | 24.576 |
| Peak RSS, GiB | 37.094 | 37.085 |
| Maximum error against matching polynomial | 0.011767 | 0.016255 |
| Physical bootstraps | 2187 | 2187 |

Evaluation falls **10.20%** and joint-gate time falls **29.17%**. The 264.85
seconds removed from joint gates account for essentially the entire 262.57
seconds saved overall. Bootstrap time is nearly unchanged. Across the two
encoding improvements, the original 3038.121-second request now takes
2310.796 seconds, a **23.94%** reduction.

The complete operation-count dictionaries, all recorded ciphertext levels,
gate degrees, shared libraries, cryptographic parameters, and payload digest
match the periodic baseline. The new encoder runs **121,975** times, exactly
matching the coefficient-row count derived independently from the payload.
Per-output polynomial errors are
`[0.000407, 0.001077, 0.002544, 0.001880, 0.016255]`.
The largest exact-model output error is `0.074393`; the execution gate is
against the matching polynomial, while generated IDs match both references.

This is one full run per variant, with the periodic baseline measured
earlier. The same-binary smoke uses repeated off/on/on/off order. These
measurements do not establish a distribution across keys/prompts or longer
encrypted horizons. The existing one-process client loop and
`security=not-set` scope remain the same. The feature stays opt-in.

The [native artifact](../../results/dgx/2026-09-22/subring-gates/m2_chain_subring-client-generation_l24_t5.json),
[campaign](../../results/dgx/2026-09-22/subring-gates/generation-campaign.json),
and [derived generation report](../../results/dgx/2026-09-22/subring-gates/generation.json)
validate with zero errors and warnings. The
[comparison](../../results/dgx/2026-09-22/subring-gates/generation-comparison.json)
records all matched conditions. Native binary SHA-256:
`1d6108e4cde1177fd852c15907ca8ed5c12689caebfcd18ac04658cc0a438b61`;
native-source SHA-256:
`16a420fc9a3085352cd1540d2471bbeb6fcadfe2aaa93f3fda5841c632b1bf52`.
The final source snapshot still matches all 92 archived project files.

The manifest is `experiments/manifests/dgx_spark_subring_gate_generation.json`.
Use the matching isolated root above and a new results/output directory for
another run. The prompt launcher accepts
`--joint-periodic-coefficients --joint-subring-encoding`.

## Background completion and parallel offline work

The running inference was left intact while a one-shot user service,
`cipher-subring-completion-20260922.service`, checked campaign completion
every 30 seconds. Its hook collected results, ran `finish_report.py`, wrote
`completion.json`, and submitted a Windows desktop notification. It completed
successfully and exited; no watcher remains active. Scripts, status and logs
are under `runs/gate-subring-serial-20260922/`.

During that wait, the separate [normalization analysis](2026-09-22-normalization-bounds.md)
derived necessary approximation degrees for all 49 public intervals and
identified 227.28 seconds spent in their stage-11 refreshes in the periodic
baseline. That is a conditional budget for the next experiment, not an
additional speedup included in this result.
