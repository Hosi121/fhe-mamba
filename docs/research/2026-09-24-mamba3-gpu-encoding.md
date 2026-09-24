# GPU plaintext encoding in the Mamba-3 packed executor

Full trained Mamba-3 SISO 187M generation falls from **1312.20 to 1115.51 s**
(**21.87 to 18.59 minutes, 14.99% less evaluation time, 1.1763×**) on DGX Spark
GB10. Both modes use the same binary, payload, CPU placement and CKKS parameters.
All four actual generated IDs match, and the candidate's maximum hidden error
is **8.83e-5** versus exact FP64, below the unchanged 0.001 gate. Each full mode
was measured once; a shorter same-binary ABBA comparison also passes.

[Raw runs, comparison script and frozen sources](../../results/dgx/2026-09-24/mamba3-gpu-encoding/).

## What changed

The previous profile showed substantial host plaintext preparation even after
bounded mask caching. Most dense weight diagonals cannot reuse that cache.
This experiment asks whether preserving OpenFHE's coefficient construction
while moving its final RNS NTT to the GPU reduces complete generation time.

Two opt-in paths share a small bridge to the pinned FIDESlib backend:

- `--fast-plaintext-upload` builds upload arrays from const limb references.
  It avoids the two by-value polynomial-vector copies in the pinned
  `GetRawArray`/`GetModuli` helpers. The standard CPU NTT remains unchanged.
- `--gpu-plaintext-ntt` includes that upload path. OpenFHE still performs the
  FFT, floating-point rounding, RNS conversion and scale handling. Private
  identity-root parameters suppress its final CPU format transform, and the
  polynomial is restored to the real context parameters in coefficient form.
  FIDESlib then performs the exact integer NTT on the GPU, batching 16 RNS
  limbs per launch. Additive constants are encoded directly at the required
  scale degree, avoiding the earlier discarded degree-one intermediate.

The bridge does not change public model coefficients, frozen polynomials,
packing, arithmetic order, refresh policy or CKKS parameters. It uses installed
FIDESlib internal headers and symbols, with no changes to that library. Client
embedding encryption retains its original path. All existing evaluator
barriers remain; upload/NTT complete before staging buffers can be destroyed.
In the pinned backend, limb allocation selects the partition stream for every
limb; uploads and batched NTT launches therefore preserve their stream order.
This is an internal-API assumption to recheck on backend upgrades.
The native report records path flags, NTT batch size, preparation/upload counts
and, with `--profile-evaluation`, separate host and upload times.

## Allocation failure and exactness gate

The first probe failed with an illegal CUDA memory access. The standard GPU
plaintext allocator omits NTT scratch storage because it expects already
transformed inputs. Reusing it for a coefficient-format NTT was invalid.
The correction allocates ordinary limbs and their auxiliary NTT workspace
before loading the plaintext. The original failed report/log and source are
retained. FIDESlib's error path returned exit code zero, but the controller
correctly rejected the absent native result; exit status alone was insufficient.

`packed_plaintext_probe` compares every GPU RNS residue with the ordinary
OpenFHE encoder. The 160 input cases cover slot counts 1/32/1024/32768,
levels 0/21/34/44, scale degrees 1/2, and zero/one/signed sparse/dense/tiny
patterns. Four NTT batch widths, 1/4/16/64, pass all cases (640 coefficient
NTT comparisons), as does upload-only conversion. The CPU coefficient
representation, converted back to evaluation form, also matches exactly,
including level, scale factor, degree and slots. GPU readback is of public
plaintext polynomials; it introduces no ciphertext decryption in the evaluator.

Encrypted multiplication over all 32768 slots passes at working levels
21/26/34, with maximum absolute error **2.52e-12**. Median plaintext
preparation time (encode + upload, ms; eight timed samples per cell):

| Level | Standard | Fast upload | GPU NTT, batch 1 | GPU NTT, batch 16 |
| --- | ---: | ---: | ---: | ---: |
| 21 | 15.233 | 9.825 | 7.127 | 6.862 |
| 26 | 8.928 | 8.136 | 5.758 | 5.515 |
| 34 | 5.695 | 5.247 | 3.728 | 3.595 |

Batching 16 limbs reduces the upload/NTT phase by about 14% relative to one
limb per launch. At these levels, batch 64 adds little benefit; the final
implementation uses 16. The full raw record also includes batch 4 and 64.
These phase timings isolate preparation, not complete model evaluation.

Timing restores a fixed nonzero encrypted input and regenerates the same
coefficient vectors per mode. Each level uses six modes in forward/reverse
order, with a warmup and four timed samples per block. The encode/upload timers
include necessary allocation; multiplication, output decryption and comparison
are outside these timers. Standard upload has allocator/size-sensitive costs,
so these local numbers must not be extrapolated directly to model latency.

## Matched prefix comparisons

The first same-binary comparison uses order A/B/C/C/B/A, where A is standard,
B is fast upload, and C adds GPU NTT with one limb per launch:

| Mode | Evaluation samples (s) | Mean (s) | Reduction vs A |
| --- | --- | ---: | ---: |
| Standard | 19.6227, 19.7053 | 19.6640 | — |
| Fast upload | 18.9000, 18.8603 | 18.8802 | 4.0% |
| GPU NTT, batch 1 | 16.6265, 16.6243 | 16.6254 | 15.5% |

A second matched comparison uses the final executable with batch 16, in
A/C/C/A order. Evaluation falls **19.7113 → 16.4227 s (16.684%, 1.2002×)**.
Its baseline samples are 19.7111/19.7116 s; candidate samples are
16.4029/16.4425 s. Mean host preparation falls 6.1848 → 3.5010 s. Both modes
make 888 preparation requests; all 888 use GPU NTT in the candidate. Every
run passes the original numerical gate, with maximum exact error 2.16e-5
across both comparisons. The separate comparisons must not be subtracted to
isolate a model-level batching gain; that choice is supported by the phase
probe above.

All prefix processes share the payload, fixture, CKKS parameters and executable
within their comparison. They use Cortex-X925 CPUs 15–19, OMP four threads,
planned/batched two-pass refresh, in-place scratch reuse, and no plaintext
cache. Host timers are enabled in all modes; no Nsight profiler is attached.
No build or other benchmark runs concurrently on the target. Every mode has
8 bootstrap calls, 181 ciphertext products, 1440 plaintext products and
1103 rotations. Setup/reference checks are outside evaluation time but are
included in the campaign ledger. Two measurements per mode support only the
reported component result, not a statistical guarantee across workloads.

## Complete generation and regression

The full comparison runs A then C with the final executable. Both modes use
all settings described above; only C enables the new plaintext path.

| Metric | Standard preparation | GPU NTT + fast upload |
| --- | ---: | ---: |
| Evaluation (s) | 1312.1985 | 1115.5140 |
| Native process wall (s) | 1352.7960 | 1155.9237 |
| Setup (s) | 26.6053 | 26.6180 |
| Host preparation (s) | 378.8377 | 218.5604 |
| Preparation requests | 61696 | 61696 |
| GPU plaintext NTTs | 0 | 61696 |
| Bootstrap calls | 726 | 726 |
| Ciphertext / plaintext products | 12005 / 107738 | 12005 / 107738 |
| Rotations | 86063 | 86063 |
| Maximum exact error | 0.000118053 | 0.000088344 |
| Maximum polynomial error | 0.000116286 | 0.000086577 |
| Process peak RSS (GiB) | 27.9649 | 27.9649 |
| Actual token IDs | `[315,279,1614,315]` | `[315,279,1614,315]` |

The measured text is **`The capital of the state of`**. Evaluation saves
196.68 seconds; native process wall falls by 14.55%. Evaluation includes the
three intermediate client feedback callbacks, but excludes setup, the final
client token selection and diagnostic reference checks. Native process wall
includes those phases; input-hash verification is additionally charged in the
campaign ledger. Network transfer and checkpoint export are outside both.

Host preparation falls by 160.28 seconds (42.31%). It originally occupied
28.9% of evaluation, which explains why a large local encoding improvement
produces a smaller full-model gain. Candidate upload/NTT takes 73.50 seconds;
the standard path's upload is uninstrumented, so its reported zero is not a
measured zero cost. Bootstrap timing still dominates substantial work
(517.03 seconds in the candidate), and includes some preparation: these timers
are nested and must not be added as disjoint categories.

The comparison preserves 7910 total/7814 evaluated nodes, 1108 logical
refreshes and 260 refresh batches. Both modes already use in-place scratch
reuse. This is a separate comparison from the earlier 24.95-minute result,
which used different CPU placement and predates scratch reuse and the new
instrumentation. The historical delta must not be attributed solely to GPU NTT.

The one-layer generation probe and four-step synthetic state probe also pass.
The latter enables the bounded cache to test compatibility; all 20 output/state
checks pass. These are correctness runs without matched baselines. Their times
must not be compared with older, differently placed CPU runs to infer speedup.

All model gates retain exact/polynomial error limits of 0.001, with zero
intermediate evaluator decryptions. The full model remains the pinned 187M
checkpoint, all 12 layers and five evaluations, prompt IDs `[791, 6864]`, and
four actual client-selected tokens. The inline client computes the vocabulary
head and encrypts the chosen embedding. State is never decrypted/re-encrypted.
The same ring 65536, 32768 slots, depth 44, scale 59 and `security=not-set`
context is retained. This does not extend empirical polynomial domains,
long-context accuracy or process-isolation/security claims.

## Reproduction and decision

The final bridge is opt-in. Build the normal packed target and enable
`--gpu-plaintext-ntt` with the existing planned/batched refresh and in-place
options. The standalone GPU gate is `packed_plaintext_probe OUTPUT.json`.
The retained `run_final.sh` and `run_comparison.py` record exact measured
launches; replace their machine-specific paths and CPU IDs before reuse.
The input payloads and their derivations are referenced from earlier evidence,
and every result is bound to program, manifest and executable hashes.

Run `compare.py` in the retained evidence directory to recheck raw result
hashes, exact source archives, invariant operation counts, token/error gates
and the derived metrics. Local checks and target build identities are retained
separately. The first failed allocation is part of the experiment, not omitted
from runtime accounting.

Local formatting/lint and **280 Python tests, including the 16 native CPU
contracts**, pass. The final GPU probe checks 160 cases at all four NTT batch
widths and encrypted multiplication. The one-layer generation case has maximum
exact/polynomial errors 2.04e-5/1.39e-5 and selects `[6864,6864]`. The cached
synthetic case records 236 hits with maximum errors 7.72e-8/5.16e-12.

All **18 attempts**, including the failed first GPU allocation, consume
**3226.63 seconds (53.78 minutes)** of charged runner time. Source archives,
compiler/dependency hashes, real-process CPU-affinity samples and raw logs are
retained. No target build or second benchmark runs during model measurement.

The earlier 7200-second campaign remains immutable. The user authorized
additional GPU hours; this experiment uses a separate 21600-second review
window. That window is a review checkpoint, not an inferred total spending cap.
Further useful candidates include reducing remaining coefficient construction
and staging work, or overlapping independent preparation with GPU execution.
Those require new measurements and buffer-lifetime proofs; this experiment
does not claim asynchronous execution or elimination of all host encoding.
