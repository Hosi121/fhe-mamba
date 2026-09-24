# Shared plaintext preparation for Mamba-2 and Mamba-3

Mamba-2 complete generation falls **38.64 → 35.70 minutes (7.59% less)** by
sharing Mamba-3's plaintext preparation improvements. Both modes use the same
binary, frozen 24-layer/five-evaluation request and numerical contract.
Mamba-3's complete generation also passes after extracting the common policy.

[Raw runs, comparison script and frozen sources](../../results/dgx/2026-09-24/shared-plaintext-preparation/).

## Implementation

Both native evaluators now use `fhemamba::PlaintextPreparation` for real coefficient encoding, additive scale-degree correction, and application plaintext upload. Its policy selects stock encoding, the existing periodic subring encoder, or the existing GPU NTT adapter. The bridge avoids two pass-by-value copies of the full RNS limb array. GPU NTT retains OpenFHE FFT, rounding, modulus conversion and scale metadata, then transforms on FIDESlib in batches of 16 limbs.

Periodic coefficients keep their small CPU NTT in every mode. Mamba-2 cache workers and client encryption retain their stock CPU path; GPU handle registration remains on the evaluator thread. Cached, streamed, scalar BSGS, fused projection and complex-constant consumers share the upload path. The Mamba-3 executor no longer maintains a separate preparation policy. No cryptographic parameters, coefficients, operation ordering, refresh policy, synchronization or error limits change. No dependency library was patched.

## Gates and measurements

The shared-policy probe runs with both Mamba-2 sparse-ternary/complex and Mamba-3 uniform-ternary/real settings. Each configuration checks 160 exact RNS cases over four GPU NTT batch sizes plus 54 shared dispatch/addend cases. Maximum encrypted arithmetic errors are below 1e-6.

The Mamba-2 one-layer/two-evaluation A/B/C/C/B/A comparison has all ordinary coefficient masks cached. It exercises fast upload and periodic encoding, but its GPU NTT count is zero. A separate two-layer/two-evaluation B/C/C/B control exercises ordinary coefficient cache misses. Candidate selection uses that latter comparison. All native outputs and original controller records remain available, including a controller validation-key error corrected without rerunning or modifying the successful native output.

The full Mamba-2 comparison uses the exact same binary, DGX Spark, 24 layers, five evaluations, prompt `The capital`, and four client-selected tokens. Evaluation falls from **2318.10 to 2142.11 seconds (38.64 to 35.70 minutes, 7.59% less, 1.0822×)**. Maximum CKKS-to-polynomial error is 0.013862 for baseline and 0.005044 for the candidate. Both generate IDs `[273, 253, 4687, 273]`, giving `The capital of the Republic of`.

| Full metric | Stock upload / CPU ordinary NTT | Shared fast upload / GPU ordinary NTT |
| --- | ---: | ---: |
| Evaluation | 2318.100 s | 2142.107 s |
| Setup | 23.181 s | 22.278 s |
| Bootstrap evaluation | 1083.549 s | 1082.829 s |
| Joint selective gates | 652.513 s | 530.812 s |
| Peak RSS | 37.094 GiB | 37.063 GiB |
| Maximum polynomial error | 0.013862 | 0.005044 |

All complete operation counters, per-token/phase counts and CKKS levels match. Both retain 2187 physical bootstraps and 121975 subring encodes. The candidate performs 23542 GPU ordinary encodes and 145588 evaluator fast uploads. Bootstrap time is almost unchanged; reduced preparation work accounts for the observed evaluation reduction. Upload timing is 214.85 s for the candidate; the disabled baseline timer is uninstrumented.

The one-layer means are A=22.7167 s, B=21.4820 s, C=21.6994 s. B isolates copy reduction (5.44% lower mean). Its entirely cached ordinary masks mean zero GPU encodes for C, so that comparison does not demonstrate GPU NTT benefit. The cache-pressure two-layer means are B=58.7917 s and C=57.7104 s (1.84% incremental reduction), with 258 GPU encodes in each C run. This selects C for the full comparison.

Mamba-3 old/new/new/old prefix evaluation means are 16.5354/16.4872 s, a 0.29% difference: common-policy extraction is effectively neutral at this scale. All operation counts and error gates pass. The cached four-step synthetic regression passes with 236 cache hits. Complete 12-layer/five-evaluation generation takes 1115.322 s (18.59 min), with maximum exact/polynomial errors 0.000091960/0.000123327, 726 bootstraps and 27.965 GiB peak RSS. It selects `[315,279,1614,315]`, giving `The capital of the state of`. This full run validates the refactor; its elapsed time is not a separately matched speedup claim.

Mamba-2 retains its 0.05 CKKS-to-frozen-polynomial gate, with zero intermediate diagnostic decryptions. Mamba-3 regression retains its separate 0.001 exact/polynomial gates. These models and their error contracts are not used as a cross-architecture speed comparison. Fresh keys produce different CKKS errors; the lower observed candidate error does not demonstrate improved numerical accuracy.

## Reproduction and scope

The native Mamba-2 executable takes `--fast-plaintext-upload 1` or `--gpu-plaintext-ntt 1`. GPU NTT includes fast upload. Use both `--joint-periodic-coefficients 1 --joint-subring-encoding 1` to preserve the measured periodic optimization. Campaign equivalents are FAST_PLAINTEXT_UPLOAD and GPU_PLAINTEXT_NTT. The prompt-to-text launcher accepts boolean flags without values. All new behavior stays opt-in.

The new controller keeps a separate six-hour review ledger under the user's existing authorization for hours of GPU work. This is an internal review window, not a newly assumed spending ceiling. It serializes GPU jobs, imposes per-job timeouts, writes completion/status hooks, and preserves failed records. Prior studies and budgets are immutable.

The full comparison uses fresh keys, one complete request per mode. It covers one frozen prompt and five evaluations, not a distribution over prompts, keys or long horizons. Setup and complete process wall time are recorded separately from evaluation. Disabled upload timers are uninstrumented, not evidence of zero upload cost. Inline client boundaries and security=not-set remain unchanged.

All 20 attempts consume 6582.387 s of charged process time (109.71 minutes)
within the 21600-second review window. Full Mamba-2 native process wall time
falls from 2346.691 to 2169.816 s; Mamba-3's full process takes 1156.049 s.
Evaluation excludes setup and reference checking. The measured launchers
retain CPU affinity, environment, binary/input hashes and timeout records.
Mamba-2 uses CPUs 0–19, OpenMP eight threads and `CUDA_LAUNCH_BLOCKING=1`;
Mamba-3 uses CPUs 15–19 and OpenMP four threads. No target build or second
benchmark runs concurrently with a model measurement.

Local formatting/lint, 280 Python tests and 16 native CPU contracts pass for
this source snapshot. Both GPU policy probes, all mirrored controls and the
complete generation gates pass. The source archive contains the measured
70 files; later direct-upload/NAF/weight changes have a separate build and
study. The completion hook collected the final raw files and submitted a
desktop notification. `compare.py` rechecks raw hashes, archived sources,
parameters, counts, levels, token IDs and numerical gates before regenerating
the summary.
