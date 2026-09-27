# GPU plaintext FFT and coefficient preparation

Moving public plaintext preparation to the GPU reduces the matched classical-128
Mamba-3 evaluation from **211.712 to 165.373 seconds (21.89%)**. That is
**41.34 seconds per generated token**, including prompt work in a request with
five encrypted evaluations and four generated tokens. The candidate is adopted
as the opt-in `--gpu-plaintext-fft` path. This is one fresh full process per arm,
preceded by a matched prefix and exact primitive qualification.

Both arms retain the [merged refresh correction](2026-09-27-b300-refresh-correction.md),
frozen model/polynomials and the [classical-128 context](2026-09-27-b300-classical128.md).
No security, precision, approximation-degree or refresh-count reduction is used.

## Mechanism and correctness boundary

Previously, CPU workers performed the special inverse FFT, scaling and rounding,
then uploaded compact coefficients for GPU RNS expansion and NTT. The shared
[`GpuSpecialInverseFFT`](../../native/fideslib_stage0/src/gpu_special_fft.cu)
now uploads real input values and performs those steps on the GPU. Rounded
coefficients feed the existing integer RNS expansion directly. Only a status
word returns to the CPU; coefficient arrays remain on the device.

Butterfly order, CPU-generated twiddles, bit reversal, normalization and
rounding follow the pinned OpenFHE implementation. Explicit round-to-nearest
double operations avoid fused arithmetic changes. Integer rounding preserves
ties away from zero. The existing compact-range proof and small-scale rejection
determine eligibility; unsupported values, scale degrees, levels and sparse
packing use the stock path. Degree-two addends apply the same rounded integer
scale multiplier as before.

The current path supports at most 65,536 slots on one device, with full packing,
64-bit limbs, FLEXIBLEAUTO and the per-limb RNS strategy. CPU workers still
prepare masks, while the evaluator thread owns GPU work and handle registration.
Synchronization preserves coefficient-buffer lifetime. The CPU member of a
GPU plaintext holds metadata only; verification must use `readback_plaintext`.

## Qualification and measured comparison

The FFT probe passes 204 bitwise comparisons across all powers of two through
65,536 slots, counting the ordinary and experimental tiled implementations.
The production path additionally passes 153 rounded-coefficient cases and
51 expected small-scale rejections, including ties, zeros and short inputs.

Real and complex RNS probes each pass 360 residue/metadata cases: 70 use the
GPU FFT (35 are degree-two addends), while 290 exercise CPU fallbacks. Each
also passes nine encrypted arithmetic cases. Maximum errors are `5.050e-12`
and `3.942e-12` under the unchanged `1e-6` primitive gate. Complex qualification
covers the shared encoder; it is not a full specialized Mamba-2 timing result.

| Measurement | CPU preparation control | GPU FFT |
| --- | ---: | ---: |
| Prefix evaluation | 13.496975 s | 10.264601 s |
| Full evaluation | 211.712029 s | 165.373386 s |
| Refresh time, included above | 61.833305 s | 51.729209 s |
| Encoding-call time | 84.771863 s | 6.725953 s |
| Preparation queue wait | 25.406752 s | 0.850914 s |
| Plaintext encodes | 16,195 | 16,195 |
| GPU FFT encodes / fallbacks | 0 / 0 | 16,193 / 2 |
| Peak process RSS | 62.869 GiB | 58.003 GiB |
| Setup/key generation | 139.062234 s | 138.941201 s |
| Complete process | 371.536246 s | 324.551559 s |

The two GPU FFT fallbacks match the control's two compact-RNS fallbacks.
The legacy `host_encoding_seconds` field measures encoding-call time; with the
new option it includes GPU work. CPU producer timings overlap evaluation, so
these rows cannot be added to derive total time. GPU preparation, including
RNS expansion and NTT, takes 6.665194 seconds. Refresh-wrapper time also falls
because its public packing/extraction masks use the common encoder.

Both full arms preserve 484 physical bootstraps, 242 merged corrections,
11,540 ciphertext products, 93,026 plaintext products and 55,426 ordinary
rotations. IDs remain `[315,279,1614,315]`. Candidate maximum exact/polynomial
errors are `4.30478e-5` / `4.88485e-6`, below both absolute `0.001` gates;
non-finite values and evaluator decryptions remain zero.

Source, executable, static libraries, program, fixture and client-head hashes
match between the full arms. The same B300, NUMA placement, four OpenMP threads,
two mask producers and 2,048-entry cache are used. Actual N=131,072 and
QP=3,376 bits pass the unchanged uniform-ternary classical-128 audit. This is
a fixed-prompt prototype with inline client feedback, not steady-state serving
latency or a process-separated deployment qualification. No statistical
significance claim is made from the single pair.

## RMSNorm attribution and remaining work

Per-node timing is joined to the frozen calibration manifest by exact interval
and coefficient values. In the candidate, RMSNorm inverse-square-root nodes
account for **20.325 seconds (12.29%)**. Of this, 12.403 seconds belong to refresh
batches initiated at those nodes; batches may also refresh other live values.
Surrounding reductions, routing and products are separate. Thus this is neither
the cost of a complete RMSNorm module nor an achievable saving estimate.

The 245 inverse-square-root nodes have degrees 15/31/63. `negative_a`, whose
maximum degree is 1,023, accounts for 15.741 seconds including 5.676 seconds
of attributed refresh. These measurements motivate evaluating layout and
polynomial strategies across both families. Sylph's prefix/outlier calibration
is not implemented; its applicability and distinction from orthogonal rotation
are discussed in the [paper review](2026-09-27-sylph-cachemir.md#rmsnorm-outliers-and-the-frozen-mamba-export).

An experimental shared-memory FFT tail remains disabled in the executor.
Its 65,536-slot round-trip samples vary: 422.2/426.4 microseconds for ordinary/tiled
in the first screen and 487.1/397.8 in the later screen. This does not qualify
an additional full-model gain. Small transforms can favor the CPU substantially.

## Reproduction

The [qualification guide](../../experiments/openfhe_ifft/README.md#gpu-coefficients)
provides build and probe commands. Use the recorded control arguments, adding
only `--gpu-plaintext-fft` for the candidate. Build with `FHE_STAGE0_GPU_RNS=ON`;
the option requires `--gpu-plaintext-rns` and rejects batched/fused RNS modes.
`experiments/run_packed_probe.py` forwards the option and rejects results that
do not prove the GPU path was used.

The [curated evidence](../../results/b300/2026-09-27/gpu-plaintext-fft/README.md)
contains both complete process records, correctness probes, matched comparisons,
polynomial profiles, frozen calibration manifest and measured source snapshots.
The reusable job runner supplies completion events and serializes GPU jobs.
Temporary orchestration scripts are not required.
