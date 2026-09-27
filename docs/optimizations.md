# Optimization mechanisms

Choose options by their validated mechanism and parameter requirements. The
[study index](research/README.md) retains matched comparisons, rejected candidates
and limitations. Historical timings with different security settings or CPU
placement are not interchangeable baselines.

| Responsibility | Maintained implementation | Qualification and use |
| --- | --- | --- |
| Ciphertext lifetime | `owned_arithmetic.hpp`, `fideslib_owned_arithmetic.hpp` | Final-use reuse and shared square dispatch; exact RNS checks for both model configurations |
| Slot routing | `packed_routing.hpp`, `rotation_steps.hpp` | NAF decomposition and staged BSGS gather/scatter; packed-runner options |
| Plaintext preparation | `fideslib_plaintext_encoder.*`, `gpu_special_fft.*`, `fideslib_plaintext_rns.cu` | Shared upload, optional GPU FFT/rounding and compact RNS/NTT; fallbacks retain the CPU path |
| Preparation overlap | `bounded_prefetch.hpp` | Bounded ordered CPU workers; `--prefetch-plaintexts --prefetch-workers 2` |
| Plaintext reuse | `plaintext_cache.hpp` | Content/level-aware bounded cache; size against available memory |
| Polynomial reuse | `chebyshev_basis_cache.hpp` | `--share-chebyshev`; reuse only with matching input and interval |
| Refresh scheduling | `packed_schedule.hpp`, `packed_depth.hpp` | Planned, batched and frontier refresh with fixed numerical gates |
| Refresh correction | `fideslib_refresh_correction.hpp` | Exact integer merging before extraction; `--merge-refresh-correction` |
| Dependency kernels | `native/fideslib_stage0/patches/` | Optional pinned-library patches; rebuild and run their qualification probes |
| Security parameters | `fideslib_security.hpp`, `experiments/security128/` | Actual QP audit for classical-128; experimental dual-ring settings are a separate profile |

Implementation paths in the table are relative to `native/fideslib_stage0/`
(`include/` for dependency-free contracts, `src/` for backend adapters).

## Reusable qualification suites

| Suite | Question |
| --- | --- |
| [OpenFHE inverse FFT](../experiments/openfhe_ifft/README.md) | Does a new plan preserve FFT and encoded coefficients? |
| [NTT warp tail](../experiments/ntt_warp_tail/README.md) | Does register exchange preserve every coefficient and improve a matched workload? |
| [Even bootstrap seed](../experiments/refresh_even_seed/README.md) | Can symmetry reduce the polynomial circuit within fixed output-error/level gates? |
| [Rotation stream chain](../experiments/rotation_stream_chain/README.md) | Can intermediate barriers be removed while preserving exact outputs? |
| [Classical-128](../experiments/security128/README.md) | Do actual context parameters, encrypted refresh and subsequent operations qualify? |
| [Refresh correction](../experiments/refresh_correction/README.md) | Does integer correction merging preserve residues, scale and follow-on arithmetic? |
| [Level and polynomial analysis](../experiments/level_schedule/README.md) | Where does a schedule leave unused capacity, and which nonlinear functions consume measured time? |

These suites do not all have the same adoption status. A primitive pass or an
optimistic cost model is not a full-model speedup. Preserve source, input,
security, precision, cache state and resource placement in each paired comparison.
The [experiment workflow](experiments.md) handles execution and publication;
the [validation guide](validation.md) defines the numerical and promotion gates.
