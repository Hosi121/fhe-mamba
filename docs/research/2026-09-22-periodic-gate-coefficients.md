# Periodic coefficients for the encrypted selective gates

Status: encrypted coefficient probe, integrated smoke comparison, and full
24-layer/five-evaluation generation gate passed. Evaluation time fell from
3038.121 to 2573.366 seconds (15.3%) on the same frozen request.

## Start from the required computation

The successful [client generation](2026-09-22-client-generation.md) took
3038.121 seconds for five encrypted evaluations of 24 layers. Its joint selective
gates took 1379.372 seconds (45.4%), with 123856 ciphertext/plaintext products and
no bootstraps inside that phase. Removing that entire phase would still leave
1658.750 seconds. This is a ceiling for optimizing this phase alone, not a lower
bound on other algorithms. Bootstrap timings overlap some other named phases;
the phase times must not all be added together.

Each layer needs only 24 gate values. The ring has 32768 complex slots because
the recurrent state is much larger. The two public per-head polynomials share
an encrypted Chebyshev basis, but each coefficient row was expanded from 24
numbers into a 32768-slot sparse plaintext and encoded again at its consuming
level. Those public coefficients do not depend on the current token.

The first experiment changes the coefficient representation. It retains every
coefficient, the public approximation domains, the PS decomposition, the write/
decay identities, all heads, the ring, and the error acceptance threshold.

## Why period 32 is sufficient

Let `M` be one on the active head lanes and zero elsewhere. The affine gate
input `u` already has this support in exact arithmetic. Use the masked basis

```text
S0 = M
S1 = u
S(2k)   = 2 Sk² - M
S(2k+1) = 2 S(k+1) Sk - u
```

Induction gives `Sn = M Tn(u)`: ordinary Chebyshev values on the active lanes
and zero elsewhere. A coefficient row packed with period 32 agrees with the
sparse coefficient row on every active lane. Multiplication by `Sn` therefore
produces the same full-slot result in exact arithmetic. The constant term also
multiplies the encrypted mask, rather than an all-one ciphertext.

For general layouts, the period is the smallest power of two covering the head
count. The period must divide both the batch size and stream stride. Coefficient
positions use `(head_offset + head) % period`, so an offset need not start at a
period boundary. The native adapter constructs the public encrypted mask once
and aligns cloned operands; it adds no final masking multiplication or depth.

CKKS introduces rounding and encryption noise, including outside the active
lanes. The algebraic argument alone does not establish encrypted accuracy. The
full-slot encrypted probe and integrated generation gate test that separately.

## Isolated falsification probe

`stage1_coefficient_encoding_probe` uses ring 65536, depth 44, scaling modulus
59, first modulus 60, sparse ternary keys, and consumption levels 21, 26, 34.
Each variant encodes changing nonzero coefficients and multiplies the same
masked encrypted input. Every result is decrypted and checked over all 32768
slots, outside the timed encoding and upload/multiply/synchronize intervals.

Two independent processes use ABBA and BAAB block order. Each block has two
warmups and eight measured samples; the table pools 32 measured samples per
variant and level and reports the median. The process sets `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`, and `OPENBLAS_NUM_THREADS=2`.

| Consumed level | Full encoding, ms | Period-32 encoding, ms | Reduction |
| --- | ---: | ---: | ---: |
| 21 | 11.918 | 7.517 | 36.9% |
| 26 | 9.439 | 5.772 | 38.9% |
| 34 | 5.941 | 3.437 | 42.1% |

The maximum active-lane error was `6.070e-11`; the maximum inactive-lane error
was `2.459e-10`, below the preselected `1e-7` threshold. At levels 26 and 34,
upload/multiply time remains about 2.8 and 1.5 ms respectively. Level-21 device
timings vary between processes; no separate GPU multiplication speedup is claimed.

The result rejects the idea that all encoding work scales with the 24 useful
values. In [OpenFHE's encoder](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/lib/encoding/ckkspackedencoding.cpp),
the inverse slot FFT uses the packing size, while the RNS polynomial still has
the full ring dimension and is converted to evaluation form. This suggests why
smaller slot packing only removes part of the cost. The measurements establish
the encoding reduction, not an overall model speedup.

Raw samples, the binary/source/library manifest, and their summary are retained
under `runs/gate-encoding-20260922/`. The successful probe binary is
`32edd8f95af88daf4a0add2569fce127def51d7ee0735e1a50461a121c21820d`.

## Integration and validation

The native switch is `--joint-periodic-coefficients 1`; the campaign environment
variable is `JOINT_PERIODIC_COEFFICIENTS=1`. The feature is opt-in. Results record
the selected mode, packing period, and masked basis.
The full prompt-to-text launcher accepts `--joint-periodic-coefficients` and
checks that the native artifact actually used the requested mode.

Host tests compare both sparse and periodic/masked PS evaluations to independent
Clenshaw evaluation through degree 1024, including tiny coefficients, domain
endpoints, inactive lanes, offset wrapping, multiple streams, and invalid period/
stride combinations. These tests pass on the local host and DGX.

The integrated smoke campaign uses the same binary and frozen payload in four
independent processes, in off/on/on/off order. All four runs passed:

| One layer, two evaluations | Off, seconds | On, seconds | Reduction |
| --- | ---: | ---: | ---: |
| Total evaluation, mean of two runs | 27.028 | 23.940 | 11.4% |
| Joint gates, mean of two runs | 7.798 | 4.894 | 37.2% |

Both variants had exactly the same recorded ciphertext levels, 22 bootstraps,
732 rotations, and 466 ciphertext/ciphertext products. The masked basis added
22 scalar ciphertext/plaintext products across the two evaluations. Maximum
CKKS-to-polynomial output errors ranged from `1.01e-4` to `3.19e-4`. The two
on-runs took 23.938 and 23.942 seconds; the off-runs took 27.053 and 27.003 seconds.
This smoke result does not establish the full-chain speedup or accuracy.

The generation campaign uses the
same five-evaluation request and payload as the 3038.121-second baseline:

```text
payload SHA-256: 9869571e4253ceb36a853aecf99ce44a0fa502d41181ef6bdcf12bee80c54375
prompt: The capital
expected generated IDs: [273, 253, 4687, 273]
maximum CKKS-to-polynomial error: <= 0.05
all output decryptions finite; zero intermediate diagnostic decryptions
```

The isolated candidate checkout/build is
`/home/kataiwa/fhemamba/gate-periodic-20260922`; it reuses the immutable baseline
payload and installed FIDESlib/OpenFHE dependencies. The validated baseline
checkout and binary are preserved. The campaign manifests are
`fhemamba/experiments/dgx_spark_periodic_gate_smoke.json` and
`fhemamba/experiments/dgx_spark_periodic_gate_generation.json`.

```text
candidate binary SHA-256: ea02d53608a7337ad94b6577bca864ef05eee876276bcc80fc009fc26b12d2b1
native/config source SHA-256: f98b1eadf70abd972344fa27be84592194dd230ec72ba7d356a143841cde12b8
```

The [tracked evidence](../../fhemamba/results/dgx/2026-09-22/periodic-gates/)
contains the raw coefficient samples, smoke artifacts, summaries, and build
identities. The exact source archives and native log are retained under
`runs/gate-encoding-20260922/` and `runs/gate-periodic-20260922/`.

## Full generation result

The measured output is again **`The capital of the Republic of`**. All four
generated IDs, `[273, 253, 4687, 273]`, match both the polynomial and exact-model
references. All five output decryptions are finite and within the unchanged
`0.05` polynomial-circuit error gate. Intermediate diagnostic decryptions are
zero; the four client output decryptions remain protocol boundaries.

| Measurement | Baseline | Periodic coefficients |
| --- | ---: | ---: |
| Evaluation, seconds | 3038.121 | 2573.366 |
| Joint selective gates, seconds | 1379.372 | 908.037 |
| Bootstrap evaluation, seconds | 1075.828 | 1082.114 |
| Mean carried-state step, seconds | 615.237 | 522.035 |
| Setup, seconds | 23.442 | 24.541 |
| Peak RSS, GiB | 37.0499 | 37.0940 |
| Maximum CKKS-to-polynomial error | 0.009192 | 0.011767 |

This removes **464.755 seconds (7.75 minutes, 15.3%)** from evaluation. The
joint-gate phase falls by **471.334 seconds (34.2%)**; the remainder increases
by 6.579 seconds, including a 6.286-second bootstrap increase. Thus the measured
reduction is concentrated in the phase changed by this patch. Setup and peak
memory do not improve; the extra encrypted mask adds roughly 45 MiB to peak RSS.

The candidate's per-evaluation polynomial errors are
`[0.000408, 0.001541, 0.001856, 0.002401, 0.011767]`. Its worst error is higher
than the baseline's, while both pass the original threshold. This is an
accuracy acceptance result, not evidence of improved numerical precision.

The payload, public polynomial degrees, shared libraries, cryptographic
parameters, and all recorded ciphertext levels match. Both runs execute
2187 physical bootstraps, 47667 rotations, and 32701 ciphertext/ciphertext
products. The masked basis adds 2460 scalar ciphertext/plaintext products
(`273213 -> 275673`) without adding depth or a final mask.

The [comparison report](../../fhemamba/results/dgx/2026-09-22/periodic-gates/generation-comparison.json)
binds the two raw artifacts by SHA-256 and checks these conditions. Native,
campaign, and derived generation artifacts all validate with zero errors and
zero warnings. The full timings are one run per variant, using the earlier
baseline; the interleaved smoke supplies the repeated comparison. This result
covers this prompt and horizon, with the existing single-process client loop
and `security=not-set`; new prompts, longer horizons, and other security
parameters remain separate gates.

For a fresh prompt-to-text run on the measured host, use the existing generation
command with `--joint-periodic-coefficients`, a new output directory, and
`--remote-root /home/kataiwa/fhemamba/gate-periodic-20260922`. The original
baseline binary remains available in its separate checkout.
