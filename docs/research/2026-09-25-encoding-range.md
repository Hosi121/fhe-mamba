# Reducing the OpenFHE Encode range scan

A local CPU prototype removes repeated logarithms from the pinned OpenFHE
64-bit CKKS encoder. Full coefficient-root encoding improves by **2.31–3.92%**
in these controls. Every tested coefficient and metadata word is unchanged.
The DGX dependency and model implementations do not include this prototype.

[Patch, full source revisions, exactness records, all samples and reproduction tools](../../results/cpu/2026-09-25/encoding-range/).

## Mechanism and scope

The original range scan scales each inverse-FFT component and computes
`ceil(log2(abs(component)))` for every nonzero real and imaginary part,
retaining the maximum. The candidate scales the same values, finds the
largest absolute component and makes that same libm call once. It retains
`log2`; an integer-exponent shortcut is not assumed equivalent near powers
because the original libm result can round to an integer.

The tested contract is finite scaled inputs, with zero and the existing
small-scale rejection covered explicitly. Nonfinite/overflow policy and the
128-bit build are not validated by this experiment. This is an isolated
prototype patch, not an unconditional upstream or application change.

The earlier extracted range scan was about 3.51 times faster locally, but that
is only one part of encoding. This full-construction comparison limits the
claim to a few percent. No model speedup or DGX result is inferred from it.

## Exactness in the actual encoder

Both builds use OpenFHE commit
`aa391988d354d4360f390f223a90e0d1b98839d7`. Its encoding file is byte-identical
to the pinned DGX encoding file; the other FIDESlib integration patches are
absent from this local CPU build. Only the PKE shared library differs between
baseline and candidate; core and BinFHE libraries and the probe executable
remain byte-identical.

The probe compares **167,772,160 integer coefficient words**, plus every
recorded modulus, format, scale, degree, level and slot metadata field. It runs
real/uniform-ternary and complex/sparse-ternary contexts at ring dimension
65,536, depth 44 and 59/60-bit scale/first modulus settings. Cases cover 512
and 32,768 slots, levels 0/21/34, degrees 1/2 and zero/dense/sparse/large/tiny
inputs. All **96 successful encodes and 24 expected small-scale failures**
match, including the large-input rescaling path.

The reference stream is compared word-for-word, with an end-of-file check.
Its 1.25-GiB payload is represented in the evidence by its exact size and hash;
the fixed fixtures reproduce it. It was removed only after comparison,
curation and hash verification. Identity roots suppress the final CPU NTT,
as in the application's shared coefficient preparation path. These tests do
not upload plaintexts or execute ciphertext arithmetic.

## Matched local measurements

Hardware is an Intel Core Ultra 7 255H on x86-64, GCC 13.3, Release `-O3`,
64-bit native integers and no machine-specific optimization setting.
Eight fresh processes run baseline/candidate/candidate/baseline twice. They
use the same probe executable and explicit immutable library directories;
`ldd` verifies the library chosen in every process. CPU affinity is 0 and
encoding uses the existing serial suppression policy. Each sample averages
24 constructions after one warm-up, with 32,768 slots, level 21 and degree 1.

| Context / fixture | Baseline mean (ms) | Candidate mean (ms) | Reduction |
| --- | ---: | ---: | ---: |
| Real / dense | 13.5593 | 13.0735 | 3.58% |
| Real / sparse | 13.0097 | 12.4998 | 3.92% |
| Complex / dense | 13.7529 | 13.4357 | 2.31% |
| Complex / sparse | 13.2618 | 12.8171 | 3.35% |

Timing covers `MakeCKKSPackedPlaintext` construction with coefficient roots.
Input creation, context setup and plaintext release are outside the timer in
both modes. There are four process-level samples per mode, and some variation
is comparable to the small improvement. These are descriptive means, not a
claim of statistical significance or a target-platform guarantee.

The probe's first compilation failed because its `Params` alias collided with
an OpenFHE class. The failed source/logs and corrected `EncodingPolyParams`
revision are retained. The corrected actual-encoding and benchmark runs pass.

## Reproduce and next decision

Recalculate the recorded comparison without rebuilding:

```bash
python3 results/cpu/2026-09-25/encoding-range/compare.py
```

For a fresh CPU build and comparison, use an unused output directory:

```bash
python3 results/cpu/2026-09-25/encoding-range/reproduce.py \
  runs/encoding-range-reproduction
```

Git, CMake and a C++20 GCC toolchain are required. The recipe fetches the pinned
source, retains separate library variants, runs the exact comparison, then
repeats the interleaved timings. It leaves its own build products for inspection.
The original experiment's temporary checkout, builds, libraries and large
reference stream have been removed after verification; portable evidence and
both source revisions remain.

Target integration still needs a finite/invalid-input policy, the actual FIDESlib
patch set, target exact-RNS checks and matched model controls. The modest local
full-encode gain should be weighed against the larger repeated-weight-preparation
candidate before starting another long GPU experiment.
