# Shared exact square dispatch

This bounded experiment preserves square identity through the shared Mamba-2
and Mamba-3 arithmetic adapter. It tests the first candidate from the
[arithmetic audit](2026-09-25-math-kernel-audit.md): selecting the existing
FIDESlib square implementation before protective clones hide that the two
operands are identical.

| Prefix workload | Baseline mean | Candidate mean | Reduction |
| --- | ---: | ---: | ---: |
| Mamba-3, one layer / three evaluations | 47.090536 s | 46.692888 s | 0.84% |
| Mamba-2, two layers / two evaluations | 53.978249 s | 53.501078 s | 0.88% |

Both corresponding pairs improve for each architecture. The implementation
is retained: it passes exact arithmetic checks and both predeclared 0.5%
prefix thresholds. These small, two-sample-per-variant differences do **not**
establish statistical significance or full-model acceleration. Full-depth
qualification remains outstanding; the previous complete-generation timings
remain the published full-model evidence. No approximation or error gate is
changed to obtain the reduction.

## Implementation

`owned_ciphertext_square` calls `EvalSquareInPlace` on a private ciphertext.
It consumes uniquely owned storage and clones a value with live aliases.
The helper waits for device completion before returning the result. Both
model adapters use it for Chebyshev basis squares and normalization; Mamba-3
also recognizes same-parent DAG products, while Mamba-2 uses it for its
explicit polynomial squaring steps. General binary arithmetic is unchanged.

The existing backend path is
`EvalSquareInPlace → Ciphertext::square → mult(*this) → binomialSquare_`.
The product kernel computes three modular products from two input words,
instead of four products from four words. Relinearization, transforms and
refresh remain. The full Mamba-3 audit found 4,386 eligible squares, but that
count is neither a measured whole-model speedup nor a new full-run result.

Separate counters record unary calls, reused inputs and protective clones.
Logical ciphertext multiplication counts retain their original meaning.
Binary ownership counters decrease when a formerly binary square moves to
the unary helper; these counters must not be compared as operation counts.

## Correctness and measurement contract

The candidate is compiled separately from the adopted baseline at `88870f9`.
The baseline source snapshot matches the source identities of the adopted
binaries. Compiler, flags, backend libraries, inputs and thread policies are
fixed and recorded in the
[artifact directory](../../results/dgx/2026-09-25/square-dispatch/).

Before timing, the GPU probe compares exact RNS coefficients and metadata
with allocating multiplication, mutable multiplication and direct `x*x`.
Each of the Mamba-2 and Mamba-3 contexts covers levels 0/21/34/39, fresh,
scalar-multiplied and previously multiplied values, plus three ownership
patterns. All 36 square cases per context pass, including unchanged live
inputs. The existing 96 binary cases per context also pass: **264 GPU cases
in total**, of which **72 exercise the new square helper**. These are exact
ciphertext comparisons; randomized CKKS decoding is not a bitwise gate.

The predeclared model controls are one ABBA block per architecture, with a
fresh process and keys each time:

- Mamba-3: one layer, three evaluations and two client-loop generated tokens;
  unchanged `0.001` exact and polynomial error gates.
- Mamba-2: two layers and two evaluations with carried encrypted state;
  unchanged `0.05` polynomial error gate.

All runs must retain logical operation counts, refresh policy, CKKS levels,
finite outputs and zero intermediate decryptions. The primary metric is
native evaluation time; request wall time is retained separately. No full
model run is scheduled by this experiment, and no extrapolated full-model
time should replace the existing README measurements.

The two model contexts retain the earlier feasibility scope (`security=not-set`,
ring 65,536, depth 44, scale 59). This change does not alter that scope.

## Samples and retained behavior

ABBA evaluation times in seconds are:

- Mamba-3: `[47.057378841, 46.521366317, 46.864410125, 47.123694148]`.
  Pair reductions are 1.14% and 0.55%.
- Mamba-2: `[54.076628747, 53.282486080, 53.719669041, 53.879869377]`.
  Pair reductions are 1.47% and 0.30%.

Mamba-3 dispatches **257 squares**, reusing 15 dying inputs and cloning 242
live inputs per candidate run. Every variant retains 673 ciphertext products,
26 bootstraps, 2,515 model rotations and 2,563 host encodes. The truncated
client loop retains IDs `[6864, 6864]`; these are not the full model's text.
Candidate maximum errors are `2.03413160706e-5` versus exact FP64 and
`1.6605251111e-5` versus the polynomial reference, below the unchanged `0.001`
gates.

Mamba-2 dispatches **330 squares**, each cloning a live input in this prefix.
All variants retain 1,008 ciphertext products, 61 bootstraps, 1,464 rotations,
522 unity-alignment products and 2,519 direct level drops. Parameters,
per-token/phase operation counts and CKKS levels match the baseline. Maximum
candidate polynomial error is `0.000595074321964`, below `0.05`. This prefix
uses carried state without an autoregressive client loop. Error variation
under fresh keys is not an accuracy improvement.

Local validation passes **286 tests**, including **18 native C++ contracts**,
and both Ruff checks. The ownership contract covers unique input reuse,
live-alias isolation, completion lifetime and null input rejection.
`summarize.py` recomputes the comparisons and verifies the retained source
archives, executable identities, native-output hashes and recorded gates.
The serial GPU jobs used 650.43 seconds including setup; the completion hook
collected their output and successfully submitted the local desktop notice.

The result supports a modest exact optimization. It does not imply that
replacing four modular products with three reduces inference time by 25%:
key switching, transforms, refresh and the rest of the circuit remain. This
trial combines specialized square dispatch and fewer copies; it does not
separately attribute time to the product microkernel.
