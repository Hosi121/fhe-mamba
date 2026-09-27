# Even refresh seed qualification

This optional [FIDESlib patch](fideslib-even-bootstrap-seed.patch) rewrites the
uniform-secret EvalMod seed as `Q(2*(x-1/2048)^2-1)`. Q has degree 44; the
polynomial's degree in the original input remains 88. It removes four general
ciphertext multiplications per backend bootstrap, preserving the six
double-angle steps and the existing setup depth budget.

The branch requires the S2C-first precomputation, K=512, R=6 and exact equality
with all 89 pinned baseline coefficients. Other contexts retain the original
evaluator. Model polynomials are unchanged. The derivation and exact
real-arithmetic seed difference bound are in the
[static study](../../docs/research/2026-09-27-refresh-even-polynomial.md).
The [B300 follow-up](../../docs/research/2026-09-27-b300-refresh-even-polynomial.md)
records encrypted qualification, the model comparison and the adoption decision.

## Separate backend installations

Use FIDESlib `cd171f20f510eeca04c71d7b0034ef073829f761` and OpenFHE
`aa391988d354d4360f390f223a90e0d1b98839d7`, retaining the existing integration,
S2C-first and B300 correctness patches. Build a baseline installation, then
apply this additional patch in a separate source/build/install tree:

```bash
git -C /path/to/isolated/fideslib apply \
  "$(pwd)/experiments/refresh_even_seed/fideslib-even-bootstrap-seed.patch"
```

Rebuild FIDESlib and relink the native executable against the candidate
archive. The normal build does not apply this patch automatically. A different
`LD_LIBRARY_PATH` does not replace statically linked code. Keep the compiler,
CUDA architecture, OpenFHE, application sources and CPU controls identical.
The retained B300 source archives, compile/link commands and binary
hashes document the actual isolated builds.

## Encrypted rejection test

Build this probe against each installation. `REFRESH_CANDIDATE` chooses the
isolated seed test; the actual EvalMod/refresh path is selected by the linked
backend. Its value must match the library variant.

```bash
cmake -S experiments/refresh_even_seed -B runs/refresh-probe-base \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=103-real \
  -DCMAKE_PREFIX_PATH='/path/to/base/fideslib;/path/to/openfhe' \
  -DREFRESH_CANDIDATE=0
cmake --build runs/refresh-probe-base -j4
cmake -S experiments/refresh_even_seed -B runs/refresh-probe-candidate \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=103-real \
  -DCMAKE_PREFIX_PATH='/path/to/candidate/fideslib;/path/to/openfhe' \
  -DREFRESH_CANDIDATE=1
cmake --build runs/refresh-probe-candidate -j4

runs/refresh-probe-base/refresh_probe runs/refresh-base.json qualification
runs/refresh-probe-candidate/refresh_probe runs/refresh-candidate.json qualification
```

Each process checks four seed/EvalMod cases over all 32,768 slots, at input
levels 0 and 3. Inputs include endpoints, phase-shift neighborhoods, uniform
interior points and neighborhoods of modular-lift integers. The seed gate is
1e-8 and the full EvalMod gate 1e-6, against the original polynomial evaluated
in long-double arithmetic. Compare output levels, scale degree and factor
between the reports; timing these few cases does not establish model speed.

The same process checks six exact NTT ring maps, 30 encrypted ring-transfer
cases, and 12 two-pass refresh circuits (including four warm-ups) with active
widths 1, 32, 768 and 1,536. Both ordinary/refresh rings, inactive slots and a
following ciphertext product are covered. Their error gate is 1e-6.

Only passing qualification should proceed to the frozen prefix and full
trained-model gates. This changes rounding/noise propagation, so identical
precision cannot be inferred from the plaintext polynomial identity. The
measured scope retains the experimental `security=not-set` parameters; the
patch is not a security qualification or a full Mamba-2 speed result.

`EvenBootstrapSeed.cuh` duplicates the header introduced by the patch so that
the baseline seed probe can compile against an unpatched library. Qualification
requires these copies and the measured candidate to be identical.

The reusable [exact polynomial derivation](derive.py) accepts coefficient literals
and `K` from a JSON input, with no machine paths or measured timing dependencies:

```bash
python experiments/refresh_even_seed/derive.py \
  results/cpu/2026-09-27/refresh-even-polynomial/input.json \
  --output runs/even-seed/derived.json
```

It reproduces the recorded coefficient bits and exact rational difference bound.
That real-arithmetic bound does not include CKKS noise or floating evaluation.
