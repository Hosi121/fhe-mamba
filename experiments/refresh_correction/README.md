# Merge two-pass refresh correction before extraction

The opt-in `--merge-refresh-correction` packed-executor option replaces
separate extraction of `first` and `second` by:

```text
combined = 4096 * first + second         # integer RNS multiplication
output   = extract(combined, mask/4096)  # the existing plaintext product
```

Integer multiplication changes residues, keeping the CKKS scale and level.
Using an ordinary CKKS scalar product for the first line would consume an
extra level. The [shared helper](../../native/fideslib_stage0/src/fideslib_refresh_correction.hpp)
uses the pinned backend's `multIntScalar`. It checks equal context, key, level,
scale degree, exact finite scale factor, slot count and ordinary Q format.
Incompatible components retain the two extraction paths. Shared inputs are
cloned before mutation.

Build `refresh_correction_probe` with this directory's CMake project against
the same FIDESlib and OpenFHE libraries used by the packed executable. The
qualified B300 libraries include the existing even bootstrap seed and rotation
stream chaining, together with the patched inverse FFT. The
[full comparison](../../docs/research/2026-09-27-b300-refresh-correction.md)
records qualification, measured commands and source identities.

```text
refresh_correction_probe result.json real
refresh_correction_probe result.json complex
```

Both variants use the audited classical-128 geometry N=131,072, QP=3,376,
uniform ternary, HYBRID digits 4, depth 44, scaling 59 and first modulus 60.
They compare all ciphertext residues to a host 128-bit modular oracle at
consumed levels 0/18/35 and scale degrees 1/2, and check preservation of shared
inputs and metadata rejection paths.

The real variant additionally checks two-pass S2C-first refresh, mixed-width
blocks and full packing, all inactive coordinates, reference/candidate
differences and following products. Block bounds range from 0.5 to 1,024.
The `1e-6` module gate is normalized by `max(1,bound)` (squared for a following
product); the report separately records the maximum absolute error including
those large products. The full model retains its original absolute `0.001`
gates and generated-token comparison.

The existing S2C-first backend requires real packing. The complex variant
therefore qualifies the shared integer primitive only; it does not claim a
complex refresh or full Mamba-2 qualification. This option requires planned
two-pass refresh, defaults off, and is also forwarded and validated by
`experiments/run_packed_probe.py`.
