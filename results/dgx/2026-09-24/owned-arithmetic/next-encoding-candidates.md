# Follow-up encoder candidates, separate from the measured ownership change

These checks ran locally while the immutable DGX arithmetic campaign was
running. They do not modify the encoder, any dependency or either GPU binary.

## Repeated public-weight preparation

`encoding_inventory.cpp` reads the exact full Mamba-3 program and calls the
shared replicated-mask builder for every live matrix. It finds 48 matrices,
4,488 unique matrix/diagonal identities and 22,440 evaluations of those
diagonals. This agrees with the earlier static inventory.

Storing one complex-double inverse-FFT vector per identity would cost
2,353,004,544 bytes (2.19140625 GiB), excluding metadata. This is independent
of scale/level because scaling follows the inverse FFT in the pinned encoder.
The expanded RNS representation at illustrative level 21 requires 24 times
as much: 52.59375 GiB for a single CPU or GPU copy. No such cache is implemented.

The maximum absolute public weight is 2.859375. The largest outward-rounded
mean absolute mask value is 0.07159273792354816. In exact inverse-FFT arithmetic,
each component is bounded by that mean; at an illustrative scale of 2^59 the
bound is below 2^56. This does not bound the floating implementation's rounding
error or prove the actual encoded coefficient range for all contexts.

A second possible representation is one signed 64-bit integer per polynomial
coefficient: also 2.19140625 GiB at one scale/level. To avoid copying OpenFHE's
rounding algorithm, a miss could run the existing encoder and then attempt to
compress its coefficient-format RNS result. A hit could reconstruct only if
the balanced coefficient recovered from one modulus exactly reproduces every
limb and remains within the supported range. Context, scale, degree, level,
packing and mask identity must remain exact cache keys; unsupported cases
must keep the current path. The range/roundtrip checks are a proposed gate,
not a completed encrypted test. GPU expansion could also reduce upload bytes,
but needs its own lifetime and exact-RNS validation.

## Encode range scan

The pinned OpenFHE 64-bit `CKKSPackedEncoding::Encode` scales each complex
coefficient and calculates `ceil(log2(abs(component)))` for every nonzero real
and imaginary component, retaining the maximum. On finite scaled inputs,
the candidate first finds the maximum absolute component and makes the same
libm call once. It preserves scaled coefficient bytes and the resulting `logc`.
The actual encoder and the compiled dependency are unchanged.

`encoding_log_probe.cpp` tests 6,356 cases, including the neighbors of every
finite binary exponent, signed zero, subnormals, several CKKS-like scale
factors and vector lengths around powers of two. It deliberately retains
`log2`: `ilogb + mantissa test` is not assumed to preserve libm rounding near
powers. Nonfinite/overflow behavior is outside this extracted test's contract
and requires a safe policy before any library integration.

The recorded local x86-64 run on an Intel Core Ultra 7 255H measures eight
ABBA/ABBA samples, 31 restored inputs per sample, with no fast-math flags.
Its means are about 241.55 → 68.84 microseconds for 32,768 complex values,
approximately 3.51 times faster for this extracted range scan. The earlier
sample is also retained; its absolute times were less stable. Neither run is
an OpenFHE full-encode benchmark, a DGX result or a model speedup. Full encoder
RNS parity, target measurements and model controls remain required.

`encoding-inventory-run.json` and `encoding-log-probe-run.json` bind input,
source, compiler and binary identities. Initial scan artifacts are preserved
under `initial-encoding-log-probe/`; the later revision guards the empty-vector
comparison without changing either timed nonempty kernel.

A [separate actual-encoder CPU follow-up](../../../../docs/research/2026-09-25-encoding-range.md)
has since completed the local RNS comparison: 167,772,160 integer coefficient
words and metadata match exactly across 120 successful/rejected cases. Its
full plaintext-construction means improve by 2.31–3.92%, substantially less
than the isolated range-scan ratio. Those local source variants and raw samples
are preserved separately. The DGX dependency remains unchanged; target
integration and matched model controls are still outstanding for that patch.
