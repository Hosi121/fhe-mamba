# Static waste audit: remaining ciphertext copies and repeated preparation

The strongest next candidate is ownership-aware reuse **inside** linear,
routing and polynomial evaluators. DAG-level last-use reuse does not cover
these private temporaries. The current Mamba-3 program contains 91,956 such
additions, each still cloning both operands: **183,912 outer Clone calls**
are candidates for removal. Mamba-2 also retains a result copy after explicitly
cloning inputs to protect live values.

This is a static audit, not an implementation or a speedup result. No new GPU
experiment was started for it. Its profile baseline is the uncached
borrowed-upload/routing run (1,004.62 s). The separate
[cache integration](2026-09-24-packed-cache-integration.md) subsequently completed
in 976.11 s; its reduced encoding/upload counts are not used in this inventory.
Counts use the frozen 12-layer, five-evaluation Mamba-3 program and the existing
24-layer, five-evaluation Mamba-2 run. These are separate workloads; their
operation counts are not a model-to-model speed comparison.

## Evidence and priority

| Priority | Candidate | Source or existing evidence | Next falsifiable check |
| --- | --- | --- | --- |
| 1 | Consume private Mamba-3 accumulator operands | 183,912 outer Clone calls in 91,956 additions | Retain identical arithmetic order, levels, error gates and live inputs; count removed copies and compare a matched prefix |
| 1 | Reuse Mamba-2's already protected operand for the result | At least 8,280 result copies in joint-gate multiplies alone; normalization and additions also use this pattern | Keep input protection, replace only the result allocation with an equivalent in-place operation; check carried state and normalization live-outs |
| 2 | Reuse public-weight encoding work | 22,440 diagonal encodes but 4,488 weight/diagonal identities in Mamba-3; host encoding totals 200.27 s | Inventory exact context/level/scale keys, then compare bounded host preparation reuse with exact RNS parity |
| 3 | Synchronize at a longer-lived ownership boundary | At least 339,141 explicit wrapper/upload synchronization calls in the existing Mamba-3 run | Keep inputs, scratch and borrowed host buffers alive until the relevant streams finish; profile wait overhead before and after a small scope change |
| 3 | Account for CPU and GPU plaintext cache residency | Mamba-2's 5 GiB budget estimates one RNS array; uploaded entries retain both representations | Count actual uploaded entries and resident limbs; measure peak memory before designing an eviction policy |
| 4 | Cache routing plans and geometry | 2,643 helper calls, 23 distinct geometries; 1,680 optimized stages plan twice | Reuse immutable metadata; first establish how much host time this actually removes |

## Private accumulator copies

[`aligned()`](../../native/fideslib_stage0/src/packed_fideslib.cpp#L72) clones both
operands. The current `inplace_ops` path removes the backend's third result copy,
but these two outer copies remain. The DAG's `binary_owned()` path is already
optimized; the internal evaluators still call `add()`.

| Internal accumulation | Additions | Outer Clone calls |
| --- | ---: | ---: |
| Replicated linear, inner | 19,920 | 39,840 |
| Replicated linear, outer | 2,280 | 4,560 |
| Routing transforms | 29,220 | 58,440 |
| Chebyshev leaf sums | 37,077 | 74,154 |
| Chebyshev recursive recombination | 3,459 | 6,918 |
| Total | 91,956 | 183,912 |

The relevant sites are [linear](../../native/fideslib_stage0/src/packed_fideslib.cpp#L358),
[transform](../../native/fideslib_stage0/src/packed_fideslib.cpp#L225) and
[Chebyshev](../../native/fideslib_stage0/src/packed_fideslib.cpp#L422).
The accumulator and newly produced term are dead after the addition. A consuming
helper can reuse them. This requires retiring temporary aliases such as a
zero-offset rotation's handle; it must not mutate cached basis values, shared
rotation inputs, squares or live DAG outputs. Keep the summation order unchanged.
The count excludes fill/fold sums, refresh interiors and other arithmetic.

Mamba-2's [aligned arithmetic wrappers](../../native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp#L1698)
call allocating `EvalAdd`, `EvalSub` and `EvalMult`.
[NormalizationOps](../../native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp#L2376)
and [JointOps](../../native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp#L3780)
first clone both inputs. In the pinned FIDESlib GPU path, allocating arithmetic
then copy-constructs a result from the first input; that copy invokes
`CopyDeviceCiphertext`. CPU storage is already shared lazily, but GPU polynomial
storage is copied. The 8,280 joint-phase ciphertext multiplications give a
conservative subset count, not the total opportunity.

A shared consuming arithmetic helper can serve both models, with each model's
level-alignment policy preserved. Changing every Mamba-2 `add_aligned` call to
mutate its first argument would be unsafe: some callers retain that handle.

## Public-weight preparation and cache size

The program evaluates 48 weight matrices five times, producing 22,440 diagonal
encodes for 4,488 identities. This exposes **up to 17,952 repeated identities**,
before considering level and scale compatibility. The current
[mask cache admission rule](../../native/fideslib_stage0/include/plaintext_cache.hpp#L14)
accepts zero plus one nonzero value; general dense-weight diagonals bypass it.
The separate mask-cache campaign therefore does not settle this candidate.
Mamba-2 likewise records only 5 replicated-weight cache hits versus 22,435 misses.

Caching every fully expanded RNS plaintext would be expensive. At the illustrative
level 21, depth 44 and ring dimension 65,536, those 4,488 identities require
**52.59 GiB for one RNS copy**. Multiple levels or CPU/GPU copies increase this.
Investigate bounded reuse of host FFT/coefficient preparation or selected hot
diagonals instead. Exact rounding, coefficient range, context, packing, scale
and level must be accounted for; same public weights alone do not prove an
encoded plaintext is interchangeable.

The [Mamba-2 budget estimate](../../native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp#L1546)
counts one coefficient array. The [shared uploader](../../native/fideslib_stage0/src/fideslib_plaintext_encoder.cpp#L178)
attaches GPU storage while retaining the CPU plaintext. Its current 356 cached
entries occupy a nominal 5 GiB; if all are uploaded, a second logical RNS copy
can approach another 5 GiB. This is not a measured RSS saving. Metadata and
readback/alignment paths still need CPU information; discarding it wholesale
would break the existing API contract.

## Synchronization and smaller candidates

The explicit synchronization lower bound is:

```
57,932 rotation steps + 107,738 ct/plain multiplies + 12,005 ct/ct multiplies
+ 91,956 identified additions + 7,814 node boundaries + 61,696 uploads
= 339,141 calls
```

These are wrapper calls derived from source and existing operation counters,
not a CUDA API trace. Additional additions, backend internals and destruction
barriers are excluded. A call can return quickly; the count is not time saved.
The existing FIDESlib lifetime patch synchronizes before recycling ciphertext
storage, and borrowed upload buffers currently survive until device completion.
Removing barriers requires event/stream-aware ownership, not merely deleting
`sync_gpu()` calls. A single linear or polynomial scope is a useful first unit.

The completed Mamba-3 run took 1,004.62 s. It recorded 513.22 s in refresh,
200.27 s in host encoding, 26.59 s in upload and 1.23 s in mask preparation.
These timers are nested; their shares must not be added. Eliminating all host
encoding would cap that category's contribution at 19.94% of current elapsed
time, and a weight cache addresses only part of it. Plan-only reuse is a lower
priority than ciphertext copies and encoding; mask preparation is already small.

The audit also found reasons to defer seemingly plausible changes:

- Mamba-2 reports 47,667 direct rotations and **zero composite steps**. Reusing
  an intermediate NAF rotation buffer cannot help the measured configuration.
- Scanning every integer Chebyshev baby size with no greater symbolic depth
  saves only **20 of 12,005 ciphertext multiplications (0.17%)**, plus 245 scalar
  multiplies. Numerical error and runtime are untested. This is a smaller
  candidate than removing accumulator copies.
- The actual Mamba-3 polynomial evaluation has **zero scalar additions of zero**;
  a generic zero-add fast path would not help this payload.
- Increasing the refresh batch cap above 16 does not address the observed maximum
  batch of 8. Slot co-location needs a separate layout/depth analysis.

## Reproduction and limits

[Preserved evidence](../../results/dgx/2026-09-24/static-waste-audit/README.md)
contains the source analyzer, full inventory, derivation, unchanged raw run
records and source/dependency hashes. The static counts agree with the existing
runtime's 12,005 ciphertext multiplications, 761 polynomial nodes, 240 linear
nodes, 2,643 routing calls and 1,680 optimized routing stages. All 111 frozen
application source hashes still match.

Dependency inspection uses FIDESlib commit
`cd171f20f510eeca04c71d7b0034ef073829f761` plus the campaign's existing patch set.
The normalized combined diff hash matches; `api/CryptoContext.cpp` and
`api/Plaintext.cpp` also match individual campaign hashes. Core `Ciphertext.cpp`
was not individually listed in that manifest and is bound through the combined
diff, not treated as an unexplained source mismatch.

No inference source or measured binary changed for this audit. Every candidate
still needs targeted ownership/numerical checks and a matched timing comparison
before adoption. The existing cryptographic and client-loop scope is unchanged.
