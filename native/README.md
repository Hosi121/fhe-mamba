# Native backend

[`fideslib_stage0/`](fideslib_stage0/) contains the optimized Mamba-2 CKKS backend
and the architecture-independent packed executor used by Mamba-3 SISO.
The directory's historical name does not denote a separate model implementation.

| Directory | Contents |
| --- | --- |
| [`src/`](fideslib_stage0/src/) | Encrypted model evaluator, runtime configuration, payload/provenance code and GPU probes |
| [`include/`](fideslib_stage0/include/) | Layout, polynomial and payload contracts |
| [`tests/`](fideslib_stage0/tests/) | FIDESlib-free C++ contract tests |
| [`patches/`](fideslib_stage0/patches/) | Patches for the pinned FIDESlib dependency |

For encrypted execution, use the [DGX Spark build guide](../docs/dgx-spark.md#build)
and [reproduction guide](../docs/reproducing.md). The backend is built separately
from the Python wheel. Source and build paths are part of its provenance, so
rebuild after updating a checkout whose layout changed.

`stage1_mamba2_decode_fideslib` executes the specialized Mamba-2 payload.
`packed_fideslib` executes the common arithmetic program exported through
`TensorOps`; it shares the replicated matrix-layout helpers with Mamba-2.
See the [Mamba-3 guide](../docs/mamba3.md) for full-checkpoint export, client
generation, bounded execution and matched routing comparisons.
Its optional `--planned-refresh --batch-refresh` schedule passes complete
trained generation; `--inplace-ops` controls scratch reuse and
`--profile-evaluation` marks the CUDA evaluation capture range. The shared
`plaintext_mask.hpp` helper uses bit-preserving contiguous copies. Measured
scopes are in the [depth study](../docs/research/2026-09-24-mamba3-depth-batching.md)
and [microkernel study](../docs/research/2026-09-24-mamba3-microkernels.md).

`--frontier-refresh` advances independent ready operations before refreshing
blocked branches, so more values can share the existing grouped refresh. It
requires `--planned-refresh --batch-refresh`. Dependency and feedback barriers
are preserved; runtime edge-use counts protect final-use ownership and pinned
outputs. The result records `frontier_deferrals` and `maximum_ready_nodes`,
and progress uses completed live-node counts. Omitting the option retains the
sequential schedule. See the
[three-candidate study](../docs/research/2026-09-25-packed-frontiers.md).
Polynomial-batching and deferred-layout prototypes are archived there; their
experimental flags are not part of the current executable.

`--hoist-rotations` shares signed-digit prefixes and sibling key-switch
preparation for the packed executor's BSGS baby rotations, using existing
rotation keys. `--share-chebyshev` shares normalized arguments and basis terms
between live polynomial nodes with the same input, width and interval. A
refreshed input invalidates the basis; the last consumer releases it. Both
options preserve the model coefficients and remain opt-in. See the
[four-candidate study](../docs/research/2026-09-25-structural-four.md).

`--gpu-dual-ring` selects N=32,768 for the packed executor's ordinary
arithmetic and retains N=65,536 for two-pass S2C-first refresh. The shared
bridge switches encrypted keys and maps bit-reversed NTT pairs on the GPU;
no evaluator decryption or ciphertext download is involved. This option
requires `--s2c-first --planned-refresh --batch-refresh`, a program declared
with 32,768 slots, and logical node widths at most 16,384. The Spark build
includes it; custom builds use `-DFHE_STAGE0_GPU_DUAL_RING=ON`. See the
[dual-ring study](../docs/research/2026-09-26-gpu-dual-ring.md). The specialized
Mamba-2 executor keeps its existing ring and refresh path.

`rotation_batch_probe OUTPUT.json [--mamba2]` checks complete RNS residues,
level/scale-degree metadata and input preservation for the shared rotation
helper. It also records alternating scalar/shared microbenchmarks. The Mamba-2
option selects complex slots and sparse-ternary keys; it does not change or
benchmark the specialized Mamba-2 model executor.

`--cache-plaintexts` enables a 64-entry LRU for encoded multiplication masks
and their GPU handles. It requires exact coefficient bits and CKKS levels,
bypasses dense diagonals/additive constants, and keeps existing synchronization.
Its entry limit is not an RSS byte limit; the native report includes cache
counters and peak RSS.

`--gpu-plaintext-ntt` keeps OpenFHE's FFT, rounding and scale handling, then
uploads coefficient-format plaintexts and runs FIDESlib's NTT on the GPU in
batches of 16 RNS limbs. It includes `--fast-plaintext-upload`, which can also
be used independently to eliminate pass-by-value CPU limb copies. Both are
opt-in; they preserve the evaluator's synchronization and buffer lifetimes.
The bridge uses the pinned FIDESlib internal API and needs no dependency patch.
Both executors use `PlaintextPreparation` in
[`fideslib_plaintext_encoder.hpp`](fideslib_stage0/src/fideslib_plaintext_encoder.hpp)
for encoding policy, additive scale repair and GPU upload. Periodic coefficients
select the existing small CPU NTT before the shared upload; enabling GPU NTT
does not replace that cheaper transform with a full-ring transform. Mamba-2's
parallel CPU cache preparation and client encryption retain their stock encoder.
GPU handle registration occurs on the evaluator thread after CPU workers join.

`--direct-plaintext-upload` additionally bypasses the pinned library's
same-type conversion and transfer vectors. The shared bridge copies its owned
uint64 staging buffer onto each limb's stream and waits before releasing it.
This option implies fast upload and works with CPU, periodic and GPU NTT
encoding. Unrecognized modulus layouts retain the library loader.

`--move-plaintext-coefficients` transfers ownership of freshly encoded OpenFHE
RNS arrays while restoring their real NTT parameters. It implies GPU plaintext
NTT; periodic subring encoding keeps its existing path. Both evaluators use
the same `PlaintextPreparationOptions` and report `moved_coefficient_encodes`.

`--borrow-plaintext-upload` transfers the pinned OpenFHE arrays directly,
removing the bridge's remaining host staging vectors. It implies direct and
fast upload. The guard requires a contiguous native vector, a standard-layout
64-bit word wrapper, ordinary moduli and 64-bit target limbs. Other layouts
fall back to direct/staged upload. CUDA reads the byte representation without
a typed alias; the CPU plaintext remains unchanged and alive until every
device has synchronized. GPU transfer and possible driver staging still occur.
Reports distinguish requested borrowing from actual `borrowed_uploads` (Mamba-2)
or `borrowed_plaintext_uploads` (packed). Both new modes remain opt-in.

The Mamba-2 native executable accepts `--fast-plaintext-upload 1` and
`--gpu-plaintext-ntt 1`; its campaign environment uses
`FAST_PLAINTEXT_UPLOAD=1` and `GPU_PLAINTEXT_NTT=1`. The prompt launcher accepts
the same flags without values. GPU NTT implies fast upload in both executors.
Mamba-2 records selected modes and total/evaluation counters in
`measurements.plaintext_encoding`; its periodic counter remains in
`parameters.joint_gate_schedule.subring_encode_calls`.
The direct path is selected by `--direct-plaintext-upload 1` or
`DIRECT_PLAINTEXT_UPLOAD=1`, with actual dispatch counted as `direct_uploads`.
The ownership and borrowing controls also accept native `0/1` values and the
campaign variables `MOVE_PLAINTEXT_COEFFICIENTS` and `BORROW_PLAINTEXT_UPLOAD`.

The packed executor also offers `--naf-rotations`, `--reuse-dead-inputs` and
`--compact-weights`. NAF uses the shared Mamba-2 signed-power decomposition and
the existing rotation keys. Last-use reuse consumes only uniquely owned DAG
inputs, protects retained outputs and aliases, and implies `--inplace-ops`.
Compact matrix storage requires an exact BF16 round trip for every coefficient;
other matrices retain doubles. Masks and arithmetic still use doubles and the
serialized program is unchanged. These switches remain opt-in.
The [resource comparison](../docs/research/2026-09-24-packed-resources.md)
records full Mamba-3 generation, individual prefix controls and Mamba-2's
shared direct-upload comparison.

`packed_resource_inventory PROGRAM.txt` runs without FIDESlib or CUDA. It
reports planned circuit rotations, last-use opportunities, live DAG slots and
lossless matrix storage. Rotation counts exclude runtime refreshes; the live
slot total is a packing opportunity, not a realizable memory or speed claim.

`--bsgs-routing-stages` reuses the existing diagonal-transform planner inside
radix-8 gather/scatter stages when it reduces actual rotation-key steps.
Destination masks preserve the selected coordinates and one multiplication
level per stage. It requires planned refresh and radix-8 routing. Runtime
reports include `optimized_routing_stages` and `routing_stage_rotations_saved`;
the CPU inventory accepts the same flag to predict model rotations, excluding
runtime refreshes. This does not allocate extra rotation keys.

`packed_plaintext_probe OUTPUT.json` checks exact CPU/GPU RNS equality and
encrypted multiplication for multiple slot counts, levels and scale degrees.
Add `--mamba2` to use sparse-ternary keys and complex slots. Both configurations
also check the shared periodic dispatch and cached/new additive scale repair.
See the [encoding study](../docs/research/2026-09-24-mamba3-gpu-encoding.md)
for the matched measurements and supported scope.

CPU contracts require CMake 3.25.2+ and a C++20 compiler, without CUDA or FIDESlib:

```bash
cmake -S native/fideslib_stage0 -B build/stage0-layout-tests \
  -DFHE_STAGE0_BUILD_KERNEL=OFF -DFHE_STAGE0_BUILD_TESTS=ON
cmake --build build/stage0-layout-tests
ctest --test-dir build/stage0-layout-tests --output-on-failure
```

These contracts include packed-program validation and slot routing against
direct indexing. They also run through the Python test suite and CI. Numerical
GPU promotion requires the separate [encrypted validation gates](../docs/validation.md#gpu-integration-gates).
