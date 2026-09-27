# B300 rotation stream chaining — 2026-09-27

**Adopt as an optional dependency patch.** A fresh full Mamba-3 ABBA
comparison improves **146.36346 → 141.76769 s (3.14%)**, or
**35.44 s/generated token**. Refresh also improves **2.72%**. The existing
even refresh seed, inverse-FFT plan and two-worker preparation remain enabled
in both arms. The [patch and probe](../../experiments/rotation_stream_chain/README.md)
and [raw evidence](../../results/b300/2026-09-27/rotation-stream-chain/README.md)
are retained; normal dependency builds do not apply this patch automatically.

## Mechanism and qualification

The pinned FIDESlib `Ciphertext::rotate(int,bool)` has four device-wide host
barriers for a reduced output. The candidate removes its first three on one
GPU. Existing stream dependencies order copy, ModUp, fused key switching /
automorphism and ModDown. The final host barrier remains, as do ciphertext
destruction/recycling barriers. Multiple GPUs and extended outputs retain
their previous paths. The patch changes no arithmetic, keys or parameters.

`copyLimb` orders source/destination streams; ModUp joins its auxiliary input
and publishes completion; fused rotation waits on source/output streams and
makes consumers wait on its work. ModDown follows those output dependencies.
The terminal synchronization preserves externally visible completion and
scratch lifetime. This is a dependency audit, not permission to remove the
remaining completion barriers.

Nine fresh rotation processes cover both rings (32,768/65,536) and both model
configurations, plus graph-disabled Mamba-3. They pass **1,080 normal-output
exact RNS/metadata comparisons, 1,080 extended-output fallback comparisons
after explicit ModDown, and 27 chains of 24 rotations followed by a scalar
product**. Levels 0/18/34/39/42, scale degrees 1/2, positive/negative shifts,
cold capture, replay, pending producers and live-input preservation are covered.
Decoded maximum error is 1.96e-12 against the unchanged 1e-6 primitive gate.
Mamba-2 uses complex slots and sparse-ternary secrets; Mamba-3 uses real slots
and uniform-ternary secrets. No full Mamba-2 performance claim is made.

Each arm also passes four seed/EvalMod cases, six exact ring maps, 30 encrypted
transfers and 12 two-pass refresh circuits including four warm-ups. Output
levels, scale degrees and factors match. Maximum refresh-circuit errors are
9.30e-8 / 8.12e-8, below 1e-6; independent fresh keys prevent an equal-error
claim. Raw seed and EvalMod retain their respective 1e-8 / 1e-6 gates.

Only CKKS `Ciphertext.cpp.o` changes among 39 archive members. The archive
contains two members named `Ciphertext.cpp.o`; the occurrence-aware audit
identifies the changed CKKS occurrence, leaving the API occurrence unchanged.
Compiler commands match the inherited build. The paired model executables
link identical application objects and OpenFHE libraries against separate
backend archives. The public patch reproduces the measured source exactly.

## Complete-model comparison

The bounded sequence is fresh baseline profiling, encrypted qualification,
one prefix pair, full ABBA, then candidate profiling. Promotion requires at
least 1% mean full reduction, both candidate samples faster than the baseline
mean, no >1% refresh regression and all correctness gates. Prefix regression
above 1% stops full testing. These are engineering screens, not significance
tests; no timing samples are discarded or automatically repeated.

One B300 GPU 4, CPU mask 64–68, NUMA 2, four OpenMP threads, CUDA 13.0.1,
GCC 13.3 and SM103 are fixed. Every full run observes the main thread on CPU
64 and producers on 65/66. The common affinity helper is measurement-only.
Resident model/dependency hashes are rechecked before and after the campaign.

The frozen 187M SISO workload has 12 layers, five encrypted evaluations and
four generated tokens for `The capital`. IDs remain `[315,279,1614,315]`.
Ordinary/refresh rings 32,768/65,536, depth 44, scale 59, first modulus 60,
HYBRID/FLEXIBLEAUTO, dnum 3, uniform-ternary secret, two-pass S2C-first refresh,
multiplier 4096, ceiling 35 and returned level 18 remain unchanged.
**This experiment retains `security=not-set`; 35.44 s/token is not a
128-bit-security result.**

| Process | Evaluation (s) | Refresh (s) |
| --- | ---: | ---: |
| Prefix A | 7.653990835 | 1.006953712 |
| Prefix B | 7.616949874 | 0.985393591 |
| Full A1 | 146.375998476 | 34.839226890 |
| Full B1 | 141.709397176 | 33.838996638 |
| Full B2 | 141.825990320 | 33.904719246 |
| Full A2 | 146.350931023 | 34.795073602 |

| Full mean metric | Baseline | Candidate |
| --- | ---: | ---: |
| Evaluation | 146.36346 s | 141.76769 s |
| Refresh wrapper | 34.81715 s | 33.87186 s |
| Evaluation minus refresh | 111.54631 s | 107.89584 s |
| Setup | 69.69872 s | 69.75323 s |
| Complete process | 233.58117 s | 229.03280 s |
| CPU preparation wait | 18.31306 s | 17.53387 s |
| Plaintext upload/NTT service | 18.06631 s | 17.95573 s |
| Peak host RSS | 26.29185 GiB | 26.29104 GiB |

Evaluation excludes setup and final verification; per-token time amortizes
evaluation over four generated tokens. Complete process includes Docker
invocation/availability checking through exit, excluding external transfers
and lock waiting. Evaluation minus refresh is an arithmetic difference.
The full saving is 4.59577 s: 0.94529 s in refresh and 3.65048 s elsewhere.
The prefix improves only 0.48%; its shape and profiling cannot predict the
full improvement. Historical results are not substituted for this baseline.

Both full arms pass the unchanged 0.001 polynomial/exact gates. Their maximum
errors are respectively 4.577e-6 / 3.222e-6 against the polynomial reference
and 4.753e-5 / 4.699e-5 against the exact reference. There are no nonfinite
values or evaluator decryptions. All 484 bootstraps, 242/484 transfers and
ordinary operation counts match. Two samples per arm and one prompt do not
establish significance or arbitrary-prompt quality.

## Nsight diagnosis and remaining bottlenecks

The new baseline trace uses the retained adopted even-seed executable; full
controls use freshly linked shared application objects. All source and binary
identities are recorded. The candidate uses the measured patched archive.

| Instrumented prefix metric | Baseline | Candidate |
| --- | ---: | ---: |
| Evaluation | 9.29497 s | 9.12195 s |
| Device synchronization calls | 54,217 | 46,825 |
| Rotation-attributed synchronization calls | 10,144 | 2,752 |
| Rotation host inclusive duration | 1.36927 s | 1.15466 s |
| GPU kernels | 305,998 | 305,998 |
| Graph instantiations / launches | 79 / 3,784 | 79 / 3,784 |
| GPU activity interval union | 2.34823 s | 2.31051 s |

The source predicts removal of three barriers from 2,464 reduced rotations:
**7,392**, exactly matching the trace. Every kernel's name/count is unchanged.
GPU sums, nested host ranges and synchronization durations overlap; required
GPU work is included in waits. Activity union is not SM utilization or ideal
headroom. Instrumented timing is separate from the full performance comparison.

The disabled `LimbPartition::multPt` graph previously discussed is reached by
the fused FIXEDMANUAL path. This FLEXIBLEAUTO workload uses `multElement` and
separate rescaling; enabling that flag alone would not help. Existing replay
already amortizes graph creation. This trial adds no new graph capture.

Candidate full linear nodes take 47.53 s (33.52%) and Chebyshev nodes take
45.04 s (31.77%). Their included refresh times are 2.26 s and 23.84 s, so the
33.87 s total refresh must not be added to those node categories. Preparation
wait remains 17.53 s. The candidate prefix still has multiplication barriers
(0.579 s), plaintext adjustment (0.383 s) and rescale (0.466 s) as reachable
costs; these are not full-workload fractions.

The [follow-up analysis; archive member `b300/2026-09-27/rotation-stream-chain/bottleneck-followups.md`](../../results/README.md#archived-provenance)
preserves next candidates and prior negative cache results. A static scan
confirms 51,240 dense weight preparations but only 10,248 identities before
level/scale. Reusing completed GPU plaintexts might additionally avoid upload
and NTT, but actual level-specific hits are unmeasured. The current loader
retains both value and auxiliary arrays: one level per identity would cost
100.08 GiB at 20 moduli or 225.18 GiB at 45, before other storage. A byte budget
and measured consumption-level trace are required. This is not a speed claim.

The first profile attempt lacked Python in its container; the shell driver
fixed it. The first probe build passed a temporary to an API requiring a
plaintext lvalue; that was fixed before qualification. Both failed preparation
attempts remain in the evidence. All gates and source audits pass. The
22 stopped study containers and regenerable remote trees are removed after
verifying a retained compiled archive.
