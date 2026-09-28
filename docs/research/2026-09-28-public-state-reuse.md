# Reusing identical public initial ciphertexts — 2026-09-28

**Adopt as an optional initialization optimization.** Two matched full-model
pairs reduce mean evaluation from **1,157.98896 to 1,078.48210 seconds (6.87%)**,
or **67.40513 seconds per generated token** for this 16-token request. Both
pairs improve, all token/error/security gates pass, and the predeclared 2%
pooled gate is met. The option remains disabled by default.

The [long-generation profile](2026-09-28-long-generation.md) identifies about
85 seconds of public-state preparation before ordinary evaluation. The frozen
16-token program has 120 live public source nodes but only two distinct values
after padding: 108 zero vectors and 12 vectors with 768 leading ones. The
maintained `packed_resource_inventory` now reports both counts.

The [records](../../results/b300/2026-09-28/public-state-reuse/README.md) retain
native outputs, qualifications, comparisons, exact commands, source snapshots
and input/library/binary identities. No additional GPU kernel optimization is
implemented in this study; the later NTT analysis identifies a future trial.

## Results

| Fixed placement and order | Control (s) | Reuse (s) | Reduction |
| --- | ---: | ---: | ---: |
| GPU 5: A1 then B2 | 1,160.41052 | 1,078.94918 | 7.02% |
| GPU 6: B1 then A2 | 1,155.56740 | 1,078.01502 | 6.71% |
| Mean | 1,157.98896 | 1,078.48210 | 6.87% |

Each request selects 16 actual tokens after 17 server evaluations of the full
12-layer trained Mamba-3 SISO model. All IDs match both frozen references.
Across the four runs, the largest polynomial error is `1.42692e-5` and the
largest exact-reference error is `6.93592e-5`, below both unchanged `0.001`
gates. The single-ring classical-128 audit remains `N=131072`, `Q=2656`,
`P=720`, `QP=3376 <= 3523`, uniform ternary secret, sigma 3.19 and four HYBRID
digits. Both bootstrap passes and merged refresh correction remain enabled.

| Mean component / resource | Control | Reuse |
| --- | ---: | ---: |
| Public-source preparation | 82.20686 s | 1.36215 s |
| Remaining evaluation | 1,075.78210 s | 1,077.11995 s |
| Refresh, included in remaining evaluation | 429.80682 s | 430.07113 s |
| Complete process wall time | 1,324.37807 s | 1,244.47240 s |
| Peak host RSS | 66.40537 GiB | 58.17141 GiB |
| Sampled peak GPU allocation | 179.41180 GiB | 179.41180 GiB |

Removing 118 public encryptions saves 80.84471 seconds in that operation group;
the complete evaluation saves 79.50686 seconds. The remaining evaluation is
slightly slower in this sample, so these results do not establish faster
steady-state arithmetic or refresh. Host RSS also falls; each clone keeps
independent GPU storage while the backend can share its immutable CPU data.

Evaluator counters in every full run record 52,547 ciphertext products,
588,248 ciphertext/plaintext products and 364,758 rotations, including 37,270
refresh-packing rotations but excluding bootstrap-internal rotations. There
are 4,426 physical bootstraps and 7,887 logical refreshes.
Cache, scheduling, liveness and the remaining count gates
also match. Peak live DAG handles remain 229; reuse temporarily retains at most
two extra immutable seeds during public-source initialization.

The separate-device two-token screen passes at 267.17779 versus 184.68592
seconds. Its larger 30.88% reduction reflects the larger share of fixed
initialization work and is not the full-request claim. All four full runs use
the same source archive, executable, static libraries and payload hashes.
Actual expanded commands and completion timestamps verify each matched pair
and its execution order.

Python checks pass 424 tests; the later profiler-analyzer change passes its
three focused tests. Native contract checks pass all 23 tests, including
mutable clone isolation, padding identity, secret-source exclusion, source
scheduling order, seed release and fresh reuse state for the next request.

## Implementation and invariants

The optional `--reuse-public-ciphertexts` path encrypts each distinct public
source once within an evaluation and clones it for equal source nodes. The
planner compares exact coefficient bits after removing trailing positive zeros,
which `encrypt()` restores to the fixed slot count. Signed zero, coefficient
order and nonzero bits remain distinct. Model-specific rules are unnecessary.

Every source receives a separate mutable ciphertext handle. The backend's
existing copy-on-write CPU representation and independent GPU copies preserve
caller ownership. An immutable seed remains available only until the last
equal source is materialized; source scheduling order does not matter. The
reuse object is local to one evaluation and public key. Neither private input
nor client feedback is eligible, and no encryption is reused across requests.

The model, weights, polynomial coefficients, refresh bounds, DAG dependencies,
operation order and cryptographic profile remain fixed. The encrypted noise
in equal public sources is now correlated, so algebraic equality alone does
not qualify numerical accuracy. The existing hidden-output and token checks
remain mandatory. This is an initialization optimization: its fractional
benefit decreases as a request generates more tokens.

## Measurement contract

Both arms use the same newly built executable, dependencies, frozen payload
and classical-128 profile, with both 0.001 error gates. The only option change
is public ciphertext reuse. Setup, final client token selection, output
validation, teardown and external transport remain outside `eval_seconds`.
Each process has fresh keys and cold evaluator caches.

The two-token prefix first screens both modes for all token, numerical and
security gates. Its separate-device timings are exploratory. Full comparison
then uses two fixed GPU/CPU/NUMA placements: one executes A then B, and the
other B then A. Both pairs must improve, and the pooled reduction must reach
2%, before retaining a performance claim. Concurrent requests on separate
placements remain a limitation; this is a descriptive comparison, not a
statistical significance test.

The contract also requires equal encrypted arithmetic, rotation and refresh
counts, cache behavior, scheduling and generated IDs. Native counters must
show 120 public encryptions in the control and two in the candidate. Completion
hooks require successful numerical qualification before dependent runs start.

This study does not qualify the previously rejected 64-token request, arbitrary
prompts, process-separated deployment or Mamba-2 full-model generation.

## Further interpretation of the existing GPU counters

During the full-run comparison, the previous raw Nsight Compute exports were
reanalyzed locally. Each row below still represents only four selected kernel
launches under replay, not an average over the model. All columns are the named
hardware unit's `avg.pct_of_peak_sustained_elapsed` counter.

| Kernel family | Instruction issue | ALU pipeline | FMA-heavy pipeline | L2 throughput |
| --- | ---: | ---: | ---: | ---: |
| NTT mode 1 | 56.79–62.29% | 51.46–52.17% | 62.32–63.11% | 8.68–11.87% |
| RNS decomposition/base extension | 58.57–59.77% | 48.27–50.49% | 60.50–64.29% | 3.51–3.92% |
| Hoisted key switching | 46.79–47.24% | 34.89–35.22% | 54.41–54.93% | 21.24–21.47% |

FMA-heavy also executes integer multiply instructions; its name does not imply
that these RNS kernels use approximate floating-point arithmetic. NVIDIA's
[pipeline definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#pipelines)
also distinguish physical and aggregated pipelines, so the columns must not
be added together. The observed instruction-issue activity strengthens the
case for examining instruction dependencies and latency hiding. It does not
identify a particular stall reason: that requires scheduler/warp-state data
and a subsequent matched implementation comparison.

The maintained Nsight CSV analyzer accepts repeated `--metric` arguments to
extract additional recorded counters with their units and input/source hashes.
These derived records preserve the original captures and their selection scope.

The same selected NTT launches expose a concrete register-residency constraint:

| NTT half | Threads/block | Registers used / allocated per thread | Block limit: registers / shared memory / warps |
| --- | ---: | ---: | ---: |
| First | 256 | 38 / 40 | 6 / 6 / 8 |
| Second | 128 | 42 / 48 | 10 / 12 / 16 |

The second half therefore admits at most ten resident blocks under the reported
limits. If a future implementation reduces actual allocation to 40 registers
per thread without spills, register capacity would admit twelve blocks, matching
the shared-memory limit (75% theoretical warp occupancy instead of 62.5%).
This is a conditional capacity calculation, not an observed performance gain.
Reducing registers alone would not increase the first half's occupancy because
shared memory already imposes the same six-block limit. Neither observation
justifies forcing a register cap that introduces local-memory spills.

The next arithmetic candidate is delayed modular normalization inside NTT
butterflies. In the earlier graph-node trace after 90 seconds, the listed
NTT/INTT variants contribute 135.474 seconds out of 218.926 summed kernel
seconds (at least 61.88%). This is a share of overlapping kernel durations
in a two-token prefix, not a full-model wall-time fraction or a speed forecast.
It supports prioritizing transforms over isolated, infrequent kernels.

The inspected FIDESlib `ALGO_SHOUP` path uses separate modular
multiplication, addition and subtraction in `CT_butterfly`, and the analogous
three operations in `GS_butterfly`. Harvey's
[Algorithms 3 and 4](https://arxiv.org/pdf/1205.2926) instead retain redundant
residues, removing two normalization steps per butterfly when the prime is
less than one quarter of the machine-word range. The final transform result
must return to the canonical residue interval before consumers that require it.
For 64-bit words this bound is `p < 2^62`; the current 59/60-bit moduli make
the direction plausible without reducing cryptographic parameters.

This is a candidate, not an implemented or measured speedup. A future trial
must audit every modulus and fused NTT input/output boundary, compare exact
RNS words across all used modes at ring 131,072, and measure instructions and
register residency. The earlier
[register/shuffle tail](2026-09-26-b300-ntt-warp-tail.md) passed exactness but
slowed its prefix by 2.31%, which is why fewer shared-memory accesses alone
are insufficient evidence. The present study implements only public-source
ciphertext reuse.
