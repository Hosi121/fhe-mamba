# Generation cost: arithmetic bounds and conditional timing limits

The reference for this calculation is the periodic-coefficient baseline:
**2573.366 seconds (42.9 minutes)** for the frozen 24-layer/five-evaluation
request. The later [subring run](2026-09-22-subring-gate-encoding.md) completes
it in 2310.796 seconds; the bounds below retain their original reference.
There is no single proven minimum
latency for this FHE model yet. We can already calculate several useful bounds,
provided their assumptions stay explicit.

The [calculator](../../fhemamba/experiments/analyze_generation_cost.py) checks
the complete payload hash against the native artifact, reads every frozen gate
coefficient, and emits a [reproducible report](../../fhemamba/results/dgx/2026-09-22/subring-gates/cost-model.json).
It performs no encrypted evaluation and changes no approximation.

The follow-up [normalization bound](2026-09-22-normalization-bounds.md) also
derives a necessary degree for any polynomial meeting the same public interval
and relative-error target, and measures the budget for removing one internal
normalization refresh.

## 1. A limit on optimizing one existing phase

Two non-overlapping measured buckets are joint gates (908.037 s) and bootstrap
(1082.114 s). The remaining evaluation time is 583.215 s. Other named phase
timers contain bootstrap time, so summing all timers would count work twice.

For gate speed factor `g` and bootstrap speed factor `b`, holding the rest fixed:

```text
T(g, b) = 908.037/g + 1082.114/b + 583.215 seconds.
```

| Hypothetical change | Evaluation time |
| --- | ---: |
| Current measured implementation | 42.89 min |
| Gates twice as fast | 35.32 min |
| Gates take zero time | 27.76 min |
| Bootstrap twice as fast | 33.87 min |
| Bootstrap takes zero time | 24.85 min |
| Both twice as fast | 26.30 min |
| Both take zero time | 9.72 min |

These are conditional Amdahl calculations, not intrinsic FHE lower bounds or
performance predictions. They establish that optimizing only coefficient
encoding cannot bring this fixed remainder below 27.76 minutes. Even that
limit removes the whole gate phase, including work beyond encoding.

## 2. Work that the periodic representation makes redundant

For ring dimension `N = 65536`, coefficient period `s = 32`, and `L` remaining
RNS primes, a radix-two full-ring NTT performs

```text
B_full = L * (N/2) * log2(N) = L * 524288 butterflies.
```

The coefficient polynomial has the form `f(X) = g(X^(N/(2s)))`. Its NTT values
are repeats of a `2s = 64` point NTT, so the same modular evaluations require

```text
B_small = L * s * log2(2s) = L * 192 butterflies.
B_full / B_small = 2730.667.
```

This is an arithmetic count for the modular transform alone. The
[implemented adapter](2026-09-22-subring-gate-encoding.md) still expands its
answer into the original full-ring plaintext. Its cost becomes
`O(L*2s*log(2s) + L*N)`, rather than `O(L*N*log(N))`. Allocation, expansion,
transfer, ciphertext products, and synchronization remain. Thus a 2731-fold
reduction of butterflies does not imply a 2731-fold latency improvement.

At consumption level 26 (`L = 19`), one row needs 3648 small-transform
butterflies instead of 9,961,472. Its compact NTT representation is **9728
bytes**, while the expanded plaintext is **9,961,472 bytes**: a **1024-fold
representation difference**. Preserving the compact form through GPU
multiplication could remove that expansion, but needs a new backend path.

The frozen matrix dimensions contain **121,975 coefficient rows** across the
five evaluations, before any exactly-zero row skipping. PS decomposition and
block scaling can change row values but do not require new encrypted inputs.
The native subring-call counter independently checks how many rows execute.
These public coefficients repeat across tokens; they are not new secret data.

## 3. A genuine algebraic depth bound

For a circuit built from additions and binary multiplications, starting with
a degree-one input and public constants, multiplication depth `d` can produce
a polynomial of degree at most `2^d`. This follows by induction: addition takes
the larger degree, while multiplication adds the two degrees.

The frozen gate computes `write = p(z)^2 * q(z)^2`. Reading the actual nonzero
coefficients for each head gives a maximum write degree of **3072**, occurring
in layers 4 and 6. Therefore the exact frozen polynomial requires at least

```text
ceil(log2(3072)) = 12 dependent ciphertext/ciphertext multiplication stages.
```

This concerns one gate's arithmetic dependency depth. It is neither a bound
of 3072 total multiplications nor a bound on the entire model's latency or
bootstrap count. Scalar products and rescaling also consume CKKS resources.
If the contract permits a different approximation within an error budget,
its required degree must be established separately; the 12-stage bound does
not prove that every sufficiently accurate implementation needs 12 stages.

A high degree does not require a dense evaluation circuit: `(1+x)^1024`, for
example, needs ten squarings after its initial addition. The present `p` and
`q` are dense fitted polynomials; the example does not factor them. Replacing
them with structured compositions is an algorithmic research direction, with
a new approximation and invariant certificate required before use.

## 4. Bootstrap count is an implementation choice, not this bound

The baseline executes **2187 physical bootstraps**, averaging **0.495 seconds**
per physical call in this run. The recorded event ledger partitions them:

| Checkpoint family | Physical bootstraps |
| --- | ---: |
| Block and gated normalization internal stages | 920 |
| Joint write and scaled readout | 480 |
| Residual and input projection | 460 |
| Carried state/FIFO | 279 |
| Remaining checkpoints | 48 |

These counts reflect the chosen public normalization schedules, precision
restoration, packing, and available modulus levels. They are not proven
minimum counts. In particular, many transient refreshes use two physical
calls for accuracy. Removing a call without an error-propagation argument
does not preserve the existing `0.05` gate by construction.

Increasing the modulus budget is also a tradeoff: fewer refreshes can increase
the number of RNS primes, per-operation work, key sizes, and the ring dimension
needed for a given security target. The objective is the complete schedule,
not simply minimizing the number of bootstrap calls.

## 5. Hardware-only ideal time needs the right throughput model

For a fixed circuit and memory schedule, a useful lower-bound model is

```text
T >= max(mandatory_DRAM_bytes / attainable_bandwidth,
         modular_work / attainable_modular_throughput,
         dependent_operation_latency).
```

The advertised DGX Spark memory bandwidth is **273 GB/s**.
[NVIDIA hardware documentation](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
At that peak rate, writing one expanded level-26 coefficient plaintext once
would take about **36.5 microseconds**. Repeating that hypothetical write for
all 121,975 rows would take **4.45 seconds**. This is a bandwidth budget example,
not a model-runtime bound: real levels vary, cache residency and mandatory
DRAM traffic have not been measured, and encoding/multiplication do more work.

Likewise, FP4 tensor FLOPS do not measure the 59-bit modular integer arithmetic
used here. An absolute ideal latency needs measured or independently derived
modular throughput and a dependency/memory ledger. Claiming a model-wide
minimum in seconds from the marketing FLOPS would be unsupported.

## Reproduction

```sh
.venv/bin/python fhemamba/experiments/analyze_generation_cost.py \
  --artifact fhemamba/results/dgx/2026-09-22/periodic-gates/m2_chain_periodic-client-generation_l24_t5.json \
  --payload runs/client-generation-20260922/payload \
  --output runs/gate-subring-serial-20260922/cost-model.json
```

The report binds the native measurements, every coefficient metadata file, the
payload, and the calculator source. Arithmetic counts, conditional timing
scenarios, and bandwidth examples are labeled separately from measurements.
