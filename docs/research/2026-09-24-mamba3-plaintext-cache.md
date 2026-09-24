# Bounded plaintext reuse in the Mamba-3 packed executor

A 64-entry cache of encoded multiplication masks reduces a matched trained
prefix from **19.5937 to 19.3164 seconds (1.415%)** on DGX Spark GB10.
All four ABBA runs pass the unchanged exact/polynomial 0.001 gates.
This is a small component result. Complete trained generation was not rerun;
the previous [24.95-minute full-session result](2026-09-24-mamba3-depth-batching.md)
does not include this change.

[Raw runs, comparison script and frozen sources](../../results/dgx/2026-09-24/mamba3-plaintext-cache/).

## Mechanism and constraints

The [previous profile](2026-09-24-mamba3-microkernels.md) attributed about a
third of prefix evaluation to host plaintext preparation. `plain()` called
OpenFHE's CKKS encoder on every request, then FIDESlib uploaded the result.
The new `--cache-plaintexts` option retains both representations for reuse.

The cache belongs to one evaluator with a fixed context and slot count.
It accepts only multiplication masks containing zeros and at most one finite
nonzero coefficient. Dense weight diagonals bypass it to avoid displacing
reusable masks; additive plaintexts retain their scale-degree correction path.
Hits require the same CKKS level, vector length and exact coefficient bits.
Hashes only filter candidates; a full bit comparison prevents false hits.
No polynomial, arithmetic order, precision or cryptographic parameter changes.

FIDESlib's `EvalMult(Ciphertext, Plaintext&)` loads an unloaded handle once.
Its `multPt(const Plaintext&)` reads the operand and performs level adjustment
on a private plaintext copy. Cached source handles therefore remain reusable.
Existing device synchronization stays in place before subsequent eviction.

LRU eviction bounds retained entries to 64, including up to 16 MiB of input
coefficient keys at 32768 slots. Encoded CPU/GPU RNS storage adds memory whose
size depends on level and backend allocation. This is an entry limit, not an
RSS byte limit. Process peak RSS was about 26.822 GiB in both modes; that peak
does not measure the cache's incremental live allocation.

## Matched comparison

All four fresh processes use the same executable, 127-node program, fixture
and CKKS parameters. Both modes enable planned/batched two-pass refresh and
in-place scratch reuse. They use Cortex-X925 CPUs 15–19, `OMP_NUM_THREADS=4`,
and host timing markers without an attached Nsight profiler. Each starts with
an empty cache; population, lookup and eviction costs are inside evaluation.
Setup and reference decryption are excluded from evaluation timing but charged
to the campaign budget. No other target build or benchmark runs concurrently.

| Run | Cache | Evaluation (s) | Host preparation calls | Cache hits |
| --- | --- | ---: | ---: | ---: |
| A1 | Off | 19.5409 | 888 | 0 |
| B1 | On | 19.3087 | 825 | 63 |
| B2 | On | 19.3242 | 825 | 63 |
| A2 | Off | 19.6466 | 888 | 0 |

Mean host encoding time falls **6.1545 → 5.8296 s**, while evaluation falls
**19.5937 → 19.3164 s (1.0144×)**. Each candidate avoids 63 preparations and
initial uploads. It still has 413 admitted misses, 349 evictions and 380 dense
bypasses. The 32 additive requests remain uncached. `host_encodes` counts
preparation requests; an additive request may internally encode twice for
scale-degree correction, as before.

Operation counts are identical: 8 bootstrap calls, 181 ciphertext products,
1440 plaintext products and 1103 rotations. All four runs pass, with maximum
exact error **2.44e-5** and polynomial error **2.33e-5**. The program evaluates
one trained layer's first prompt token and checks its final hidden vector;
it does not select tokens. Two samples per mode establish only this narrow
measurement, not a guarantee for other workloads or hardware.

## Regression and adoption

The same executable also passes a four-step synthetic mixer with all 20 output
and carried-state checks, maximum exact error **7.72e-8**, polynomial error
**6.47e-12**, and no evaluation decryption. It records 236 cache hits and
33.5684 s evaluation. This is a correctness run without a matched baseline;
its time must not be compared with earlier unpinned runs as cache speedup.

Local verification passes **280 Python tests and 16 native CPU contracts**.
The cache contract also passes on the target ARM CPU. Tests cover deliberate
hash collisions, signed zero, differing lengths/levels, LRU eviction, handle
ownership and bypasses. An initial campaign-resume self-test rejected its own
changing source fingerprint because the test log was written into the
untracked evidence directory. Redirecting live logs to ignored `runs/` and
rerunning the unchanged suite resolves that interference; both logs are kept.

Keep the option **off by default**. It produces a modest prefix improvement,
but avoids only 7.1% of preparation requests and leaves most encoding work.
The cache policy is not a solution to dense, mostly unique encodings. Further
work should measure reuse distance and unique-encoding cost before enlarging
the cache or changing the NTT path; those are separate untested candidates.
Full-generation latency and token selection with this flag remain unverified.

These five attempts consume 247.47 s of the existing campaign. The complete
ledger totals **7174.24 / 7200 s across 29 attempts**, leaving 25.76 s.
No further GPU run fits with setup and the termination reserve. Security and
inline-client limitations remain unchanged (`security=not-set`).

## Reproduction

Build the packed target and add `--cache-plaintexts` to the existing
`experiments/run_packed_probe.py` command. Native JSON records the enabled flag,
capacity, retained entries, hits, misses, bypasses and evictions.
`run_abba.py` and `run_synthetic.sh` retain the measured launchers; their CPU
numbers and dependency paths are specific to this Spark. Reuse requires an
explicitly authorized budget and prepared payloads, not resetting this ledger.
Run the retained `compare.py` against the saved results to verify the input,
binary, arithmetic counts and error gates before regenerating the summary.
