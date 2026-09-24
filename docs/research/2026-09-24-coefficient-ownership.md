# Owning plaintext coefficient arrays

Moving freshly encoded RNS arrays into their real-parameter polynomial reduces
the matched Mamba-3 prefix from **15.4315 to 15.1540 seconds (1.80%)**. Its
complete 12-layer/five-evaluation candidate passes in **1038.777 seconds
(17.31 minutes)**, with the same four selected tokens and unchanged 0.001
exact/polynomial gates. That full run validates completion and parity; the
incremental timing comparison is the same-binary prefix ABBA.

[Raw runs, frozen sources and comparison script](../../results/dgx/2026-09-24/coefficient-ownership/).

## Change and comparison

The GPU-NTT encoder asks OpenFHE to produce coefficient values with identity
roots, then restores the original moduli and roots. Previously, the restoration
copied each newly owned limb vector. The opt-in path moves the temporary's
unique vector through OpenFHE's validated `SetValues(VecType&&, COEFFICIENT)`
setter. It preserves coefficients, format, scale, noise degree and slots; it
does not move from caller data or context parameters. The pinned library is
unchanged.

Both evaluators use the shared implementation. `--move-plaintext-coefficients`
implies GPU plaintext NTT and fast upload. Periodic subring coefficients,
parallel CPU cache preparation and client encryption retain their earlier
paths. Reports count actual moved coefficient encodes separately from uploads.

The baseline already enables direct upload. Both Mamba-3 modes also retain NAF
rotations, final-use reuse and lossless compact weights. Fresh keys are generated
for each run; the binary, payload, placement, parameters and operation counts
are fixed within each ABBA.

| Model/control | Baseline samples (s) | Move samples (s) | Mean change |
| --- | --- | --- | --- |
| Mamba-3 trained prefix | 15.4186, 15.4445 | 15.1214, 15.1866 | 15.4315→15.1540, −1.80% |
| Mamba-2, two layers/two evaluations | 55.4072, 55.3116 | 55.9637, 55.2868 | 55.3594→55.6252, +0.48% |

Mamba-2 does not demonstrate a model-time improvement in this control. Its
full follow-up was therefore skipped, and subsequent Mamba-2 transfer tests
keep coefficient moves disabled. The path remains an explicit option; a
faster local encoder is not sufficient evidence of a faster model.

## Correctness and full completion

The final binary passes, for both parameter configurations, 160 original
exact-RNS inputs, 160 moved-coefficient inputs and 126 shared-policy cases.
These cover multiple slot counts, levels, scale degrees, periodic coefficients
and cached/new degree-two addends. The encrypted arithmetic gate stays at
`1e-6`. The separate initial prototype retains its mirrored phase samples:
median preparation reductions range from 3.1% to 7.1% across the tested levels
and configurations. Those local figures are not model speedups.

The full Mamba-3 candidate executes 61,696 moved encodes, retaining all model
operation counts and 726 bootstraps. Maximum exact/polynomial errors are
0.000171286/0.000169519 and peak RSS is 27.3139 GiB. Tokens
`[315,279,1614,315]` decode to `The capital of the state of`. The cached
four-step state regression also passes, with 236 hits and maximum exact error
7.7227e-8.

The model campaign's recorded process wall totals 1695.205 seconds, including
setup and all 12 attempts, within its two-hour internal review interval.
Completion hooks collected results, decoded tokens and checked the artifacts.
The final Python suite (286 tests) and 17 CPU contracts pass. The initial
runner test retained a hardcoded argument-list length after adding a flag;
the repaired assertion and its initial failure are preserved. An initial CTest
invocation omitted the project's opt-in test-build flag and found no tests;
the corrected 17-test run is separate. Neither is represented as a GPU failure.

This is one frozen prompt on DGX Spark, using an inline client and
`security=not-set`. No error gate was relaxed. The two models have different
weights, architecture sizes and numerical contracts, so these measurements
are not an architecture speed ranking.
