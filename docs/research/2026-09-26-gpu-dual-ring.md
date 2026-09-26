# GPU ordinary/refresh ring switching

This study tests one representation change: ordinary encrypted arithmetic at
N=32,768 and the existing two-pass S2C-first refresh at N=65,536. It passes
primitive, prefix and full-model gates. Matched full ABBA processes reduce
native evaluation **613.83 → 430.51 s (29.86%)** on DGX Spark GB10, with the
same generated tokens and both 0.001 error gates. Adopt the mechanism as
**opt-in**; the default execution path remains separately qualified.

The baseline is `e6bde4d` / native implementation `b6817f0`, with rotation
sharing, Chebyshev basis sharing, GPU plaintext RNS preparation and S2C-first
refresh. The frozen model weights, polynomial coefficients, prompt, token IDs,
59-bit scale, depth 44 and both 0.001 model gates remain fixed. Changing the
ordinary ring is the hypothesis under test. The experiment retains the
`security=not-set` scope; it does not establish equal security for the two
ring sizes or a production encryption parameter set.

## Encrypted transfer

The smaller context uses exactly the large context's Q primes and their
squared NTT roots. Ordinary secrets are dense uniform ternary in their own
rings. The small secret is embedded as `s(X²)` only for transfer; refresh
continues to use its original independent dense secret.

In a bit-reversed negacyclic NTT, adjacent large-ring entries evaluate a
polynomial at `X` and `-X`. The map to the smaller ring averages each pair
modulo q; it selects the even coefficients. Embedding in the opposite
direction duplicates each small-ring entry. CUDA performs these maps directly
in the NTT representation. There is no ciphertext download, coefficient NTT
round trip or evaluator decryption in a transfer.

Entering refresh embeds the two ciphertext components and switches from the
embedded small secret to the large secret. Leaving refresh switches to the
embedded secret before projecting the components. Key generation is client
setup and is timed separately. Every recurring allocation, synchronization,
map and key switch is included in model evaluation timing. The standalone
component timers exclude encryption, client checks and an artificial fixture
level drop before refresh; their sum is not a complete process time.

A general large-ring vector projects to the average of its two slot halves.
The model route instead embeds the same small vector in both halves, so its
projection preserves the vector. The smaller ring carries 16,384 independent
slots, not 32,768. Public node values and weights stay the same; replicated
linear layouts and refresh grouping must obey the smaller capacity.

The reason projection works on ciphertexts is the module identity
`T(a(X) b(X²)) = T(a)(Y) b(Y)`, where `T` selects even coefficients and
`Y = X²`. After switching to the embedded small secret, applying `T` to both
ciphertext components also applies it to their decryption relation. Direct
projection under an unrelated dense large-ring secret would not satisfy this
identity. The required key switch is part of the measured transfer cost.

The model returns the first refresh result and the residual-correction result
to the small ring separately. This costs one upward and two downward switches
per refresh boundary but preserves the existing unpacking multiplication
depth and refreshed consumed level 18. Combining the results before transfer
would require a different level analysis and is outside this mechanism.

## Qualification and comparison

The standalone probe checks NTT maps against an independent coefficient-domain
oracle at three Q primes. Encrypted tests cover logical widths 1, 32, 768,
1,536 and 16,384; consumed levels 18, 26 and 34; scale degrees one and two;
independent signed values in both large-ring halves; and all inactive slots.
Each small circuit checks the complete final vector at 1e-6 after ordinary
products, both refresh passes, transfers and a subsequent encrypted product.

OpenFHE's host NTT tables are keyed by modulus, so setup explicitly selects
the corresponding CPU ring before generating keys, encrypting or checking
client outputs. GPU contexts have their own tables. The model evaluator uses
only the small-ring CPU encoder after setup; refresh stays on the GPU.

Matched runs use the existing DGX Spark GB10, CUDA 13, CPU affinity 15–19 and
four OpenMP threads. Baseline and candidate keep both previous shared-work
options enabled. Warm-ups are excluded, and process order is alternated.
Builds finish before GPU timing begins. Prefix/full decisions use complete
native evaluation and retain setup-inclusive process time separately.

The three final primitive processes each pass six exact NTT-map cases and
30 paired encrypted-transfer cases. In the two timing processes, ABBA and
BAAB order provide four measured samples per mode after warm-ups:

| Component mean | N=65,536 throughout | N=32,768 ordinary / N=65,536 refresh |
| --- | ---: | ---: |
| Eight ordinary square/bias steps | 43.802 ms | 18.132 ms |
| Upward encrypted transfer | — | 2.719 ms |
| Existing two-pass refresh | 791.474 ms | 790.415 ms |
| Downward encrypted transfer | — | 3.888 ms |

The ordinary component saves 58.60%. The measured upward/downward pair costs
6.607 ms. These are component measurements with the exclusions above. The
model's separate correction transfer adds another downward call per boundary,
so this table must not be substituted for model evaluation time.

The one-layer model prefix passes in all four fresh ABBA processes. Native
evaluation falls **30.6116 → 20.0039 s (34.65%)**, including refresh and all
transfers. Both candidate processes retain token IDs `[6864, 6864]`, the
0.001 error gates and 16 physical bootstrap calls. The result qualified the
one mechanism for the full comparison; it is not an estimate of full speedup.

The smaller ring has less packing capacity. In the prefix, plaintext products
increase **5,432 → 6,872** and rotations **2,530 → 2,731**. Ring reduction
therefore does not simply halve every cost. These additional operations,
encoding, context switches and refresh packing are included in the result.

## Full-model result and decision

The same frozen 12-layer SISO 187M request performs five encrypted evaluations
and generates four tokens. All four fresh processes pass:

| Process order | Native evaluation | Refresh wrapper |
| --- | ---: | ---: |
| Baseline 0 | 613.8714 s | 239.0050 s |
| Candidate 1 | 430.0679 s | 217.6819 s |
| Candidate 2 | 430.9611 s | 218.0550 s |
| Baseline 3 | 613.7908 s | 239.0119 s |

The mean saves **183.32 s**. Disjoint category means subtract nested refresh
from operation totals:

| Category | Baseline | Candidate |
| --- | ---: | ---: |
| Refresh wrapper | 239.01 s | 217.87 s |
| Linear transforms | 95.29 s | 86.57 s |
| Polynomial evaluation | 118.20 s | 49.81 s |
| Routing | 146.88 s | 69.38 s |
| Other evaluation work | 14.46 s | 6.88 s |
| **Complete native evaluation** | **613.83 s** | **430.51 s** |

Ordinary work falls **374.82 → 212.65 s (43.27%)**. The refresh wrapper falls
**239.01 → 217.87 s (8.84%)**, including transfers and smaller-ring packing
and unpacking. The existing large-ring bootstrap circuit is unchanged; the
primitive table does not show a corresponding core-bootstrap acceleration.
Host encoding is nested in these totals and falls **64.66 → 47.21 s**.

The full operation counts are:

| Count | Baseline | Candidate |
| --- | ---: | ---: |
| Ciphertext products | 11,540 | 11,540 |
| Plaintext products | 106,481 | 135,279 |
| Rotations outside bootstrap internals | 62,369 | 66,464 |
| Physical bootstrap calls | 484 | 484 |
| Logical refreshes | 1,335 | 1,334 |
| Refresh batches | 210 | 210 |
| Upward / downward ring transfers | 0 / 0 | 242 / 484 |

Both candidates produce IDs `[315, 279, 1614, 315]` and the full text
`The capital of the state of`. The worst candidate exact/polynomial errors
are **4.72545e-5 / 5.98173e-7**, with zero non-finite values and zero evaluator
decryptions. Fresh-key error differences are not accuracy improvements.

Setup increases **26.40 → 31.79 s**. Complete process time, including parsing,
setup and final validation, falls **654.52 → 476.39 s (27.21%)**. The candidate
averages **107.63 s per generated token**, or **119.10 s including process
overhead**, amortized over this five-evaluation/four-token request. This is
not steady-state token latency. Peak process RSS stays about **26.21 GiB**;
the peak includes key setup, so it does not establish a GPU-memory reduction.

Adopt `--gpu-dual-ring` for the qualified packed Mamba-3 configuration: both
complete candidate samples improve on both controls while preserving the
frozen numerical gates, scale, depth and refresh circuit. The experiment ends
after this one mechanism's qualification and full comparison. The final binary
also passes a default-path prefix without the flag and three early rejection
checks. Local release checks pass **292 tests**, **87.60% coverage**, and the
Spark build passes all **21 C++ contracts**.

Two samples per mode on one prompt do not establish statistical significance,
arbitrary-prompt quality, longer carried-state horizons or a full Mamba-2
speedup. The smaller ordinary ring changes encryption parameters; the retained
`security=not-set` scope does not imply equal security at both ring sizes.
`remaining-cost-notes.json` records two unimplemented follow-ups from static
review: final-use ownership in the downward transfer and dense-linear layouts
under reduced packing capacity. Neither has a measured speedup claim.

## Reproduction

The implementation is opt-in with `--gpu-dual-ring`, requiring
`--s2c-first --planned-refresh --batch-refresh`. The Spark build compiles it;
custom native builds use `-DFHE_STAGE0_GPU_DUAL_RING=ON` and the existing
patched FIDESlib. Unsupported program capacity is rejected before key setup.

The standalone source is `experiments/gpu_dual_ring/`. It links the same
`fideslib_dual_ring.hpp` and CUDA mapping kernel as the packed evaluator.
The campaign controller records exact commands, binary/source identities,
errors and timing, including failed runs. The CPU-only ring-switch study from
September 25 remains historical evidence for a different backend.

For the frozen full generation payload produced by the
[Mamba-3 guide](../mamba3.md), run from a checkout on the Spark host with its
documented OpenFHE/CUDA library path. Choose a fresh output directory:

```bash
OMP_NUM_THREADS=4 taskset -c 15-19 python3 experiments/run_packed_probe.py \
  --binary "$HOME/fhemamba/spark/kernel/packed_fideslib" \
  --payload /path/to/mamba3-full-generation-payload \
  --output runs/mamba3-dual-ring-001 --timeout 2400 --tolerance 0.001 \
  --planned-refresh --batch-refresh --frontier-refresh --s2c-first \
  --inplace-ops --profile-evaluation --naf-rotations --reuse-dead-inputs \
  --compact-weights --gpu-plaintext-ntt --direct-plaintext-upload \
  --move-plaintext-coefficients --borrow-plaintext-upload \
  --bsgs-routing-stages --cache-plaintexts --gpu-plaintext-rns \
  --hoist-rotations --share-chebyshev --gpu-dual-ring
```

Remove only `--gpu-dual-ring` for a comparison using the same newly built
binary. The recorded ABBA campaign instead uses the separately preserved
released baseline executable; the final-binary default-path regression
checks the refactored code without the flag.

The native implementation commit is `c2d7d4d1441f48ee17b25327093e68acf4b868df`.
All 92 native source files match the full-comparison source manifest and
archive byte for byte. The prefix uses the same arithmetic; its only source
difference is the missing new flag in the usage help string, recorded in
`prefix-to-full-source.diff`. The final full comparison uses the committed
source. The existing baseline executable is preserved separately.

The [artifact directory](../../results/dgx/2026-09-26/gpu-dual-ring/) contains
`prefix-*/run.json` and `full-*/run.json` with the exact process commands.
Each command uses its original frozen payload, four OpenMP threads and CPU
affinity 15–19. The preceding ownership, routing, cache, GPU-RNS, S2C-first,
rotation-sharing and basis-sharing flags are enabled in both modes; the
candidate adds `--gpu-dual-ring`. Source archives retain the exact controller
and probe revisions used at each stage, including the initial standalone
qualification before the stronger asymmetric-half and NTT-oracle checks.

Two initial compilation failures are retained: the host NTT class namespace
and a conflicting CUDA synchronization declaration. Both were corrected
before inference qualification. They are build failures, not numerical
results. The full comparison has a completion/failure hook; a dependent hook
checks the final binary's default path only after comparison processes end.
