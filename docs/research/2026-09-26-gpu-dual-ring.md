# GPU ordinary/refresh ring switching

This study tests one representation change: ordinary encrypted arithmetic at
N=32,768 and the existing two-pass S2C-first refresh at N=65,536. The decision
requires numerical qualification, a matched model prefix, and then a matched
full model comparison if the prefix qualifies. Measurements are in progress;
the current source does not yet establish a full-model improvement.

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
