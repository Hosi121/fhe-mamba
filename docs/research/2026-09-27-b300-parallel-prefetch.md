# B300 ordered parallel plaintext preparation — 2026-09-27

**Adopt the optional two-worker path.** Full Mamba-3 ABBA improves
**173.16 → 148.80 s (14.07%)**, or **37.20 s/generated token** for this
four-token request. The CPU-preparation wait falls **39.84 → 17.95 s**.
Accuracy, generated tokens, cryptographic settings and refresh remain fixed.
This result uses the common, verified CPU-placement control described below.

This is one mechanism: prepare two independent public plaintext operands on
two CPU workers, then feed the existing GPU evaluator in the original order.
The adopted single-producer pipeline, inverse-FFT plan and unmodified FIDESlib
backend are the baseline. The rejected forward-NTT shuffle patch is absent.
The [raw evidence](../../results/b300/2026-09-27/parallel-prefetch/README.md)
contains frozen sources, commands, all ABBA samples and the adoption decision.

## Why this can help

Each replicated BSGS linear operation consumes a sequence of public diagonal
masks. The masks and their CPU CKKS coefficient encodings are independent once
the baby ciphertext levels are fixed. One producer can leave the GPU consumer
waiting for the next mask, even though another CPU core is available.

The candidate reserves consecutive indices under one mutex and lets two
workers prepare them independently. Completed results enter a two-slot ring;
the consumer takes only its next expected index. It performs the same cache
lookup, upload, multiplication, additions, rotations and synchronization in
the same order. It neither changes arithmetic nor overlaps two GPU evaluators.

For ideal uniform items, steady-state preparation service can change from
`Tprepare` to `Tprepare / 2`, while consumer service stays `Tconsume`.
The pipeline throughput is bounded by their maximum, plus handoff and finite
window effects. This is a conditional model, not a twofold inference speedup.
The measurable target is exposed `prefetch_wait_seconds`; the sum of encoding
service times already overlaps other work and is not removable wall time.

## Ordered ownership and bounded storage

`assigned - consumed <= capacity` holds under the queue mutex. Reservation
counts both in-flight and completed items. A consumer clears its exact slot
before advancing `consumed`, so the next item using the same ring position
cannot overwrite a live result. GPU and plaintext-registry ownership stays
on the consumer. All encoder geometry and public weights are initialized
before the producers start; each producer owns its output coefficients.

The capacity stays two, plus one operand currently held by the consumer.
The largest measured retained operand payload is 393,216 bytes, so those
three payloads account for at most 1.125 MiB, as in the single-producer path.
This is not a total RSS or encoder-scratch bound: two active encoders also
have temporary allocations and thread stacks. Peak process RSS is retained
in the raw and derived reports.

The first producer error cancels admission and wakes the consumer. Destruction
joins every worker before captured input views can die. The original serial
implementation remains the default, including support for move-only producers.
Parallel producers must be copyable and their shared captures must be safe
for concurrent reads.

## Qualification and unchanged model conditions

| Check | Coverage |
| --- | ---: |
| Native CPU suite | 22 tests pass |
| Python runner validation and forwarding | 29 tests pass |
| Queue ASan + UBSan | Pass |
| Exact RNS coefficients and metadata | 960 cases pass |
| Encrypted primitives | 36 cases pass |

The queue checks include a blocked first callback while a second makes
progress, ordered consumption, admission capacity, move-only payloads,
producer exceptions, cancellation and empty input. ThreadSanitizer compiled
but could not start in this local environment (`unexpected memory mapping`,
exit 66); it is recorded as unavailable, not a passing race check.

GPU primitive coverage includes Mamba-2 and Mamba-3 patterns at both deployed
rings, 32,768 and 65,536. Full timing is Mamba-3 SISO 187M, 12 layers, five
encrypted evaluations and four generated tokens from the frozen `The capital`
prompt. Both arms retain ordinary ring 32,768, refresh ring 65,536, 16,384
slots, depth 44, 59-bit scale, 60-bit first modulus, `FLEXIBLEAUTO`, `HYBRID`,
three large digits, real data, uniform ternary secret, and experimental
`security=not-set`. The two 0.001 error gates and zero evaluation decryptions
remain fixed. This study does not establish a new cryptographic security level.

Full IDs remain `[315, 279, 1614, 315]`; each run has 484 bootstraps, 51,240
prefetched encodes in 240 groups, 78,369 total host encodes, 75,712 compact
uploads and 2,657 fallbacks. No extra rotation, refresh, approximation or
plaintext-cache change is introduced. Full Mamba-2 and other-platform speed
remain unmeasured.

## Matched comparison

| ABBA process | Evaluation (s) | CPU preparation wait (s) |
| --- | ---: | ---: |
| `prefix-0-base` | 8.881335047 | 1.843007 |
| `prefix-1-candidate` | 7.654855588 | 0.771258 |
| `prefix-2-candidate` | 7.800140171 | 0.776790 |
| `prefix-3-base` | 8.884950169 | 1.868373 |
| `full-0-base` | 173.135351160 | 39.729561 |
| `full-1-candidate` | 148.521933951 | 17.691467 |
| `full-2-candidate` | 149.086651202 | 18.214535 |
| `full-3-base` | 173.186675116 | 39.957067 |

The prefix mean improves **8.88314 → 7.72750 s (13.01%)**. Full evaluation
saves **24.3567 s**, of which the measured queue-pop wait reduction accounts
for **21.8903 s**. Linear-node time falls **73.28 → 49.33 s**; refresh time
is **37.82 → 37.70 s**. These are descriptive timings, not independent
additive categories or a causal partition.

Summed host encoding service rises **91.00 → 93.09 s** while evaluation
gets faster: more of that work overlaps the consumer. It must not be
subtracted from wall time. Remaining queue-pop wait is **12.06%** of candidate
evaluation; eliminating only that wait under fixed other costs would save
at most that fraction, not imply an achievable next speedup.

The maximum candidate full polynomial/exact errors are **5.4864e-7 /
4.7078e-5**, both below 0.001. Mean peak RSS is **26.292 → 26.291 GiB**.
Mean setup is **69.23 → 69.29 s**; complete container wall time is
**259.85 → 235.63 s**, including setup and client validation. Thus 37.20
s/generated token describes the evaluation interval; complete process wall
time divided by four is **58.91 s/token**, not 37.20.

The predefined gate requires a mean reduction greater than 0.5% on prefix
ABBA before full ABBA, then the same threshold for full adoption. Both arms
use B300 GPU 4, CPU mask 64–68, NUMA 2, four OpenMP threads and the same
CUDA 13.0.1 image with SM103 code. Builds finish before timing. Each sample
is a fresh process; profiling occurs after all uninstrumented samples.
Durable completion hooks serialize every GPU job. No host clock/governor
setting is changed.

### CPU placement is controlled and checked

The preceding NTT trial discovered that the first serialized OpenMP region
overwrites a foreign `std::thread`'s requested affinity. A separate CPU probe
reproduces the transition from explicitly pinned CPU 65 back to CPU 64.
It also shows that initializing OpenMP once before the affinity call preserves
CPU 65 across subsequent serialized regions.

Both measured arms apply the same helper: warm each producer's OpenMP state,
then pin producer 0 to CPU 65 and producer 1, when present, to CPU 66. Set
`OMP_PROC_BIND=true`, `OMP_PLACES={64},{66},{67},{68}`. The main thread remains
on CPU 64. The observer checks named producer threads during the actual jobs;
it verifies CPUs and allowed masks after initialization, not just the requested
configuration. The second producer uses a core already inside the unchanged
five-core process mask, also available to an OpenMP worker in other phases.

This helper is a **measurement control**, not a portable product affinity
policy. It is archived with both exact measured source trees. The product
option does not hard-code this machine's CPU IDs. Performance outside this
verified placement remains to be measured. CPU frequency samples are retained
without frequency normalization; most generic main-thread observations also
include setup, while named producer samples identify preparation.

The historical 202.33-second pipeline result used different observed CPU
placement. It is not the denominator for this trial's reduction and cannot
isolate the contribution of the new worker count. Two samples per arm on one
prompt establish this comparison only, not statistical significance or a
general speed guarantee.

## GPU activity diagnosis

Both instrumented prefixes record **308,734 kernels**, with summed kernel
duration **2.94133 → 2.94088 s**. Recorded GPU activity interval union is
**2.38736 → 2.40072 s**, while instrumented evaluation falls
**10.58439 → 9.43799 s**. Their activity fractions are **22.56% → 25.44%**.
This supports reducing gaps in work supply, rather than faster GPU arithmetic.
The uninstrumented ABBA, not these profiler times, determines adoption.

The trace's GPU activity interval union counts time with recorded kernels or
copies. Dividing it by instrumented evaluation duration is an activity fraction,
not SM utilization, percentage of theoretical peak, or full-model utilization.
Kernel duration sums can overlap; tracing changes the elapsed duration.
No Nsight Compute replay is needed because the GPU kernels are unchanged.

## Use and reproduce

Enable the shared executor's optional two-worker path with
`--prefetch-plaintexts --prefetch-workers 2` and coefficient encoding, for
example `--gpu-plaintext-rns`. The Python `experiments/run_packed_probe.py`
runner forwards the option in both budgeted and unbudgeted runs. The default
worker count is one; two workers without plaintext prefetch are rejected.

The retained `base-measured-sources.tar.gz` and
`candidate-measured-sources.tar.gz` include the common CPU-placement helper.
The two identical `compile_commands.json` files and generated link commands
record the comparison build against the same static
libraries. Input and library hashes were freshly checked on B300; binaries
are also hashed. The optional inverse-FFT dependency patch remains enabled
in both arms. `baseline-command.json` gives the complete feature set;
the candidate command adds only `--prefetch-workers 2`.

`candidate-product-sources.tar.gz` retains the final portable sources and
Python runner. Its only native difference from the measured candidate,
after removing the common measurement control, is leading indentation in
the queue header. The recorded source audit verifies this explicitly. No C++ token
or line changes are hidden behind that formatting cleanup.

Verify the public evidence files from the repository root:

```bash
python -m fhemamba.benchmarks verify results
```

This checks published file and archive-member hashes. The original study verification checked the original source identities, raw/derived reports, numerical/token
gates and observed CPU placement; its original-byte manifest is distinct from
the public publication manifest. Neither command reruns encrypted inference.

The original study archive passed all 191 file hashes and its offline verifier.
A first analysis attempt
misclassified four container-entrypoint samples as evaluator threads; its failure record remains available. The corrected analysis retains those
startup samples separately. No GPU timing or placement observation changed.
