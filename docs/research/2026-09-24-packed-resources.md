# Packed resource costs and shared transfer improvements

Complete trained Mamba-3 SISO 187M generation falls **1114.927→1054.857 s
(18.58→17.58 minutes, 5.39% less)** with the same binary and frozen
12-layer/five-evaluation payload. All four actual tokens match, and maximum
exact/polynomial errors stay below the unchanged 0.001 gates. Matrix storage
falls from 673.3 to 168.3 MiB without changing coefficient values. There is
one complete fresh-key run per mode and a separate mirrored prefix control.

[Raw runs, comparison script and frozen source](../../results/dgx/2026-09-24/packed-resources/).

The full Mamba-3 program's static inventory reproduces 78,740 binary rotations versus 59,900 signed-power NAF rotations (23.93% fewer, runtime refresh work excluded), 2,345 of 2,902 add/multiply nodes with a final-use input, and 88,252,416 matrix coefficients exactly representable in BF16. These are counts and storage facts; target runtime comparisons decide speed claims.

## Implementation and invariants

- The Mamba-2 NAF decomposition is now shared with the packed executor. Its routing planner uses the selected key-switch cost. Existing signed power-of-two keys suffice; no key, polynomial, modulus, slot count or depth change is needed.
- DAG liveness pins all requested outputs. Final binary operands are moved only when their handle is uniquely owned; aliases are cloned. A repeated input such as a square retains two independent mutable buffers. This counts actual avoided clones, not just eligible DAG nodes.
- Both evaluators share a direct plaintext upload path. The existing bridge's uint64 staging arrays are transferred to the same FIDESlib limb streams without its same-type conversion and transfer vectors. Buffers live until synchronization. Coefficient-format inputs retain NTT scratch. Unknown modulus layouts use the original loader, and narrow limbs keep their conversion path. No dependency source is patched.
- Public matrix weights are stored as BF16 only after every coefficient round-trips exactly. Other matrices keep doubles. Shared BSGS masks expand selected coefficients back to double, retaining the arithmetic and input file hashes. Matrix storage changes from 706,019,328 to 176,504,832 bytes (673.3 to 168.3 MiB); this is logical matrix capacity, not a 4x process RSS claim or a compressed on-disk payload.
- Stateful CPU references copy only the retained SSM/conv slices. In the two-layer, batch-two, 64-token Mamba-2 reproducer, each SSM's backing storage changes from 262,144 to 4,096 bytes and conv from 51,456 to 2,304 bytes. Prefill and continued decode are bit-identical. This storage comparison uses no-grad inference; autograd can retain separate graph intermediates.

## Measurement

The predecessor's [shared-preparation study](2026-09-24-shared-plaintext-preparation.md) finished before this cycle built a separate source snapshot. The serial controller keeps GPU measurements separate from builds and other runs. Prefix modes (baseline, NAF, lifetime, direct upload, compact weights, combined) run in mirrored order. The full Mamba-3 comparison uses the same new binary and the same frozen 12-layer/five-evaluation payload. Mamba-2 direct upload has a two-layer/two-evaluation ABBA control; its improved mean qualified the full direct-upload completion run.

The exact/polynomial Mamba-3 gates remain 0.001 and its four generated IDs must match. Mamba-2 retains 0.05 versus its frozen polynomial circuit and actual token equality. The different models and gates are not an architecture speed comparison. All experimental switches remain opt-in.

## Complete Mamba-3 generation

| Metric | Base | Combined |
| --- | ---: | ---: |
| Evaluation (s) | 1114.927 | 1054.857 |
| Native process wall (s) | 1155.961 | 1095.675 |
| Setup (s) | 26.669 | 26.755 |
| Host encoding (s) | 216.858 | 215.662 |
| Upload (s) | 73.414 | 42.865 |
| Bootstrap evaluation (s) | 517.219 | 515.390 |
| Rotations counted by evaluator | 86,063 | 67,052 |
| Of those, refresh pack/unpack rotations | 7,323 | 7,152 |
| Binary-input Clone calls avoided | 0 | 3,898 |
| Public matrix storage (MiB) | 673.3125 | 168.328125 |
| Process peak RSS (GiB) | 27.9648 | 27.3143 |
| Maximum exact error | 0.00008697 | 0.00012660 |
| Maximum polynomial error | 0.00006570 | 0.00012483 |

Both modes retain 726 physical bootstraps, 1108 logical refreshes, 260 refresh
batches, 12005 ciphertext products and 107738 plaintext products. Each makes
61696 GPU plaintext NTTs; all candidate uploads use the direct path. Subtracting
refresh pack/unpack gives **78740→59900 model rotations**, exactly matching
the static inventory. Rotations inside FIDESlib's bootstrap implementation
are not included in either evaluator count.

The 3898 count measures avoided outer binary-input Clone calls; it is not a
count of unique DAG nodes or all device copies. There are 2902 arithmetic
nodes, 2345 with at least one final input, and a node can contribute two
avoided clones. Buffer aliases and retained outputs remain protected.

Both full runs select `[315,279,1614,315]`, producing
`The capital of the state of`. Evaluation saves **60.070 s** and native process
wall saves 60.286 s. The upload phase falls 41.61%; bootstrap timing is nearly
unchanged. These phase timers can be nested and do not partition total time.
The combined full comparison does not assign all savings to NAF: the separate
prefix controls below measure individual effects.

This is one frozen prompt and fresh keys for each run, not a distribution over
keys or contexts. Both use CPUs 15–19, OpenMP four threads, planned/batched
two-pass refresh, in-place scratch, GPU plaintext NTT and profiling. The cache
is disabled in both full modes. Setup, input parsing and final reference checks
are outside evaluation but inside native process wall. The inline client and
`security=not-set` scope remain unchanged. No approximation gate was relaxed.

## Individual prefix controls

Two samples per mode use the mirrored order base/NAF/reuse/direct/compact/all,
then its reverse. All runs pass the unchanged 0.001 gates.

| Mode | Evaluation samples (s) | Mean (s) | Reduction vs base |
| --- | --- | ---: | ---: |
| Base | 16.3172, 16.3564 | 16.3368 | — |
| NAF | 15.9943, 16.0028 | 15.9986 | 2.07% |
| Last-use reuse | 16.3379, 16.4670 | 16.4025 | −0.40% |
| Direct upload | 15.8610, 15.9481 | 15.9046 | 2.65% |
| Compact weights | 16.3991, 16.4982 | 16.4486 | −0.68% |
| Combined | 15.4985, 15.4506 | 15.4745 | 5.28% |

NAF reduces actual prefix rotations 1103→909, including three unchanged
refresh-packing rotations. Reuse avoids 52 clones; direct upload handles all
888 evaluator preparations. Matrix storage changes from 58,834,944 to
14,708,736 bytes. Clone reuse and compact storage do not demonstrate isolated
latency improvements in this control. They remain explicit resource options;
the combined result does not establish a gain from each member individually.

The profiled prefix upload phase changes from 1.1963 to 0.7089 s with direct
upload alone (40.7% less), while bootstrap timing remains near 5.33 s. Compact
weights reduce average process peak RSS from 26.8224 to 26.7738 GiB on this
prefix. The much smaller process-memory change, compared with matrix storage,
reflects the unchanged keys, ciphertexts and backend allocations. Phase timers
can be nested inside operation/bootstrap timers and must not be summed.

Both configuration probes pass 160 exact-RNS input cases at four NTT batch
widths, with ordinary and direct upload, plus 90 shared-policy cases. Maximum
encrypted arithmetic errors are 1.30e-12 (Mamba-2) and 2.17e-12 (Mamba-3).
For Mamba-3, median preparation at levels 21/26/34 with GPU NTT batch 16 changes
from 7.168/5.824/3.600 ms to 6.402/4.968/3.267 ms with direct upload. Each cell
has eight timed samples, mirrored modes and excluded warmups. These local
preparation timings do not predict the complete model's speedup.

## Shared Mamba-2 transfer control

The two-layer, two-evaluation base/direct/direct/base sequence keeps GPU NTT,
subring encoding, parameters, complete operation counters and levels fixed.
Base samples are 58.9675/59.0956 s and direct samples 55.7689/55.7264 s:
**59.0315 → 55.7476 s (5.56% less)**. All runs perform 258 evaluator GPU
NTTs, so this isolates the additional transfer-copy removal. All numerical
gates pass. The full direct-upload run is a separate completion/parity check;
this short mirrored comparison supports the incremental speed claim.

The full 24-layer/five-evaluation run completes in **2029.903 s (33.83
minutes)**, with process wall 2060.957 s, maximum CKKS-to-polynomial error
0.00642355 and peak RSS 37.0618 GiB. Its IDs `[273,253,4687,273]` decode to
`The capital of the Republic of`. It retains 2187 bootstraps, all operation
counts and levels from the preceding full GPU-NTT result, and zero intermediate
diagnostic decryptions. The earlier full run used a different executable, so
its 35.70-minute timing is not the isolated comparison for this change.

The combined cached Mamba-3 state regression also passes: 236 cache hits,
210 avoided lifetime clones, maximum exact error 7.7223e-8 and polynomial
error 4.46e-12. This confirms composition with caching on the synthetic
four-step workload; it is not a new cached full-model timing result.

## Persistent state packing

At the maximum of 174 live DAG handles, the program has 249,288 logical occupied slots out of 5,701,632 (4.37%). Maximum logical demand at any point is 251,592 slots. Dividing by 32,768 gives an ideal capacity of eight ciphertexts, ignoring levels, aliases, layout and access costs. It does not prove those independent states can be evaluated in eight ciphertexts or that doing so is faster. Runtime already batches refreshes by packing compatible values. Persistent co-location needs a separate access schedule with mask/rotation/depth costs; this cycle records the opportunity without changing encrypted state layout.

## Reproduce

The native/runner packed flags are `--naf-rotations --reuse-dead-inputs --direct-plaintext-upload --compact-weights`. The reused-input flag implies `--inplace-ops`; direct upload implies fast upload and composes with `--gpu-plaintext-ntt`. Mamba-2 takes `--direct-plaintext-upload 1` or `DIRECT_PLAINTEXT_UPLOAD=1` through the campaign shell, and a boolean flag through the prompt launcher.

Build `packed_resource_inventory` with `FHE_STAGE0_BUILD_KERNEL=OFF` and pass the frozen `program.txt` for the static count/storage report. Its compiled planners are shared with evaluation. Native GPU reports expose actual rotations and refresh rotations separately, avoided lifetime clones, matrix storage bytes and direct upload dispatch. The controller checks static versus executed non-refresh rotation counts in every packed run.

The original CPU storage harness accidentally called the installed architecture dispatch for both labels. Its invalid output is preserved and marked invalid; the corrected harness calls the two mixer functions directly and reproduces the 256 KiB-to-4 KiB difference.

## Validation and completion

All 286 Python tests and 17 C++ CPU contracts passed. Both exact-RNS probes,
all mirrored controls, the cached state regression and all three full runs
passed. The recorded native process time totals 5306.806 seconds (88.45
minutes), within the four-hour internal review interval; setup is included
in that ledger. Completion hooks collected the results, submitted the desktop
notification and ran source/hash comparisons, token decoding and CLI artifact
validation successfully. Source snapshots and executables from the preceding
study were retained unchanged. No dependency source changes were required.
