# Mamba-3 SISO support

Mamba-3 now has a complete mixer and trained-model encrypted generation path.
Model math is shared between the exact PyTorch reference, its frozen
polynomial surrogate and a packed FIDESlib program. The measured Mamba-2-130M
generation path remains available through its existing optimized native kernel.

The trained **Mamba-3 SISO 187M** loader, all 12 blocks and client generation
loop pass full CPU/reference parity and complete encrypted generation: five
evaluations and four actual client-selected tokens.
The ready-node refresh implementation completes in **12.68 minutes**.
The final-source comparison measures **941.95 → 761.00 seconds (19.21% reduction)**,
with bootstrap calls falling from 726 to 476. All four generated IDs and both
`0.001` gates are preserved; maximum exact/polynomial errors are
`0.0000809575` / `0.0000607799`. The 20% target remains unmet. The three-way
polynomial/layout/scheduling combination takes 792.40 seconds and is not
retained in the active executor. See the
[refresh scheduling study](research/2026-09-25-packed-frontiers.md) for all
samples, rejected candidates and final-source qualification.

The preceding upload/routing comparison saves 9,120 model rotations
(59,900 → 50,780); the selected configuration also retains the 64-entry cache.
See the [upload/routing study](research/2026-09-24-borrowed-plaintext.md)
and [cache integration](research/2026-09-24-packed-cache-integration.md).

The [recorded small probes](research/2026-09-24-mamba3-siso.md) pass at 4 and 8
steps: 102.68 s and 195.71 s of encrypted evaluation respectively on Spark.

## Supported boundary

| Component | Mamba-2 | Mamba-3 |
| --- | --- | --- |
| Architecture dispatch and state allocation | `architectures.py` | `architectures.py` |
| State | Convolution FIFO, SSM | Accumulated angle, SSM, previous rotated B, previous x |
| Plaintext formula | `reference.py` | `mamba3.py` |
| Shared arithmetic | `tensor_ops.py`, `ops.py` | `tensor_ops.py`, `ops.py` |
| Encrypted execution | Optimized Mamba-2 native kernel | Architecture-independent `packed_fideslib` executor |
| Validation scope | Recorded 130M generation experiments | Synthetic mixer; trained 187M CPU parity and complete 12-layer encrypted client generation |

The Mamba-3 formula includes input/output projections, B/C RMSNorm and learned
biases, input-dependent negative A (including its floor), softplus dt, sigmoid
trapezoidal mixing, tanh/dt rotary angles, lagged and current writes, D skip and
SiLU output gating. Both rotary fractions (0.5 and 1.0) and optional headwise
normalization before the output gate are implemented in the reference.
MIMO and multiple B/C groups raise explicit errors. Chunked Mamba-3 prefill is
not implemented; its supported scan schedule is `loop`.

The native executor knows arithmetic operations and slot maps, not Mamba model
versions. It packs each tensor into one ciphertext and frees values after their
last use. Broadcasts use sparse placement followed by doubling rotations. It
uses balanced Chebyshev evaluation and can refresh ciphertexts as levels run
low. Encrypted export requires batch size one. The extended format supports
non-power-of-two widths through zero padding. The trained path keeps the SSM
as exact rank-one writes, so a 196608-coordinate state does not need to be
expanded into one ciphertext. This is a finite-history representation with
storage and readout cost growing with the number of tokens; no ranks are dropped.

Monotone slot maps use radix-8 routing: each stage partitions coordinates by
up to three displacement bits and evaluates the masks at the same ciphertext
level. Small maps (at most 32 diagonals) use a direct transform. This reduces
multiplicative depth at the cost of extra rotations; `--legacy-routing` on the
runner restores the original bitwise routing for matched comparisons. Native
reports include time and refresh counts by operation.
On the same trained one-layer fixture, routing reduced evaluation time from
130.17 to 100.83 seconds (22.5%). That routing comparison is separate from
the later full-model depth/batching comparison.

The runner also accepts `--planned-refresh --batch-refresh`. The depth planner
accounts for each operation, eliminates unused nodes and hoists selected
refreshes before fan-out. Algebraic rewrites save levels without changing the
frozen polynomials; small routing maps reuse baby-step/giant-step rotations.
Batch refresh packs live tensors into disjoint slot ranges using their public
bounds, applies the existing two-pass correction and unpacks them. These flags
require replicated linear transforms, radix-8 routing and two bootstrap passes.
One-pass refresh is an experimental native option and fails the trained
one-layer accuracy gate; it is not a supported acceleration setting.

`--frontier-refresh` advances ready nodes that fit their actual CKKS levels
before refreshing a blocked branch. It requires `--planned-refresh --batch-refresh`
and preserves the two-pass refresh. Runtime use counts protect pinned outputs
and duplicate input edges under the changed order; feedback epochs remain
barriers. Omitting this option retains the sequential schedule. The native
report includes `frontier_deferrals` and `maximum_ready_nodes`.

`--hoist-rotations` shares existing signed-digit prefixes and sibling ModUp
preparation within BSGS baby rotations, without adding rotation keys.
`--share-chebyshev` shares the normalized argument and requested basis terms
between compatible polynomial nodes, including sin/cos pairs. A refresh or
level/degree change invalidates that basis; its last consumer releases it.
Both switches are opt-in and preserve the frozen model coefficients. See the
[four-candidate comparison](research/2026-09-25-structural-four.md) for measured
scope, correctness probes and alternative-backend trials.

`--gpu-dual-ring` moves ordinary encrypted arithmetic to N=32,768 while
keeping the existing N=65,536 S2C-first refresh circuit. It requires
`--s2c-first --planned-refresh --batch-refresh` and the matching Spark build.
Logical node widths must fit 16,384 slots; the executor regenerates its
replicated layouts for that capacity. Model coefficients and error gates
stay fixed. See the [GPU dual-ring study](research/2026-09-26-gpu-dual-ring.md)
for qualification and the limitations of this experimental parameter set.

`--profile-evaluation` adds host encoding and mask preparation timers and CUDA
profiler start/stop markers around evaluation. Use an Nsight Systems wrapper
with `--capture-range=cudaProfilerApi --capture-range-end=stop` to exclude setup
from the trace. Profiled timings include instrumentation overhead and should
not be compared directly with unprofiled throughput measurements.

`--inplace-ops` reuses privately cloned GPU scratch ciphertexts during add,
multiply and composite rotations. It leaves original program values intact.
It reduced a matched short trained prefix by 1.8% and a one-layer generation
loop by 2.2% in separate ABBA comparisons; it is
opt-in and is not included in the 24.95-minute full-generation measurement.
Both modes of the later 21.87/18.59-minute encoding comparison include it.
The shared coefficient-mask helper now uses contiguous SIMD-capable copies,
with bitwise parity and a 12.4× local CPU result at 32768 coefficients. See the
[microkernel study](research/2026-09-24-mamba3-microkernels.md) for scope and profiles.

The same `--inplace-ops` option now consumes final-use accumulator terms
through the shared ownership helper. Aliases detach before alignment and
right-input storage stays live through GPU completion. The
[ownership comparison](research/2026-09-24-owned-arithmetic.md) records
exact-RNS checks and a fresh full baseline/candidate pair.

`--cache-plaintexts` is an optional, evaluator-local cache for multiplication
masks containing zeros and one finite nonzero coefficient. It retains up to
64 encoded plaintexts, including their GPU handles, and uses LRU eviction.
Only exact coefficient bits at the same CKKS level can hit; dense weight
diagonals and additive plaintexts keep their original encoding paths. The
existing GPU barriers protect evicted handles. Capacity limits entry count,
not RSS bytes; native results record hits, misses, bypasses, evictions and RSS.
`host_encodes` counts uncached plaintext preparation requests regardless of
profiling; `host_encoding_seconds` requires `--profile-evaluation`.
The [cache study](research/2026-09-24-mamba3-plaintext-cache.md) measures a 1.4%
reduction on a matched trained prefix and passes a four-step state regression.
See the latest [cache integration study](research/2026-09-24-packed-cache-integration.md)
for its controls, full follow-up decision and final recommendation.

`--gpu-plaintext-ntt` prepares coefficients with OpenFHE, then runs their RNS
NTT on the GPU in batches of 16 limbs. FFT, rounding, scale, coefficients and
cryptographic parameters are preserved. It includes `--fast-plaintext-upload`,
which independently removes two CPU limb-vector copies from the standard
upload path. The new encoder also creates additive constants at their required
scale degree directly. Both paths are opt-in and use the pinned FIDESlib API;
existing device barriers protect staging and cached plaintext lifetimes.
Native reports include `gpu_ntt_encodes`, `fast_plaintext_uploads` and
`plaintext_ntt_batch`. Upload timing requires `--profile-evaluation`; its zero
value in the standard path means uninstrumented, not zero work. Exact RNS
probes and matched model measurements are in the
[encoding study](research/2026-09-24-mamba3-gpu-encoding.md).

Further opt-in controls are `--naf-rotations`, `--reuse-dead-inputs`,
`--direct-plaintext-upload` and `--compact-weights`. They respectively use
signed-power rotation keys, consume uniquely owned final DAG inputs, remove
redundant upload staging copies, and store exactly representable matrix weights
as BF16. Weight expansion and all arithmetic remain double precision. The
[native guide](../native/README.md) describes their counters and CPU inventory
tool. The [resource study](research/2026-09-24-packed-resources.md) records
the complete 18.58→17.58-minute comparison, separate prefix controls and
673.3→168.3 MiB matrix storage. Reuse and compact storage alone do not show
isolated latency gains in the prefix control. Persistent state co-location
remains an inventory result, without a new encrypted layout.

Additional measured controls are `--move-plaintext-coefficients`,
`--borrow-plaintext-upload` and `--bsgs-routing-stages`. They remove
coefficient/staging copies and reuse rotations inside radix-8 stages.
The [upload/routing study](research/2026-09-24-borrowed-plaintext.md)
records their separate/combined controls and full-model validation.
The shared policy checks layout and lifetime, falls back when necessary,
and records actual dispatch.

## CPU usage

```bash
uv sync --locked --extra experiments
uv run --no-sync python examples/mamba3_cpu_smoke.py
```

```python
import torch
from fhemamba.architectures import init_mixer_state
from fhemamba.mamba3 import Mamba3Mixer

mixer = Mamba3Mixer(32, d_state=16, headdim=32).eval()
state = init_mixer_state(mixer)
with torch.no_grad():
    output = mixer(torch.randn(1, 4, 32), state=state)
    next_output = mixer(torch.randn(1, 1, 32), state=state)
```

`mixer3_forward` can also read weights directly from an upstream `Mamba3`
module. The local module uses matching parameter names/shapes, so a compatible
SISO mixer's `state_dict` can be loaded with `strict=True`.

## Trained checkpoint and generation

Download the pinned public weights and Llama-3.1 tokenizer:

```bash
uv run --no-sync python scripts/download_mamba3_checkpoint.py
```

The [reproduction manifest](../config/mamba3-reproduction.json) pins the official
State Spaces checkpoint and NVIDIA's public Llama-3.1 tokenizer files. No chat
template or BOS token is added in this example. The client embedding/vocabulary
head is tied exactly as in the checkpoint.

```python
from transformers import AutoTokenizer
from fhemamba.mamba3_lm import Mamba3LM

model = Mamba3LM.from_pretrained("checkpoints/mamba3-siso-187m")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/mamba3-siso-187m/tokenizer")
prompt = tokenizer.encode("The capital", add_special_tokens=False)
tokens, hidden_trace = model.generate(prompt, 4)
print(tokenizer.decode(prompt + tokens))
# The capital of the state of
```

Compile all layers on the development host:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --no-sync python experiments/export_mamba3_lm.py \
    --output runs/mamba3-lm --prompt 'The capital' --new-tokens 4
```

This writes a full-backbone program, lossless FP32 client vocabulary weights,
references and their hashes. The default payload is about 1.6 GiB. Independent
calibration prompts determine polynomial domains; the chosen evaluation prompt
is then checked without refitting. The compiler rejects domain violations and
changed greedy tokens. A per-node refresh bound is derived from the compiler
proxy with eightfold headroom; this is an empirical fixture assumption, **not**
an interval certificate for arbitrary prompts.

Transfer that directory to the GPU host. Use the build and library environment
shown below, then run with an explicit wall-time cap (seconds):

```bash
OMP_NUM_THREADS=4 python3 experiments/run_packed_probe.py \
  --binary /path/to/packed_fideslib --payload /path/to/mamba3-lm \
  --output /path/to/mamba3-lm-run --planned-refresh --batch-refresh --frontier-refresh \
  --inplace-ops --gpu-plaintext-ntt \
  --naf-rotations --reuse-dead-inputs --direct-plaintext-upload --compact-weights \
  --move-plaintext-coefficients --borrow-plaintext-upload --bsgs-routing-stages --cache-plaintexts \
  --timeout 2400 \
  --budget-file /path/to/campaign-budget.json --budget-seconds 21600
```

The recorded full comparison also pins Spark CPUs 15–19 and enables
`--profile-evaluation` in both modes; CPU numbering is machine-specific.
The example cap is a command-line setting, not a measured runtime. Reuse the
same budget file for every attempt in a campaign: the runner serializes access,
charges elapsed runner time (including setup and failed attempts), and caps the
next process timeout by the remaining allowance. It rejects changes to an
existing budget limit. Keep the runner alive until its child exits so the
charge is recorded. The runner
requires hidden-vector errors below `0.001` against both frozen polynomials and
the original float64 model, and exact equality of all generated token IDs.
The server arithmetic has no secret key. An inline client callback decrypts
only the final hidden vector, computes the full vocabulary head, chooses the
actual greedy token and freshly encrypts its embedding. Reference next-token
embeddings are **not** serialized into feedback nodes. State is never decrypted
or re-encrypted. Process isolation is not provided by this feasibility runner.

Copy the run results back and decode the actual encrypted-client token IDs:

```bash
uv run --no-sync python experiments/report_mamba3_generation.py \
  --payload runs/mamba3-lm --run runs/mamba3-lm-run \
  --tokenizer checkpoints/mamba3-siso-187m/tokenizer \
  --output runs/mamba3-generation.json
```

`--probe-layers 1` compiles only a component for debugging and cost measurement.
Its `complete_backbone` flag is false and its text is not full-model generation.
The exact-history implementation currently limits a session to 32 evaluations.
Neither this limit nor successful short probes establish long-context accuracy.

## Reproduce the encrypted probe

Export on the Python development host:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-sync python experiments/export_mamba3_probe.py \
    --output runs/mamba3-probe --tokens 4
```

The default is width 32, inner width 64, two heads, state size 16 and 1024 slots.
Weights are random and public. 128 independent calibration inputs determine
polynomials; held-out inputs drive encrypted evaluation. Every nonlinear site
must have a polynomial and stay in its frozen domain. The fit report gives
sampled errors on 8193 points, **not** an interval certificate. Exact-reference
error and CKKS-to-polynomial error are measured separately for output, angle,
SSM, previous B and previous x after every step.

Build on Spark with the [existing pinned dependencies](dgx-spark.md#build):

```bash
scripts/build_dgx_spark.sh
```

This also builds `~/fhemamba/spark/kernel/packed_fideslib`. Transfer the complete
payload directory to the Spark host. From the repository checkout on that host:

```bash
export LD_LIBRARY_PATH="$HOME/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=4 python3 experiments/run_packed_probe.py \
  --binary "$HOME/fhemamba/spark/kernel/packed_fideslib" \
  --payload /path/to/mamba3-probe \
  --output runs/mamba3-encrypted-001 --timeout 600
```

Use a new output directory for every attempt. The runner verifies payload
hashes, records the executable hash, retains native output and terminates the
process group at the timeout. The native log prints progress and a rough ETA.
The error gate is `0.001` for both exact and polynomial references, including
the carried states. This is a bounded feasibility check, not a throughput claim.

The context uses ring dimension 65536, depth 44, scale 59, uniform ternary keys
and `security=not-set`. The client fixture encrypts inputs and decrypts recorded
outputs after evaluation; the evaluator never decrypts or re-encrypts a carried
state. It is an inline test, not a deployed client/server protocol. Public
refresh bounds are fixed for the synthetic probe and checked by the exporter;
they do not certify arbitrary user inputs or long-context accuracy.

## Upstream reference parity

The independent parity runner loads the original upstream CPU reference
functions without importing Triton or CuteDSL. It tests both rotary fractions,
both output-normalization settings, nontrivial B/C biases and carried states.

```bash
git clone https://github.com/state-spaces/mamba.git /tmp/mamba-reference
git -C /tmp/mamba-reference checkout e9594ce1c732d97440f0332fdc43170a2294dbfa
uv run --no-sync python experiments/check_mamba3_upstream.py \
  --source /tmp/mamba-reference --output runs/mamba3-upstream-parity.json
```

The runner records all three source hashes and the checkout revision. It uses
a `1e-7` tolerance because upstream preprocessing casts A to float32 while the
local float64 oracle retains double precision. Fused CUDA-kernel compatibility
on GB10 is not needed by this FHE executor.
