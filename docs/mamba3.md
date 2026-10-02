# Mamba-3 SISO support

The trained **Mamba-3 SISO 187M** path shares model arithmetic between the
PyTorch reference, polynomial surrogate and packed CKKS executor.
Start with [Python installation](reproducing.md#1-install-the-python-environment).

The current classical-128 baseline qualifies **16 generated tokens on B300**
for the recorded prompt. The 64-token candidate fails at selection 24;
arbitrary prompts remain unqualified. See [current state](status.md) for
parameters and the open numerical problem.

## CPU usage

```bash
uv run --no-sync python examples/mamba3_cpu_smoke.py
```

This plaintext example checks prefill/decode parity with random weights.
To use the mixer directly:

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

Compatible upstream SISO mixer weights load with `strict=True`.
`mixer3_forward` can also read weights from an upstream `Mamba3` module.

## Supported boundary

| Area | Scope |
| --- | --- |
| Architecture | SISO; rotary fractions 0.5/1.0 and optional headwise output normalization |
| Unsupported variants | MIMO and multiple B/C groups raise errors; prefill uses `loop`, not chunked scan |
| Encrypted export | Batch size one; up to 64 generated tokens and 128 total evaluations |
| State | Default exact rank-one history; optional fixed-size tiles and phasor rotation |
| Numerical gates | Final-hidden error <= 0.001 against both exact and polynomial references; identical generated IDs |
| Protocol | Public weights; encrypted activations/state; inline client token selection |

Export limits are resource bounds, not accuracy guarantees. The default exact
history grows with session length. Tiled state retains every SSM coordinate;
neither representation drops ranks.

Polynomial fits use independent calibration inputs and sampled error checks,
not interval certificates. Export rejects observed domain violations or changed
greedy IDs. Refresh bounds use empirical headroom, so new prompts still need
domain and encrypted validation.

## Trained checkpoint and generation

Download the pinned weights and tokenizer:

```bash
uv run --no-sync python scripts/download_mamba3_checkpoint.py
```

[config/mamba3-reproduction.json](../config/mamba3-reproduction.json) pins the
State Spaces checkpoint and compatible Llama-3.1 tokenizer. Plaintext generation:

```python
from transformers import AutoTokenizer
from fhemamba import load_model

model = load_model("checkpoints/mamba3-siso-187m")
tokenizer = AutoTokenizer.from_pretrained(
    "checkpoints/mamba3-siso-187m/tokenizer", local_files_only=True,
)
prompt = tokenizer.encode("The capital", add_special_tokens=False)
result = model.generate(prompt, max_new_tokens=4)
print(tokenizer.decode(prompt + result.generated_ids, clean_up_tokenization_spaces=False))
# The capital of the state of
```

No chat template or BOS token is added. The embedding/vocabulary head is tied
as in the checkpoint.

Use the [generation API and CLI](generation.md) for text files, raw token IDs,
request preparation and exact/polynomial/CKKS comparison. The recipes below
retain the specialized export and research controls.

### Export a four-token workload

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --no-sync python -m fhemamba workload export-mamba3 \
    --output runs/mamba3-lm --prompt 'The capital' --new-tokens 4
```

The default payload is about 1.6 GiB: a full-backbone program, FP32 client
weights, references and hashes. Use fresh output directories. Follow the
[client-head copy rules](experiments.md#shared-local-client-heads) when
transferring a payload to another host.

`--probe-layers 1` exports a component for diagnosis; its
`complete_backbone=false` output is not full-model generation.

### Run on Spark

Build the [pinned native backend](dgx-spark.md#build) and install `fhemamba`
on the GPU host. The following four-token recipe uses experimental
**`security=not-set`**, separate from the classical-128 configuration below:

```bash
export LD_LIBRARY_PATH="$HOME/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=4 python -m fhemamba benchmark packed \
  --binary "$HOME/fhemamba/spark/kernel/packed_fideslib" \
  --payload /path/to/mamba3-lm --output runs/mamba3-lm-run \
  --security not-set --planned-refresh --batch-refresh --frontier-refresh \
  --inplace-ops --gpu-plaintext-ntt \
  --naf-rotations --reuse-dead-inputs --direct-plaintext-upload --compact-weights \
  --move-plaintext-coefficients --borrow-plaintext-upload --bsgs-routing-stages --cache-plaintexts \
  --timeout 2400 \
  --budget-file /path/to/campaign-budget.json --budget-seconds 21600
```

Set an explicit timeout and campaign budget. The shared budget file charges
elapsed runner time, including failed attempts, and limits subsequent runs.
Keep the runner alive until its child exits so charging completes.

The inline client decrypts final hidden vectors, computes vocabulary logits,
chooses actual greedy tokens and freshly encrypts their embeddings.
Reference next-token embeddings are not substituted into feedback. State stays
encrypted; process-separated client/server deployment is not implemented.

Copy results back and decode the measured IDs:

```bash
uv run --no-sync python -m fhemamba benchmark generation-report \
  --payload runs/mamba3-lm --run runs/mamba3-lm-run \
  --tokenizer checkpoints/mamba3-siso-187m/tokenizer \
  --output runs/mamba3-generation.json
```

### Classical-128, 16-token configuration

Export fixed-size state tiles with phasor rotation:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --no-sync python -m fhemamba workload export-mamba3 \
    --output runs/mamba3-long16 --prompt 'The capital' --new-tokens 16 \
    --state-representation tiled --rotary phasor --calibration-tokens 65 \
    --normalization-margin 1 --positive-lower-fraction 0.25
```

Calibration uses six independent 65-token generations. Phasor updates keep
polynomial inputs at one step's angular increment; all original normalizations
and state coordinates remain.

The recorded [B300 job](../results/b300/2026-09-28/public-state-reuse/jobs/full-b1.json)
specifies the native options. Reproduction requires the matching
[build and settings](../results/b300/2026-09-28/public-state-reuse/README.md),
adapted to your paths and allocated device through the
[job workflow](experiments.md). This is a different hardware and security
profile from the Spark recipe above. The measured request uses about
179.41 GiB of GPU memory; see its
[timing and resource scope](research/2026-09-28-public-state-reuse.md).

Retain two S2C-first refresh passes, merged correction and frontier limit 256.
A limit of 128 fails on the full model despite passing a component probe.
The limit controls speculative live values, not GPU allocation.
The result reporter supports `--validate-only --security 128-classic` for
qualification without a tokenizer.

The separate [64-step recurrence probe](../experiments/recurrent_state/README.md)
qualifies a component, not 64-token full-model generation.

## Reproduce the encrypted probe

Export a small random mixer on the Python host:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-sync python -m fhemamba workload export-mixer \
    --output runs/mamba3-probe --tokens 4
```

Defaults are width 32, inner width 64, two heads, state size 16 and 1024 slots.
Independent calibration and held-out inputs are used. Both exact-reference
and CKKS-to-polynomial errors cover output and every carried state.

Transfer the payload to Spark and use the build/library setup above:

```bash
OMP_NUM_THREADS=4 python -m fhemamba benchmark packed \
  --binary "$HOME/fhemamba/spark/kernel/packed_fideslib" \
  --payload /path/to/mamba3-probe \
  --output runs/mamba3-encrypted-001 --timeout 600 --security not-set
```

This feasibility probe uses N=65,536, depth 44, scale 59 and uniform ternary
keys. Both error gates are 0.001. The runner verifies payload hashes, records
binary identity and terminates the process group on timeout. Use a fresh
output directory per attempt.

## Runner controls

Use `python -m fhemamba benchmark packed --help` for options and the
[native guide](../native/README.md) for mechanisms and counters.

| Control | Requirement or boundary |
| --- | --- |
| `--planned-refresh --batch-refresh` | Replicated transforms, radix-8 routing and two bootstrap passes |
| `--frontier-refresh` | Planned/batched refresh; preserves feedback and dependency barriers |
| `--reuse-public-ciphertexts` | Reuses only identical public sources within one request/key; requires workload qualification |
| `--cache-plaintexts` | Exact coefficient bits/levels; entry count is not a memory-byte limit |
| GPU encoding/upload and operand reuse | Opt-in; follow the matching native build and qualification suite |
| `--hoist-rotations --share-chebyshev` | Opt-in sharing with fixed coefficients and compatible inputs |
| `--gpu-dual-ring` | Experimental profile; requires S2C-first, planned/batched refresh and matching build; incompatible with classical-128 |
| `--profile-evaluation` | Adds timers and CUDA capture markers; instrumented timings have a different scope |

One-pass refresh fails the trained one-layer accuracy gate. Indexed mask
lookup remains unqualified and disabled by default. Use the
[optimization map](optimizations.md) and [research index](research/README.md)
for candidate-specific comparisons.

## Upstream reference parity

The independent runner loads upstream CPU functions without Triton or CuteDSL:

```bash
git clone https://github.com/state-spaces/mamba.git /tmp/mamba-reference
git -C /tmp/mamba-reference checkout e9594ce1c732d97440f0332fdc43170a2294dbfa
uv run --no-sync python -m fhemamba workload check-mixer \
  --source /tmp/mamba-reference --output runs/mamba3-upstream-parity.json
```

It covers both rotary fractions, output-normalization modes, B/C biases and
carried states, recording source hashes and revision. The 1e-7 tolerance
accounts for upstream float32 preprocessing of A versus the float64 oracle.
