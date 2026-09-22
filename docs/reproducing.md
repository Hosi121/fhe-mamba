# Reproducing the research snapshot

This guide starts with a new checkout. Commands run from the repository root
unless a step explicitly says to run on Spark. The small CPU example and
recorded-artifact inspection need no GPU or model download. New encrypted
generation needs a DGX Spark host and several minutes per evaluated token.

## 1. Install the Python environment

```bash
git clone https://github.com/Hosi121/fhe-native-mamba3.git
cd fhe-native-mamba3
uv sync --locked --extra experiments
uv run --no-sync python examples/cpu_smoke.py
```

The example uses random two-layer weights and performs no encryption. A
successful output has `passed: true` and both errors below `1e-4`. Use
`uv sync --locked --extra dev` for the test/lint tools as well.

## 2. Obtain the exact public checkpoint

```bash
uv run --no-sync python scripts/download_checkpoint.py
```

This downloads about 516 MB of weights plus tokenizer/configuration files
from [AntonV/mamba2-130m-hf](https://huggingface.co/AntonV/mamba2-130m-hf),
an independently converted copy of the original State Spaces checkpoint.
The revision is `05e8773fc4ac1cd067e8a18a5c45372ce5178405`.
[config/reproduction.json](../config/reproduction.json) records SHA-256 for
every required file and both public coefficient bundles. The downloader
checks the hashes and refuses to replace an existing different checkpoint.
Weights are not redistributed in this repository.

To check an existing download without network access:

```bash
uv run --no-sync python scripts/download_checkpoint.py --verify-only
```

The native and Python references depend on the converted checkpoint's exact
layout; substituting a similarly named checkpoint does not reproduce it.

## 3. Check the full checkpoint on a CPU

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-sync python fhemamba/experiments/run_parity.py \
  --checkpoint checkpoints/mamba2-130m-hf --device cpu \
  --output runs/parity-mamba2.json
```

This compares the reference model with Transformers. It does not perform
encrypted evaluation or establish polynomial approximation quality.

## 4. Export a stabilized payload

The frozen coefficients used by the measured candidate are included:

- `config/mamba2-130m-normalization-20260921.json`: 49 inverse-square-root recipes.
- `config/mamba2-130m-gates-20260921.npz`: public coefficients for 24 joint gates.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-sync python fhemamba/experiments/export_m1_payload.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --normalization-bundle config/mamba2-130m-normalization-20260921.json \
  --stabilized-gate-bundle config/mamba2-130m-gates-20260921.npz \
  --output runs/stabilized-payload --tokens 2 \
  --cal-tokens 128 --bound-cal-tokens 128 --device cpu
```

Use a new output directory. The export writes about 0.9 GB of weights,
metadata and references and rechecks the coefficient certificates. The
built-in calibration text has **34 tokens**; `128` is a cap. This reproduces
the short calibration policy and is not a guarantee for arbitrary prompts
or long sessions. Larger calibration/quality studies are in [testing](testing.md).

Frozen real-arithmetic certificates are conditional on their public domains;
CKKS noise and model quality require their separate checks. Raw measurements
retain the original payload hashes and paths. A fresh export/run records its
own hashes; historical artifacts should never be edited to match it.

## 5. Build on Spark

Follow the [fresh-host build instructions](dgx-spark.md#build) on a DGX Spark
with CUDA 13.0. The expected layout is:

```text
/home/YOUR_USER/fhemamba/
  cipher/       repository checkout
  spark/        isolated native build and dependency manifest
```

The local and remote checkouts must use the same revision. The build must
complete its C++ contracts, and the runner verifies its source/library/binary
hashes. The measured run peaks near 37 GiB process RSS; campaign preflight
requires 65 GiB host memory available and an idle GPU. The complete native
build on another machine is not part of the CPU CI gate.

## 6. Run the measured prompt-to-text workload

Configure ordinary SSH access, then set these two values for your host:

```bash
export FHEMAMBA_SSH_HOST=your-user@your-spark
export FHEMAMBA_REMOTE_ROOT=/home/your-user/fhemamba

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-sync python fhemamba/experiments/run_dgx_generation.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --base-chain runs/stabilized-payload \
  --output-dir runs/generation-001 \
  --prompt 'The capital' --generate-tokens 4 \
  --ssh-host "$FHEMAMBA_SSH_HOST" \
  --remote-root "$FHEMAMBA_REMOTE_ROOT" \
  --joint-periodic-coefficients --joint-subring-encoding
```

The launcher requires `ssh` and `rsync`. It copies a new payload, invokes
`cipher/scripts/run_dgx_spark.sh` remotely, collects the measured IDs, and
decodes them locally. Local and remote output directories must be unused.
The remote checkout normally has its own `.git`; the historical
`--remote-git-dir` workaround is unnecessary for a regular clone.

The reference continuation is `The capital of the Republic of`, with IDs
`[273, 253, 4687, 273]`. Acceptance also requires all five outputs to decrypt,
error at most `0.05` against the polynomial reference, and zero intermediate
diagnostic decryptions. New keys can change numerical error and timing.
The recorded 38.5 minutes covers native evaluation; setup, Python reference
preparation and transfer are additional. See the
[measurement scope](research/2026-09-22-subring-gate-encoding.md#complete-generation-result).

Inspect the result:

```bash
uv run --no-sync fhemamba validate-artifacts runs/generation-001/generation.json
```

This is a single-process client-loop experiment with public weights and
`security=not-set`. It does not establish a separated secret-key-free server,
full-chain 128-bit security or production privacy guarantees.

## Included evidence and local-only data

Curated small JSON results, failed controls, configuration and coefficient
bundles are versioned. Checkpoints, exported payloads, build products,
secret/evaluation keys and transient logs are not required in Git. Historical
study paths under `runs/` describe the original execution environment; use
the portable commands above to create fresh artifacts.

The [cost](research/2026-09-22-generation-cost-bounds.md) and
[normalization](research/2026-09-22-normalization-bounds.md) reports are offline
analyses, not additional encrypted benchmark results.
