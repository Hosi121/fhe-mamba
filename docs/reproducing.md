# Getting started

Run commands from the repository root. Python references and recorded-result
inspection work on a CPU; encrypted execution needs a separately built native
backend.

## 1. Install the Python environment

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Hosi121/fhe-mamba.git
cd fhe-mamba
uv sync --locked --extra experiments
uv run --no-sync python examples/cpu_smoke.py
uv run --no-sync python examples/mamba3_cpu_smoke.py
```

Both examples use random weights, perform plaintext calculations and print
`passed: true`. They need no checkpoint or GPU. For development, use
`uv sync --locked --extra dev`.

| Next step | Guide and requirements |
| --- | --- |
| Trained Mamba-3 SISO 187M | [Load, export and run](mamba3.md); checkpoint download, plus native GPU build for encryption |
| Small encrypted Mamba-3 mixer | [Probe](mamba3.md#reproduce-the-encrypted-probe); Spark build, random weights, experimental `not-set` security |
| CPU CKKS primitives | [Recipes](../experiments/local_ckks/README.md); separate OpenFHE Python installation, toy security |
| Recorded Mamba-2-130M generation | Steps 2–6 below; DGX Spark, CUDA 13.0, SSH and rsync |
| Inspect published results | [Artifact validation](testing.md#artifact-validation); no GPU |

The current classical-128 baseline is the **B300 Mamba-3 16-token request**.
See [current state](status.md) for its limits. The Mamba-2 walkthrough below
uses `security=not-set` and an inline client loop; it does not establish
process-separated deployment or full-chain 128-bit security.

## 2. Obtain the exact public checkpoint

```bash
uv run --no-sync python scripts/download_checkpoint.py
# Verify an existing download without network access:
uv run --no-sync python scripts/download_checkpoint.py --verify-only
```

The download contains about 516 MB of weights plus tokenizer/configuration
files from [AntonV/mamba2-130m-hf](https://huggingface.co/AntonV/mamba2-130m-hf).
[config/reproduction.json](../config/reproduction.json) pins its revision,
file hashes and coefficient bundles. The downloader verifies them and refuses
to replace a different checkpoint. Other conversions may have incompatible layouts.

## 3. Check the full checkpoint on a CPU

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-sync python experiments/quality/run_parity.py \
  --checkpoint checkpoints/mamba2-130m-hf --device cpu \
  --output runs/parity-mamba2.json
```

This compares the reference with Transformers. Polynomial approximation quality
and encrypted accuracy have [separate gates](validation.md).

## 4. Export a stabilized payload

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-sync python experiments/export/export_m1_payload.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --normalization-bundle config/mamba2-130m-normalization-20260921.json \
  --stabilized-gate-bundle config/mamba2-130m-gates-20260921.npz \
  --output runs/stabilized-payload --tokens 2 \
  --cal-tokens 128 --bound-cal-tokens 128 --device cpu
```

Use a fresh output directory. The export writes about 0.9 GB and verifies the
normalization and joint-gate certificates. The built-in calibration
text has **34 tokens**; 128 is a cap. Certificates are conditional on their
domains, so this short calibration does not qualify arbitrary prompts or long
sessions. New workloads need independent calibration and quality checks.

## 5. Build on Spark

Follow the [Spark build guide](dgx-spark.md#build). Local and remote checkouts
must use the same revision. The build runs C++ contracts; the runner checks
source, library and binary identities.

The Mamba-2 campaign preflight requires an idle GPU and **65 GiB available host
memory**. Python installation and CPU CI do not qualify the native GPU build.

## 6. Run the measured prompt-to-text workload

Set your SSH destination and the absolute build root on Spark
(default: `~/fhemamba` on that host):

```bash
export FHEMAMBA_SSH_HOST=your-user@your-spark
export FHEMAMBA_REMOTE_ROOT=/path/on/spark/fhemamba

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-sync python experiments/execution/run_dgx_generation.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --base-chain runs/stabilized-payload \
  --output-dir runs/generation-001 \
  --prompt 'The capital' --generate-tokens 4 \
  --ssh-host "$FHEMAMBA_SSH_HOST" \
  --remote-root "$FHEMAMBA_REMOTE_ROOT" \
  --joint-periodic-coefficients --joint-subring-encoding \
  --gpu-plaintext-ntt --direct-plaintext-upload --borrow-plaintext-upload
```

The launcher uses SSH and rsync to copy a new payload, execute the remote
kernel and collect actual generated IDs. Both output directories must be
unused. Use `--prompt-file` for a UTF-8 file; new prompts must pass domain checks.

The recorded continuation is `The capital of the Republic of`, with IDs
`[273, 253, 4687, 273]`. Acceptance requires matching IDs, all five outputs
decrypting with polynomial-reference error at most **0.05**, and zero diagnostic
intermediate decryptions. Setup and transfer add to the native evaluation
time; see the [measured configuration](research/2026-09-24-owned-arithmetic.md).

```bash
uv run --no-sync fhemamba validate-artifacts runs/generation-001/generation.json
```

## Included evidence and local-only data

Use fresh paths under `runs/` for output and `config/local/` for machine
settings. Published results retain their original numerical values and failure
statuses; new runs get new identities. See the [experiment workflow](experiments.md).

For the older `research-2026-09-22` snapshot, use its pinned source and the
[path migration map](repository.md#previous-layout).
