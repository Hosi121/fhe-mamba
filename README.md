# FHE Mamba

[![CI](https://github.com/Hosi121/fhe-mamba/actions/workflows/ci.yml/badge.svg)](https://github.com/Hosi121/fhe-mamba/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research software for **Mamba-2 and Mamba-3 SISO inference with encrypted
activations and recurrent state**: Python/PyTorch references, polynomial
operators and CKKS GPU execution using OpenFHE and FIDESlib. See the
[Mamba-3 guide](docs/mamba3.md) for its shared arithmetic backend and trained
187M generation commands. Mamba-3 uses sampled polynomial fits; the certified
operator recipes belong to the Mamba-2 experiments.

Complete encrypted-backbone generation runs cover **Mamba-2-130M** and
**Mamba-3 SISO 187M** on **DGX Spark**. The project is **FHE Mamba** (`fhe-mamba`); the Python package
and CLI are both `fhemamba`.

[Reproduce](docs/reproducing.md) · [Documentation](docs/README.md) ·
[Research and results](docs/research/README.md) · [Contribute](CONTRIBUTING.md)

## Quick start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from the repository root:

```bash
git clone https://github.com/Hosi121/fhe-mamba.git
cd fhe-mamba
uv sync --locked --extra experiments
uv run --no-sync python examples/cpu_smoke.py
```

This CPU example creates a tiny random Mamba-2 and checks forward/decode parity
against Transformers. It needs no GPU or checkpoint and reports `passed: true`
with errors below `1e-4`. It performs plaintext calculations; the initial
PyTorch dependency installation can be large.

Inspect a recorded encrypted result locally:

```bash
uv run --no-sync fhemamba validate-artifacts --require-commit \
  results/dgx/2026-09-24/borrowed-plaintext/m2-full-borrow/generation.json
```

For a new encrypted run, follow the [reproduction guide](docs/reproducing.md)
and [Spark build instructions](docs/dgx-spark.md#build).

## Measured result and scope

With Mamba-2, the prompt `The capital` produces `The capital of the Republic of`:
**24 layers, five encrypted evaluations, four generated tokens** on DGX Spark.

| Implementation | Evaluation time | Maximum CKKS-to-polynomial error |
| --- | ---: | ---: |
| [Initial pipeline](docs/research/2026-09-22-client-generation.md) | 50.6 min | 0.009192 |
| [Periodic coefficients](docs/research/2026-09-22-periodic-gate-coefficients.md) | 42.9 min | 0.011767 |
| [Subring encoding](docs/research/2026-09-22-subring-gate-encoding.md) | 38.5 min | 0.016255 |
| [Shared plaintext preparation](docs/research/2026-09-24-shared-plaintext-preparation.md) | 35.70 min | 0.005044 |
| [Direct plaintext upload](docs/research/2026-09-24-packed-resources.md) | 33.83 min | 0.006424 |
| [Borrowed plaintext upload](docs/research/2026-09-24-borrowed-plaintext.md) | **32.94 min** | **0.010484** |

These use the same frozen payload, with one full run per variant. A full
same-binary comparison measures **38.64 → 35.70 min (7.59%)** after sharing
Mamba-3's fast upload/GPU NTT path with Mamba-2. Additional transfer-copy
removal passes full generation; its isolated two-layer ABBA measures
**59.03 → 55.75 s (5.56%)**. That direct-upload candidate peaks at
**37.06 GiB RSS**; encoder optimizations remain opt-in. Setup and transfer are
additional. The `0.05` error gate compares CKKS with the matching
polynomial circuit, not the original floating-point model. The
[evidence registry](docs/evidence.md) includes raw results, repeated probes,
quality measurements and failed controls.

The additional borrowed-upload Mamba-2 control measures
**56.03 → 54.50 seconds
(2.73%)** in ABBA order; the complete candidate
passes in **32.94 minutes**, peak RSS **37.06 GiB**.
Coefficient moves remain off for Mamba-2 after their non-improving control.

**Mamba-3 SISO 187M** generates `The capital of the state of` through 12 layers
× five encrypted evaluations. The preceding resource comparison measures
**18.58 → 17.58 min (5.39%)** with NAF rotations, final-use input reuse,
direct plaintext upload and lossless compact weights. All four generated IDs
match; maximum hidden error is **0.0001266** versus exact FP64 and
**0.0001248** versus frozen
polynomials, below the unchanged `0.001` gate. Each full mode was measured once;
a separate mirrored prefix comparison measures a 5.28% reduction. Public matrix
storage falls **673.3 → 168.3 MiB**, and process peak RSS falls
**27.9648 → 27.3143 GiB**. See the
[resource comparison](docs/research/2026-09-24-packed-resources.md) and earlier
[GPU encoding comparison](docs/research/2026-09-24-mamba3-gpu-encoding.md).

The latest selected complete Mamba-3 configuration takes **16.27
minutes**, with maximum exact/polynomial errors
`0.0001666146` / `0.0001565546`
and the same four generated IDs. The upload/routing full pair is
**17.36 → 16.74 minutes
(3.53% reduction)**. The cache follow-up completes in 16.27 minutes on the same binary; its uncached full baseline precedes intervening Mamba-2 validation.
Each full mode was measured once; mirrored prefix controls are separate.
Model rotations fall **59,900 → 50,780**, excluding refresh. Both numerical
gates stay at 0.001. See the [upload/routing study](docs/research/2026-09-24-borrowed-plaintext.md)
and [cache integration](docs/research/2026-09-24-packed-cache-integration.md).

The earlier [depth/batching study](docs/research/2026-09-24-mamba3-depth-batching.md)
measured 46.20 → 24.95 min under its own matched conditions. That historical
timing must not be used to isolate the new encoding change.
The Mamba-2 and Mamba-3 models and accuracy contracts differ; these are not
architecture speed rankings.

The benchmark is a **single-process client loop** with public weights and
`security=not-set`. The client decrypts the final hidden vector to select each
next token. It does not establish full-chain 128-bit security, a secret-key-free
full-model server, or production private chat. See the
[protocol and remaining work](docs/design.md) and [cost bounds](docs/research/2026-09-22-generation-cost-bounds.md).

## Repository

| Directory | Contents |
| --- | --- |
| [`src/fhemamba/`](src/fhemamba/) | Installable Python model, operators, layouts and CLI |
| [`native/`](native/README.md) | Encrypted GPU kernel, probes and C++ contracts |
| [`examples/`](examples/) / [`config/`](config/README.md) | CPU demo, pinned checkpoint metadata and frozen coefficients |
| [`experiments/`](experiments/README.md) | Research runners, campaign manifests and Slurm launchers |
| [`results/`](results/README.md) | Curated measurements and historical evidence |
| [`tests/`](tests/) / [`scripts/`](scripts/README.md) | Tests, checks, build and setup helpers |
| [`docs/`](docs/README.md) | Guides, design, research notes and archives |

For development, use `uv sync --locked --extra dev` and
`scripts/run_checks.sh`; see [contributing](CONTRIBUTING.md) and
[testing](docs/testing.md). CI covers Python 3.10/3.12, native CPU contracts,
the example, recorded-artifact validation and installed-wheel execution.

Code and original coefficient bundles are [MIT licensed](LICENSE).
Checkpoints and native dependencies are downloaded separately; see
[third-party notices](THIRD_PARTY_NOTICES.md). The immutable
[September 22 research snapshot](https://github.com/Hosi121/fhe-mamba/releases/tag/research-2026-09-22)
retains the original measured layout; [path changes](docs/repository.md#previous-layout)
are documented for readers following older commands.
