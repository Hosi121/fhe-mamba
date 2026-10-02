# FHE Mamba

[![CI](https://github.com/Hosi121/fhe-mamba/actions/workflows/ci.yml/badge.svg)](https://github.com/Hosi121/fhe-mamba/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research tools for **Mamba-2 and Mamba-3 SISO inference with encrypted
activations and recurrent state**. Compare PyTorch references with polynomial
approximations, export CKKS programs, and reproduce GPU experiments using
OpenFHE and FIDESlib. Model weights are public.

The project is **FHE Mamba**; its Python package and CLI are `fhemamba`.

## Quick start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from the repository root:

```bash
git clone https://github.com/Hosi121/fhe-mamba.git
cd fhe-mamba
uv sync --locked --extra experiments
uv run --no-sync python examples/cpu_smoke.py
uv run --no-sync python examples/mamba3_cpu_smoke.py
```

Both examples run **plaintext** calculations on small random models, with no
GPU or checkpoint download. Expect `passed: true`; the Mamba-2 example compares
against Transformers. Initial PyTorch installation can be large.

Inspect a recorded encrypted result without running inference:

```bash
uv run --no-sync fhemamba validate-artifacts --require-commit \
  results/dgx/2026-09-24/owned-arithmetic/m2-full-candidate/generation.json
```

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Generate from text or token IDs; compare exact, polynomial and CKKS | [Generation API and CLI](docs/generation.md) |
| Set up Python and choose a model | [Getting started](docs/reproducing.md) |
| Use Mamba-3 SISO 187M or export an encrypted workload | [Mamba-3 guide](docs/mamba3.md) |
| Reproduce Mamba-2-130M generation on Spark | [Mamba-2 reproduction](docs/reproducing.md#2-obtain-the-exact-public-checkpoint) |
| Compare experiments and publish evidence | [Experiment workflow](docs/experiments.md) |
| Find Python APIs and commands | [Package guide](docs/package.md) |
| Develop or contribute | [Contributing](CONTRIBUTING.md) · [Current state](docs/status.md) |

## Capabilities and limits

- The common API accepts local Mamba-1, Mamba-2 and Mamba-3 SISO checkpoints.
  Mamba-1 supports CPU exact generation; inspect backend requirements with
  `fhemamba inspect-model --model /path/to/checkpoint`.
- Encrypted runs cover Mamba-2-130M and Mamba-3 SISO 187M on the recorded
  prompts. Native GPU builds are separate from Python installation.
- The current classical-128 Mamba-3 configuration qualifies **16 generated
  tokens on B300**. Its 64-token candidate fails the hidden-error gate at
  selection 24. Arbitrary prompts and longer sessions remain unqualified.
- The client decrypts the final hidden vector to select each next token.
  The inline client loop does not provide a process-separated private-chat service.
- Mamba-3 uses sampled polynomial fits. Mamba-2's operator certificates are
  conditional on their declared input domains; model quality and CKKS error
  require separate checks.

## Recorded performance

The qualified B300 16-token request averages **1,078.48 s of native evaluation**
(**67.41 s per generated token**, across two matched pairs). Its complete
process averages 1,244.47 s, with sampled peak GPU allocation about **179.41 GiB**.
Evaluation excludes setup/key generation and final validation; the per-token
figure is a request average, not steady-state throughput.
[Measurement and conditions](docs/research/2026-09-28-public-state-reuse.md).

Other configurations and failed controls are in the [evidence registry](docs/evidence.md)
and [research index](docs/research/README.md). Use [current state](docs/status.md)
for the qualified parameters and open accuracy problem.

See the [documentation index](docs/README.md) and [repository map](docs/repository.md)
for further guides and file ownership. Code and original coefficient bundles
are [MIT licensed](LICENSE); downloaded models and dependencies retain their
[upstream terms](THIRD_PARTY_NOTICES.md).
