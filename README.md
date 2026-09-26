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
  results/dgx/2026-09-24/owned-arithmetic/m2-full-candidate/generation.json
```

For a new encrypted run, follow the [reproduction guide](docs/reproducing.md)
and [Spark build instructions](docs/dgx-spark.md#build).

## Measured results and scope

Both trained models complete **five encrypted evaluations and four generated
tokens** on **DGX Spark GB10**. The latest Mamba-3 trial moves ordinary
arithmetic to N=32,768 while retaining the existing N=65,536 two-pass
S2C-first refresh, with encrypted transfers between the rings on the GPU.

| Model | Layers | Study baseline → candidate | Reduction | Generated text |
| --- | ---: | ---: | ---: | --- |
| Mamba-2-130M | 24 | 1959.30 → **1942.99 s** (32.38 min) | **0.83%** | `The capital of the Republic of` |
| Mamba-3 SISO 187M | 12 | 613.83 → **430.51 s** (7.18 min) | **29.86%** | `The capital of the state of` |

The [GPU dual-ring study](docs/research/2026-09-26-gpu-dual-ring.md) adopts
`--gpu-dual-ring` as **opt-in** after primitive, prefix and full-model gates.
Four fresh processes in baseline/candidate/candidate/baseline order average
the values above. Ordinary evaluation falls **374.82 → 212.65 s (43.27%)**;
the refresh wrapper, including packing and ring transfers, falls
**239.01 → 217.87 s (8.84%)** with **484 bootstrap calls**. The large-ring
bootstrap circuit is unchanged. Peak process RSS stays about **26.21 GiB**.

Weights, model polynomials, the 59-bit scale, depth 44, all four generated IDs
and both `0.001` error gates are unchanged. The candidate's maximum
exact/polynomial errors are **4.72545e-5 / 5.98173e-7**. Its mean is
**107.6 s per generated token**, or **119.1 s including process setup and
validation**, amortized over this five-evaluation/four-generated-token request.
Complete process time falls **654.52 → 476.39 s (27.21%)**. These rates are
not steady-state token latency. The smaller ordinary ring changes encryption
parameters; this experimental `security=not-set` comparison does not establish
equal security for both ring sizes.

After rebuilding on Spark, add `--gpu-dual-ring` alongside
`--hoist-rotations --share-chebyshev --gpu-plaintext-rns --s2c-first`
and `--planned-refresh --batch-refresh --frontier-refresh`. The
[study's reproduction command](docs/research/2026-09-26-gpu-dual-ring.md#reproduction)
contains the complete configuration; [raw commands and results](results/dgx/2026-09-26/gpu-dual-ring/)
retain source, binary and payload hashes. Programs require 32,768 declared
slots and logical node widths at most 16,384. The final binary also passes a
prefix regression without the new flag and rejects unsupported configurations
before key setup.

The preceding [four-candidate study](docs/research/2026-09-25-structural-four.md)
reduced **619.41 → 614.11 s (0.86%)** with rotation and Chebyshev basis sharing.
Its 32-bit profile failed accuracy and its CPU ring-switch route lost on cost;
the new GPU transfer is a separate implementation. Earlier
[GPU RNS](docs/research/2026-09-25-gpu-rns.md),
[S2C-first](docs/research/2026-09-25-s2c-first.md) and
[ready-node refresh](docs/research/2026-09-25-packed-frontiers.md) studies retain
their own matched controls. The new comparison reruns the released baseline.

These are native evaluation times, including encoding, GPU upload and client
feedback inside the evaluation loop. Setup/key generation, input parsing,
post-evaluation validation and external transport are excluded. Compare
variants within each study: model sizes, weights and numerical contracts
differ, so this table does not rank architectures. Two full samples per mode
on one fixed prompt do not establish statistical significance or arbitrary-prompt
performance. Fresh-key error differences do not establish accuracy improvements.
Local release checks pass **292 tests**, including **21 native C++ contracts**.
Three fresh GPU primitive processes each pass six exact NTT maps and 30 paired
encrypted-transfer cases, covering levels, scale degrees and inactive slots.

The Mamba-2 row retains the earlier
[shared ownership study](docs/research/2026-09-24-owned-arithmetic.md):
151,315 backend result copies removed, maximum error `0.003651` below `0.05`,
one fresh full pair and eight interleaved prefix samples. The small time
difference has no statistical significance claim. GPU RNS uses shared plaintext
machinery, qualified for both model configurations; the specialized Mamba-2
executor and its full-depth measurement are unchanged in this study.

The [shared square dispatch](docs/research/2026-09-25-square-dispatch.md)
passes 72 additional exact-RNS cases and has separate prefix controls for both
models. It is included in the new Mamba-3 baseline; this study does not isolate
its full-depth contribution.

Earlier milestones remain reproducible:

- Mamba-2's [initial complete run](docs/research/2026-09-22-client-generation.md)
  took 50.6 minutes. [Periodic coefficients](docs/research/2026-09-22-periodic-gate-coefficients.md),
  [subring encoding](docs/research/2026-09-22-subring-gate-encoding.md) and
  [shared plaintext preparation](docs/research/2026-09-24-shared-plaintext-preparation.md)
  reduced successive runs to 42.9, 38.5 and 35.70 minutes.
- The [resource comparison](docs/research/2026-09-24-packed-resources.md)
  measures Mamba-3 at 18.58 → 17.58 minutes with NAF rotations, final-use reuse,
  direct upload and lossless compact weights. Public matrix storage falls
  **673.3 → 168.3 MiB**.
- [Borrowed upload and routing](docs/research/2026-09-24-borrowed-plaintext.md)
  removes more transfer copies and 9,120 Mamba-3 model rotations. The
  [bounded mask cache](docs/research/2026-09-24-packed-cache-integration.md)
  remains enabled in the latest Mamba-3 configuration. Coefficient moves stay
  off for Mamba-2 after their non-improving control.

Compare variants within each study's matched conditions; historical times do
not isolate an individual optimization. The [evidence registry](docs/evidence.md)
links all claims and their validation boundaries. A later
[compact weight-coefficient cache](docs/research/2026-09-25-weight-coefficient-cache.md)
passed exact RNS/GPU checks but made the cold three-step probe **1.26% slower**;
it was not adopted. Actual level changes limited reuse. The
[inverse-FFT reuse candidate](docs/research/2026-09-25-preparation-design.md)
remains unimplemented. The [encoder range-scan prototype](docs/research/2026-09-25-encoding-range.md)
is a separate local CPU experiment and is not applied to DGX inference.

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
