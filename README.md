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
tokens** on **DGX Spark GB10**. The measured shared ownership change reuses
private ciphertext buffers in both models while preserving arithmetic order,
operation counts, selected tokens and the existing error gates.

| Model | Layers | Fresh baseline → candidate | Reduction | Generated text |
| --- | ---: | ---: | ---: | --- |
| Mamba-2-130M | 24 | 1959.30 → **1942.99 s** (32.38 min) | **0.83%** | `The capital of the Republic of` |
| Mamba-3 SISO 187M | 12 | 976.48 → **949.26 s** (15.82 min) | **2.79%** | `The capital of the state of` |

These are native evaluation times, including plaintext encoding and GPU upload
performed during evaluation. Context/key setup, payload export, input loading
and external transport are outside that timer. Each full mode has one fresh
process/key sample. Separate interleaved prefix controls show a **2.76%**
reduction for Mamba-3 and **0.69%** for Mamba-2 across its eight samples,
including reversed controls. The small Mamba-2 difference does not establish
statistical significance. Model sizes, weights and numerical contracts differ;
this table does not rank architectures.

Mamba-2 removes **151,315 backend result copies** and has maximum
CKKS-to-polynomial error **0.003651**, below its unchanged `0.05` gate.
Mamba-3 reuses **183,912 temporary input buffers**; maximum errors are
**0.0001395992** versus exact FP64 and **0.0001251919** versus the frozen
polynomial circuit, both below `0.001`. These comparisons use fresh keys;
variation in error between runs is not evidence of an accuracy improvement.
The change passes **286 tests**, including **18 C++ contracts**, plus
**192 GPU cases with exactly matching ciphertext coefficients and metadata**.
See the [ownership study](docs/research/2026-09-24-owned-arithmetic.md) for
commands, raw measurements, failed controls and source/binary identities.

A later [shared square dispatch](docs/research/2026-09-25-square-dispatch.md)
passes **72 additional exact-RNS cases**. In four interleaved prefix runs per
model, mean evaluation time falls **47.09 → 46.69 s (0.84%)** for Mamba-3
and **53.98 → 53.50 s (0.88%)** for Mamba-2, with unchanged error gates and
operation counts. These small prefix results have no significance claim;
the square change has **not yet been measured at full depth**. The full-model
table above therefore retains the earlier measurements.

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
