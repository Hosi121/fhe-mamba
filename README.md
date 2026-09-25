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
tokens** on **DGX Spark GB10**. The latest Mamba-3 trial shares rotation
preparation and same-input Chebyshev bases while retaining GPU plaintext RNS
preparation, S2C-first refresh and two-pass error correction.

| Model | Layers | Study baseline → candidate | Reduction | Generated text |
| --- | ---: | ---: | ---: | --- |
| Mamba-2-130M | 24 | 1959.30 → **1942.99 s** (32.38 min) | **0.83%** | `The capital of the Republic of` |
| Mamba-3 SISO 187M | 12 | 619.41 → **614.11 s** (10.24 min) | **0.86%** | `The capital of the state of` |

The [four-candidate study](docs/research/2026-09-25-structural-four.md) adopts
`--hoist-rotations --share-chebyshev` as **opt-in**. Four full processes in
baseline/candidate/candidate/baseline order average the values above. Ordinary
evaluation falls **380.94 → 375.77 s (1.36%)**; refresh stays about **238.4 s**
with **484 bootstraps**. Ciphertext products fall **12,005 → 11,540** and
rotations **62,969 → 62,369**. Peak process RSS stays about **26.21 GiB**.

Weights, model polynomials, CKKS parameters, all four generated IDs and both
`0.001` error gates are unchanged. The candidate's maximum exact/polynomial
errors are **4.65294e-5 / 9.65161e-7**. Its mean is **153.5 s per generated
token**, or **163.7 s including process setup and validation**, amortized over
this five-evaluation/four-generated-token request. This is a modest additional
gain, not a large speedup or steady-state token-latency result.

All four candidate mechanisms have implementations and small-circuit trials.
The nominal 59-bit **32-bit RNS profile fails numerical validation**; its
64-bit control and separate lower-precision 32-bit control pass. The **smaller
ring CPU route passes accuracy but its refresh boundary costs about 55 s**,
outweighing ordinary-arithmetic savings. Both are rejected for integration;
[a GPU ring switcher remains unimplemented](docs/research/2026-09-25-structural-four.md#smaller-ordinary-ring).
Their source, controls and limitations are archived with the study.

After rebuilding on Spark, add `--hoist-rotations --share-chebyshev` alongside
`--gpu-plaintext-rns --s2c-first --planned-refresh --batch-refresh --frontier-refresh`
to the packed runner. The [recorded controller](results/dgx/2026-09-25/structural-four/full_controller.py)
contains the complete flags and exact process commands; the
[study](docs/research/2026-09-25-structural-four.md#reproduction-and-provenance)
links the build instructions and portable alternative-backend recipes.

The preceding [GPU RNS study](docs/research/2026-09-25-gpu-rns.md) reduced
750.41 → 618.95 s (17.52%), with ordinary evaluation down 23.96%. The preceding
[S2C-first study](docs/research/2026-09-25-s2c-first.md) reduced 759.70 → 747.63 s
(1.59%); its refresh saving was largely offset by ordinary preparation work.
The [ready-node refresh study](docs/research/2026-09-25-packed-frontiers.md)
reduced 941.95 → 761.00 s (19.21%), short of that earlier campaign's 20% target.
Each study retains its own controls; the new comparison reruns the released
GPU-RNS executable rather than treating an older sample as its control.

These are native evaluation times, including encoding, GPU upload and client
feedback inside the evaluation loop. Setup/key generation, input parsing,
post-evaluation validation and external transport are excluded. Compare
variants within each study: model sizes, weights and numerical contracts
differ, so this table does not rank architectures. Two full samples per mode
on one fixed prompt do not establish statistical significance or arbitrary-prompt
performance. Fresh-key error differences do not establish accuracy improvements.
Local release checks pass **290 tests**, including **21 native C++ contracts**;
the new rotation helper passes **552 complete RNS/metadata cases** across both
model configurations. The previous GPU-RNS qualification remains recorded.

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
