# FHE Mamba

[![CI](https://github.com/Hosi121/fhe-native-mamba3/actions/workflows/ci.yml/badge.svg)](https://github.com/Hosi121/fhe-native-mamba3/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research software for Mamba-2 inference with encrypted activations and recurrent
state. It includes a Python/PyTorch reference, certified polynomial operators,
and a CKKS GPU implementation using OpenFHE and FIDESlib.

The current checkpoint is **Mamba-2-130M**. The measured backend is **DGX Spark**
(CUDA 13, SM121). The package is called `fhemamba`; the repository name
`fhe-native-mamba3` is historical and does not imply Mamba-3 model support.

## Try it on a CPU

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from the repository root:

```bash
git clone https://github.com/Hosi121/fhe-native-mamba3.git
cd fhe-native-mamba3
uv sync --locked --extra experiments
uv run --no-sync python examples/cpu_smoke.py
```

The example creates a tiny random Mamba-2 locally and compares this project's
reference and stateful decode against Transformers. It downloads no checkpoint
and reports `passed: true` with errors below `1e-4`. This is a plaintext CPU
example; it does not perform encryption or produce useful natural-language text.
The first dependency installation includes PyTorch and can be large.

Inspect a recorded encrypted experiment without GPU hardware:

```bash
uv run --no-sync fhemamba validate-artifacts --require-commit \
  fhemamba/results/dgx/2026-09-22/subring-gates/generation.json
```

This validates the recorded result's schema and acceptance criteria. To run
new experiments, follow [reproducing the research snapshot](docs/reproducing.md).

## What is included

- Mamba-2 forward, recurrent decode, polynomial substitutions and packing/layout code.
- Public inverse-square-root schedules and joint write/decay polynomials, with
  conditional interval certificates and the frozen coefficient bundles.
- A native encrypted GPU kernel, coefficient encoders, bootstrap scheduling,
  isolated probes and CPU C++ contract tests.
- Build/payload provenance, experiment runners and raw success/failure artifacts.
- Arithmetic lower bounds and measured cost analyses to guide further work.

The package exposes model and layout utilities; GPU execution uses the native
build and experiment scripts from this repository checkout.

## Measured generation

On DGX Spark, the fixed prompt `The capital` produces
`The capital of the Republic of`: **24 layers, five encrypted evaluations,
four generated tokens**. All three variants use the same frozen payload.

| Implementation | Evaluation | Maximum CKKS-to-polynomial error |
| --- | ---: | ---: |
| [Initial complete pipeline](docs/research/2026-09-22-client-generation.md) | 50.6 min | 0.009192 |
| [Periodic coefficients](docs/research/2026-09-22-periodic-gate-coefficients.md) | 42.9 min | 0.011767 |
| [Subring coefficient encoding](docs/research/2026-09-22-subring-gate-encoding.md) | **38.5 min** | **0.016255** |

The subring run has identical generated IDs, operation counts and ciphertext
levels to the periodic baseline, with peak RSS **37.08 GiB**. Both encoder
optimizations are opt-in. Full timing is one run per variant; the isolated
probes and interleaved smoke runs provide additional evidence. The existing
error threshold is `0.05` against the matching polynomial circuit, not the
original floating-point model. See the [evidence registry](docs/evidence.md)
for source hashes, quality measurements, failed controls and claim boundaries.

## Execution and security scope

The current full-model benchmark runs a **single-process client loop**. Model
weights are public; activations and carried state remain encrypted during
model evaluation. The client decrypts the final hidden vector, selects the
next token with the public vocabulary projection, and encrypts its embedding.

```text
client                                  encrypted model evaluation
tokenize, embed, encrypt  ------------->  24 Mamba-2 blocks + final RMSNorm
                                         retain encrypted state/FIFO
decrypt final hidden     <-------------  encrypted final hidden
select token, embed, encrypt ---------->  next evaluation
```

The full-chain results use `security=not-set` and establish numerical/systems
feasibility, **not a full-chain 128-bit security guarantee**. Separate 128-bit
normalization probes do not change that scope. A secret-key-free full-model
server, encrypted token selection, output-security protocol and long-session
encrypted accuracy remain open. This implementation is not ready for
interactive private chat or production handling of sensitive prompts.

## Reproduce and contribute

| Start here | What it covers |
| --- | --- |
| [Reproduction guide](docs/reproducing.md) | Pinned checkpoint, included coefficients, CPU parity, payload export and generation |
| [DGX Spark runbook](docs/dgx-spark.md) | Native dependencies, isolated builds, campaigns and machine requirements |
| [Package guide](fhemamba/README.md) | Python modules, operators and experiments |
| [Testing](docs/testing.md) | Local contracts, approximation and GPU acceptance gates |
| [Design](docs/design.md) | Protocol invariants and algorithm choices |
| [Cost bounds](docs/research/2026-09-22-generation-cost-bounds.md) | Conditional runtime limits and normalization depth bounds |
| [Roadmap](docs/roadmap.md) / [backlog](docs/backlog.md) | Remaining research and implementation work |

For development (CMake and a C++17 compiler are needed for the native CPU tests):

```bash
uv sync --locked --extra dev
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  CHECK_JOBS=2 scripts/run_checks.sh
uv build
```

CI runs Python 3.10/3.12 tests, C++ contracts, the CPU example, artifact
validation and an installed-wheel smoke check. It does not run the GPU
benchmarks. See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and evidence
requirements. Report reproducible problems through [GitHub issues](https://github.com/Hosi121/fhe-native-mamba3/issues).

Code and original coefficient bundles are provided under [MIT](LICENSE).
Checkpoints and native dependencies are downloaded separately; their upstream
licenses and attribution are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
Historical B300 code/results and the retired compatibility stack are described
in [maintenance](docs/maintenance.md). Package version **0.5.0** is a research
version and does not denote completion of the security or protocol work.
