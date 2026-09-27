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
**Mamba-3 SISO 187M** on **DGX Spark**, with the Mamba-3 workload also verified
on [B300](results/b300/2026-09-26/mamba3-dual-ring/). The project is **FHE Mamba** (`fhe-mamba`); the Python package
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

## Capabilities and limits

| Component | Supported scope |
| --- | --- |
| Python references | Mamba-2 and trained Mamba-3 SISO; shared polynomial and layout operators |
| Encrypted execution | CKKS GPU backend with encrypted activations and recurrent state, using public model weights |
| Classical-128 prototype | Audited single-ring Mamba-3 parameter profile; actual QP checked before and after evaluation |
| Client generation | Five encrypted evaluations and four generated tokens on the recorded fixed prompts |
| Reusable experiments | Declarative jobs, deadlines, completion events, isolated local settings and verifiable public evidence |

The client decrypts the final hidden vector to choose each next token. The
current inline client loop does not establish a process-separated private-chat
service. Polynomial quality on arbitrary prompts and deployment security remain
separate from numerical parity and the classical RLWE parameter audit.

## Recorded performance

| Workload and platform | Native evaluation | Security profile | Evidence |
| --- | ---: | --- | --- |
| Mamba-3 SISO 187M, B300 | 235.75 s / 58.94 s per generated token | Classical-128, N=131,072, QP=3,376 bits | [Parameter audit and optimization](docs/research/2026-09-27-b300-classical128.md) |
| Mamba-3 SISO 187M, B300 | 141.77 s / 35.44 s per generated token | Experimental `not-set`, dual ring | [Rotation stream chaining](docs/research/2026-09-27-b300-rotation-stream-chain.md) |
| Mamba-2-130M, DGX Spark | 1,942.99 s / 485.75 s per generated token | Experimental `not-set` | [Shared ownership](docs/research/2026-09-24-owned-arithmetic.md) |

These are separate studies with different models, hardware and security
settings. Each request has five encrypted evaluations and four generated tokens;
setup/key generation and final validation are excluded. The studies contain
matched baselines, sample counts, precision gates, memory requirements and failed
controls. These rates do not establish steady-state latency or arbitrary-prompt
performance. See the [evidence registry](docs/evidence.md) for the full record.

## Experiments and contributions

Use the [experiment workflow](docs/experiments.md) to run a versioned job with
local settings and completion events, then publish reviewed measurements.
Machine-specific settings belong in ignored `config/local/`; raw output belongs
in ignored `runs/`. Public records distinguish original and published hashes.

```bash
uv run --no-sync python -m fhemamba.benchmarks verify results
```

The [optimization guide](docs/optimizations.md) maps mechanisms to maintained
code and qualification suites. Dated technical reports are indexed under
[research](docs/research/README.md). Superseded one-off scripts are deleted;
shared utilities, necessary measured source snapshots and failure records remain.

| Directory | Contents |
| --- | --- |
| [`src/fhemamba/`](src/fhemamba/) | Installable model, operators, layout and benchmark utilities |
| [`native/`](native/README.md) | GPU backend, dependency patches, probes and C++ contracts |
| [`examples/`](examples/) / [`config/`](config/README.md) | CPU examples, public pins and configuration templates |
| [`experiments/`](experiments/README.md) | Maintained runners, manifests and qualification suites |
| [`results/`](results/README.md) | Public measurements, provenance bundle and hash manifest |
| [`docs/`](docs/README.md) | Usage, design, methods and measured studies |

For development, run `uv sync --locked --extra dev` and `scripts/run_checks.sh`.
See [contributing](CONTRIBUTING.md) and [testing](docs/testing.md) for the local
and hardware validation gates. The cleanup changes file organization and
publication, without changing inference kernels or benchmark numbers.

Code and original coefficient bundles are [MIT licensed](LICENSE). Checkpoints
and native dependencies are downloaded separately; see [third-party notices](THIRD_PARTY_NOTICES.md).
