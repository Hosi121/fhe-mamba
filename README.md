# Encrypted Mamba-2 Inference with CKKS

[![CI](https://github.com/Hosi121/fhe-native-mamba3/actions/workflows/ci.yml/badge.svg)](https://github.com/Hosi121/fhe-native-mamba3/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research prototype for evaluating the real, open-weight `mamba2-130m`
checkpoint under fully homomorphic encryption. The reference and lowering
pipeline is written in Python/PyTorch; encrypted execution uses CKKS through
OpenFHE and FIDESlib-GPU.

The active research trunk is [`fhemamba/`](fhemamba/README.md), with the native
GPU kernel in [`native/fideslib_stage0/`](native/fideslib_stage0/). The retired
pre-rebuild implementation and its orchestration are preserved on
`archive/pre-compat-retirement-20260811`, not shipped from `main`.

## Current status

Evidence recorded through **2026-07-14**, at package version `0.4.5`:

| Gate | Result | Evidence state |
|---|---|---|
| Model quality | WikiText-2 PPL **22.307 -> 22.333** (**+0.12%**) over 280 windows | Tracked result JSON |
| Lowering parity | **3.1e-5** against the reference decode schedule over five verified tokens | Tracked result JSON |
| Full encrypted chain | Reported pass, errors **0.01295 / 0.01173 / 0.03475** over 24 layers and three sequential tokens | **Documented only**; raw B300 JSON is missing and the result is not independently verifiable from this repository |
| Full-chain runtime | Reported **145.75 s** evaluation; **26.38 s** average for the two warm carried-state steps | Same documented-only B300 measurement |
| 128-bit parameters | Layer 0, two tokens: errors **0.012 / 0.031**, about **197 s/token** | Tracked raw JSON with legacy provenance fields |
| Key separation | Three-process probe passes at **1.79e-12** round-trip error | Tracked raw JSON |

The promoted B300 configuration combines:

- input-replicated true BSGS;
- FIDESlib fused linear transforms for `out_proj` only;
- complex real/imaginary pairing for recurrent-state refresh;
- a fully synchronized FIDESlib build;
- ring `2^16`, `security=not-set`, and a 65 GiB plaintext cache.

The B300 result is feasibility and systems evidence, not a 64-bit or 128-bit
security claim. See the [evidence registry](docs/evidence.md) for the exact
provenance state and the
[bottleneck survey](docs/research/2026-07-13-fhe-mamba-bottleneck-survey.md)
for measured comparisons and negative results.

## Claim boundary

This repository does **not** yet claim:

- a complete 128-bit-secure protocol, including return-path noise flooding;
- a 24-layer run at 128-bit parameters;
- long-horizon or interactive encrypted generation;
- process-separated autoregressive execution of the full Mamba kernel;
- a measured full-kernel client/server round trip;
- support for models beyond `mamba2-130m`.

The immediate milestone is a five-step B300 autoregressive run using the
promoted fused-output and paired-state configuration. Full-kernel process
separation and a 128-bit full-chain run follow it. The canonical work order is
in the [backlog](docs/backlog.md) and [roadmap](docs/roadmap.md).

## Architecture

```text
client                              server (GPU CKKS)
------                              -----------------
tokenize + embed
encrypt hidden state       ----->   24 x Mamba-2 block + final RMSNorm
decrypt final hidden state <-----   encrypted hidden state
lm_head + token selection
```

Weights are public. Prompt tokens, intermediate activations, recurrent state,
and returned hidden states are the protected values. Embedding and `lm_head`
remain client-side because evaluating the public 50k-vocabulary head under FHE
would add cost without protecting an additional secret.

The encrypted path compares against the same polynomial circuit used by the
reference implementation. Exact-model approximation error and CKKS execution
error are reported separately, and decrypted diagnostics are never fed back
into ciphertext execution.

## Repository layout

```text
fhemamba/                    active reference, lowering, payload, and experiments
native/fideslib_stage0/      active FIDESlib/OpenFHE GPU kernel and native tests
scripts/                     local, DGX, B300, and artifact helpers
fhemamba/results/            small tracked correctness and benchmark artifacts
docs/research/               current measurement-driven research notes
tests/                       repository/native integration contracts
runs/                        ignored historical/generated experiment outputs
```

## Local development

Python 3.10 or newer is required. Reference tests do not require a GPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
scripts/run_fast_checks.sh
```

The required release gate adds coverage:

```bash
scripts/run_checks.sh
```

Checkpoint parity and perplexity require a local checkpoint:

```bash
python fhemamba/experiments/run_parity.py \
  --checkpoint checkpoints/mamba2-130m-hf
python fhemamba/experiments/run_ppl_ladder.py \
  --checkpoint checkpoints/mamba2-130m-hf
```

See [docs/testing.md](docs/testing.md) for test tiers and GPU limitations.
The [maintenance boundary](docs/maintenance.md) identifies the canonical code
paths. See the [legacy archive record](docs/legacy-archive.md) for retired code.

## B300 five-token campaign

The promoted long-horizon candidate is pinned in
[`b300_autoregressive_prompt2_generate4.json`](fhemamba/experiments/b300_autoregressive_prompt2_generate4.json).
Before launching, add prompt-2/generate-4 client assets to the chain payload:

```bash
python fhemamba/experiments/export_autoregressive_client_payload.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --chain-dir /home/kataiwa/fhemamba-b300/payloads/m2_chain_payload_sqnewton_wiki512_t2 \
  --prompt-tokens 2 \
  --generate-tokens 4
```

On the B300 host, run the resumable campaign:

```bash
python fhemamba/experiments/run_dgx_campaign.py \
  --manifest fhemamba/experiments/b300_autoregressive_prompt2_generate4.json \
  --runner scripts/run_b300_mamba2.sh \
  --output-json /home/kataiwa/fhemamba-b300/results/b300-p2-g4-campaign.json \
  --resume
```

Promotion requires all five steps to decrypt, every polynomial-circuit error
to remain at or below `0.05`, generated token IDs to match, and the artifact to
record zero intermediate decrypts, per-token timing, bootstrap counts, peak
RSS, repository commit, and binary SHA-256.

## Versioning

- `0.4.x`: real Mamba-2 weights under real encrypted execution.
- `0.4.5`: package version for the documented 24-layer, three-token B300
  milestone. The Git tag remains pending until its raw success artifact is
  recovered or the baseline is rerun.
- `1.0.0`: an interactive encrypted-generation demo at 128-bit parameters with
  reproducible benchmark artifacts.

See [CONTRIBUTING.md](CONTRIBUTING.md) for artifact and review requirements.
Licensed under the [MIT License](LICENSE).
