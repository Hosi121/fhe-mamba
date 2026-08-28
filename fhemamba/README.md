# `fhemamba`: active encrypted Mamba-2 trunk

`fhemamba` contains the reference formula, FHE-friendly operator
substitutions, CKKS lowering, payload export, and experiment drivers for the
real `mamba2-130m` checkpoint. The native encrypted implementation lives in
[`../native/fideslib_stage0/`](../native/fideslib_stage0/).

The design rule is simple: language quality is checked first, then encrypted
execution must match the same polynomial circuit without clamping values or
feeding decrypted diagnostics back into the computation.

## Current result

The recorded results below were produced under package version `0.4.5` and include one
documented-only B300 milestone whose raw success artifact is still missing:

- WikiText-2 PPL `22.307 -> 22.333` over 280 windows;
- decode lowering parity `3.1e-5` from the reference over five verified tokens;
- 24 encrypted Mamba-2 layers over three sequential token steps;
- per-token polynomial-circuit errors `0.01295`, `0.01173`, and `0.03475`;
- 145.75 seconds total evaluation, with 26.38 seconds average for the two warm
  carried-state steps;
- 469 physical bootstraps and 120.24 GiB peak RSS on NVIDIA B300.

The full-chain run used ring `2^16` with `security=not-set`. A separate
layer-0/two-token run passes at 128-bit parameters, but no 24-layer 128-bit
claim is made. The three-token B300 values are documented measurements whose
raw success JSON still needs to be recovered or regenerated; this distinction
is tracked in [`../docs/evidence.md`](../docs/evidence.md).

## Promoted and experimental paths

The B300 runner promotes `out_proj` fusion, complex-paired normalized-state
refresh, refresh interval 1, a fully synchronized FIDESlib build, and a 65 GiB
plaintext cache.

The following remain opt-in experiments:

- fusion of both input and output projections;
- shared dt/decay head expansion;
- reduced synchronization builds;
- state-refresh intervals greater than one.

They have useful component-level or short-session results, but none currently
beats the promoted accuracy/runtime trade-off over the complete measured
session.

## Design rules

1. `src/fhemamba/reference.py` is the single Mamba formula.
2. Exact, range-recording, and polynomial behavior use injectable `Ops`, not
   separate model forks.
3. Closed-loop perplexity is the model-quality gate; MSE is diagnostic.
4. CKKS-incompatible clamping is forbidden. Range misses are measured.
5. Exact-model approximation error and encrypted polynomial-circuit error are
   separate metrics.
6. Small result JSON is tracked; large payloads and generated campaigns stay
   ignored unless explicitly curated.

## Layout

```text
src/fhemamba/
  reference.py             exact and FHE-lowerable Mamba-1/Mamba-2 formula
  ops.py                   exact, range, and polynomial operators
  lowering.py              CKKS operation and level schedule
  m1_payload.py            checkpoint payload and reference export
  bsgs_layout.py           replicated/interleaved BSGS slot layouts
  state_layout.py          recurrent-state packing and refresh plans
  rotation_keys.py         direct/composite rotation-key planning
tests/                     independent unit and parity tests
experiments/               local probes and resumable GPU campaigns
results/                   curated correctness, quality, and performance JSON
```

## Local checks

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest fhemamba/tests -q
```

Run parity and the PPL ladder against a local checkpoint:

```bash
python fhemamba/experiments/run_parity.py \
  --checkpoint checkpoints/mamba2-130m-hf
python fhemamba/experiments/run_ppl_ladder.py \
  --checkpoint checkpoints/mamba2-130m-hf
```

## Payload preparation

Add autoregressive client assets to an exported chain payload:

```bash
python fhemamba/experiments/export_autoregressive_client_payload.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --chain-dir /path/to/m2_chain_payload \
  --prompt-tokens 2 \
  --generate-tokens 4
```

Audit carried state and convolution FIFO bounds before an expensive run:

```bash
python fhemamba/experiments/audit_autoregressive_bounds.py \
  --checkpoint checkpoints/mamba2-130m-hf \
  --chain-dir /path/to/m2_chain_payload \
  --output-json /path/to/autoregressive-bound-audit.json \
  --ring-dim 65536 \
  --state-margin 1.1
```

A bound failure is a candidate failure, not an infrastructure failure.

## GPU campaigns

The generic campaign runner continues after numerical candidate failures,
fails on missing/malformed artifacts, performs an idle-resource preflight, and
supports `--resume`:

```bash
python fhemamba/experiments/run_dgx_campaign.py \
  --manifest fhemamba/experiments/b300_autoregressive_prompt2_generate4.json \
  --runner scripts/run_b300_mamba2.sh \
  --output-json /home/kataiwa/fhemamba-b300/results/b300-p2-g4-campaign.json \
  --resume
```

Manifests with an `acceptance` object are promotion gates: every artifact must
match the requested layer/token geometry, numerical tolerance, decrypt and
autoregressive-token checks, intermediate-decrypt policy, and synchronization
profile. A missed criterion makes the campaign artifact and process fail.
Exploratory manifests without `acceptance` retain the infrastructure-only exit
semantics. Resume reuses an artifact only when its schema, version, repository
commit (including a dirty-tree content fingerprint), binary SHA-256, geometry,
synchronization profile, and prior campaign's effective environment match the
current run; stale artifacts are rejected and rerun. Resume without a matching
prior campaign report reruns rather than trusting a standalone artifact.

In autoregressive mode, decrypting a completed `final_norm` output for client
token selection is an explicit protocol-boundary operation. It is counted
separately and does not violate `zero_intermediate_decrypts`; debug probes and
debug client re-encryption do.

`config/b300-platform.env` pins the promoted image/CUDA/SM, FIDESlib commit,
full-sync profile, and binary path for the image builder, native builder,
runner, and manifest. The runner validates the build metadata sidecar before
allocating the GPU and attaches that provenance to the raw result. Named
reduced-synchronization builds use separate source snapshots and are never
promotion candidates without a passing full 24-layer multi-token gate.

Replicated-BSGS projection plaintexts use persistent per-layer hit-first
handles. A compatible cache hit is resolved before allocating or scanning the
32,768-slot host mask; misses and level bypasses retain the existing lazy mask
and encode path. Raw artifacts expose replicated cache hits/misses/bypasses,
mask builds, materialized bytes, and mask-build seconds under `pt_cache`.
