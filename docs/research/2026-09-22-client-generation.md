# Stabilized prompt-to-text generation

The first functional milestone connects a complete text prompt to encrypted
Mamba-2 recurrence, client token selection and decoded output. Runtime
optimization is deferred until this path has a measured baseline.

## Experiment

The input is exactly `The capital`, token IDs `[510, 5347]`, with four greedy
output tokens requested. This requires five sequential evaluations through
all 24 layers and final normalization. The input is not truncated to fit a
preselected token count. Generation has a fixed length rather than EOS stopping.

`run_dgx_generation.py` prepares a separate payload and runs the existing
validated native binary through the campaign runner. The base payload's
normalization recipes, joint gates, activation fits and carried-state bounds
remain unchanged. There is no new kernel optimization or relaxed error gate.

| Identity | SHA-256 |
|---|---|
| Frozen base payload | `b40d33b30e018ef91e894aa9e2bd26adb6be6d5e1ae49873e356d78795d5c745` |
| Generation payload | `9869571e4253ceb36a853aecf99ce44a0fa502d41181ef6bdcf12bee80c54375` |
| Native binary | `57636814d5792c3bc87e1c6721a8aaaaaf3aa79946e54ea9e4c956fcc95ce085` |
| Native source/configuration | `35472e2c0fc9ee91b6b8c38ef0b72cc5fa3deefd4c04b40e02db20986f69d131` |

The generation payload adds full-vocabulary client weights and exact/polynomial
reference traces to the frozen base. Both plaintext references select
`[273, 253, 4687, 273]`. The native loop independently selects the same IDs.
The new trace has zero observed escapes across 215,040 convolution-SiLU inputs,
184,320 gate-SiLU inputs, 125 block/final normalization inputs, 120 gated-norm
inputs and 2,880 joint-gate inputs. Inactive legacy softplus/decay fits have no
observations. Domain membership on this trace is not whole-model closure.

The native settings retain ring 65,536, depth 44, scale 59, `security=not-set`,
full synchronization, 5 GiB plaintext cache, group-normalized recurrent state,
Meta-BTS alpha 12 and no intermediate diagnostic decrypts. The experiment
uses fresh keys. The unchanged acceptance gate requires every output to
decrypt, every polynomial-circuit error to be at most 0.05, and all generated
IDs to match the polynomial reference.

## Result

**Pass.** The [native result](../../fhemamba/results/dgx/2026-09-22/client-generation/m2_chain_client-generation-20260922_l24_t5.json),
[campaign](../../fhemamba/results/dgx/2026-09-22/client-generation/client-generation-campaign.json)
and [decoded report](../../fhemamba/results/dgx/2026-09-22/client-generation/generation.json)
record four generated IDs `[273, 253, 4687, 273]`, matching both references:

```text
Prompt:     The capital
Completion:  of the Republic of
Full text:  The capital of the Republic of
```

This is the requested four-token prefix, not a complete sentence or a claim
of instruction-following quality. Every final output decrypts with zero
non-finite slots, and all five polynomial-circuit errors are below 0.05.

| Evaluation | Polynomial-circuit error | Exact-model hidden-output error |
|---|---:|---:|
| 0 | 0.000474228 | 0.0116941 |
| 1 | 0.004326422 | 0.0213753 |
| 2 | 0.004230646 | 0.0183543 |
| 3 | 0.002326486 | 0.0071206 |
| 4 | 0.009191800 | 0.0725723 |

Maximum CKKS-to-polynomial error is **0.009192**. Exact-model hidden-output
error is a separate quantity and reaches **0.072572** despite matching token
choices. No intermediate diagnostic decryptions occur; the four client
output decryptions are the token-selection boundary.

Evaluation takes **3,038.12 s (50.64 min)** after **23.44 s** setup, with
**37.05 GiB** peak RSS. The first step takes **577.17 s** and the four carried
steps average **615.24 s**. These kernel times exclude local reference
preparation and payload transfer. Operation counts are **2,187** physical
bootstraps, **47,667** rotations, **273,213** ct-pt products and **32,701** ct-ct
products. Bootstrap evaluation totals **1,075.83 s**; client token selection
and embedding feedback take **0.217 s** in this one-process run.

This establishes one short prompt-to-text baseline with fresh keys. It does
not demonstrate a speedup or cover new prompts, repeated keys or long sessions.

## Protocol boundary

This is a one-process client-loop simulation. Native code decrypts completed
final-normalization outputs, computes logits over the entire vocabulary,
selects the greedy token and freshly encrypts its embedding. Recurrent state
and convolution FIFO stay encrypted. The text report decodes the measured
selected IDs, including when they differ from the references.

The result does not establish a secret-key-free server, encrypted token
selection, long-session stability or a full-chain 128-bit security parameter
gate. Those remain separate milestones.

## Reproduction and checks

The [Spark runbook](../dgx-spark.md#complete-prompt-to-text-generation) contains
the complete command. The experiment directory is
`runs/client-generation-20260922/`; the remote copy is under
`/home/kataiwa/fhemamba/gate-integration-20260921/generation/`.
It retains the request, exact input hash, Python source archive and source
manifest, native log, raw numerical artifact, campaign and decoded report.
The decoded report was reconstructed only to add explicit input-mode and
claim-scope metadata; the original report is retained as `generation-launch.json`.
`report_reconstruction` binds the original report and archived reporting
sources. Native and campaign files are copied without modifying their bytes.
All three curated artifacts validate with **zero errors and warnings**.

Local validation passes **246 tests**, including **12 native C++ contracts**,
with **86.91%** coverage. New regression tests cover whole-prompt handling,
immutable base data, checkpoint mismatch, final-normalization domain checks,
stale/truncated evidence and reporting actual failed selections. The exporter
now applies the final normalization domain check when adding autoregressive
assets to an existing payload, as well as during the original export.
