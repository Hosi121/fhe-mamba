# Current state and handoff

Research baseline: 2026-09-28. This page records qualification and the next
investigation; new GPU work needs its own scope and budget.

## Qualified baseline

- Mamba-3 SISO 187M: **16 generated tokens**, B300, classical-128 profile.
  Public initial-state reuse takes **1,078.48 s / 67.41 s per generated token**
  across matched comparisons. This is a request-level evaluation average,
  not a steady-state throughput claim.
- Ring dimension 131,072; Q/P/QP bits 2,656/720/3,376; classical bound 3,523;
  uniform ternary secret, sigma 3.19, four HYBRID digits, depth 44, scale 59.
- Keep two S2C-first refresh passes, merged correction and frontier limit 256.
  Both final-hidden errors must be at most **0.001**, with identical token IDs
  and no diagnostic intermediate decryptions in qualifying generation.
- [Baseline study](research/2026-09-28-public-state-reuse.md) and
  [baseline evidence](../results/b300/2026-09-28/public-state-reuse/README.md)
  identify the measured source, command and input hashes. Earlier `not-set`
  measurements have a different security scope.

## Open accuracy problem

The frozen **64-token** request fails at selection 24. The final-layer SSM error
is localized to batch refresh of original nodes **75898 and 75904**.
Retained-input clone replay is accurate, but happens after the original call
and does not establish pre/post input immutability. Cache, allocation,
temporary-buffer state and mutation remain hypotheses. There is no minimal
standalone failing case or qualified repair.

Next, compare RNS inputs and metadata before and after the **first** refresh,
then packing, both bootstrap passes and extraction. Repair a demonstrated
invariant violation before ordinary 64-token qualification. Fresh encryption,
split groups, doubled bounds and unmerged correction have not established a fix.
See the [diagnosis](research/2026-09-28-long-accuracy.md) and
[repair proposal](../results/b300/2026-09-28/long-accuracy/repair-proposal.json).

## Last optimization decisions

The two-candidate exploration ended; neither candidate was adopted.

| Candidate | Result | Decision |
| --- | --- | --- |
| Delayed NTT modular reduction | Exact RNS match; short screen 1.32% slower | Rejected |
| Indexed mask cache | One full pair 2.20% faster; another candidate fails accuracy | Unqualified; default off |

The [result index](../results/b300/2026-09-28/long-accuracy/README.md)
retains controls, profiles, diagnostic vectors and source identities.

## Continuing development

The [generation API and CLI](generation.md) accept text or token IDs for Mamba-2
and Mamba-3, detected from the local checkpoint. Preparation binds one input and
generation length. Mamba-2 requires a frozen base chain and explicit
`mamba2-experimental` profile (`security=not-set`); Mamba-3 retains classical-128.
Exact, polynomial and CKKS execution share a result format while preserving
their existing acceptance gates. CPU tests cover request identity, reference
parity and failed native reports. This adds no GPU qualification or
arbitrary-prompt accuracy claim.

Model loading, preparation and native adapters now use an explicit registry;
model-specific arithmetic and acceptance remain in their integrations.
`inspect-model` reports capabilities without loading weights. See the
[integration contract](model-integration.md) for adding models and preparation options.

Mamba-3 client heads now use verified read-only blobs shared by sibling payloads;
original paths remain readable through relative symlinks. See the
[storage and transfer rules](experiments.md#shared-local-client-heads) before
moving one payload. Existing numerical payload bytes and hashes are unchanged.

Use the [Python tool map](package.md#research-tools),
[experiment index](../experiments/README.md) and [maintenance contracts](maintenance.md).
Check `git status` and `git log` for local work.
Run `CHECK_JOBS=2 scripts/run_checks.sh` for CPU checks; GPU qualification is separate.
