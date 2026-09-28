# Current state and handoff

Updated 2026-09-28. Start here; the dated research reports provide evidence
on demand. This page records state, not authorization for a new GPU campaign.

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

The frozen **64-token** request fails at selection 24. It has not qualified.
The abrupt final-layer SSM error is localized to batch refresh of original
nodes **75898 and 75904**. Injecting only these two observed outputs into the
CPU circuit reproduces the final vector within `7.30e-7` in one trace and
`1.90e-6` in another.

In the full-history diagnostic, the original refresh has maximum change
`0.05397`; identical-group replay on retained-input clones gives `4.66e-6`.
All four controls pass, so split groups, doubled bounds and unmerged correction
are not established repairs. The raw clone check occurs after the original
call: it does not prove pre/post input immutability. Cache, allocation,
temporary-buffer state and unintended mutation remain hypotheses.

The next useful investigation is a pre/post RNS and metadata comparison around
that **first** refresh, then stage comparisons through packing, both bootstrap
passes and extraction. Fix a demonstrated invariant violation before an ordinary
64-token qualification. Fresh encryption alone does not reproduce the fault.
There is no minimal standalone failing case or qualified repair yet.

Read the [diagnosis](research/2026-09-28-long-accuracy.md) and
[repair proposal](../results/b300/2026-09-28/long-accuracy/repair-proposal.json).
Do not rerun unrelated historical optimization campaigns to reconstruct context.

## Last optimization decisions

The agreed two-candidate exploration ended; neither candidate was adopted.

| Candidate | Result | Decision |
| --- | --- | --- |
| Delayed NTT modular reduction | 373M exact RNS words match; all-layer short screen 1.32% slower | Rejected implementation |
| Indexed mask cache | Passing full pair 2.20% faster; another full candidate fails accuracy | Unqualified; default off |

The [result index](../results/b300/2026-09-28/long-accuracy/README.md) retains
failed controls, Nsight exports, diagnostic vectors and source identities.

## Continuing development

Use [the Python tool map](package.md#research-tools) and
[experiment entry points](../experiments/README.md). Common diagnostics,
profiling, recurrent probes, qualification and build identity are installed
modules. CPU CKKS primitive and error-growth studies now use
[JSON recipes and one shared command](../experiments/local_ckks/README.md).
Remaining specialized scripts are grouped by purpose.

For local checks run `CHECK_JOBS=2 scripts/run_checks.sh`; GPU verification
is a separate step. The maintenance reorganization did not rerun encrypted
inference or change the qualified native implementation. Check `git status`
and `git log` for the checkout's current state before continuing.
