# Recorded results

Curated execution artifacts and derived reports live here. Read the
[evidence registry](../docs/evidence.md) for the claims each result supports,
its security/execution scope, and failed controls. Raw measurements retain
their original bytes, input paths, commits and hashes.

## Complete prompt-to-text workload

All three directories record the same 24-layer, five-evaluation request with
four generated tokens. Each contains raw backend output, campaign records and
text/comparison reports.

| Run | Recorded files | Study |
| --- | --- | --- |
| Initial pipeline | [`client-generation/`](dgx/2026-09-22/client-generation/) | [Baseline](../docs/research/2026-09-22-client-generation.md) |
| Periodic coefficients | [`periodic-gates/`](dgx/2026-09-22/periodic-gates/) | [Periodic encoding](../docs/research/2026-09-22-periodic-gate-coefficients.md) |
| Subring encoding | [`subring-gates/`](dgx/2026-09-22/subring-gates/) | [Subring encoding](../docs/research/2026-09-22-subring-gate-encoding.md) |

Inspect the latest recorded completion without a GPU:

```bash
uv run --no-sync fhemamba validate-artifacts --require-commit \
  results/dgx/2026-09-22/subring-gates/generation.json
```

This validates a recorded artifact; it does not rerun encrypted inference.

## Other evidence

| Location | Contents |
| --- | --- |
| [`dgx/2026-09-21/`](dgx/2026-09-21/) | Spark integration, normalization, vector RMS, state and diagnostic experiments |
| [`dgx/`](dgx/) | Earlier encrypted probes and feasibility runs |
| [`b300/`](b300/) | Historical B300 artifacts, including failed full-chain controls |
| Top-level JSON files | Plaintext parity, polynomial quality, certificates, layouts and derived budgets |
| [`archive/`](archive/README.md) | Outputs from the retired implementation, previously tracked under `runs/` |

The current model revision and frozen public coefficients live in
[`config/`](../config/README.md). Older quality reports may use different
coefficients; they do not automatically validate the latest native payload.
Some legacy JSON predates the current provenance schema.

## Recording new work

Write new local output under ignored `runs/<experiment>/`. Promote only reviewed,
claim-bearing evidence here and link it from the corresponding study. Do not
edit a recorded result to match a new source or payload hash, and do not drop
failed controls. See [contribution requirements](../CONTRIBUTING.md#benchmark-artifacts)
and the [old-to-new path map](../docs/repository.md#previous-layout).
