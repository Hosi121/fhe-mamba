# Four structural optimization trials

The [study](../../../../docs/research/2026-09-25-structural-four.md) implements
one mechanism for each of four candidates and records numerical qualification,
matched timing, integration of eligible candidates and adoption decisions.
The frozen baseline is the preceding GPU-RNS/S2C-first implementation, `c90c54d`.
The packed-executor implementation is `b6817f0`.

| Candidate | Implemented path | Small-trial decision |
| --- | --- | --- |
| Shared rotations | Signed-digit prefix trie and sibling ModUp hoisting | Exact RNS/metadata parity; eligible for full comparison |
| Shared polynomial bases | Same-input, same-interval Chebyshev cache | Frozen-polynomial and refresh-invalidation gates pass; eligible |
| 32-bit arithmetic | Cheddar nominal 59-bit composite-rescale profile | Reject: numerical failure; matched 64-bit and lower-precision 32-bit controls pass |
| Smaller ordinary ring | Lattigo CPU N=32,768 → N=65,536 refresh → N=32,768 | Reject this CPU route: accuracy passes, boundary cost outweighs savings |

The full ABBA mean is **619.4101 → 614.1094 s (0.86%)**. Both shared-work
options are adopted as opt-in; ordinary evaluation falls 1.36% while the
refresh circuit and 484 calls remain unchanged. See `selection.json` for the
complete decisions. This campaign does not demonstrate a large speedup.

The two alternative backends are encrypted microcircuits, not model runs.
Their negative decisions do not rule out other 32-bit profiles or a future
GPU ring-switch implementation. Failed timings are not used as speedups.

## Evidence map

- `contract.json`: frozen stopping rule, model gates and baseline identity.
- `native-sources.tar.gz`, `native-source-manifest.json`: final measured native
  source; `source-verification.json` checks it against the implementation commit.
- `prefix-native-*`: earlier source archive for the prefix and rotation probes;
  final changes are include-only cleanup. Both archives are retained.
- `backend-source-verification.json`: 91 backend files unchanged from S2C-first.
- `dependency-verification-before/after.json`: canonical source/library identities.
- `rotation-probe-mamba2/`, `rotation-probe-mamba3/`: 552 complete RNS cases and
  matched batch microbenchmarks, with unchanged live inputs.
- `actual-basis-coefficients.json`, `basis-probe.txt`, `basis-probe-*`:
  production sin/cos coefficients and the shared-basis numerical probe.
- `basis-refresh-probe.txt`, `basis-refresh-*`: a refresh between consumers;
  the candidate records both cache invalidation and subsequent reuse.
- `prefix-*`: eight independent processes in a mirrored four-mode order.
- `full-*`: four independent full processes in baseline/candidate/candidate/baseline
  order; five encrypted evaluations and four generated tokens per process.
- `alternative-backends/`: selected mechanisms, failed-source variants, pins,
  parameter arrays, upstream notices and portable build instructions.
- `composite-final-32/64`: final-source nominal 59-bit small-circuit pair;
  exact commands and executable/shared-library hashes are in `run.json`.
- `composite-stock32.*`: upstream 40-bit 32-bit profile control, different
  precision/depth; `composite-diagnostic.*` is client-only failure localization.
- `small-ring-dense-15/16`: passing dense-secret CPU probes, four widths each.
- `small-selection.json`: prerequisite selection before any full-model run.
- `summary.json`, `summarize.py`: checked gates, archive integrity and disjoint
  cost categories; host encoding is nested, not added twice.
- `selection.json`, `validation.json`, `cleanup.json`, `local-cleanup.json`: final
  decisions, checks and disposal of owned build trees after retaining sources
  and measured binaries.
- `build-identity/`, `alternative-source-verification.json`: compiler/link details
  and verification of the archived adapter against the compiled source.
- `composite-controls-provenance.json`: labels reconstructed control invocations
  and missing diagnostic-build hashes explicitly.

The controllers record their exact original host paths. `full_controller.py`
shows the complete production flags; the alternative-backend README provides
portable recipes. Model payload manifests contain hashes, not weight tensors.
No client secret keys are archived.

## Failed controls

`first-pair-build-attempt1/` records the missing CMake target before reconfigure.
`first-pair-attempt2/` preserves the synthetic tiny-coefficient fixture that
also failed on the released baseline; its earlier rotation microbenchmarks
included an unnecessary zero-rotation clone and are not used in the summary.
`small-ring-15/16` use the unsuitable default K=16 dense-secret configuration.
`cheddar-build-log/` preserves the initial GMP link error; `composite-attempt1/`
preserves the default key-level error and debugging log. `composite-32/64`
are earlier paired numerical qualification, superseded for source identity
by `composite-final-32/64`. None is silently relabeled as a successful run.

Recheck the curated evidence without a GPU:

```bash
python3 results/dgx/2026-09-25/structural-four/summarize.py
uv run --no-sync fhemamba validate-artifacts --require-commit \
  results/dgx/2026-09-25/structural-four/summary.json
```

The benchmark retains the existing fixed prompt, public checkpoint/polynomials,
0.001 model gates, two-pass S2C-first refresh and `security=not-set` client-loop
scope. The specialized Mamba-2 full executor is unchanged.
