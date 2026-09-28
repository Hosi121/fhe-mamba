# Long-generation diagnosis and two optimization trials

See the [study](../../../../docs/research/2026-09-28-long-accuracy.md) for the
numerical interpretation and adoption decisions. Neither optimization is
promoted. Classical-128 parameters and the final-hidden `0.001` gates remain
fixed; diagnostic decryptions never qualify as ordinary generation.

| Evidence | Purpose |
| --- | --- |
| [Contract](contract.json), [decisions](decision.json) | Two-candidate limit, fixed conditions and final adoption decisions |
| [Operator localization](accuracy-localization.json) | Before/after batch refresh and local downstream arithmetic errors |
| [Fine analysis](analyze-operations-diagnostic-fine-r3/analysis.json) | All 684 observation summaries for the 524 selected nodes |
| [CPU error propagation](refresh-counterfactual.json), [reproduction inputs](counterfactual-provenance.json) | Two damaged refresh outputs explain the final vector within 7.30e-7 |
| [Original-history controls](refresh-replay-comparison.json), [repair proposal](repair-proposal.json) | Original refresh fails; identical-group clone replay passes. Execution-state/ownership diagnosis remains open |
| [Replay origins](replay-origins.json) | Distinguishes fresh-input controls from full-history ciphertext clones |
| [Fresh-input result](replay-refresh/native.json), [case](isolation-refresh/case.json) | The standalone logical-input case does not reproduce the failure |
| [Reconstruction check](isolation-refresh/reconstruction-check.json) | Published observed vectors reconstruct the exact case bytes |
| [NTT archive audit](archive-verification.json), [exact GPU oracle](oracle-p-r3-ntt/native.json) | Unchanged non-NTT archive members and 373,293,056 matching words |
| [Nsight summary](ntt-profile-summary.json) | Register residency and selected kernel-replay measurements |
| [Passing cache pair](full-cache-passing-pair.json), [failed candidate](full-b1/completion.json) | Descriptive 2.20% reduction cannot override the failed accuracy gate |
| [Validation](validation.json), [source identities](measured-native-source.json) | Complete checks and the distinction between measured and later diagnostic code |
| [Setup attempts](setup-attempts.json) | Resolved build/profiler/probe setup errors, distinct from numerical failures |

The [publication manifest](publication.json) records original and public hashes.
Logs, measured source snapshots, the extracted case and selected public-fixture
observations are in [provenance.tar.gz](provenance.tar.gz). Environment-specific
identities are normalized. Disposable orchestration scripts are excluded;
reusable tools are under `experiments/packed_diagnostics/`,
`experiments/ntt_lazy/` and `src/fhemamba/benchmarks/`.

```bash
python -m fhemamba.benchmarks verify results/b300/2026-09-28/long-accuracy
python -m fhemamba.benchmarks extract \
  results/b300/2026-09-28/long-accuracy runs/long-accuracy-evidence
```

Raw profiler recordings remain local; reviewed counter exports preserve the
recording identities. The artifact verifier checks hashes, not the truth of
performance or accuracy claims. Failed and timed-out runs remain failed.
