# GPU plaintext preparation: matched complete generation

**21.87 → 18.59 minutes (−14.99%, 1.1763×)** for the same trained 12-layer,
five-evaluation Mamba-3 SISO 187M payload. All four actual tokens match and the
unchanged exact/polynomial 0.001 gates pass. There is one full run per mode.
See the [study](../../../../docs/research/2026-09-24-mamba3-gpu-encoding.md).

- `full/a1/`, `full/c2/`: standard preparation and GPU NTT/fast upload, using the
  same executable. Raw runner/native JSON and logs are byte-preserved;
  `generation.json` decodes the actual selected IDs with the pinned tokenizer.
- `prefix/`: final same-binary A/C/C/A comparison, **19.7113 → 16.4227 s**.
- `prefix-v2/`: earlier A/B/C/C/B/A comparison of standard, fast upload and
  single-limb GPU NTT. B isolates the CPU-copy change within this comparison.
- `probe-final/`: 160 exact-RNS input cases, four NTT batch sizes, encrypted
  multiplication and per-sample encode/upload times. All 640 coefficient NTT
  comparisons match. `probe-v2/` and `probe-v3/` preserve intermediate probes;
  quantitative final phase claims use `probe-final/`.
- `probe-initial/`: failed allocation with no native result. FIDESlib exited
  zero after a CUDA fault; the controller correctly marks the missing result
  as failed. The corrected path allocates NTT auxiliary workspace.
- `layer/`, `synthetic/`: trained one-layer token loop and four-step carried-state
  checks. The synthetic case additionally checks compatibility with caching.
- `compiled-sources-{initial,v2,final}.{json,tar.gz}`: 61 source files per
  snapshot, including native sources, contracts, runner and package version.
  The intermediate v3 probe differs from final only in report/open/header
  cleanup; its exact source archive is not retained and it supports no final
  numerical claim absent from the archived final probe.
- `comparison.json`, `compare.py`: derived comparisons with raw hash, token,
  tolerance, arithmetic-count, input-identity and archived-source checks.
  Run `python3 compare.py` from any directory to regenerate the summary.
- `run_final.sh`, `run_comparison.py`, `run_ntt_probe.py`: measured launchers.
  `run_comparison.py` was installed as `run_encoding_comparison.py` on the
  target. Paths and CPU IDs are machine-specific; use new output directories
  and an authorized budget when reproducing.
- `*-affinity.json`, `process-*.json`: requested and sampled actual CPU
  placement. Both full processes run on CPUs 15–19 with OMP four threads.
- `build*.log`, `target-provenance.json`: build outputs, compiler commands,
  source/dependency/executable identity. This bridge needs no FIDESlib patch.
- `checks*.log`: successful local gates before and after the final native
  changes; 280 Python tests include invocation of 16 CPU native contracts.
- `campaign-plan.json`, `budget.json`: separately authorized GPU campaign;
  **3226.63 / 21600 s across 18 attempts**. The six-hour limit is a review window.
  The previous 7200-second ledger remains immutable.
- `provenance.json`: repository/source state, references and artifact hashes.

Payload manifests and fixtures are already retained under
[`../mamba3-trained-generation/full-payload/`](../mamba3-trained-generation/full-payload/),
[`../mamba3-trained-generation/layer1-payload/`](../mamba3-trained-generation/layer1-payload/),
[`../mamba3-microkernels/prefix-payload/`](../mamba3-microkernels/prefix-payload/)
and [`../mamba3-siso/tokens4/`](../mamba3-siso/tokens4/).
Their hashes match the corresponding new runs. Checkpoints, full programs and
client vocabulary weights are reproduced using the pinned export instructions;
this directory does not duplicate their large binaries.

Evaluation and process-wall timing boundaries differ; the study states both.
The standard path's upload timer is uninstrumented. Peak RSS is a process
measurement, not incremental NTT scratch usage. This remains an opt-in,
inline-client, `security=not-set` experiment on one frozen prompt.
