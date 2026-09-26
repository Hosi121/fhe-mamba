# GPU ordinary/refresh ring switching

This directory records one GPU mechanism: ordinary arithmetic at N=32,768,
encrypted transfer into the existing N=65,536 two-pass S2C-first refresh,
then encrypted transfer back. The complete model is trained Mamba-3 SISO
187M: 12 layers, five encrypted evaluations and four generated tokens.

The [study](../../../../docs/research/2026-09-26-gpu-dual-ring.md) explains the
maps, numerical contract, timing boundaries and adoption decision. The
implementation commit is `c2d7d4d1441f48ee17b25327093e68acf4b868df`; the comparison
baseline is `e6bde4d` / native implementation `b6817f0`. Runtime selection is
opt-in with `--gpu-dual-ring`. Smaller-ring security equivalence and the
specialized Mamba-2 executor are outside this experiment.

Full ABBA native means are **613.8311 → 430.5145 s (29.86% reduction)**;
complete process means are **654.5168 → 476.3945 s (27.21%)**. All four
processes preserve both 0.001 error gates and the same generated IDs. The
mechanism is adopted as opt-in for this fixed workload.

## Evidence map

| Files | Purpose |
| --- | --- |
| `contract.json`, `full-selection.json` | Frozen question, one-mechanism stopping rule, gates and prefix eligibility decision |
| `micro-{qualification,abba,baab}/` | Three fresh primitive processes; each checks six exact NTT maps and 30 paired encrypted transfers; alternating component timings exclude warm-ups |
| `prefix-{0-baseline,1-candidate,2-candidate,3-baseline}/` | Matched one-layer prefix ABBA with original native JSON, commands, process time and logs |
| `full-{0-baseline,1-candidate,2-candidate,3-baseline}/` | Matched full-model ABBA with the same four generated IDs and both 0.001 gates |
| `default-path-regression/`, `post-validation.json` | Final candidate binary without the flag, plus native early rejection of missing S2C, wrong slot capacity and oversized logical nodes |
| `summary.json`, `summarize.py` | Derived means, disjoint timing categories, source/gate assertions and hashes of the supporting files |
| `native-sources.tar.gz`, `full-source-manifest.json`, `source-verification.json` | All 92 native files match the measured full source and implementation commit byte for byte |
| `prefix-native-sources.tar.gz`, `small-source-manifest.json`, `prefix-to-full-source.diff` | Exact prefix sources; only the usage help string changes before the full comparison |
| `small-probe-sources.tar.gz`, `full-controller-sources.tar.gz` | Exact controller/probe revisions used by the small and full campaigns |
| `qualification-1/`, `qualification-1-sources.tar.gz` | Earlier standalone qualification; source hashes preserved, not used in final timing means |
| `backend-build-sources.tar.gz`, `published-backend.json`, `backend-source.patch`, `backend-source-verification.json` | Isolated FIDESlib sources, pinned revision plus patch and byte identity with the preceding study's 91 backend files |
| `repo-snapshot.tar.gz` | Complete copied source tree used on DGX, archived before cleanup |
| `canonical-dependencies-{before-full,after}.json` | All 17 canonical dependency files remain unchanged |
| `full-payload-verification.json` | Frozen model payload hashes checked before the full campaign |
| `build*.log`, `*build-attempt1.log`, `final-*.txt`, `final-flags.make`, `probe-*.txt`, `probe-flags.make` | Build commands and metadata; the two initial compiler failures remain recorded |
| `completion_hook.py`, `hook-events.jsonl`, `postprocess_hook.py` | Completion/failure events and default-path validation after the serialized comparison |
| `validation.json`, `pytest.log`, `cleanup.json`, `cleanup.py` | Release checks and verified cleanup; measured executables remain on the original host |
| `remaining-cost-notes.json` | Static ownership/layout follow-ups, not implemented or credited with a speedup |

The initial compiler failures were an incorrect OpenFHE NTT class namespace
and a conflicting CUDA function declaration. They occurred before successful
qualification. Raw logs and result JSON retain their original bytes and
paths, including baseline commit labels recorded before the new source was
committed. The separate source verification binds those runs to the actual
measured code; it does not rewrite their metadata.

The complete backend backup named in `cleanup.json` remains on the measurement
host. The published source archive omits only the unrelated upstream BERT
checkpoint. `published-backend.json` records the original/archive hashes and
the omitted file hash; the report verifies every retained source file against
the original cleanup manifest.

## Recompute and reproduce

From the repository root, derive and validate the report without a GPU:

```bash
python3 results/dgx/2026-09-26/gpu-dual-ring/summarize.py
uv run --no-sync fhemamba validate-artifacts --require-commit \
  results/dgx/2026-09-26/gpu-dual-ring/summary.json
```

For a new GPU run, build with the [Spark guide](../../../../docs/dgx-spark.md)
and use the [primitive recipe](../../../../experiments/gpu_dual_ring/README.md)
or the [Mamba-3 model guide](../../../../docs/mamba3.md). The exact recorded
native commands and environment are in each `run.json`. Replace their host
paths with fresh output directories and the corresponding frozen payload;
do not overwrite archived results. `campaign.py` serializes ABBA and enforces
the small-to-full eligibility gate.

Complete model times include ordinary work, host preparation, GPU upload,
refresh packing, every ring transfer and client feedback inside evaluation.
Setup/key generation, initial parsing and final validation are included only
in process time. The micro component sum additionally excludes its artificial
pre-refresh level-drop fixture and must not be described as complete process
latency. The model returns the two bootstrap components separately (one up,
two down); the micro returns their sum (one up, one down).

The `refresh` category covers packing/unpacking, transfers and both bootstrap
passes. Operation categories subtract their nested refresh time to avoid
double counting; host encoding is nested within those categories. Peak RSS
includes key setup and is not a direct GPU-allocation measurement. Rates per
generated token amortize this five-evaluation/four-token request and are not
steady-state token latency. Two samples per mode on one prompt do not establish
statistical significance, arbitrary-prompt quality or an accuracy improvement.
