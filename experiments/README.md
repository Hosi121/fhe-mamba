# Experiments

Run commands from the repository root after
`uv sync --locked --extra experiments`. Start with the
[reproduction guide](../docs/reproducing.md) for a complete supported path.
Use `--help` on a Python runner for its inputs and output options.

## Entry points

| Task | Runners | Guidance |
| --- | --- | --- |
| Checkpoint parity | [`run_parity.py`](run_parity.py) | Compare the reference with Transformers on a CPU |
| Mamba-3 mixer | [`export_mamba3_probe.py`](export_mamba3_probe.py), [`run_packed_probe.py`](run_packed_probe.py), [`check_mamba3_upstream.py`](check_mamba3_upstream.py) | [SISO reference parity and encrypted small-block verification](../docs/mamba3.md) |
| Trained Mamba-3 | [`export_mamba3_lm.py`](export_mamba3_lm.py), [`check_mamba3_lm_upstream.py`](check_mamba3_lm_upstream.py), [`report_mamba3_generation.py`](report_mamba3_generation.py) | Full checkpoint export, upstream CPU parity and complete encrypted client generation; [matched full speedup](../docs/research/2026-09-24-mamba3-depth-batching.md) and [microkernel profiles](../docs/research/2026-09-24-mamba3-microkernels.md) |
| Export and generate | [`export_m1_payload.py`](export_m1_payload.py), [`run_dgx_generation.py`](run_dgx_generation.py) | Frozen coefficients to the measured prompt-to-text workload |
| Run a GPU campaign | [`run_dgx_campaign.py`](run_dgx_campaign.py) | Execute a [manifest](manifests/README.md), record provenance and apply acceptance gates |
| Check payload quality | [`run_payload_quality.py`](run_payload_quality.py), [`run_ppl_ladder.py`](run_ppl_ladder.py), [`diagnose_payload_domain.py`](diagnose_payload_domain.py) | Separate plaintext approximation quality from encrypted numerical error |
| Certify gates and normalization | [`fit_dissipative_gates.py`](fit_dissipative_gates.py), [`plan_normalization_schedules.py`](plan_normalization_schedules.py), [`certify_payload_ranges.py`](certify_payload_ranges.py) | Fit and certify public polynomial candidates |
| Measure normalization | [`build_normalization_probe.py`](build_normalization_probe.py), [`run_normalization_probe.py`](run_normalization_probe.py), [`run_normalization_campaign.py`](run_normalization_campaign.py) | Isolated native probes; see [validation](../docs/validation.md#isolated-encrypted-normalization) |
| Analyze runtime and bounds | [`analyze_generation_cost.py`](analyze_generation_cost.py), [`analyze_normalization_bounds.py`](analyze_normalization_bounds.py) | Offline analyses of recorded data; no GPU execution |
| Analyze state/layout | [`run_budget.py`](run_budget.py), [`run_noise_flow.py`](run_noise_flow.py), [`audit_state_scale_coverage.py`](audit_state_scale_coverage.py) | Operation budgets, error sensitivity and calibration coverage |

Other `probe_*`, `export_*`, `audit_*` and `screen_*` scripts support the
[dated research studies](../docs/research/README.md). The two
`run_*_local.py` CKKS probes require separately installed OpenFHE Python
bindings and are outside the normal Python dependency installation.

The Mamba-2 prompt launcher also accepts `--fast-plaintext-upload` and
`--gpu-plaintext-ntt`. Its native executable takes explicit `0`/`1` values;
campaigns set `FAST_PLAINTEXT_UPLOAD` and `GPU_PLAINTEXT_NTT`. Both executors
use the same plaintext preparation implementation and retain opt-in defaults.
`--direct-plaintext-upload` (native: `1`, environment:
`DIRECT_PLAINTEXT_UPLOAD=1`) also removes redundant same-type staging vectors
inside the pinned library. It implies fast upload.

The packed runner forwards `--gpu-plaintext-ntt` and
`--fast-plaintext-upload` for the optional plaintext preparation paths.
The [GPU encoding study](../docs/research/2026-09-24-mamba3-gpu-encoding.md)
retains the matched launchers, exact-RNS probe, source snapshots and budget
ledger. `--gpu-plaintext-ntt` implies fast upload; both preserve the model's
existing exact/polynomial and generated-token gates.
The packed runner additionally forwards `--direct-plaintext-upload`,
`--naf-rotations`, `--reuse-dead-inputs` and `--compact-weights`, including through
the budgeted execution path. Commands and binary/payload hashes are recorded
for each run; these switches do not relax validation.

`--hoist-rotations` shares rotation prefixes and sibling key-switch preparation
in the packed BSGS paths. `--share-chebyshev` reuses bases for polynomial nodes
with the same encrypted input and normalization interval. Both are opt-in and
are forwarded through ordinary and budgeted execution. Their counters and
qualification are described in the
[four-candidate study](../docs/research/2026-09-25-structural-four.md).

## Supporting files

- [`manifests/`](manifests/README.md): versioned campaign parameters and acceptance gates.
- [`slurm/`](slurm/): historical cluster launchers; inspect their host/environment assumptions before adapting them.
- `dgx_mamba2_common.sh`: shared native arguments, sourced by the platform runners.
- `manage_dgx_build.py`, `manage_b300_build_metadata.py`: build identity and validation.
- `run_dgx_layer_ladder.sh`, `run_dgx_process_separated.sh`: lower-level historical host workflows; current Spark execution starts with [`scripts/run_dgx_spark.sh`](../scripts/run_dgx_spark.sh).

Use `runs/<experiment>/` for new local output. Publish reviewed artifacts under
[`results/`](../results/README.md). A passing CPU probe or short encrypted smoke
does not establish a full-model result; follow the
[research validation gates](../docs/validation.md).
