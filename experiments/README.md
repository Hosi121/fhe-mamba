# Experiments

Run commands from the repository root after
`uv sync --locked --extra experiments`. Start with the
[reproduction guide](../docs/reproducing.md) for a complete supported path.
Use `--help` on a Python runner for its inputs and output options.

Use the [shared experiment workflow](../docs/experiments.md) for job deadlines,
resource locks, completion events and public evidence. Machine-specific settings
belong in ignored `config/local/`; the installed `fhemamba.benchmarks` package
owns common orchestration. Do not copy a dated controller out of `results/`.

## Entry points

| Task | Runners | Guidance |
| --- | --- | --- |
| Qualify classical-128 packed inference | [`security128/`](security128/README.md), `run_packed_probe.py --security 128-classic` | Actual QP/context audit, rejected-parameter controls and encrypted refresh probes; [B300 prototype](../docs/research/2026-09-27-b300-classical128.md) |
| Checkpoint parity | [`run_parity.py`](run_parity.py) | Compare the reference with Transformers on a CPU |
| Mamba-3 mixer | [`export_mamba3_probe.py`](export_mamba3_probe.py), [`run_packed_probe.py`](run_packed_probe.py), [`check_mamba3_upstream.py`](check_mamba3_upstream.py) | [SISO reference parity and encrypted small-block verification](../docs/mamba3.md) |
| Trained Mamba-3 | [`export_mamba3_lm.py`](export_mamba3_lm.py), [`check_mamba3_lm_upstream.py`](check_mamba3_lm_upstream.py), [`report_mamba3_generation.py`](report_mamba3_generation.py) | Full checkpoint export, upstream CPU parity and complete encrypted client generation; [matched full speedup](../docs/research/2026-09-24-mamba3-depth-batching.md) and [microkernel profiles](../docs/research/2026-09-24-mamba3-microkernels.md) |
| Export and generate | [`export_m1_payload.py`](export_m1_payload.py), [`run_dgx_generation.py`](run_dgx_generation.py) | Frozen coefficients to the measured prompt-to-text workload |
| Run a GPU campaign | [`run_dgx_campaign.py`](run_dgx_campaign.py) | Execute a [manifest](manifests/README.md), record provenance and apply acceptance gates |
| Check payload quality | [`run_payload_quality.py`](run_payload_quality.py), [`run_ppl_ladder.py`](run_ppl_ladder.py), [`diagnose_payload_domain.py`](diagnose_payload_domain.py) | Separate plaintext approximation quality from encrypted numerical error |
| Certify gates and normalization | [`fit_dissipative_gates.py`](fit_dissipative_gates.py), [`plan_normalization_schedules.py`](plan_normalization_schedules.py), [`certify_payload_ranges.py`](certify_payload_ranges.py) | Fit and certify public polynomial candidates |
| Measure normalization | [`build_normalization_probe.py`](build_normalization_probe.py), [`run_normalization_probe.py`](run_normalization_probe.py), [`run_normalization_campaign.py`](run_normalization_campaign.py) | Isolated native probes; see [validation](../docs/validation.md#isolated-encrypted-normalization) |
| Analyze runtime and bounds | [`analyze_generation_cost.py`](analyze_generation_cost.py), [`analyze_normalization_bounds.py`](analyze_normalization_bounds.py) | Offline analyses of recorded data; no GPU execution |
| Check an OpenFHE inverse-FFT plan | [`openfhe_ifft/`](openfhe_ifft/README.md) | Separate-library bitwise FFT and integer-coefficient comparisons; [B300 profiling study](../docs/research/2026-09-26-b300-nsight-ifft.md) |
| Check forward NTT register exchange | [`ntt_warp_tail/`](ntt_warp_tail/README.md) | Separate-backend coefficient oracle and CPU lane check; [B300 trial](../docs/research/2026-09-26-b300-ntt-warp-tail.md) rejects the candidate on prefix timing |
| Qualify an even bootstrap seed | [`refresh_even_seed/`](refresh_even_seed/README.md) | Optional FIDESlib patch and encrypted seed/EvalMod/dual-ring probe; [B300 full single pair](../docs/research/2026-09-27-b300-refresh-even-polynomial.md) reduces refresh 8.22% and complete evaluation 1.62%, with fixed error/level gates |
| Chain single-GPU rotation phases | [`rotation_stream_chain/`](rotation_stream_chain/README.md) | Optional FIDESlib patch; exact RNS checks for both model/ring configurations and graph modes; [B300 full ABBA](../docs/research/2026-09-27-b300-rotation-stream-chain.md) reduces evaluation 3.14% with the even refresh seed preserved |
| Run bounded CPU plaintext preparation | [`run_packed_probe.py`](run_packed_probe.py) with `--prefetch-plaintexts` and coefficient encoding | [B300 two-stage study](../docs/research/2026-09-26-b300-rns-pipeline.md); exact-qualified RNS batch/fusion flags are also forwarded, but fusion is not adopted for full inference |
| Prepare two ordered public operands concurrently | Add `--prefetch-workers 2` to the bounded CPU path | [B300 follow-up](../docs/research/2026-09-27-b300-parallel-prefetch.md): same two-item window, full ABBA 14.07% reduction with verified CPU placement; default remains one worker |
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

The packed runner also forwards `--gpu-dual-ring` through ordinary and
budgeted execution. It requires the S2C-first/planned/batch refresh flags and
the corresponding native build option. The independent
[GPU transfer probe](gpu_dual_ring/README.md) checks the same implementation
against coefficient-domain and encrypted reference cases before model timing.

`--security 128-classic` selects the audited single-ring profile and rejects
that experimental bridge. `--security-digits` defaults to four in this mode.
The shared encoder also supports opt-in `--gpu-addend-rns` with
`--gpu-plaintext-rns`, and `--plaintext-cache-capacity N` with
`--cache-plaintexts`. The latter keeps a default of 64 and accepts 0..4096;
choose capacity against measured device memory. The [secure study](../docs/research/2026-09-27-b300-classical128.md)
retains exact coefficient qualification and cold-cache model comparisons.

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
