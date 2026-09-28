# Experiment entry points

Start with [current state](../docs/status.md) and the
[reproduction guide](../docs/reproducing.md). Run from the repository root
with `uv sync --locked --extra experiments` (use `--extra dev` for tests).

## Reusable tools

These are installed modules with ordinary Python APIs and one command tree:

```bash
python -m fhemamba --help
python -m fhemamba benchmark --help
python -m fhemamba diagnose --help
python -m fhemamba profile --help
python -m fhemamba recurrent --help
python -m fhemamba workload --help
```

| Task | Entry point | Guide |
| --- | --- | --- |
| GPU jobs, deadlines, completion events, comparisons, publication | `fhemamba benchmark` | [Workflow](../docs/experiments.md) |
| Packed inference and qualification | `fhemamba benchmark packed` | [Mamba-3](../docs/mamba3.md) |
| Actual generated-token qualification | `fhemamba benchmark generation-report` | [Model export](../docs/mamba3.md#trained-checkpoint-and-generation) |
| Native build identity | `fhemamba benchmark spark-build` / `b300-build` | [Build scope](../docs/dgx-spark.md); B300 helper is historical |
| Frozen references, operator errors and replay cases | `fhemamba diagnose` | [Diagnostics](packed_diagnostics/README.md) |
| CPU CKKS primitives and recurrent error attribution | `fhemamba diagnose ckks --recipe ... --output ...` | [Recipes](local_ckks/README.md) |
| Nsight SQLite/CSV summaries | `fhemamba profile` | [Profiling](profiling/README.md) |
| Recurrent-state fixtures and comparison | `fhemamba recurrent` | [Component studies](recurrent_state/README.md) |
| Mamba-3 model/mixer export and upstream parity | `fhemamba workload` | [Mamba-3](../docs/mamba3.md) |

Use each subcommand's `--help` for options. The old standalone entry points
were removed; arguments remain the same. Common hashing, JSON and repository
lookup are in `fhemamba.benchmarks.io`. CPU CKKS probe arithmetic is in
`fhemamba.ckks_probes`. Do not import another experiment script or copy helpers.

The packed runner's `--security 128-classic` enables the audited profile.
`--indexed-mask-cache` is experimental and **unqualified**, disabled by default.
See [optimization mechanisms](../docs/optimizations.md) for other opt-in paths;
[the current baseline](../docs/status.md) identifies the qualified combination.

## Specialized studies

Distinct experimental workflows remain here, grouped by purpose. Their shared
math and infrastructure are imported from `fhemamba`; they are not extra APIs.

| Directory | Purpose |
| --- | --- |
| [analysis/](analysis/) | Cost bounds, state coverage, noise-flow and telemetry reports |
| [quality/](quality/) | Checkpoint parity, PPL and polynomial-domain diagnostics |
| [calibration/](calibration/) | Frozen gate fits, state coordinates and range certificates |
| [export/](export/) | Mamba-2 payloads and normalization/debug fixtures |
| [normalization/](normalization/) | Build, run and compare isolated normalization probes |
| [algebra/](algebra/) | Phase, decay, selective-gate and recurrence experiments |
| [execution/](execution/) | Existing manifest-driven Mamba-2 campaigns and remote generation |
| [local_ckks/](local_ckks/README.md) | Declarative CPU feasibility recipes; separately installed OpenFHE bindings required |
| [manifests/](manifests/README.md) | Versioned workloads and acceptance gates |

Native candidates retain their probe/build assets in [ntt_lazy/](ntt_lazy/README.md),
[ntt_warp_tail/](ntt_warp_tail/README.md), [openfhe_ifft/](openfhe_ifft/README.md),
[gpu_dual_ring/](gpu_dual_ring/README.md), [refresh_even_seed/](refresh_even_seed/README.md)
and [rotation_stream_chain/](rotation_stream_chain/README.md).
[security128/](security128/README.md) and [level_schedule/](level_schedule/README.md)
explain their parameter and circuit analyses. Historical shell launchers remain
under `slurm/`; shared Mamba-2 native argument assembly is `dgx_mamba2_common.sh`.

Write fresh output to ignored `runs/`, machine settings to `config/local/` and
private handoff notes to `.local/`. Publish reviewed evidence to `results/` using
the shared workflow. Delete superseded operational scripts instead of archiving
them. Published source snapshots describe the measured code before this
reorganization; recorded hashes and failures remain unchanged.
