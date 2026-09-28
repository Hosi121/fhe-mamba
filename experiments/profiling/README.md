# GPU timeline diagnosis

Use a short, frozen payload with the production evaluator and options. Derive a
generation prefix with `export_mamba3_lm.py --prefix-source` rather than refitting
its coefficients. Capture `--profile-evaluation` with Nsight Systems:

```sh
nsys profile --trace=cuda,nvtx,osrt --sample=process-tree \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node -o trace \
  packed_fideslib PROGRAM OUTPUT 0.001 0.001 --profile-evaluation OTHER_OPTIONS
nsys export --type=sqlite --output=trace.sqlite trace.nsys-rep
python -m fhemamba profile nsys trace.sqlite --output=summary.json
```

Use `--start-seconds N` to report a later capture interval separately. Choose
the boundary from the workload's initialization markers or timing evidence and
retain the full summary as well. Activities crossing the boundary are clipped,
including API waits; CPU samples are filtered to the selected interval.

The stdlib-only analyzer reports the union of overlapping GPU activities,
ten-second timeline bins, kernel/API totals and available CPU leaf samples.
It rejects multiple-device captures because runtime API attribution would be
ambiguous. Graph-level captures are also supported, but their activity envelopes
include internal gaps. Neither kind of coverage is SM occupancy or bandwidth
utilization. Summed kernel/API times overlap and must not be added as fractions
of wall time. Use Nsight Compute on selected expensive kernels for hardware
counters, then separate unprofiled runs to evaluate any optimization.

Export selected Nsight Compute launches with `ncu --import profile.ncu-rep
--page raw --csv > metrics.csv`. Run `python -m fhemamba profile ncu
metrics.csv --output counters.json` to retain durations, SM/DRAM throughput,
occupancy, geometry and register/shared-memory use with their original units.
Kernel replay and the profiler's cache policy affect these measurements; keep
the launch filters and avoid extrapolating four launches to every kernel level.
Use repeated `--metric RAW_COUNTER_NAME` arguments to extract additional
columns already present in the CSV, with their units and strict finite-value
checks. A requested but absent counter is an error. For example, select
`sm__issue_active.avg.pct_of_peak_sustained_elapsed` for instruction issue
activity; it is distinct from achieved occupancy and from individual execution
pipeline throughput. See NVIDIA's
[metric and section guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)
before interpreting scheduler or warp-stall counters.

Raw profiler files can contain machine details. Keep them on the measurement
host; publish reviewed summaries, exact commands and hashes through the common
benchmark evidence publisher. The analyzer never changes the SQLite export.

Tool options: [Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
and [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
