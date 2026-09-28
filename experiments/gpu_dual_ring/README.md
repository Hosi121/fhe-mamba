# GPU dual-ring qualification

This standalone GPU probe uses the same bridge and CUDA NTT maps as
`packed_fideslib --gpu-dual-ring`. It checks one fixed experimental profile:
ordinary N=32,768, refresh N=65,536, depth 44, scale 59, dense uniform ternary
keys, two-pass S2C-first refresh, and a final 1e-6 gate. It is not a model
benchmark or a security-parameter certification.

Build the pinned, patched FIDESlib through `scripts/build_dgx_spark.sh` first.
Set `FIDESLIB_PREFIX` to the resulting `spark/install-...` directory and
`OPENFHE_PREFIX` to the OpenFHE installation. On the measured Spark host:

```bash
cmake -S experiments/gpu_dual_ring -B build/gpu-dual-ring \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES=121-real \
  -DCMAKE_PREFIX_PATH="$FIDESLIB_PREFIX;$OPENFHE_PREFIX"
cmake --build build/gpu-dual-ring -j 4
python -m fhemamba benchmark run experiments/gpu_dual_ring/probe-job.json \
  --settings config/local/dual-ring.json \
  --output runs/dual-ring-qualification --events runs/completions.jsonl
```

Before running, create the ignored `config/local/dual-ring.json`, replacing
paths and choosing allocated CPUs on your host:

```json
{
  "repo": "/path/to/fhe-mamba",
  "binary": "/path/to/fhe-mamba/build/gpu-dual-ring/gpu_dual_ring",
  "library_path": "/path/to/openfhe/lib:/usr/local/cuda-13.0/lib64",
  "cpu_affinity": "15-19",
  "threads": "4",
  "mode": "qualification"
}
```

The measured Spark configuration used affinity `15-19` and four threads.
The shared runner requires a fresh output directory and
records executable/source hashes, command, explicit environment, exit code
and process time. Its deadline is 1,800 seconds; process cleanup and completion
events also cover failures. `probe.json` contains accuracy and component times;
`run.log` retains setup and failure diagnostics. The modes `abba` and `baab`
compare large/small rings in alternating order. Negative iteration indices
are warm-ups and must be excluded from timing aggregates.

Each process independently verifies six NTT maps against coefficient-domain
oracles and 30 paired encrypted transfer cases (both directions, with all
slots checked). The general downward map averages the two large-ring slot
halves. The compute/refresh circuit embeds repeated halves, so that projection
preserves its logical values. The model returns the two bootstrap components
separately to preserve its existing multiplicative depth; the small circuit
combines them before returning. Their transfer counts therefore differ.

The dated `campaign.py` and `run_probe.py` were retired. Their frozen source is
available in Git at `5d1e520`; the completed study's measured records remain
unchanged. For new model comparisons use `fhemamba benchmark packed
--gpu-dual-ring` with its required S2C-first/planned/batch flags and the shared
[comparison workflow](../../docs/experiments.md#compare-without-a-study-specific-script).
Run a small qualification before the full model, keeping the study's payload,
token and error gates explicit. A process completion event alone is not model
qualification.

See the [study](../../docs/research/2026-09-26-gpu-dual-ring.md) for comparison
conditions, measured scope and the adoption decision.
