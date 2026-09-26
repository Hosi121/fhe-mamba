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
LD_LIBRARY_PATH="$OPENFHE_PREFIX/lib:/usr/local/cuda-13.0/lib64" \
  python3 experiments/gpu_dual_ring/run_probe.py \
  --binary build/gpu-dual-ring/gpu_dual_ring \
  --output runs/dual-ring-qualification --mode qualification \
  --commit "$(git rev-parse HEAD)"
```

The wrapper pins CPU affinity 15–19 and four OpenMP threads, requires a fresh
output directory and records executable/source hashes, command, exit code
and process time. `probe.json` contains accuracy and component times;
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

`campaign.py ROOT small|full` is the dated study controller. It consumes the
recorded `prefix-baseline-command.json` and `full-baseline-command.json` under
ROOT; those commands must refer to the local frozen payload and binaries.
The full stage additionally requires the small stage's successful status and
an explicit `full-selection.json` eligibility decision. Ordinary users can
run a model with `experiments/run_packed_probe.py --gpu-dual-ring` and its
required S2C-first/planned/batch flags instead.

See the [study](../../docs/research/2026-09-26-gpu-dual-ring.md) for comparison
conditions, measured scope and the adoption decision.
