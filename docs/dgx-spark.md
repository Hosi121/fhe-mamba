# DGX Spark runbook

This build targets **aarch64 / NVIDIA GB10 / CUDA 13.0 / SM121**.
Spark's CPU and GPU share 128 GB nominal memory. Check host `MemAvailable`
and GPU utilization; Mamba-2 campaign preflight requires 65 GiB available and
an idle GPU. The B300 classical-128 baseline has separate
[build and workload records](../results/b300/2026-09-28/public-state-reuse/README.md).

## Build

Install CUDA 13.0, CMake >= 3.25.2, GCC/G++ 13, Make, Git and Python 3.
Use fresh dependency checkouts:

```bash
mkdir -p "$HOME/fhemamba" "$HOME/fhe-deps"
git clone https://github.com/Hosi121/fhe-mamba.git "$HOME/fhemamba/cipher"
git clone --no-checkout https://github.com/CAPS-UMU/FIDESlib.git "$HOME/fhe-deps/FIDESlib"
git -C "$HOME/fhe-deps/FIDESlib" checkout --detach cd171f20f510eeca04c71d7b0034ef073829f761
(
  cd "$HOME/fhe-deps/FIDESlib/deps"
  ./build.sh "$HOME/fhe-deps/openfhe-fides"
)
cd "$HOME/fhemamba/cipher"
scripts/build_dgx_spark.sh
```

The dependency script builds patched OpenFHE v1.4.2 and recreates
`deps/openfhe-src`. For existing matching dependencies, set
`FIDESLIB_SOURCE_DIR` and `OPENFHE_PREFIX` instead of repeating that step.

The project build creates an isolated FIDESlib snapshot and kernels under
`~/fhemamba/spark/`, runs C++ contracts, and writes build identities.
`FHEMAMBA_REMOTE_ROOT` overrides `~/fhemamba`; `BUILD_JOBS` defaults to 8.
The runner validates source, executable and library hashes. Rebuild after
native source changes.

| Executable in `spark/kernel/` | Workload |
| --- | --- |
| `stage1_mamba2_decode_fideslib` | Specialized Mamba-2 payloads |
| `packed_fideslib` | Packed Mamba-3 programs |

Optional GPU RNS, S2C-first refresh and dual-ring support are compiled by the
Spark build. Runtime selection and prerequisites are documented in the
[native guide](../native/README.md) and [Mamba-3 guide](mamba3.md).
Custom CMake builds must select the corresponding features and GPU architecture.

## Payload

Prepare payloads on the Python host:

- Mamba-2: follow [checkpoint download and stabilized export](reproducing.md#2-obtain-the-exact-public-checkpoint).
- Mamba-3: follow [trained export or mixer probe](mamba3.md).

Use calibration data separate from the evaluation prompt. Normalized Mamba-2
state requires `carried_bounds.source=calibration_text` and
`state_head_abs_max` in each layer. Regenerate incompatible old payloads;
do not invent missing bounds. The [quality gates](validation.md) also check
head pruning and approximation-domain coverage.

### Scheduled normalization integration

`export_m1_payload.py --normalization-bundle <bundle.json>` validates all
normalization certificates and checkpoint epsilon, regenerates references,
and records domains and the bundle hash. The frozen bundle and complete
command are in [payload export](reproducing.md#4-export-a-stabilized-payload).

For normalization-only experiments, use `dgx_spark_scheduled_norm_smoke.json`
and `dgx_spark_scheduled_norm_chain.json` under `experiments/manifests/`.
The [integration study](research/2026-09-21-normalization-integration.md)
retains its failed refresh control and passing short gates. Its matching
payload fails the 1,024-token plaintext quality screen.

### Joint gates and public activation domains

Add `--stabilized-gate-bundle config/mamba2-130m-gates-20260921.npz` alongside
the normalization bundle for the joint write/decay gates and public-envelope
SiLU fits. The exporter verifies certificates and regenerates references;
older native binaries reject this payload format.

Use `dgx_spark_stabilized_smoke.json` and `dgx_spark_stabilized_chain.json`
for component and full-chain fixed-input checks. Their
[study](research/2026-09-21-stabilized-native.md) records quality, failures and
the `not-set` security scope. Fixed-input success does not qualify generation.

## Run and compare

### Complete prompt-to-text generation

Use the [complete Mamba-2 command](reproducing.md#6-run-the-measured-prompt-to-text-workload)
with a matching build and fresh local/remote output directories.
For Mamba-3, use its [generation workflow](mamba3.md#trained-checkpoint-and-generation).

Mamba-2 generation consumes the complete prompt and makes
`prompt_tokens + generated_tokens - 1` server evaluations. It uses greedy
selection for a fixed length; EOS does not stop it early. `generation.json`
contains measured IDs and separately labelled reference text. The gate requires
all outputs to decrypt, polynomial error <= 0.05, matching IDs and zero
diagnostic intermediate decryptions.

The inline client decrypts final hidden outputs and re-encrypts selected
embeddings. State stays encrypted, but this runner does not establish a
process-separated server or full-chain classical-128 Mamba-2 security.

### Kernel comparisons

From the Spark checkout:

```bash
python3 experiments/execution/run_dgx_campaign.py \
  --manifest experiments/manifests/dgx_spark_replication_ab.json \
  --runner scripts/run_dgx_spark.sh \
  --env BINARY="$HOME/fhemamba/spark/kernel/stage1_mamba2_decode_fideslib" \
  --env INPUT_CHAIN="$HOME/fhemamba/payloads/mamba2-130m" \
  --output-json runs/spark-replication-ab.json \
  --resume
```

`--env KEY=VALUE` overrides manifest settings and records the resolved values.
For another workspace also set `FHEMAMBA_REMOTE_ROOT` and `RESULTS_DIR`.
Identity checks remain active on resume.

Keep payload, binary, precision, security and placement fixed within a
comparison. A one-token screen must be followed by the
[five-step state gate](../experiments/manifests/dgx_spark_autoregressive.json).
ABBA order reduces simple ordering bias; it does not establish significance.

State-coordinate experiments use `fhemamba calibrate state` or
`state-regularize` and the row-state manifests. Row scaling requires
`num_heads * head_dim` calibrated bounds per layer and remains opt-in.
`COEFFICIENT_AWARE_PS=1` selects the Chebyshev split planner with
`dgx_spark_chebyshev_ab.json`. See [validation](validation.md) for gates and
the [evidence registry](evidence.md) for matched results and failed controls.
