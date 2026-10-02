# Generate from your own input

The public API detects local **Mamba-2** and **Mamba-3 SISO** checkpoints from
`config.json`. Inputs are token IDs; the CLI can also tokenize text.
All backends use greedy, fixed-length generation:
`max_new_tokens` excludes the prompt, and EOS does **not** stop generation early.
There is no automatic download, sampling or plaintext fallback from CKKS.

Inspect a checkpoint before loading weights:

```bash
fhemamba inspect-model --model checkpoints/mamba2-130m-hf --json
```

This reports implemented backends, preparation requirements and security profiles
from configuration alone. It does not validate weights or qualify GPU execution.
The Python equivalent is `fhemamba.inspect_model(path)`.

## CPU generation

Follow the [Python setup](reproducing.md#1-install-the-python-environment) and
download a [Mamba-2](reproducing.md#2-obtain-the-exact-public-checkpoint) or
[Mamba-3](mamba3.md#trained-checkpoint-and-generation) checkpoint. Mamba-2 and
text input require the `experiments` extra. The CLI finds a tokenizer in
`MODEL/tokenizer` or `MODEL`; override it with `--tokenizer`.

```bash
fhemamba generate --model checkpoints/mamba2-130m-hf \
  --prompt 'The capital' --max-new-tokens 4 --backend exact
```

Replace `--prompt` with `--prompt-file prompt.txt` or `--input-ids tokens.json`.
The latter reads a JSON array such as `[791, 6864]` without decoding or
retokenizing it. Both file options accept `-` for stdin. Text is preserved,
including newlines; no BOS token or chat template is added.

The command prints generated text when a tokenizer is available, otherwise
generated IDs. `--json` returns IDs, backend, validation status and a report.
Exit codes are `0` for success, `1` for failed execution/validation and `2` for
invalid input or setup. Progress and errors go to stderr.

```python
import fhemamba

model = fhemamba.load_model("checkpoints/mamba2-130m-hf")
result = model.generate([510, 5347], max_new_tokens=4, backend="exact")
print(result.generated_ids)
assert result.passed and not result.encrypted
```

`input_ids` accepts a nonempty list, tuple, or one-dimensional integer NumPy/Torch
array. IDs must fit the checkpoint vocabulary. Each call starts fresh recurrent
state. The CPU references retain their existing arithmetic: float32 for Mamba-2,
float64 for Mamba-3. Both use the same API; token IDs belong to each model's tokenizer.

## Prepare once for one input

Preparation requires a source checkout and a fresh output directory. Each
architecture retains its own coefficients, security settings and acceptance gates.

For Mamba-3, use the pinned 187M checkpoint and local tokenizer. Preparation
calibrates on six independent prompts and checks the requested input:

```bash
fhemamba prepare --model checkpoints/mamba3-siso-187m \
  --prompt 'The capital' --max-new-tokens 16 --output runs/my-request
```

The `classical-128` profile uses the documented tiled state, phasor rotary
representation and 65-token calibration recipe. Domain failures, changed greedy
IDs or hidden error above `0.001` stop preparation. Its 64-generated-token and
128-evaluation limits are resource limits, not accuracy guarantees. The current
[qualification and known failures](status.md) still apply.

For Mamba-2, first [export a stabilized base chain](reproducing.md#4-export-a-stabilized-payload)
from the same checkpoint. Preparation copies its frozen coefficients and bounds,
checks operator domains and adds references for the requested token IDs:

```bash
fhemamba prepare --model checkpoints/mamba2-130m-hf \
  --base-chain runs/stabilized-payload --profile mamba2-experimental \
  --prompt 'The capital' --max-new-tokens 4 --output runs/m2-request
```

`mamba2-experimental` must be selected explicitly: its existing native recipe
uses `security=not-set`, not classical-128. It checks CKKS against the frozen
polynomial model with error at most `0.05`; exact-model differences are reported
separately. Preparation does not refit the base chain for a new prompt.

A prepared request binds the exact input IDs, generation length, checkpoint,
polynomials and file hashes. Changing the input requires a new request. These
payloads contain plaintext input/reference data and public model weights; they
are local research artifacts, not private server deployment packages. Follow
the [client-head transfer rules](experiments.md#shared-local-client-heads) when
moving them. No private or evaluation keys are saved in a prepared request.

## Compare polynomial and encrypted execution

Recompute the saved polynomials on CPU with the same request:

```bash
fhemamba generate --model checkpoints/mamba2-130m-hf \
  --prepared runs/m2-request --backend polynomial --json
```

| Backend | Computation | Validation |
| --- | --- | --- |
| `exact` | Original nonlinearities, plaintext | Finite hidden states, requested length |
| `polynomial` | Frozen polynomial coefficients, plaintext | Mamba-2: domains, polynomial IDs/error ≤ `0.05`; Mamba-3: exact and polynomial IDs, exact hidden error ≤ `0.001` |
| `ckks` | Encrypted polynomial program | Model-specific precision, protocol and security gates |

Polynomial execution evaluates the model again; it does not replay saved IDs or
simulate CKKS noise. Preparation success does not qualify encrypted execution.

On a host with the native GPU backend, run the saved request with an explicit
binary, fresh output and time limit. Mamba-2 uses
[`stage1_mamba2_decode_fideslib`](dgx-spark.md#build):

```bash
fhemamba generate --prepared runs/m2-request --backend ckks \
  --binary /path/to/stage1_mamba2_decode_fideslib \
  --output runs/m2-run --timeout 2400 --json
```

Mamba-3 uses [`packed_fideslib`](mamba3.md#classical-128-16-token-configuration)
with its recorded classical-128 settings: two S2C-first refresh passes, merged
correction and frontier limit 256. Each invocation starts a native process with
fresh keys. GPU selection uses the native environment;
there is no automatic remote connection. The client selects tokens from decrypted
final hidden vectors in that process; client/server process separation is not
implemented.

The run directory retains the process log, `native.json` when produced, `run.json`
and the common `generation.json`. Failed results retain actual returned IDs,
which may be partial or empty. Inspect `passed` before using them. Never replace
a failed generation with its reference completion.

The corresponding Python operations are:

```python
from fhemamba.models.mamba2 import Mamba2Preparation

prepared = model.prepare(
    [510, 5347], max_new_tokens=4, output="runs/m2-request",
    options=Mamba2Preparation(base_chain="runs/stabilized-payload"),
    profile="mamba2-experimental",
)
polynomial = model.generate(
    [510, 5347], max_new_tokens=4, backend="polynomial", prepared=prepared,
)
prepared = fhemamba.load_prepared("runs/m2-request")
encrypted = prepared.generate(
    binary="/path/to/stage1_mamba2_decode_fideslib", output="runs/m2-run", timeout=2400,
)
```

Preparation options belong to the model integration. The CLI accepts the same
fields through `--prepare-options options.json`, for example
`{"base_chain": "runs/stabilized-payload"}` for Mamba-2. Paths resolve from the
working directory. Unknown fields are rejected. The existing `base_chain=` and
`--base-chain` remain supported; use either these or `options`, not both.

`load_prepared` also accepts complete legacy `workload export-mamba3` payloads.
Loading and CKKS execution work outside the checkout without Torch/Transformers
imports; text decoding optionally needs the tokenizer. Specialized export,
diagnosis and campaign controls remain under the existing research commands.
