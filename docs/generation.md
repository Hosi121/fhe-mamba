# Generate from your own input

The public API supports local **Mamba-3 SISO** checkpoints. Inputs are token IDs;
the CLI can also tokenize text. All backends use greedy, fixed-length generation:
`max_new_tokens` excludes the prompt, and EOS does **not** stop generation early.
There is no automatic download, sampling or plaintext fallback from CKKS.

## CPU generation

Follow the [Python setup](reproducing.md#1-install-the-python-environment) and
[checkpoint download](mamba3.md#trained-checkpoint-and-generation). Text input
requires the `experiments` extra and a local tokenizer, found automatically in
`MODEL/tokenizer` or supplied with `--tokenizer`.

```bash
fhemamba generate --model checkpoints/mamba3-siso-187m \
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

model = fhemamba.load_model("checkpoints/mamba3-siso-187m")
result = model.generate([791, 6864], max_new_tokens=4, backend="exact")
print(result.generated_ids)
assert result.passed and not result.encrypted
```

`input_ids` accepts a nonempty list, tuple, or one-dimensional integer NumPy/Torch
array. IDs must fit the checkpoint vocabulary. Each call starts fresh recurrent
state. The model loads on CPU in float64, using the existing reference arithmetic.

## Prepare once for one input

Preparation currently requires a source checkout, the pinned 187M checkpoint
and its local tokenizer. It calibrates on the six existing independent prompts,
checks the requested input, and exports the complete backbone and references.
Use a fresh output directory:

```bash
fhemamba prepare --model checkpoints/mamba3-siso-187m \
  --prompt 'The capital' --max-new-tokens 16 --output runs/my-request
```

The `classical-128` profile uses the documented tiled state, phasor rotary
representation and 65-token calibration recipe. Domain failures, changed greedy
IDs or hidden error above `0.001` stop preparation. Its 64-generated-token and
128-evaluation limits are resource limits, not accuracy guarantees. The current
[qualification and known failures](status.md) still apply.

A prepared request binds the exact input IDs, generation length, checkpoint,
polynomials and file hashes. Changing the input requires a new request. These
payloads contain plaintext input/reference data and public model weights; they
are local research artifacts, not private server deployment packages. Follow
the [client-head transfer rules](experiments.md#shared-local-client-heads) when
moving them. No private or evaluation keys are saved in a prepared request.

## Compare polynomial and encrypted execution

Recompute the saved polynomials on CPU with the same request:

```bash
fhemamba generate --model checkpoints/mamba3-siso-187m \
  --prepared runs/my-request --backend polynomial --json
```

| Backend | Computation | Validation |
| --- | --- | --- |
| `exact` | Original nonlinearities, plaintext | Finite hidden states, requested length |
| `polynomial` | Frozen polynomial coefficients, plaintext | Domain checks, reference IDs, hidden error ≤ `0.001` |
| `ckks` | Encrypted polynomial program | Existing native precision, protocol and classical-128 gates |

Polynomial execution evaluates the model again; it does not replay saved IDs or
simulate CKKS noise. Preparation success does not qualify encrypted execution.

On a host with the [native GPU backend](mamba3.md#classical-128-16-token-configuration),
run the saved request with an explicit binary, fresh output and time limit:

```bash
fhemamba generate --prepared runs/my-request --backend ckks \
  --binary /path/to/packed_fideslib --output runs/my-run --timeout 2400 --json
```

CKKS uses the recorded classical-128 execution settings, including two S2C-first
refresh passes, merged correction and frontier limit 256. Each invocation starts
a native process with fresh keys. GPU selection uses the native environment;
there is no automatic remote connection. The client selects tokens from decrypted
final hidden vectors in that process; client/server process separation is not
implemented.

The run directory retains `native.log`, `native.json` when produced, `run.json`
and the common `generation.json`. Failed results retain actual returned IDs,
which may be partial or empty. Inspect `passed` before using them. Never replace
a failed generation with its reference completion.

The corresponding Python operations are:

```python
prepared = model.prepare(
    [791, 6864], max_new_tokens=16, output="runs/my-request",
)
polynomial = model.generate(
    [791, 6864], max_new_tokens=16, backend="polynomial", prepared=prepared,
)
prepared = fhemamba.load_prepared("runs/my-request")
encrypted = prepared.generate(
    binary="/path/to/packed_fideslib", output="runs/my-run", timeout=2400,
)
```

`load_prepared` also accepts complete legacy `workload export-mamba3` payloads.
Loading and CKKS execution work outside the checkout without Torch/Transformers
imports; text decoding optionally needs the tokenizer. Specialized export,
diagnosis and campaign controls remain under the existing research commands.
