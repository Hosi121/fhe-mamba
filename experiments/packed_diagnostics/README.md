# Frozen packed-circuit diagnostics

This tool rebuilds a Mamba-3 tiled/phasor fixture using its frozen polynomial
coefficients and checkpoint. It exports references for nonlinear inputs and
outputs, block residuals and carried state. It never refits the circuit.

```bash
python -m fhemamba diagnose export \
  --checkpoint /path/to/checkpoint --manifest /path/to/payload/manifest.json \
  --fixture /path/to/payload/fixture.npz --program /path/to/payload/program.txt \
  --output runs/references --steps 22 23 24
python -m fhemamba diagnose verify \
  --program /path/to/payload/program.txt --metadata runs/references/references.json
```

Step numbers are zero-based server evaluations. With a two-token prompt,
selection 24 uses step 24. The verifier requires byte-identical node-prefix
content and the original full-program digest. `--all-nodes` also exports every
intermediate node in the selected steps for a finer subsequent investigation.
Repeat `--node-range FIRST STOP` to restrict those observations to half-open
intervals of original node IDs without altering the graph or its prefix hash.
`--program` also checks each reconstructed operator, width, parent and public
constant exactly. It retains the original public refresh bounds verbatim when
CPU reduction order introduces tiny bound-rounding differences (at most
`1e-12` relative/absolute); larger differences are rejected. The count and
maximum relative drift are recorded. The native run always reads the original
program, and the final prefix verifier still requires byte-identical nodes.

Run the ordinary native command with the additional options:

```text
--diagnostic-reference runs/references/references.bin
--diagnostic-output runs/observations.jsonl
```

This explicitly enables client-side intermediate decryption of ciphertext
copies, records those decryptions, and marks the result `diagnostic_only`.
It is not a zero-intermediate-decryption or timing qualification. The normal
generation reporter rejects it. Original encrypted feedback and numerical
failure gates remain active. Each observation contains the original node ID,
event, level, scaling degree, error and decrypted vector. No diagnostic value
is substituted into the encrypted evaluation. Treat these files as sensitive
if using nonpublic inputs; the research fixtures use public test prompts.

Analyze a completed observation file with the same frozen manifest:

```bash
python -m fhemamba diagnose analyze \
  --metadata runs/references/references.json \
  --manifest /path/to/payload/manifest.json \
  --program /path/to/payload/program.txt \
  --observations runs/observations.jsonl --output runs/analysis.json
```

The summary omits vectors and distinguishes input drift from the local
polynomial evaluation error by evaluating the frozen polynomial on the
observed input. It also reports refresh changes and input-domain violations.
With `--program`, it evaluates elementary operations on the latest observed
parents and verifies the original program digest. Missing parents are left
unclassified rather than replaced with CPU reference values.
`step` identifies a node's reference creation step; a carried node can be
refreshed during a later step. Records remain in actual observation order.
Intermediate quantities have different scales: the final-hidden `0.001`
qualification threshold is not a universal threshold for internal tensors.

## Isolate one operation

For a local arithmetic error or refresh change, extract its observation order:

```bash
python -m fhemamba diagnose extract \
  --metadata runs/references/references.json \
  --program /path/to/payload/program.txt \
  --observations runs/observations.jsonl --order 123 --output runs/operation-case
cmake -S experiments/packed_diagnostics -B build/diagnostics \
  -DCMAKE_PREFIX_PATH="/path/to/fideslib;/path/to/openfhe" \
  -DCMAKE_CUDA_ARCHITECTURES=103-real
cmake --build build/diagnostics --target packed_operation_replay -j4
build/diagnostics/packed_operation_replay runs/operation-case/case.txt runs/replay.json
```

The probe uses the executor's arithmetic, alignment and routing, the classical
128-bit context, and fresh encryption of the observed parent vectors at their
recorded levels and scaling degrees. It supports add/multiply, public operands,
gather/scatter/repeat and an individual two-pass refresh. The JSON reports errors
without treating an internal tensor threshold as a generation qualification.
It does **not** retain original RNS noise, keys, or companions in a batch refresh.
A passing isolated operation therefore does not refute the full-run failure.

`extract_case.py --refresh-group` requires both consecutive before/after
observations of a two-input group occupying all 65,536 physical slots. It
preserves both public bounds and companions in the fresh-input probe.
For a harness smoke check, pass `--in-context-controls` to
`packed_operation_replay`; the controls clone that probe's freshly encrypted
inputs. This does not recreate the long request's encrypted history.

When fresh encryption does not reproduce the fault, add
`--diagnostic-refresh-pair FIRST SECOND` to the original diagnostic native run.
Both nodes must have references and fill the physical slots. Immediately after
that group executes, the client verifies exact RNS copies of the original
inputs and tries the identical group, doubled public bounds, separate groups,
and the unmerged correction path. It writes `OBSERVATIONS.refresh-replay.json`.
These are isolated diagnostic controls; none replaces a model value. They
retain the original context and encrypted history without serializing keys.
The raw clone check happens after the original group; it does not check whether
that original call mutated its retained inputs. A passing replay alone cannot
distinguish such mutation from cache, allocation or workspace-state differences.
All extra client decryptions are counted, and the run remains ineligible for
generation or performance qualification. A useful control is a repair proposal,
not a qualified full-model change.

If OpenFHE rejects an intermediate decode because its approximation error is
too high, the diagnostic observer records `decryption_error` with null values
and errors, then leaves the encrypted execution unchanged. It does not disable
the decoder's check. Other exceptions and ordinary generation failures still
terminate the run. The refresh controls report their comparison basis explicitly:
`observed_input` when that input decoded, or `cpu_reference` when it did not.
A CPU-reference comparison cannot measure the refresh's added error separately
from input drift. Neither basis is used to replace encrypted state.

## Propagate a measured error on the CPU

To test whether selected refresh outputs explain a later error, use the frozen
reference binary and original program:

```bash
python -m fhemamba diagnose propagate \
  --metadata runs/references/references.json \
  --references runs/references/references.bin \
  --program /path/to/payload/program.txt \
  --observations runs/observations.jsonl --inject-order 625 626 \
  --target-node 75969 --output runs/counterfactual.json
```

This NumPy diagnostic replaces only those two CPU values with the observed
after-refresh values, then evaluates their downstream arithmetic with frozen
polynomials and weights. Other inputs remain CPU references; later encrypted
arithmetic and refresh errors are excluded. A control with no injected error
must reproduce all affected frozen references within `1e-7`. Program and
reference digests, shapes and finite values are checked. The output compares
the simulated target with its CPU reference and, when available, the observed
target. It is an attribution calculation, not an encrypted repair or speed test.
