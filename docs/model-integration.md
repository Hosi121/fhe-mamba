# Adding a model

Keep `load_model`, `prepare`, `generate` and their result format stable as models
are added. A model integration owns its numerical implementation and preparation
requirements. The shared API owns input validation, checkpoint/request identity,
output paths and result persistence.

## Boundaries

| Component | Responsibility |
| --- | --- |
| `ModelAdapter` | Checkpoint hashes/loading, vocabulary, CPU reference execution and capabilities |
| Optional `FHEImplementation` | Typed preparation options, frozen payload export/validation and native result interpretation |
| `PreparationProfile` | Named security settings and preparation requirements; the implementation retains the acceptance gates |
| `ModelRegistration` | Configuration predicate, lazy implementation import and owned prepared schemas |

The protocols live in [`models/contracts.py`](../src/fhemamba/models/contracts.py).
Each model keeps its own state, arithmetic, layouts and calibration path.
Mamba-1 and Mamba-3 share the packed native runner and classical-128 gates;
Mamba-2 retains its separate frozen-chain implementation. A CPU-only adapter
sets `fhe = None`; the API then rejects polynomial/CKKS before preparation.
Shared helpers remain useful without requiring identical model internals.

Mamba-1 lowers its convolution FIFO and channel-specific selective recurrence
through `TensorOps`/`PackedProgram`, with the original FP32 reference as its
acceptance oracle. Channel tiling preserves every state coordinate. Its
decay uses `exp(x / 2**s)` followed by `s` squarings, with per-layer counts frozen
from independent calibration. This handles the trained model's large negative
exponents without clipping the recurrence or fitting an overflowing interval. Its
`mamba1-experimental` profile is explicit because GPU qualification is pending;
the classical-128 parameters and `0.001` precision gates are unchanged.

## Registration

Implement an adapter under `src/fhemamba/models/`, then add a registration to
[`models/registry.py`](../src/fhemamba/models/registry.py). Keep the adapter module
lightweight: import Torch, Transformers and numerical exporters inside the methods
that need them. Discovery, inspection and prepared native execution must work
without importing model libraries.

Python callers can also explicitly register an installed integration before using
the API or calling the CLI's `main` function:

```python
from fhemamba.models import ModelRegistration, register_model

register_model(ModelRegistration(
    architecture="my_mamba",
    implementation="my_package.integration:ADAPTER",
    matches=lambda config: config.get("model_type") == "my_mamba",
    prepared_schemas=("my-mamba-request-v1",),
))
```

The import path refers to a stateless adapter object. Configuration predicates
must distinguish supported checkpoint formats; ambiguous matches and duplicate
architecture/schema registrations fail explicitly. Checkpoint files never
provide executable import paths. There is no automatic external plugin discovery.

## Preparation and validation

Each FHE implementation owns its options dataclass and parser. `prepare(options=...)`
accepts that type or a mapping; `--prepare-options` passes a JSON object through
the same parser. Reject unknown fields and invalid values before writing output.
Existing `base_chain` arguments are compatibility aliases for Mamba-2's options.

A prepared `manifest.json` includes `schema`, `architecture`, `profile`,
`checkpoint.files_sha256`, `prompt_ids`, `generated_tokens` and
`tokenizer_files_sha256`. Model-specific fields and file formats stay with the
integration. Its validator must check all payload bytes used for execution and
the reference's agreement with the request. The core writes `request.json` with
the selected profile and manifest digest. Loading a request uses its registered
schema and never loads model weights. Legacy Mamba-3 exports retain their default
profile without changing their bytes.

New packed integrations set `client_protocol: packed-greedy-v1` and provide
`client_head.f32` with tied embedding/head weights. This selects actual client
feedback in the shared runner without architecture switches. Other client layouts
need their own protocol implementation; cached reference IDs are never feedback.

Native execution returns actual tokens and failed/partial results in
`GenerationResult`; the core persists `generation.json`. Use the shared subprocess
runner for deadlines and cleanup, preserve original native reports, and retain
the profile's safety and numerical gates. An implementation's availability is
separate from qualification of a particular checkpoint, request and GPU build.

## Validation when adding an integration

- Compare CPU computation with the upstream model, including repeated calls with
  fresh state and more than one geometry where applicable.
- Exercise raw token IDs and the shared CLI. Test unsupported modes and invalid
  options before output creation or native execution.
- For FHE, verify prepared identity, coefficient/reference preservation and
  failed native results. CPU tests do not establish GPU qualification.
- Run `CHECK_JOBS=2 scripts/run_checks.sh` and update the generation/support guide.

The registry tests add a separate fixture integration through registration alone,
then exercise the public API, CLI, typed options and prepared-payload validation.
