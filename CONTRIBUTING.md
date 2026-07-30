# Contributing

This repository is an implementation-first FHE research prototype. A claim is
complete only when code, tests, configuration, and evidence agree.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pre-commit install
```

Use the fast gate while iterating:

```bash
scripts/run_fast_checks.sh
```

Before a release, tag, or claim-changing merge, run:

```bash
scripts/run_checks.sh
```

GPU probes are separate because they require OpenFHE/FIDESlib, dedicated
hardware, and substantial memory. A passing local suite does not validate an
encrypted B300 claim.

## Active and compatibility code

- `fhemamba/` and `native/fideslib_stage0/` are the active Mamba-2 path.
- `src/fhe_native_mamba3/` remains a shipped compatibility package for
  historical scripts and the current CLI.
- New Mamba-2 formula, lowering, packing, and runtime work belongs in the
  active path. Do not create a third implementation.
- Removing the compatibility package requires a planned breaking release and
  migration of its remaining tools.

## Definition of done

A change is done when:

- the write scope and claim boundary are explicit;
- code and tests pass the appropriate local gate;
- hardware-backed changes have a measured artifact, or clearly state why one
  is pending;
- direct result JSON records repository commit and binary identity;
- README, evidence registry, roadmap, and backlog are updated when behavior or
  claims change;
- the next measured bottleneck is named.

Documentation is not a substitute for missing raw evidence. When a measurement
is known only from notes, label it as documented and create a recovery/rerun
PBI instead of reconstructing a fake backend artifact.

## Benchmark artifacts

Use `fhemamba/results/` for small curated current artifacts. Large payloads,
logs, transient campaigns, and historical outputs remain ignored unless a
specific review requires them.

A direct backend artifact should include:

- artifact/package version and repository commit;
- native binary SHA-256 where applicable;
- backend, hardware, CKKS parameters, and security mode;
- exact input/payload identity;
- pass/fail status and numerical tolerance;
- per-token error and decrypt status;
- setup/evaluation/decrypt timing;
- rotations, ct-pt/ct-ct products, and bootstrap counts;
- peak RSS and key/cache configuration;
- explicit measurement scope and non-claims.

Validate curated artifacts with:

```bash
python scripts/validate_artifacts.py --require-commit path/to/result.json
```

## Versioning and tags

Use SemVer for package versions.

- Patch versions cover fixes, tests, process updates, and narrow optimizations
  inside a capability boundary.
- Minor versions mark a new runnable capability such as a longer encrypted
  horizon, process-separated full-kernel execution, or 128-bit full-chain
  execution.
- `1.0.0` is reserved for reproducible interactive generation at 128-bit
  parameters with an explicit protocol-security statement.

Do not create a release tag until:

1. the package version is consistent;
2. the full local gate passes;
3. claim-bearing raw artifacts are tracked and validator-clean;
4. the evidence registry links every headline result.

Package version `0.4.5` currently has no tag because the corresponding
three-token B300 success artifact is still awaiting recovery or an exact rerun.

## Review priorities

For low-level FHE changes, review these first:

- Is each ciphertext slot layout explicit and type-safe?
- Does the rotation inventory match the executing implementation?
- Are level drops and bootstrap placement visible in telemetry?
- Are reference and encrypted paths evaluating the same polynomial circuit?
- Are exact-model approximation and CKKS execution errors separated?
- Can a debug decrypt influence subsequent encrypted execution?
- Is a partial probe described as partial?

For documentation and artifacts:

- Does every number have a source?
- Is the source raw execution, a derived report, or prose-only measurement?
- Are security and process-separation boundaries stated next to the result?
- Does the backlog contain the next executable gate?
