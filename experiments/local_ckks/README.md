# CPU CKKS recipes

The installed `fhemamba diagnose ckks` command runs both local feasibility
studies. OpenFHE Python bindings must be installed separately in the same
interpreter. `--help` works without OpenFHE, NumPy or PyTorch imports.

```bash
python -m fhemamba diagnose ckks \
  --recipe experiments/local_ckks/primitives.json \
  --output runs/ckks/primitives-01.json

python -m fhemamba diagnose ckks \
  --recipe experiments/local_ckks/error-growth.json \
  --output runs/ckks/growth-01.json
```

These replace `run_ckks_primitives_local.py` and `run_error_growth_local.py`.
Both scripts were deleted. Recipes contain geometry, input distributions,
polynomial degrees, intervals and comparison conditions; shared implementations
are in [`fhemamba.diagnostics`](../../src/fhemamba/diagnostics/ckks.py), with
[Chebyshev/Newton arithmetic](../../src/fhemamba/ckks_probes.py).
Copy and edit a recipe for a new set of conditions; another runner is unnecessary.
Curve kinds are `silu`, `softplus`, `exp`, `rsqrt` and `sq_rsqrt`. Inverse-square-root
curves also accept `iterations` and `damping`; unknown fields are rejected.

## Preserved studies

| Recipe | Conditions and measurements |
| --- | --- |
| [primitives.json](primitives.json) | SiLU degree 96; squared-softplus degree 64; squared-exp degree 24 with three squarings; inverse-square-root degrees 47/31 with four Newton iterations; 128-slot reductions, broadcast and replication; chained depth/refresh probe |
| [error-growth.json](error-growth.json) | Eight recurrent steps; refresh, ciphertext multiplication, polynomial decay, additive updates, Meta-BTS amplification 16/12 and compensated recurrence |

Both use ring 16,384, fully packed batch 8,192, depth 40, first modulus 60,
uniform ternary secrets and **`HEStd_NotSet` toy security**. They cannot qualify
the classical-128 GPU configuration. The primitive study starts at scale 40;
the growth study uses scale 59. Numerical defaults and RNG order are preserved.
Encrypted-vs-polynomial error and polynomial-vs-exact error remain separate.

The primitive chain folds `-2*x` into the exponential and `90*x+5` into the
inverse-square-root tail. It refreshes when the estimated tail no longer fits.
A failed or inaccurate scale-40 refresh triggers the same scale-59 replay.
Historical OpenFHE 1.5.1 observations are retained in each recipe's `notes`:
scalar rescaling is lazy, operand levels auto-align, and the scale-40/first-60
gap requires correction 20 but gave unusable CPU refresh precision. The local
bootstrap budget remains `[4,4]`. Growth probes compress to 18 towers before
refresh because a shallow bootstrap can otherwise preserve levels and fail to
model depth-exhausted refresh. These observations are backend-specific.

## Partial runs and records

```bash
python -m fhemamba diagnose ckks \
  --recipe experiments/local_ckks/error-growth.json --arms A,B,C,D,E \
  --output runs/ckks/baseline-01.json
python -m fhemamba diagnose ckks \
  --recipe experiments/local_ckks/error-growth.json --arms F16,F12,G \
  --resume runs/ckks/baseline-01.json --output runs/ckks/combined-01.json
```

Output paths must be fresh. Merging is explicit and requires an identical
recipe SHA-256, so a changed seed, geometry or polynomial cannot silently reuse
old arms. Only selected arms run again. Without A–E, the full attribution fields
are `null`/empty; refresh-strategy comparisons also require A. Recipes retain
these baseline roles for attribution; custom recurrence conditions can be run
independently. `total_seconds` and bootstrap timings describe the current
invocation, while merged arms keep their own timings.

Existing default result field names are retained. Added `recipe_sha256` binds
the settings; failed chain decryptions now use JSON `null` in place of invalid
`NaN`, and bootstrap `decode_failed` remains explicit. Old unbound result files
are not accepted for merging. Console output is a compact per-case/step trace.
Results remain diagnostic evidence, never a model qualification.

CPU tests compare operation graphs and numeric records captured from the old
scripts, using array arithmetic and synthetic refresh noise. They cover both
Meta-BTS conditions, compensation and decode-failure fallback. The tests do not
simulate CKKS security or establish new encrypted accuracy measurements.
