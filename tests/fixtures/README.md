# CPU CKKS compatibility fixture

`ckks-recipes.json` records the unmodified local primitive and error-growth
scripts at commit `2bd2fd53caf1d18ff29ed95929a4d6a1d92ec815`, before their deletion.
The producer used `TraceBackend` from `test_ckks_recipes.py`, the production
seed, input sizes, polynomial fits and recurrence settings. The primitive
context injected a decode failure on bootstrap; its sibling context succeeded.

The trace SHA-256 covers the ordered operation names and operand identities
(including encryption, diagnostic decryption, setup and key generation).
It excludes numeric scalar values to avoid pinning least-squares coefficient
rounding across BLAS implementations. The refactor audit also compared every
scalar operand exactly on the same environment. Numeric result fields use
tolerances in the test; timing and narrative notes are excluded, while per-step
diagnostic notes remain covered. Legacy `NaN` failures are represented as `null`.

This fixture is a compatibility oracle with synthetic perturbations, not an
OpenFHE measurement, level model or security claim. It is separate from the
published encrypted evidence under `results/`.
