# Encoded mask cache: bounded component experiment

Same-binary ABBA on a 127-node trained prefix: **19.5937 → 19.3164 s (−1.415%)**.
All runs pass unchanged 0.001 exact/polynomial gates. This is not a full-session
generation measurement. See the [study](../../../../docs/research/2026-09-24-mamba3-plaintext-cache.md).

- `a1/`, `b1/`, `b2/`, `a2/`: raw runner/native JSON and logs. Both modes use
  in-place scratch, host timing and CPU affinity 15–19; only B enables caching.
- `comparison.json`, `compare.py`: derived metrics with hash, operation-count
  and correctness checks. Run the script from any directory to reproduce it.
- `synthetic/`: four-step mixer, 20 output/state checks; a correctness probe,
  not a matched speed comparison.
- `compiled-sources.tar.gz`, `compiled-sources.json`: exact native sources,
  CPU contracts, runner and version used for the frozen executable.
- `run_abba.py`, `run_synthetic.sh`, `affinity.json`, `abba.log`, `synthetic.log`:
  measured invocations and controller outputs. Host paths/CPU IDs are specific
  to this DGX Spark; use a separately authorized budget for any reproduction.
- `build.log`, `target-build-contract.log`, `target-provenance.txt`: target
  build, ARM contract, compiler command, binary/runner and dependency hashes.
- `checks.log`, `native-tests.log`: 280 passing Python tests and 16 native
  contracts. `checks-initial.log` preserves a failed campaign-resume self-test:
  writing its live log into an untracked directory changed the source hash.
  Repeating with output in ignored `runs/` passed without changing that test.
- `budget.json`: unchanged 7200-second campaign cap, 7174.24 s used over all
  29 attempts; these five use 247.47 s. Setup and all attempts are charged.
- `provenance.json`: source state, identity, references and all artifact hashes.

The prefix payload and exact derivation are already retained in
[`../mamba3-microkernels/prefix-payload/`](../mamba3-microkernels/prefix-payload/)
and [`../mamba3-microkernels/prefix-derivation.json`](../mamba3-microkernels/prefix-derivation.json).
The synthetic fixture/manifest/compressed program are retained in
[`../mamba3-siso/tokens4/`](../mamba3-siso/tokens4/). Their hashes match these runs.
The measured candidate is optional and does not alter the previously frozen
full-generation executables or their evidence.
