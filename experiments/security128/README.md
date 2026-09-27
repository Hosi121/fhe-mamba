# Classical-128 parameter and refresh qualification

This experiment accompanies the packed executor's opt-in `--security
128-classic` profile. It preserves depth 44, 59-bit scale primes, a 60-bit first
prime, two-pass S2C-first refresh and the frozen model's numerical contract.
The initial profile uses one ring of dimension 131,072 with 65,536 slots and
four HYBRID digits. The experimental dual-ring bridge is rejected in this mode.

The shared [audit](../../native/fideslib_stage0/src/fideslib_security.hpp)
checks the actual maximum Q and special-prime product P against the
[published 2024 guideline, Table 5.2](https://cic.iacr.org/p/1/4/26/pdf).
It also requires the library's classical-128 setting, uniform ternary secret
and Gaussian error sigma at least 3.19. It runs before keys are generated.
This is a classical RLWE parameter qualification; it does not certify an entire
deployment or arbitrary-prompt model accuracy.

Build with the same pinned FIDESlib/OpenFHE installation as the packed executor:

```bash
cmake -S experiments/security128 -B build/security128 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/matching/g++ \
  -DCMAKE_PREFIX_PATH="/path/to/fideslib;/path/to/openfhe"
cmake --build build/security128 --target security_parameters security_refresh -j4
build/security128/security_parameters parameters.json
build/security128/security_refresh refresh.json 4
```

The linked backend still requires its CUDA driver library for the parameter
probe, even though that probe does not generate keys or evaluate ciphertexts.
The refresh probe uses a GPU and the optional S2C-first backend patch. For an
optimized installation, reproduce the dependency patches documented in the
[B300 rotation study](../../docs/research/2026-09-27-b300-rotation-stream-chain.md).

`security_parameters` checks accepted and rejected ring/digit combinations and
rejects unsupported secret/security assumptions. `security_refresh` checks
15 two-pass refresh cases, their following products and 60 rotations, including
all padding slots, against a client-side reference. Its standalone tolerance
is `1e-6`; model inference still uses both `0.001` output gates and fixed token
IDs. Probe decryption is validation, not an evaluator operation.

For a frozen packed model, append `--security 128-classic --security-digits 4`
to the existing packed-runner command, removing `--gpu-dual-ring`. The runner
independently rejects missing/mismatched security evidence. The native result
records the original payload slot count, effective ring/slot dimensions and
actual Q/P audit; `RESULT.parameters.json` is written before key generation.

Five- and six-digit configurations are performance candidates, not weaker
security modes. Every context must pass the same audit. Compare evaluation
separately from setup/key generation, and keep source, dependency, input and
device identities with each result.
