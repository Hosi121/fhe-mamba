# Compact plaintext RNS evidence

One candidate reduces host preparation while retaining the qualified
S2C-first refresh circuit, two-pass correction, model, levels and scheduler.
The [study](../../../../docs/research/2026-09-25-gpu-rns.md) describes the
signed-lift condition and the measured scope.

Adopted as an opt-in path: full native evaluation **750.4135 → 618.9536 s
(17.52%)**; ordinary evaluation **501.0506 → 381.0208 s (23.96%)**.
Both output gates, all generated IDs and operation counts pass unchanged.

- `contract.json`: frozen parameters, workload and one-candidate stop rule.
- `native-sources.tar.gz`, `native-source-manifest.json`: measured native source overlay.
- `baseline-backend-manifest.json`, `backend-source-verification.json`: unchanged S2C-first backend sources.
- `openfhe-source.json`: pinned FFT/rounding implementation and source hashes.
- `controller.py`, `build.json`, `status.json`: commands, executable identities and job state.
- `probe-*`: 720 exact-RNS cases, including 105 compact and 615 fallback cases,
  27 encrypted arithmetic checks, and alternating preparation microbenchmarks.
- `prefix-*`: independent ABBA processes using the released baseline and candidate.
- `full-*`: final twelve-layer comparison, one fresh process per mode.
- `existing-plaintext-*`, `post_validation.py`: both existing configuration regressions, run after full timing.
- `legacy-*`: old backend/CUDA-feature-disabled build and early flag rejection.
- `validation.json`: local checks and qualification; failed build logs remain separate.
- `summary.json`, `summarize.py`: checked source archive, payload parity, gates and cost categories.
- `selection.json`: final adoption decision and limits of the measured scope.
- `source-verification.json`, `dependency-verification.json`: delivered source identity and canonical dependency preservation.
- `build_backend.sh`, `cleanup_remote.py`, `cleanup.json`: study build and removal of owned temporary build trees.

The source overlay requires the pinned backend described in the preceding
[S2C-first collection](../s2c-first/README.md); CUDA and OpenFHE are not vendored.
Payload manifests identify the existing checkpoint exports without redistributing weights.

Recheck the collected data without a GPU:

```bash
python3 results/dgx/2026-09-25/gpu-rns/summarize.py
```
