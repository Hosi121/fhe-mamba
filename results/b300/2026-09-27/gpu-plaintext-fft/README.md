# GPU plaintext FFT evidence

The [study](../../../../docs/research/2026-09-27-b300-gpu-plaintext-fft.md)
qualifies the shared GPU encoder and reports the matched full comparison:
211.712 → 165.373 seconds with classical-128 parameters.

- `full-fft-{control,candidate}-r3/` contains native JSON, actual parameters,
  job/input identities and completion events. Only `--gpu-plaintext-fft` changes.
- `full-comparison.json` checks the security, numerical and workload contract,
  additional operation-count invariants and matching job inputs/commands.
- `prefix-fft-*-r2/` and `prefix-comparison.json` contain the preceding screen.
- `probe-fft-r2/` checks transforms, rounding and small-scale rejection.
  `probe-rns-{real,complex}-r2/` checks RNS/metadata and encrypted arithmetic.
  `probe-fft-r1/` preserves the earlier tiled-transform screen.
- `polynomial-{control,candidate}.json` contains per-function timing attribution;
  `polynomial-inventory.json` and `calibration-manifest.json` preserve the
  static source of the function labels and intervals.

Revision r3 adds per-polynomial profiling to r2; the GPU arithmetic sources
are unchanged. Both full arms use the same r3 executable. Build commands,
source snapshots and the common measurement-affinity patch are in the
provenance bundle. Disposable controllers are excluded.

```bash
python -m fhemamba.benchmarks verify results/b300/2026-09-27/gpu-plaintext-fft
python -m fhemamba.benchmarks compare \
  --baseline results/b300/2026-09-27/gpu-plaintext-fft/full-fft-control-r3/native.json \
  --candidate results/b300/2026-09-27/gpu-plaintext-fft/full-fft-candidate-r3/native.json \
  --contract config/packed-classical128-comparison.json
python -m fhemamba.benchmarks extract \
  results/b300/2026-09-27/gpu-plaintext-fft runs/gpu-plaintext-fft-inspect
```

`publication.json` distinguishes original from published hashes. Logs and
measured source snapshots reside in `provenance.tar.gz`; no model checkpoint
or large frozen program is duplicated here. Input hashes bind those artifacts.
