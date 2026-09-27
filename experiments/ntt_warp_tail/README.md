# Forward NTT register/shuffle experiment

This experiment replaces the last five shared-memory butterfly stages in the
pinned FIDESlib 64-bit Shoup forward NTT with register values exchanged by
warp shuffle. It does not change inverse NTT, butterfly arithmetic, moduli,
refresh parameters or the public plaintext preparation pipeline. Blocks below
32 threads retain the shared-memory path.

The patch is a research candidate, not applied by the normal build. See the
[B300 study](../../docs/research/2026-09-26-b300-ntt-warp-tail.md) and its
[raw results](../../results/b300/2026-09-26/ntt-warp-tail/README.md) for the
adoption decision, environment and limitations.

## CPU layout check

```bash
python3 experiments/ntt_warp_tail/check_lane_mapping.py
```

This checks the register exchange against an independent shared-array
butterfly calculation for 192 block/seed combinations. It checks lane mapping;
the GPU oracle below checks the actual modular kernels.

## Independent GPU oracle

Use FIDESlib `cd171f20f510eeca04c71d7b0034ef073829f761` and OpenFHE 1.4.2
`aa391988d354d4360f390f223a90e0d1b98839d7`. Build and save a probe linked
against the unmodified backend, then rebuild the same backend with
[`fideslib-ntt-warp-tail.patch`](fideslib-ntt-warp-tail.patch) and relink/save the
candidate. Keep the compiler, CUDA architecture and all other sources fixed.
The study retains exact compile/link commands and archive member hashes.

```bash
cmake -S experiments/ntt_warp_tail -B build/ntt-warp-tail \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH='/path/to/fideslib;/path/to/openfhe' \
  -DCMAKE_CUDA_ARCHITECTURES=103-real
cmake --build build/ntt-warp-tail

# Separate executables linked against the two separately built libraries:
mkdir -p runs/ntt-oracle
for ring in 1024 2048 4096 32768 65536; do
  /path/to/base/ntt_warp_tail "runs/ntt-oracle/base-$ring.json" \
    "runs/ntt-oracle/ring-$ring.bin" "$ring" --record
  /path/to/candidate/ntt_warp_tail "runs/ntt-oracle/candidate-$ring.json" \
    "runs/ntt-oracle/ring-$ring.bin" "$ring"
done
```

The five golden streams require approximately 2.32 GB. They contain synthetic
public modular coefficients, not model data or keys. Each ring covers six
forward-NTT fusion modes, both halves, 1/4/44 active moduli and four input
patterns (zero, modulus minus one, alternating extremes, mixed values).
Every output word is compared, and untouched secondary outputs are checked.
The 1,024/2,048 rings also exercise the unchanged sub-warp fallback.

Reports include diagnostic CUDA-event samples for the high-modulus ordinary
and rescale modes. A `--bench-only` option restricts execution to those four
mode/half cases. Use a separate golden file for that option. Neither these
microbenchmarks nor instrumented Nsight runs establish model speedup.

For integration, run `packed_rns_probe` with `--prefetch --strategy limb` for
both model patterns and both deployed ring sizes, followed by unchanged model
accuracy/token gates. The study uses a small ABBA performance gate before
allowing a full ABBA comparison. Its host-specific controllers and a separate
common CPU-affinity measurement patch are archived with the results.
