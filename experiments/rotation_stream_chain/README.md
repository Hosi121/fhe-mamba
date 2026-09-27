# Chain the phases of a GPU rotation

This experiment removes three intermediate host barriers from the pinned
FIDESlib single-GPU rotation path. The existing copy, ModUp graph, fused
key-switch/automorphism and ModDown operations keep their order through
stream dependencies. The final `cudaDeviceSynchronize` remains. Extended
outputs (`moddown=false`), multiple GPUs and ciphertext recycling keep the
previous behavior.

The [patch](fideslib-rotation-stream-chain.patch) targets FIDESlib
`cd171f20f510eeca04c71d7b0034ef073829f761` with this repository's existing
dependency patches. It is independent of the optional
[even refresh seed](../refresh_even_seed/README.md); the B300 comparison keeps
that seed enabled in both arms. The normal dependency build does not apply
this experiment automatically.

The [probe](probe.cpp) compares the compiled rotation with the original
barrier-separated sequence, using the same encrypted inputs and keys. It
compares every RNS coefficient and scale/level/key metadata, checks live
inputs and a following scalar product, and exercises cold graphs and replay.
For each model configuration and ring, it checks 120 reduced-output cases,
120 extended-output fallback cases, and three chains of 24 rotations. The
Mamba-2 configuration uses complex slots and sparse-ternary secrets; Mamba-3
uses real slots and uniform-ternary secrets. All use the existing experimental
`security=not-set`, depth 44, 59-bit scale, 60-bit first modulus and dnum 3.
These probes are not a full Mamba-2 model benchmark.

Apply the patch to a separate checkout of that dependency, then rebuild and
install it with the same compiler, architecture and flags as the baseline:

```bash
git -C /path/to/fideslib apply "$PWD/experiments/rotation_stream_chain/fideslib-rotation-stream-chain.patch"
```

Build against a separately installed baseline or patched dependency:

```bash
cmake -S experiments/rotation_stream_chain -B build/rotation-probe \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/path/to/matching/g++ \
  -DCMAKE_PREFIX_PATH='/path/to/fideslib;/path/to/openfhe'
cmake --build build/rotation-probe --target rotation_probe -j4
build/rotation-probe/rotation_probe probe.json 32768 mamba3
```

Repeat with N=65536 and `mamba2`, using independent processes. The B300
campaign also checks the candidate with `FIDESLIB_USE_GRAPH_CAPTURE=0`, the
adopted seed/EvalMod and two-pass refresh circuits, a frozen model prefix,
and a gated full ABBA comparison. Model accuracy tolerances remain 0.001;
the primitive numerical gate is 1e-6, in addition to exact coefficient tests.
The [B300 study](../../docs/research/2026-09-27-b300-rotation-stream-chain.md)
adopts this optional patch: full ABBA means **146.36 → 141.77 s (3.14%)**,
or **35.44 s/generated token**; refresh improves **2.72%**. Both rings and
model configurations pass exact checks, including a separate graph-disabled
Mamba-3 process. Full-model timing is only for the frozen Mamba-3 workload.
Use the [shared evidence and comparison tools](../../docs/experiments.md) to
verify public files and compare recorded samples. Study-specific controllers
and verification wrappers have been removed.
