# Native backend

[`fideslib_stage0/`](fideslib_stage0/) contains the active Mamba-2 CKKS backend.
The directory's historical name does not denote a separate model implementation.

| Directory | Contents |
| --- | --- |
| [`src/`](fideslib_stage0/src/) | Encrypted model evaluator, runtime configuration, payload/provenance code and GPU probes |
| [`include/`](fideslib_stage0/include/) | Layout, polynomial and payload contracts |
| [`tests/`](fideslib_stage0/tests/) | FIDESlib-free C++ contract tests |
| [`patches/`](fideslib_stage0/patches/) | Patches for the pinned FIDESlib dependency |

For encrypted execution, use the [DGX Spark build guide](../docs/dgx-spark.md#build)
and [reproduction guide](../docs/reproducing.md). The backend is built separately
from the Python wheel. Source and build paths are part of its provenance, so
rebuild after updating a checkout whose layout changed.

CPU contracts require CMake 3.25.2+ and a C++20 compiler, without CUDA or FIDESlib:

```bash
cmake -S native/fideslib_stage0 -B build/stage0-layout-tests \
  -DFHE_STAGE0_BUILD_KERNEL=OFF -DFHE_STAGE0_BUILD_TESTS=ON
cmake --build build/stage0-layout-tests
ctest --test-dir build/stage0-layout-tests --output-on-failure
```

These contracts also run through the Python test suite. Numerical GPU promotion
requires the separate [encrypted validation gates](../docs/validation.md#gpu-integration-gates).
