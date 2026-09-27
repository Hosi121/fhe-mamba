# Repository helpers

Run from the repository root. Research-specific runners are indexed in
[`experiments/`](../experiments/README.md).

| Task | Scripts | Guide |
| --- | --- | --- |
| Local verification | [`run_fast_checks.sh`](run_fast_checks.sh), [`run_checks.sh`](run_checks.sh) | [Testing](../docs/testing.md) |
| Checkpoint download/verification | [`download_checkpoint.py`](download_checkpoint.py) | [Reproduction](../docs/reproducing.md#2-obtain-the-exact-public-checkpoint) |
| DGX Spark build/execution | [`build_dgx_spark.sh`](build_dgx_spark.sh), [`run_dgx_spark.sh`](run_dgx_spark.sh) | [Spark runbook](../docs/dgx-spark.md) |
| Historical B300 build/execution | `b300_platform.sh`, `build_b300_fideslib.sh`, `build_b300_fideslib_image.sh`, `launch_b300_fideslib_build.sh`, `run_b300_mamba2.sh` | [Maintenance](../docs/maintenance.md) |
| Dependency inspection | `probe_fideslib.sh` | Set `FIDESLIB_DIR` to inspect a local dependency checkout |

Checks and the checkpoint verifier run locally. Build/run helpers may compile
native dependencies or launch expensive GPU work. Remote utilities require an
explicitly configured SSH host; inspect their environment options when adapting
them to your machine.

One-off host synchronization and remote package-install wrappers have been
retired. Clone the repository on the target and follow the same installation
commands. B300 launchers accept `ROOT_DIR` and require an explicit `GPU_DEVICE`;
job settings and completion events use the [shared workflow](../docs/experiments.md).
