# Repository helpers

Run from the repository root. Research-specific runners are indexed in
[`experiments/`](../experiments/README.md).

| Task | Scripts | Guide |
| --- | --- | --- |
| Local verification | [`run_fast_checks.sh`](run_fast_checks.sh), [`run_checks.sh`](run_checks.sh) | [Testing](../docs/testing.md) |
| Checkpoint download/verification | [`download_checkpoint.py`](download_checkpoint.py) | [Reproduction](../docs/reproducing.md#2-obtain-the-exact-public-checkpoint) |
| DGX Spark build/execution | [`build_dgx_spark.sh`](build_dgx_spark.sh), [`run_dgx_spark.sh`](run_dgx_spark.sh) | [Spark runbook](../docs/dgx-spark.md) |
| Historical B300 build/execution | `b300_platform.sh`, `build_b300_fideslib.sh`, `build_b300_fideslib_image.sh`, `launch_b300_fideslib_build.sh`, `run_b300_mamba2.sh` | [Maintenance](../docs/maintenance.md) |
| Historical host utilities | `probe_fideslib.sh`, `bootstrap_remote_dev.sh`, `sync_high.sh` | These retain host defaults and are not part of the fresh-checkout reproduction path |

Checks and the checkpoint verifier run locally. Build/run helpers may compile
native dependencies or launch expensive GPU work. Remote utilities require an
explicitly configured SSH host; inspect their environment options when adapting
them to your machine.
