# Campaign manifests

These JSON files configure [`run_dgx_campaign.py`](../run_dgx_campaign.py).
Paths passed on the command line are relative to the repository root. Keep
numerical parameters and acceptance gates explicit when adapting a manifest.

The full reproduction path uses [`run_dgx_generation.py`](../run_dgx_generation.py),
which prepares a fresh prompt payload and derives its campaign from
[`dgx_spark_stabilized_generation.json`](dgx_spark_stabilized_generation.json).
Enable both encoder flags as shown in the [reproduction guide](../../docs/reproducing.md).

| Workload | Manifests |
| --- | --- |
| Stabilized baseline | [`generation`](dgx_spark_stabilized_generation.json), [`chain`](dgx_spark_stabilized_chain.json), [`smoke`](dgx_spark_stabilized_smoke.json) |
| Periodic coefficients | [`generation`](dgx_spark_periodic_gate_generation.json), [`smoke`](dgx_spark_periodic_gate_smoke.json) |
| Subring encoding | [`generation`](dgx_spark_subring_gate_generation.json), [`smoke`](dgx_spark_subring_gate_smoke.json) |
| Projection/layout comparison | [`replication A/B`](dgx_spark_replication_ab.json), [`smoke`](dgx_spark_smoke.json) |
| Normalization | [`scheduled chain`](dgx_spark_scheduled_norm_chain.json), [`scheduled smoke`](dgx_spark_scheduled_norm_smoke.json), [`Chebyshev A/B`](dgx_spark_chebyshev_ab.json) |
| State coordinates | [`row/group A/B`](dgx_spark_row_state_ab.json), [`confirmation`](dgx_spark_row_state_confirm.json), [`scale floor`](dgx_spark_row_state_floor.json) |
| Diagnostics | [`layer`](dgx_spark_diagnostic.json), [`numerical`](dgx_spark_numerical_diagnostic.json), [`state`](dgx_spark_state_diagnostic.json) |

`dgx_spark_autoregressive.json` and the other `dgx_*`/`b300_*` manifests retain
earlier experiments. They may depend on older payloads, host paths, or security
settings. Their existence is not a successful benchmark or candidate promotion;
check the [evidence registry](../../docs/evidence.md) for measured status.
