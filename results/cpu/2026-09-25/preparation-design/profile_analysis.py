"""Observed category costs and refresh counts; no achievable-speed claim."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
native_path = ROOT / "m3-native.json"
native = json.loads(native_path.read_text())
run = json.loads((ROOT / "m3-run.json").read_text())
assert run["passed"] and native["passed"] and all(run["checks"].values())
assert run["native_sha256"] == hashlib.sha256(native_path.read_bytes()).hexdigest()
assert native["bootstrap_passes"] == 2 and native["bootstraps"] % 2 == 0
events = native["bootstraps"] // 2
singles = events - native["refresh_batches"]
grouped = native["logical_refreshes"] - singles
assert singles >= 0 and grouped >= 2 * native["refresh_batches"]
result = {
    "native_sha256": run["native_sha256"],
    "eval_seconds": native["eval_seconds"],
    "host_encoding_seconds": native["host_encoding_seconds"],
    "upload_seconds": native["plaintext_upload_seconds"],
    "refresh_seconds": native["bootstrap_seconds"],
    "eligible_repeat_weight_identities": 17952,
    "all_host_encodes": native["host_encodes"],
    "repeat_identity_fraction_of_all_encodes": 17952 / native["host_encodes"],
    "first_linear_seconds_outside_refresh": native["operation_stats"]["linear"]["seconds"]
        - native["operation_stats"]["linear"]["bootstrap_seconds"],
    "repeated_linear_seconds_outside_refresh": native["operation_stats"]["linear_ref"]["seconds"]
        - native["operation_stats"]["linear_ref"]["bootstrap_seconds"],
    "two_pass_refresh_events": events,
    "grouped_refresh_events": native["refresh_batches"],
    "single_value_refresh_events": singles,
    "values_in_grouped_refreshes": grouped,
    "mean_values_per_grouped_refresh": grouped / native["refresh_batches"],
    "largest_group": native["largest_refresh_batch"],
    "scope": "Derived from an existing completed run. Category timers overlap/nest. Cache hits and savings are unmeasured. Singleton refreshes are not asserted to be jointly schedulable.",
}
(ROOT / "profile-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
