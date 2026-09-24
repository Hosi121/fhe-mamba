"""Validate the matched raw prefix runs and derive the reported comparison."""

import hashlib
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
names = ("a1", "b1", "b2", "a2")
runs = [json.loads((root / name / "run.json").read_text()) for name in names]
native = [json.loads((root / name / "native.json").read_text()) for name in names]
identity = ("binary_sha256", "program_sha256", "manifest_sha256")
for key in identity:
    assert len({run[key] for run in runs}) == 1, key
common = (
    "security",
    "slots",
    "ring_dimension",
    "depth",
    "scale_bits",
    "nodes",
    "evaluated_nodes",
    "bootstraps",
    "ct_ct_mul",
    "ct_pt_mul",
    "rotations",
    "linear_method",
    "routing_method",
    "refresh_policy",
    "bootstrap_passes",
    "batch_refresh",
    "logical_refreshes",
    "planned_logical_refreshes",
    "refresh_batches",
    "largest_refresh_batch",
    "inplace_ops",
    "scratch_clones_eliminated",
    "profile_evaluation",
    "evaluation_decryptions",
    "client_output_decrypt_count",
    "generated_token_ids",
    "exact_tolerance",
    "polynomial_tolerance",
)
for key in common:
    assert all(row[key] == native[0][key] for row in native), key
for name, run, row in zip(names, runs, native, strict=True):
    assert run["passed"] is True
    assert row["passed"] is True
    assert run["returncode"] == 0
    assert run["timed_out"] is False
    assert row["non_finite"] == 0
    assert 0 <= row["max_abs_error_vs_exact"] <= row["exact_tolerance"] == 0.001
    assert 0 <= row["max_abs_error_vs_polynomial"] <= row["polynomial_tolerance"] == 0.001
    assert row["cache_plaintexts"] == name.startswith("b")
    assert row["host_encodes"] + row["plaintext_cache_hits"] == 888
    digest = hashlib.sha256((root / name / "native.json").read_bytes()).hexdigest()
    assert digest == run["native_sha256"]
baseline = [native[0], native[3]]
candidate = [native[1], native[2]]
a = mean(row["eval_seconds"] for row in baseline)
b = mean(row["eval_seconds"] for row in candidate)
comparison = {
    "schema": "fhemamba-plaintext-cache-comparison-v1",
    "scope": "127-node trained one-layer prefix, first evaluation; no generation",
    "order": list(names),
    "identity": {key: runs[0][key] for key in identity},
    "common": {key: native[0][key] for key in common},
    "baseline_eval_seconds": [row["eval_seconds"] for row in baseline],
    "candidate_eval_seconds": [row["eval_seconds"] for row in candidate],
    "baseline_mean_seconds": a,
    "candidate_mean_seconds": b,
    "reduction_percent": 100 * (1 - b / a),
    "speedup": a / b,
    "baseline_host_encoding_mean_seconds": mean(row["host_encoding_seconds"] for row in baseline),
    "candidate_host_encoding_mean_seconds": mean(row["host_encoding_seconds"] for row in candidate),
    "baseline_host_encodes": [row["host_encodes"] for row in baseline],
    "candidate_host_encodes": [row["host_encodes"] for row in candidate],
    "candidate_cache_hits": [row["plaintext_cache_hits"] for row in candidate],
    "baseline_peak_rss_gib": [row["peak_rss_gib"] for row in baseline],
    "candidate_peak_rss_gib": [row["peak_rss_gib"] for row in candidate],
    "max_abs_error_vs_exact": max(row["max_abs_error_vs_exact"] for row in native),
    "max_abs_error_vs_polynomial": max(row["max_abs_error_vs_polynomial"] for row in native),
    "note": "Both modes enable host timers, without an attached Nsight profiler; two samples each.",
}
(root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison, indent=2))
