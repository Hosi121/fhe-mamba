"""Verify one compact-RNS candidate and derive non-overlapping evaluation costs."""

import hashlib
import json
import math
import tarfile
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def categories(data):
    operations = data["operation_stats"]

    def ordinary(names):
        return sum(
            operations[n]["seconds"] - operations[n]["bootstrap_seconds"]
            for n in names
            if n in operations
        )

    costs = {
        "refresh": data["bootstrap_seconds"],
        "linear": ordinary(["linear", "linear_ref"]),
        "polynomial": ordinary(["cheb"]),
        "routing": ordinary(["gather", "scatter", "repeat", "sum"]),
    }
    costs["other_and_timer_gaps"] = data["eval_seconds"] - sum(costs.values())
    return costs


def inference(name, full=False):
    run = read(ROOT / name / "run.json")
    data = read(ROOT / name / "native.json")
    assert run["exit_code"] == 0
    assert data["passed"]
    assert data["s2c_first"]
    assert data["bootstrap_passes"] == 2
    assert (data["refresh_ceiling"], data["refreshed_level"]) == (35, 18)
    assert (data["ring_dimension"], data["slots"], data["depth"], data["scale_bits"]) == (
        65536,
        32768,
        44,
        59,
    )
    assert data["security"] == "not-set"
    assert data["exact_tolerance"] == data["polynomial_tolerance"] == 0.001
    assert data["evaluation_decryptions"] == data["non_finite"] == 0
    assert data["generated_token_ids"] == ([315, 279, 1614, 315] if full else [6864, 6864])
    assert bool(data.get("gpu_plaintext_rns")) == name.endswith("candidate")
    for key in ["max_abs_error_vs_exact", "max_abs_error_vs_polynomial"]:
        assert math.isfinite(data[key])
        assert 0 <= data[key] <= 0.001
    assert all(
        math.isfinite(v) and 0 <= v <= 0.001
        for errors in data["per_output_errors"]
        for v in errors.values()
    )
    if data.get("gpu_plaintext_rns"):
        assert data["compact_rns_encodes"] == data["compact_rns_uploads"] > 0
        assert data["compact_rns_fallbacks"] > 0
    result = {
        k: data[k]
        for k in [
            "eval_seconds",
            "bootstrap_seconds",
            "bootstraps",
            "logical_refreshes",
            "ct_ct_mul",
            "ct_pt_mul",
            "rotations",
            "evaluated_nodes",
            "refresh_batches",
            "frontier_deferrals",
            "host_encoding_seconds",
            "peak_rss_gib",
            "max_abs_error_vs_exact",
            "max_abs_error_vs_polynomial",
        ]
    }
    result.update({k: v for k, v in data.items() if k.startswith("compact_rns_")})
    result["ordinary_seconds"] = data["eval_seconds"] - data["bootstrap_seconds"]
    result["categories_seconds"] = categories(data)
    result["process_wall_seconds"] = run["wall_seconds"]
    result["binary_sha256"] = run["binary_sha256"]
    result["native_sha256"] = sha(ROOT / name / "native.json")
    parity = read(ROOT / name / "parity.json")
    payload_name = "lm-full" if full else "lm-layer1"
    manifest_path = ROOT / "payload-manifests" / payload_name / "manifest.json"
    manifest = read(manifest_path)
    assert parity["program_sha256"] == manifest["files_sha256"]["program.txt"]
    assert parity["manifest_sha256"] == sha(manifest_path)
    return result


def compare(a, b):
    for key in [
        "bootstraps",
        "logical_refreshes",
        "ct_ct_mul",
        "ct_pt_mul",
        "rotations",
        "evaluated_nodes",
        "refresh_batches",
        "frontier_deferrals",
    ]:
        assert a[key] == b[key], (key, a[key], b[key])
    return {
        "baseline": a,
        "candidate": b,
        "reduction_percent": 100 * (1 - b["eval_seconds"] / a["eval_seconds"]),
        "ordinary_reduction_percent": 100 * (1 - b["ordinary_seconds"] / a["ordinary_seconds"]),
        "same_schedule": True,
    }


def main():
    manifest = read(ROOT / "native-source-manifest.json")
    with tarfile.open(ROOT / "native-sources.tar.gz") as handle:
        files = {
            m.name.removeprefix("native-source/"): hashlib.sha256(
                handle.extractfile(m).read()
            ).hexdigest()
            for m in handle
            if m.isfile()
        }
    assert files == manifest
    probes = {}
    for name in ["probe-t4-mamba3", "probe-t1-mamba3", "probe-t4-mamba2"]:
        data = read(ROOT / name / "native.json")
        assert read(ROOT / name / "run.json")["exit_code"] == 0
        assert data["passed"]
        assert data["exact_rns_cases"] == 240
        assert data["encrypted_cases"] == 9
        assert data["compact_cases"] == 35
        assert data["fallback_cases"] == 205
        assert math.isfinite(data["max_abs_error"])
        assert data["max_abs_error"] <= 1e-6
        timings = {}
        for level in [18, 26, 34]:
            samples = [s for s in data["samples"] if s["level"] == level and s["iteration"] >= 0]
            costs = {
                "candidate" if compact else "baseline": mean(
                    s["encode_ms"] + s["upload_ms"] for s in samples if s["compact"] == compact
                )
                for compact in [False, True]
            }
            timings[str(level)] = {
                **costs,
                "reduction_percent": 100 * (1 - costs["candidate"] / costs["baseline"]),
            }
        probes[name] = {k: v for k, v in data.items() if k != "samples"}
        probes[name]["mean_warm_preparation_ms"] = timings
    prefixes = {
        "baseline": [inference(f"prefix-{i}-baseline") for i in [0, 3]],
        "candidate": [inference(f"prefix-{i}-candidate") for i in [1, 2]],
    }
    for samples in prefixes.values():
        for sample in samples:
            compare(prefixes["baseline"][0], sample)
    regressions = {}
    for name in ["existing-plaintext-mamba3", "existing-plaintext-mamba2"]:
        data = read(ROOT / name / "native.json")
        assert read(ROOT / name / "run.json")["exit_code"] == 0
        assert data["passed"]
        assert data["encrypted"]
        assert data["exact_rns_cases"] == data["moved_coefficient_cases"] == 160
        assert data["shared_policy_cases"] == 162
        assert data["borrowed_rns_cases"] == 320
        assert data["tolerance"] == 1e-6
        assert math.isfinite(data["max_abs_error"])
        assert 0 <= data["max_abs_error"] <= data["tolerance"]
        regressions[name] = {k: v for k, v in data.items() if k != "samples"}
    assert read(ROOT / "status.json")["state"] == "completed"
    assert read(ROOT / "post-validation.json")["state"] == "completed"
    dependencies = read(ROOT / "dependency-verification.json")
    assert dependencies["passed"]
    assert (
        dependencies["files_sha256"]
        == read(ROOT / "prior-dependency-verification.json")["files_sha256"]
    )
    means = {mode: mean(s["eval_seconds"] for s in samples) for mode, samples in prefixes.items()}
    result = {
        "scope": "One GPU RNS expansion mechanism; S2C-first and both correction passes retained",
        "source_archive_sha256": sha(ROOT / "native-sources.tar.gz"),
        "verified_source_files": len(manifest),
        "probes": probes,
        "existing_plaintext_regressions": regressions,
        "prefixes": prefixes,
        "prefix_means_seconds": means,
        "prefix_reduction_percent": 100 * (1 - means["candidate"] / means["baseline"]),
    }
    a, b = inference("full-baseline", True), inference("full-candidate", True)
    build = read(ROOT / "build.json")
    assert a["binary_sha256"] == build["baseline_binary_sha256"]
    assert b["binary_sha256"] == build["candidate_binary_sha256"]
    result["full"] = compare(a, b)
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"prefix": means, "full": result["full"]}, indent=2))


if __name__ == "__main__":
    main()
