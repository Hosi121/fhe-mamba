"""Check the four-candidate study's gates and derive non-overlapping costs."""

import hashlib
import json
import math
import tarfile
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def verify_archive(prefix):
    manifest = read(ROOT / f"{prefix}source-manifest.json")
    with tarfile.open(ROOT / f"{prefix}sources.tar.gz") as archive:
        actual = {
            m.name.removeprefix("native-source/"): hashlib.sha256(
                archive.extractfile(m).read()
            ).hexdigest()
            for m in archive
            if m.isfile()
        }
    assert actual == manifest


def inference(name, full=False):
    data = read(ROOT / name / "native.json")
    record = read(ROOT / name / "run.json")
    assert record["exit_code"] == 0
    assert data["passed"]
    assert data["generated_token_ids"] == ([315, 279, 1614, 315] if full else [6864, 6864])
    assert data["evaluation_decryptions"] == data["non_finite"] == 0
    assert data["exact_tolerance"] == data["polynomial_tolerance"] == 0.001
    for k in ("max_abs_error_vs_exact", "max_abs_error_vs_polynomial"):
        assert math.isfinite(data[k])
        assert 0 <= data[k] <= 0.001
    assert all(
        math.isfinite(v) and 0 <= v <= 0.001 for e in data["per_output_errors"] for v in e.values()
    )
    assert data["bootstrap_passes"] == 2
    assert data["s2c_first"]
    assert data["gpu_plaintext_rns"]
    assert (data["ring_dimension"], data["slots"], data["depth"], data["scale_bits"]) == (
        65536,
        32768,
        44,
        59,
    )
    assert (data["refresh_ceiling"], data["refreshed_level"]) == (35, 18)
    if full:
        parity = read(ROOT / name / "parity.json")
        assert parity["passed"]
        manifest = ROOT / "full-payload-manifest.json"
        assert parity["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert parity["program_sha256"] == read(manifest)["files_sha256"]["program.txt"]
        candidate = "candidate" in name
        assert data.get("hoist_rotations", False) == candidate
        assert data.get("share_chebyshev", False) == candidate
        if candidate:
            assert data["shared_basis_hits"] > 0
            assert data["rotation_sibling_batches"] > 0
    costs = {"refresh": data["bootstrap_seconds"]}
    for label, names in {
        "linear": ("linear", "linear_ref"),
        "polynomial": ("cheb",),
        "routing": ("gather", "scatter", "repeat", "sum"),
    }.items():
        costs[label] = sum(
            v["seconds"] - v["bootstrap_seconds"]
            for k, v in data["operation_stats"].items()
            if k in names
        )
    costs["other"] = data["eval_seconds"] - sum(costs.values())
    return {
        "name": name,
        **{
            k: data[k]
            for k in (
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
            )
        },
        "ordinary_seconds": data["eval_seconds"] - data["bootstrap_seconds"],
        "categories_seconds": costs,
        "process_wall_seconds": record["wall_seconds"],
        "shared_basis_hits": data.get("shared_basis_hits", 0),
        "shared_basis_invalidations": data.get("shared_basis_invalidations", 0),
        "binary_sha256": record["binary_sha256"],
    }


def aggregate(samples):
    return {
        key: mean(s[key] for s in samples)
        for key in ("eval_seconds", "bootstrap_seconds", "ordinary_seconds", "process_wall_seconds")
    }


def main():
    verify_archive("native-")
    verify_archive("prefix-native-")
    source = read(ROOT / "source-verification.json")
    assert source["passed"]
    assert source["native_source_sha256"] == read(ROOT / "native-source-manifest.json")
    alternate = ROOT / "alternative-backends"
    assert (
        source["alternative_source_manifest_sha256"]
        == hashlib.sha256((alternate / "source-manifest.json").read_bytes()).hexdigest()
    )
    for name, digest in read(alternate / "source-manifest.json").items():
        assert hashlib.sha256((alternate / name).read_bytes()).hexdigest() == digest, name
    before = read(ROOT / "dependency-verification-before.json")
    after = read(ROOT / "dependency-verification-after.json")
    assert before["passed"]
    assert before == after
    assert len(before["files_sha256"]) == 17
    backend = read(ROOT / "backend-source-verification.json")
    assert backend["passed"]
    assert backend["identical_files"] == 91
    assert read(ROOT / "full-status.json")["state"] == "completed"
    rotation = {}
    for mode in ("mamba3", "mamba2"):
        data = read(ROOT / f"rotation-probe-{mode}/native.json")
        assert data["passed"]
        assert data["exact_rns_cases"] == 276
        assert data["live_inputs_unchanged"]
        timings = {}
        for level in (18, 34):
            rows = [s for s in data["samples"] if s["level"] == level and s["iteration"] >= 0]
            a = mean(s["ms"] for s in rows if not s["hoisted"])
            b = mean(s["ms"] for s in rows if s["hoisted"])
            timings[str(level)] = {
                "scalar_ms": a,
                "shared_ms": b,
                "reduction_percent": 100 * (1 - b / a),
            }
        rotation[mode] = {"exact_cases": 276, "timings": timings}
    for case in (
        "basis-probe-baseline",
        "basis-probe-share",
        "basis-refresh-baseline",
        "basis-refresh-share",
    ):
        data = read(ROOT / case / "native.json")
        assert data["passed"]
        assert data["max_abs_error_vs_polynomial"] <= 0.001
        if case == "basis-refresh-share":
            assert data["shared_basis_hits"] > 0
            assert data["shared_basis_invalidations"] > 0
    modes = ("baseline", "hoist", "share", "both", "both", "share", "hoist", "baseline")
    prefixes = [inference(f"prefix-{i}-{mode}") | {"mode": mode} for i, mode in enumerate(modes)]
    prefix_means = {
        m: aggregate([s for s in prefixes if s["mode"] == m]) for m in sorted(set(modes))
    }
    for _m, v in prefix_means.items():
        v["reduction_percent"] = 100 * (
            1 - v["eval_seconds"] / prefix_means["baseline"]["eval_seconds"]
        )
    full = [
        inference(f"full-{i}-{mode}", True) | {"mode": mode}
        for i, mode in enumerate(("baseline", "candidate", "candidate", "baseline"))
    ]
    for key in (
        "bootstraps",
        "logical_refreshes",
        "evaluated_nodes",
        "refresh_batches",
        "frontier_deferrals",
    ):
        assert len({s[key] for s in full}) == 1, key
    contract = read(ROOT / "contract.json")
    baseline_hashes = {s["binary_sha256"] for s in full if s["mode"] == "baseline"}
    candidate_hashes = {s["binary_sha256"] for s in full if s["mode"] == "candidate"}
    assert baseline_hashes == {contract["baseline_binary_sha256"]}
    assert len(candidate_hashes) == 1
    assert candidate_hashes != baseline_hashes
    full_means = {
        m: aggregate([s for s in full if s["mode"] == m]) for m in ("baseline", "candidate")
    }
    for mode in ("baseline", "candidate"):
        for key in ("ct_ct_mul", "ct_pt_mul", "rotations", "shared_basis_hits"):
            assert len({s[key] for s in full if s["mode"] == mode}) == 1, (mode, key)
    a, b = full_means["baseline"], full_means["candidate"]
    summary = {
        "stage": "structural-four-report",
        "version": "0.5.0",
        "repo_commit": source["implementation_commit"],
        "passed": True,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": True,
            "claim": (
                "Four implemented mechanisms with bounded qualification; two shared-intermediate "
                "paths compared in four full, fixed-payload Mamba-3 processes. Model gates and "
                "generated IDs pass. Report validation passing does not mean all candidates "
                "succeeded: the nominal 59-bit 32-bit adapter fails and the CPU ring-switch "
                "route is not competitive. No Mamba-2 full-speed, arbitrary-prompt, separated "
                "server, or 128-bit full-chain claim."
            ),
            "hardware": "DGX Spark GB10, CUDA 13.0, CPU affinity 15-19, OMP_NUM_THREADS=4",
            "security": "not-set",
            "layers": 12,
            "encrypted_evaluations": 5,
            "generated_tokens": 4,
        },
        "rotation": rotation,
        "prefix_samples": prefixes,
        "prefix_means": prefix_means,
        "full_samples": full,
        "full_means": full_means,
        "full_reduction_percent": 100 * (1 - b["eval_seconds"] / a["eval_seconds"]),
        "ordinary_reduction_percent": 100 * (1 - b["ordinary_seconds"] / a["ordinary_seconds"]),
        "amortized_native_seconds_per_generated_token": b["eval_seconds"] / 4,
        "amortized_process_seconds_per_generated_token": b["process_wall_seconds"] / 4,
        "small_ring": {},
        "composite": {},
        "selection": read(ROOT / "small-selection.json"),
    }
    for logn in (15, 16):
        data = read(ROOT / f"small-ring-dense-{logn}/native.json")
        assert data["passed"]
        assert data["max_abs_error"] <= 1e-6
        assert data["scale_bits"] == 59
        assert data["tolerance"] == 1e-6
        assert data["intermediate_evaluator_decryptions"] == 0
        assert data["ordinary_log_n"] == logn
        assert data["refresh_log_n"] == 16
        assert data["bootstrap_passes"] == 2
        assert len(data["samples"]) == 4
        summary["small_ring"][str(logn)] = data
    assert summary["small_ring"]["15"]["ordinary_q"] == summary["small_ring"]["16"]["ordinary_q"]
    assert summary["small_ring"]["15"]["refresh_q"] == summary["small_ring"]["16"]["refresh_q"]
    for bits in (32, 64):
        record = read(ROOT / f"composite-final-{bits}/run.json")
        path = ROOT / f"composite-final-{bits}/native.json"
        data = read(path)
        assert data["passed"] == (bits == 64)
        assert record["exit_code"] == (0 if bits == 64 else 3)
        assert data["ordinary_rescale_bits"] == 59
        assert data["tolerance"] == 1e-6
        assert (
            data["intermediate_evaluator_decryptions"] == data["diagnostic_client_validations"] == 0
        )
        assert not data["stock_profile_control"]
        assert data["bootstrap_passes"] == 2
        assert len(data["samples"]) == 4
        for sample in data["samples"]:
            error = sample["max_abs_error"]
            assert math.isfinite(error)
            assert error >= 0
            assert (error <= 1e-6) == (bits == 64)
        summary["composite"][str(bits)] = {"run": record, "native": data}
    for key in ("binary_sha256", "library_sha256"):
        assert summary["composite"]["32"]["run"][key] == summary["composite"]["64"]["run"][key]
    stock = read(ROOT / "composite-stock32.json")
    assert stock["passed"]
    assert stock["word_bits"] == 32
    assert stock["stock_profile_control"]
    assert stock["ordinary_rescale_bits"] == 40
    assert stock["tolerance"] == 1e-6
    assert all(0 <= sample["max_abs_error"] <= 1e-6 for sample in stock["samples"])
    summary["stock32_control"] = stock
    directories = [s["name"] for s in prefixes + full] + [
        "rotation-probe-mamba2",
        "rotation-probe-mamba3",
        "basis-probe-baseline",
        "basis-probe-share",
        "basis-refresh-baseline",
        "basis-refresh-share",
        "small-ring-dense-15",
        "small-ring-dense-16",
        "composite-final-32",
        "composite-final-64",
    ]
    raw_paths = [ROOT / d / f for d in directories for f in ("native.json", "run.json")]
    raw_paths += [ROOT / f for f in ("full-payload-manifest.json", "composite-stock32.json")]
    summary["raw_artifacts_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_paths
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "full_means",
                    "full_reduction_percent",
                    "ordinary_reduction_percent",
                    "amortized_native_seconds_per_generated_token",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
