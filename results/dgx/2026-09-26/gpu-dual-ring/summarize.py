"""Validate the bounded GPU dual-ring study and derive disjoint cost totals."""

import hashlib
import json
import math
import tarfile
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_check(archive_name, manifest_name):
    with tarfile.open(ROOT / archive_name) as archive:
        files = {
            item.name.removeprefix("native/fideslib_stage0/"): hashlib.sha256(
                archive.extractfile(item).read()
            ).hexdigest()
            for item in archive
            if item.isfile()
        }
    assert files == read(manifest_name)


def archive_files(name):
    with tarfile.open(ROOT / name) as archive:
        return {
            item.name: hashlib.sha256(archive.extractfile(item).read()).hexdigest()
            for item in archive
            if item.isfile()
        }


def inference(name, full):
    data, run = read(f"{name}/native.json"), read(f"{name}/run.json")
    candidate = name.endswith("candidate")
    assert run["exit_code"] == 0
    assert data["passed"]
    assert data["non_finite"] == data["evaluation_decryptions"] == 0
    assert data["exact_tolerance"] == data["polynomial_tolerance"] == 0.001
    assert data["generated_token_ids"] == ([315, 279, 1614, 315] if full else [6864, 6864])
    assert data["s2c_first"]
    assert data["bootstrap_passes"] == 2
    assert data["gpu_plaintext_rns"]
    assert data["hoist_rotations"]
    assert data["share_chebyshev"]
    assert (
        data["depth"],
        data["scale_bits"],
        data["refresh_ceiling"],
        data["refreshed_level"],
    ) == (44, 59, 35, 18)
    assert (data["ring_dimension"], data["slots"]) == (
        (32768, 16384) if candidate else (65536, 32768)
    )
    assert data["security"] == "not-set"
    assert data["encrypted"]
    assert run["command"][:3] == ["taskset", "-c", "15-19"]
    assert run["environment"]["OMP_NUM_THREADS"] == "4"
    reference = read("full-baseline-command.json" if full else "prefix-baseline-command.json")
    expected = reference["command"].copy()
    expected[3], expected[5] = run["command"][3], run["command"][5]
    if candidate:
        expected.append("--gpu-dual-ring")
    else:
        assert run["binary_sha256"] == reference["binary_sha256"]
    assert run["command"] == expected
    for key in ("max_abs_error_vs_exact", "max_abs_error_vs_polynomial"):
        assert math.isfinite(data[key])
        assert 0 <= data[key] <= 0.001
    for error in data["per_output_errors"]:
        assert all(math.isfinite(v) and 0 <= v <= 0.001 for v in error.values())
    if candidate:
        assert data["gpu_dual_ring"]
        assert data["refresh_ring_dimension"] == 65536
        assert (
            data["ring_switch_down_calls"] == data["bootstraps"] == 2 * data["ring_switch_up_calls"]
        )
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
    assert all(math.isfinite(v) and v >= 0 for v in costs.values())
    return {
        "name": name,
        "candidate": candidate,
        **{
            k: data[k]
            for k in (
                "eval_seconds",
                "bootstrap_seconds",
                "setup_seconds",
                "ct_ct_mul",
                "ct_pt_mul",
                "rotations",
                "bootstraps",
                "logical_refreshes",
                "evaluated_nodes",
                "refresh_batches",
                "host_encoding_seconds",
                "peak_rss_gib",
                "max_abs_error_vs_exact",
                "max_abs_error_vs_polynomial",
            )
        },
        "ring_switch_up_calls": data.get("ring_switch_up_calls", 0),
        "ring_switch_down_calls": data.get("ring_switch_down_calls", 0),
        "ordinary_seconds": data["eval_seconds"] - data["bootstrap_seconds"],
        "categories_seconds": costs,
        "wall_seconds": run["wall_seconds"],
        "binary_sha256": run["binary_sha256"],
    }


def aggregate(samples):
    result = {}
    for mode in (False, True):
        rows = [r for r in samples if r["candidate"] == mode]
        result["candidate" if mode else "baseline"] = {
            key: mean(r[key] for r in rows)
            for key in (
                "eval_seconds",
                "ordinary_seconds",
                "bootstrap_seconds",
                "setup_seconds",
                "host_encoding_seconds",
                "wall_seconds",
                "peak_rss_gib",
            )
        }
        result["candidate" if mode else "baseline"]["categories_seconds"] = {
            k: mean(r["categories_seconds"][k] for r in rows) for k in rows[0]["categories_seconds"]
        }
    result["reduction_percent"] = 100 * (
        1 - result["candidate"]["eval_seconds"] / result["baseline"]["eval_seconds"]
    )
    return result


def main():
    assert read("small-status.json")["state"] == read("full-status.json")["state"] == "completed"
    archive_check("prefix-native-sources.tar.gz", "small-source-manifest.json")
    archive_check("native-sources.tar.gz", "full-source-manifest.json")
    source = read("source-verification.json")
    assert source["passed"]
    assert source["files_sha256"] == read("full-source-manifest.json")
    before, after = (
        read("canonical-dependencies-before-full.json"),
        read("canonical-dependencies-after.json"),
    )
    assert before == after
    assert before["passed"]
    assert read("backend-source-verification.json")["passed"]
    assert read("full-payload-verification.json")["passed"]
    assert read("post-validation.json")["passed"]
    assert read("validation.json")["passed"]
    cleanup = read("cleanup.json")
    assert cleanup["passed"]
    for item in cleanup["archives"]:
        name, expected_sha, expected_files = item["archive"], item["sha256"], item["files_sha256"]
        if name == "backend-sources.tar.gz":
            published = read("published-backend.json")
            assert published["passed"]
            assert published["original_sha256"] == expected_sha
            for omitted, digest in published["omitted_files"].items():
                assert expected_files[omitted] == digest
            expected_files = {
                k: v for k, v in expected_files.items() if k not in published["omitted_files"]
            }
            name, expected_sha = published["published_archive"], published["published_sha256"]
        assert sha(ROOT / name) == expected_sha
        files = {path.removeprefix("./"): digest for path, digest in archive_files(name).items()}
        assert files == expected_files
    micro_samples = []
    max_transfer_error = 0.0
    primitive_sources = archive_files("small-probe-sources.tar.gz")
    primitive_sources.update(archive_files("prefix-native-sources.tar.gz"))
    initial_sources = {
        k.removeprefix("qualification-1-source/"): v
        for k, v in archive_files("qualification-1-sources.tar.gz").items()
    }
    assert initial_sources == read("qualification-1/run.json")["source_sha256"]
    for mode in ("qualification", "abba", "baab"):
        data, run = read(f"micro-{mode}/probe.json"), read(f"micro-{mode}/run.json")
        assert data["passed"]
        assert run["exit_code"] == 0
        assert data["exact_ntt_maps"] == 6
        assert data["transfer_cases"] == 30
        assert data["max_abs_error"] <= data["tolerance"] == 1e-6
        assert data["evaluator_decryptions"] == 0
        for path, digest in run["source_sha256"].items():
            assert primitive_sources[path] == digest, path
        max_transfer_error = max(max_transfer_error, data["max_abs_error"])
        if mode != "qualification":
            micro_samples += [r for r in data["samples"] if r["iteration"] >= 0]
    micro = {}
    for small in (False, True):
        rows = [r for r in micro_samples if r["small"] == small]
        assert len(rows) == 4
        micro["candidate" if small else "baseline"] = {
            k: mean(r[k] for r in rows)
            for k in ("ordinary_seconds", "up_seconds", "refresh_seconds", "down_seconds")
        }
    order = ("baseline", "candidate", "candidate", "baseline")
    prefixes = [inference(f"prefix-{i}-{mode}", False) for i, mode in enumerate(order)]
    full = [inference(f"full-{i}-{mode}", True) for i, mode in enumerate(order)]
    assert full[1]["binary_sha256"] == full[2]["binary_sha256"]
    assert prefixes[1]["binary_sha256"] == prefixes[2]["binary_sha256"]
    assert read("default-path-regression/run.json")["binary_sha256"] == full[1]["binary_sha256"]
    total = aggregate(full)
    result = {
        "stage": "gpu-dual-ring-report",
        "version": "0.5.0",
        "repo_commit": source["repo_commit"],
        "passed": True,
        "measurement_scope": {
            "artifact_level_report": True,
            "full_model_correctness_claimed": True,
            "claim": (
                "One GPU dual-ring mechanism, fixed-payload Mamba-3 SISO 187M; "
                "no arbitrary-prompt, full Mamba-2, production protocol or "
                "security-equivalence claim."
            ),
            "security": "not-set",
            "hardware": "DGX Spark GB10, CUDA 13, affinity 15-19, OMP 4",
            "layers": 12,
            "encrypted_evaluations": 5,
            "generated_tokens": 4,
        },
        "micro_component_means": micro,
        "micro_max_error": max_transfer_error,
        "micro_timing_scope": (
            "Component sum excludes encryption, validation, and the artificial "
            "fixture level drop; full model timing includes its complete evaluator."
        ),
        "prefix_samples": prefixes,
        "prefix": aggregate(prefixes),
        "full_samples": full,
        "full": total,
        "adopted_opt_in": total["reduction_percent"] > 0,
        "native_seconds_per_generated_token": total["candidate"]["eval_seconds"] / 4,
        "process_seconds_per_generated_token": total["candidate"]["wall_seconds"] / 4,
    }
    result["files_sha256"] = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p.name not in ("summary.json", "README.md")
        and "__pycache__" not in p.parts
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "full": total}, indent=2))


if __name__ == "__main__":
    main()
