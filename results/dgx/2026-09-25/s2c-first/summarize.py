"""Verify frozen sources and numerical gates; keep prefix and full times separate."""

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
    ops = data["operation_stats"]

    def ordinary(names):
        return sum(ops[n]["seconds"] - ops[n]["bootstrap_seconds"] for n in names if n in ops)

    result = {
        "refresh": data["bootstrap_seconds"],
        "linear": ordinary(["linear", "linear_ref"]),
        "polynomial": ordinary(["cheb"]),
        "routing": ordinary(["gather", "scatter", "repeat", "sum"]),
    }
    result["other_and_timer_gaps"] = data["eval_seconds"] - sum(result.values())
    return result


def inference(name, full=False):
    path = ROOT / name
    run, data = read(path / "run.json"), read(path / "native.json")
    assert run["exit_code"] == 0, name
    assert data["passed"], name
    assert data["evaluation_decryptions"] == data["non_finite"] == 0
    assert data["polynomial_tolerance"] == data["exact_tolerance"] == 0.001
    assert (data["ring_dimension"], data["slots"], data["depth"], data["scale_bits"]) == (
        65536,
        32768,
        44,
        59,
    )
    assert data["security"] == "not-set"
    assert data["bootstrap_passes"] == 2
    assert data["generated_token_ids"] == ([315, 279, 1614, 315] if full else [6864, 6864])
    for key in ("max_abs_error_vs_exact", "max_abs_error_vs_polynomial"):
        assert math.isfinite(data[key])
        assert 0 <= data[key] <= 0.001
    assert all(
        math.isfinite(v) and 0 <= v <= 0.001 for e in data["per_output_errors"] for v in e.values()
    )
    keys = [
        "eval_seconds",
        "bootstrap_seconds",
        "bootstraps",
        "logical_refreshes",
        "ct_ct_mul",
        "ct_pt_mul",
        "rotations",
        "host_encoding_seconds",
        "peak_rss_gib",
        "max_abs_error_vs_exact",
        "max_abs_error_vs_polynomial",
    ]
    result = {k: data[k] for k in keys}
    result["categories_seconds"] = categories(data)
    result["native_sha256"] = sha(path / "native.json")
    result["process_wall_seconds"] = run["wall_seconds"]
    return result


def main():
    archives = {}
    for label in ("trial", "final"):
        manifest = read(ROOT / f"{label}-source-manifest.json")
        archive = ROOT / f"{label}-sources.tar.gz"
        with tarfile.open(archive) as handle:
            members = {
                m.name: hashlib.sha256(handle.extractfile(m).read()).hexdigest()
                for m in handle
                if m.isfile()
            }
        for group, files in manifest.items():
            for name, digest in files.items():
                assert members[f"{group}/{name}"] == digest, (label, group, name)
        archives[label] = {
            "sha256": sha(archive),
            "verified_files": sum(map(len, manifest.values())),
        }

    probe = {}
    for name in ("initial-baseline", "initial-s2c", "s2c-level8", "s2c-level38"):
        run, data = read(ROOT / name / "run.json"), read(ROOT / name / "native.json")
        assert run["exit_code"] == 0
        assert data["passed"]
        assert all(
            math.isfinite(s["two_pass_error"]) and s["two_pass_error"] <= 0.001
            for s in data["samples"]
        )
        probe[name] = {"max_error": data["max_error"], "samples": data["samples"]}
    rejected = ROOT / "s2c-level39-rejected"
    assert read(rejected / "run.json")["exit_code"] == 3
    assert "requires five remaining levels" in (rejected / "run.log").read_text()

    prefixes = {}
    for mode, indices in (("baseline", [0, 3]), ("s2c", [1, 2])):
        samples = [inference(f"prefix-{i}-{mode}") for i in indices]
        prefixes[mode] = {
            "samples": samples,
            "mean_eval_seconds": mean(s["eval_seconds"] for s in samples),
        }
    prefix_reduction = 100 * (
        1 - prefixes["s2c"]["mean_eval_seconds"] / prefixes["baseline"]["mean_eval_seconds"]
    )
    result = {
        "scope": "One circuit; measured prefix and full workloads remain separate",
        "archives": archives,
        "probes": probe,
        "prefixes": prefixes,
        "prefix_reduction_percent": prefix_reduction,
    }

    if (ROOT / "final-candidate-full/native.json").exists():
        baseline = inference("final-baseline-full", full=True)
        candidate = inference("final-candidate-full", full=True)
        build = read(ROOT / "final-build.json")
        for mode in ("baseline", "candidate"):
            run = read(ROOT / f"final-{mode}-full/run.json")
            assert run["binary_sha256"] == build[f"{mode}_binary_sha256"]
            parity = read(ROOT / f"final-{mode}-full/parity.json")
            manifest = read(ROOT / "payload-manifests/lm-full/manifest.json")
            assert parity["program_sha256"] == manifest["files_sha256"]["program.txt"]
            assert parity["manifest_sha256"] == sha(
                ROOT / "payload-manifests/lm-full/manifest.json"
            )
        result["full"] = {
            "baseline": baseline,
            "candidate": candidate,
            "reduction_percent": 100 * (1 - candidate["eval_seconds"] / baseline["eval_seconds"]),
        }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "prefix_reduction_percent": prefix_reduction,
                "full": result.get("full", "not yet collected"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
