"""Validate saved GPU-encoding experiments and regenerate derived comparisons."""

import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path
from statistics import mean, median

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
IDENTITY = ("binary_sha256", "program_sha256", "manifest_sha256")
COMMON = (
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
    "cache_plaintexts",
    "host_encodes",
    "evaluation_decryptions",
    "client_output_decrypt_count",
    "generated_token_ids",
    "exact_tolerance",
    "polynomial_tolerance",
)


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def gate(path, mode, tokens, cache=False, batch=16):
    run, row = read(path / "run.json"), read(path / "native.json")
    assert run["passed"] is True, path
    assert row["passed"] is True, path
    assert run["returncode"] == 0, path
    assert run["timed_out"] is False, path
    assert run["native_sha256"] == sha(path / "native.json"), path
    assert row["encrypted"] is True, path
    assert row["non_finite"] == 0, path
    for reference in ("exact", "polynomial"):
        error = row[f"max_abs_error_vs_{reference}"]
        assert math.isfinite(error)
        assert 0 <= error <= row[f"{reference}_tolerance"] == 0.001
        assert max(r[reference] for r in row["per_output_errors"]) == error
    assert row["evaluation_decryptions"] == 0
    assert row["generated_token_ids"] == tokens
    assert row["client_output_decrypt_count"] == len(tokens)
    assert row["cache_plaintexts"] is cache
    assert row["gpu_plaintext_ntt"] == (mode == "c")
    assert row["fast_plaintext_upload"] == (mode in ("b", "c"))
    assert row["gpu_ntt_encodes"] == (row["host_encodes"] if mode == "c" else 0)
    assert row["fast_plaintext_uploads"] == (row["host_encodes"] if mode != "a" else 0)
    if "plaintext_ntt_batch" in row:
        assert row["plaintext_ntt_batch"] == (batch if mode == "c" else 0)
    assert row["inplace_ops"]
    assert row["profile_evaluation"]
    return run, row


def comparison(directory, names, tokens, batch=16):
    runs, rows = zip(
        *(gate(Path(directory) / n, n[0], tokens, batch=batch) for n in names), strict=True
    )
    for key in IDENTITY:
        assert len({run[key] for run in runs}) == 1, (directory, key)
    for key in COMMON:
        assert all(row[key] == rows[0][key] for row in rows), (directory, key)
    manifest_path = (
        "../mamba3-trained-generation/full-payload/manifest.json"
        if directory == "full"
        else "../mamba3-microkernels/prefix-payload/manifest.json"
    )
    manifest = read(manifest_path)
    assert runs[0]["manifest_sha256"] == sha(manifest_path)
    assert runs[0]["program_sha256"] == manifest["files_sha256"]["program.txt"]
    assert all(len(row["per_output_errors"]) == len(manifest["output_names"]) for row in rows)
    groups = {}
    for mode in sorted({n[0] for n in names}):
        selected = [row for name, row in zip(names, rows, strict=True) if name[0] == mode]
        groups[mode] = {
            "eval_seconds": [row["eval_seconds"] for row in selected],
            "mean_seconds": mean(row["eval_seconds"] for row in selected),
            "process_wall_seconds": [
                run["wall_seconds"]
                for name, run in zip(names, runs, strict=True)
                if name[0] == mode
            ],
            "host_encoding_mean_seconds": mean(row["host_encoding_seconds"] for row in selected),
            "upload_mean_seconds": mean(row["plaintext_upload_seconds"] for row in selected),
            "setup_seconds": [row["setup_seconds"] for row in selected],
            "peak_rss_gib": [row["peak_rss_gib"] for row in selected],
        }
    a = groups["a"]["mean_seconds"]
    for group in groups.values():
        group["reduction_percent"] = 100 * (1 - group["mean_seconds"] / a)
        group["speedup"] = a / group["mean_seconds"]
    return {
        "order": list(names),
        "identity": {k: runs[0][k] for k in IDENTITY},
        "common": {k: rows[0][k] for k in COMMON},
        "groups": groups,
        "max_abs_error_vs_exact": max(r["max_abs_error_vs_exact"] for r in rows),
        "max_abs_error_vs_polynomial": max(r["max_abs_error_vs_polynomial"] for r in rows),
    }


# Byte-bind the archived implementation, including the failed first allocation.
for version in ("initial", "v2", "final"):
    files = read(f"compiled-sources-{version}.json")
    with tarfile.open(ROOT / f"compiled-sources-{version}.tar.gz") as archive:
        assert set(archive.getnames()) == set(files)
        for name, digest in files.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest

failure = read("probe-initial/run.json")
assert failure["passed"] is False
assert not (ROOT / "probe-initial/native.json").exists()
probe = read("probe-final/native.json")
probe_run = read("probe-final/run.json")
assert probe_run["native_sha256"] == sha("probe-final/native.json")
assert probe["passed"]
assert probe_run["passed"]
assert probe["exact_rns_cases"] == 160
assert probe["ntt_batches"] == [1, 4, 16, 64]
assert math.isfinite(probe["max_abs_error"])
assert probe["max_abs_error"] < 1e-6
micro = []
for level in (21, 26, 34):
    for mode, name in enumerate(probe["modes"]):
        samples = [s for s in probe["samples"] if s["level"] == level and s["mode"] == mode]
        assert len(samples) == 10
        assert all(
            math.isfinite(s["max_abs_error"]) and 0 <= s["max_abs_error"] < 1e-6 for s in samples
        )
        timed = [s for s in samples if s["iteration"] >= 0]
        assert len(timed) == 8
        micro.append(
            {
                "level": level,
                "mode": name,
                "samples": len(timed),
                "median_encode_ms": median(s["encode_ms"] for s in timed),
                "median_upload_ms": median(s["upload_ms"] for s in timed),
                "median_total_ms": median(s["encode_ms"] + s["upload_ms"] for s in timed),
            }
        )

summary = {
    "schema": "fhemamba-gpu-encoding-comparison-v1",
    "prefix_v2": comparison("prefix-v2", ("a1", "b2", "c3", "c4", "b5", "a6"), [], batch=1),
    "prefix_final": comparison("prefix", ("a1", "c2", "c3", "a4"), []),
    "full": comparison("full", ("a1", "c2"), [315, 279, 1614, 315]),
    "micro_probe": {
        "exact_rns_input_cases": probe["exact_rns_cases"],
        "ntt_batches": probe["ntt_batches"],
        "encrypted_max_abs_error": probe["max_abs_error"],
        "timing": micro,
    },
    "regressions": {},
    "limitations": [
        "Full generation has one fresh-key run per mode on one frozen prompt; "
        "prefix has two per mode.",
        "Host timers enabled in both model modes, with no attached profiler.",
        "Full comparison uses identical CPU affinity and in-place scratch; cache is disabled.",
        "Standard upload time is uninstrumented in model JSON, not zero work.",
        "Evaluation excludes setup, final token selection and diagnostic reference checks; "
        "process wall includes them. Payload hash verification is charged only to the ledger.",
        "Inline client and security=not-set; no longer-context or new-prompt guarantee.",
    ],
}
for name, tokens, cache, outputs in (
    ("layer", [6864, 6864], False, 3),
    ("synthetic", [], True, 20),
):
    run, row = gate(Path(name), "c", tokens, cache=cache)
    assert len(row["per_output_errors"]) == outputs
    summary["regressions"][name] = {
        "identity": {key: run[key] for key in IDENTITY},
        **{
            key: row[key]
            for key in (
                "eval_seconds",
                "max_abs_error_vs_exact",
                "max_abs_error_vs_polynomial",
                "generated_token_ids",
                "evaluation_decryptions",
                "plaintext_cache_hits",
                "gpu_ntt_encodes",
                "plaintext_ntt_batch",
            )
        },
    }

for label, modes, cache in (
    ("prefix", "a,b,c,c,b,a", False),
    ("final-prefix", "a,c,c,a", False),
    ("full", "a,c", False),
    ("layer", "c", False),
    ("synthetic", "c", True),
):
    affinity = read(f"gpu-encoding-{label}-affinity.json")
    assert affinity["allowed_cpus"] == [15, 16, 17, 18, 19]
    assert affinity["omp_num_threads"] == 4
    assert affinity["profile"] is True
    assert affinity["modes"] == modes
    assert affinity["cache"] == cache

# Distinguish the complete backbone from the separately labelled one-layer probe.
assert all(read(Path("full") / name / "run.json")["complete_backbone"] for name in ("a1", "c2"))
assert not read("layer/run.json")["complete_backbone"]
ledger = read("budget.json")
assert math.isclose(
    ledger["used_seconds"], sum(r["charged_seconds"] for r in ledger["runs"]), abs_tol=1e-6
)
summary["campaign"] = {
    "used_seconds": ledger["used_seconds"],
    "attempts": len(ledger["runs"]),
    "review_window_seconds": ledger["limit_seconds"],
}
(ROOT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
