"""Recompute prefix comparisons and verify their retained identities and gates."""
import hashlib
import json
from pathlib import Path
from statistics import mean
import tarfile

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert read(ROOT / "completion.json")["passed"]
for prefix in ("baseline", "compiled"):
    manifest = read(ROOT / f"{prefix}-sources.json")
    with tarfile.open(ROOT / f"{prefix}-sources.tar.gz") as archive:
        members = {x.name: x for x in archive.getmembers() if x.isfile()}
        assert set(members) == set(manifest["files"])
        for path, expected in manifest["files"].items():
            assert hashlib.sha256(archive.extractfile(members[path]).read()).hexdigest() == expected

provenance = read(ROOT / "target-provenance.json")
baselines = list(provenance["baseline_binaries_sha256"].values())
candidates = provenance["candidate_binaries_sha256"]
for arch in ("mamba3", "mamba2"):
    folder = ROOT / ("probe-rns-" + arch)
    record, native = read(folder / "run.json"), read(folder / "native.json")
    assert record["passed"] and all(record["checks"].values())
    assert record["native_sha256"] == sha(folder / "native.json")
    assert record["binary_sha256"] == candidates["owned_arithmetic_probe"]
    assert native["square_cases"] == 36 and native["cases"] == 96

out = {"gpu_cases": 264, "square_gpu_cases": 72, "architectures": {}}
for arch, binary in (("m3", "packed_fideslib"), ("m2", "stage1_mamba2_decode_fideslib")):
    rows = []
    for i, candidate in enumerate((False, True, True, False), 1):
        folder = ROOT / f"{arch}-{i}-{'candidate' if candidate else 'base'}"
        record, native = read(folder / "run.json"), read(folder / "native.json")
        assert record["passed"] and all(record["checks"].values())
        assert record["native_sha256"] == sha(folder / "native.json")
        assert record["candidate"] == candidate
        assert record["binary_sha256"] in ([candidates[binary]] if candidate else baselines)
        assert record["source_manifest_sha256"] == sha(ROOT / ("compiled-sources.json" if candidate else "baseline-sources.json"))
        if arch == "m3":
            error = {"exact": native["max_abs_error_vs_exact"], "polynomial": native["max_abs_error_vs_polynomial"]}
            square = {key: native.get("square_arithmetic_" + key, 0) for key in ("calls", "reused_inputs", "cloned_inputs")}
            counts = {key: native[key] for key in ("ct_ct_mul", "ct_pt_mul", "bootstraps", "rotations", "host_encodes")}
            tokens = native["generated_token_ids"]
        else:
            error = {"polynomial": native["measurements"]["max_abs_error"]}
            square = native["measurements"].get("square_arithmetic", {})
            counts = native["operation_counts"]
            tokens = None
        rows.append({"name": folder.name, "candidate": candidate, "eval_seconds": record["eval_seconds"],
                     "wall_seconds": record["wall_seconds"], "error": error, "square": square,
                     "counts": counts, "generated_token_ids": tokens})
    a, b = mean(rows[i]["eval_seconds"] for i in (0, 3)), mean(rows[i]["eval_seconds"] for i in (1, 2))
    reduction = 100 * (1 - b / a)
    assert reduction == read(ROOT / (arch + "-short-comparison.json"))["reduction_percent"]
    out["architectures"][arch] = {"samples": rows, "baseline_mean_seconds": a,
                                  "candidate_mean_seconds": b, "reduction_percent": reduction}
out["passed"] = True
out["scope"] = "Prefix evidence only; no full-model timing or significance claim."
(ROOT / "verified-summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: {f: v for f, v in d.items() if f != "samples"}
                  for k, d in out["architectures"].items()}, indent=2))
