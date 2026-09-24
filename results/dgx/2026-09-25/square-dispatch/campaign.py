"""Exact square parity first; one fixed ABBA prefix per architecture, no full run."""
import json
import math
import subprocess
import traceback
from statistics import mean

from runlib import ROOT, base_env, run, save, sha, stamp

PREVIOUS = ROOT.parent / "owned-arithmetic-20260924"
M3 = ROOT.parent / "mamba3-20260924/lm-layer1"
FLAGS = ["--planned-refresh", "--batch-refresh", "--inplace-ops", "--gpu-plaintext-ntt",
         "--profile-evaluation", "--naf-rotations", "--reuse-dead-inputs", "--direct-plaintext-upload",
         "--compact-weights", "--move-plaintext-coefficients", "--borrow-plaintext-upload",
         "--bsgs-routing-stages", "--cache-plaintexts"]
M3_KEYS = ("nodes", "evaluated_nodes", "bootstraps", "ct_ct_mul", "ct_pt_mul", "rotations",
           "refresh_rotations", "logical_refreshes", "refresh_batches", "ring_dimension", "slots",
           "depth", "scale_bits", "bootstrap_passes", "refresh_policy", "batch_refresh",
           "public_weight_count", "public_weight_bytes", "host_encodes", "plaintext_cache_hits",
           "plaintext_cache_misses", "optimized_routing_stages", "routing_stage_rotations_saved")
M2_KEYS = ("parameters", "ckks_levels", "operation_counts", "operation_counts_by_token", "phase_operation_counts")
references = {}

def read(path):
    return json.loads(path.read_text())

def metadata(candidate):
    source = ROOT / ("compiled-sources.json" if candidate else "baseline-sources.json")
    return {"candidate": candidate, "source_manifest_path": str(source),
            "source_manifest_sha256": sha(source), "campaign_sha256": sha(__file__)}

def m3(name, candidate):
    manifest = read(M3 / "manifest.json")
    assert all(sha(M3 / p) == h for p, h in manifest["files_sha256"].items())
    binary = ROOT / "build/packed_fideslib" if candidate else PREVIOUS / "build-final/packed_fideslib"
    command = ["taskset", "-c", "15-19", binary, M3 / "program.txt", "{output}/native.json", ".001", ".001"]
    command += FLAGS + ["--client-head", M3 / "client_head.f32"]
    env = base_env()
    env["OMP_NUM_THREADS"] = "4"
    def check(d):
        checks = {
            "native": d["passed"], "finite": d["non_finite"] == 0,
            "gates": d["polynomial_tolerance"] == d["exact_tolerance"] == .001,
            "errors": all(math.isfinite(d[k]) and 0 <= d[k] <= .001 for k in
                          ("max_abs_error_vs_polynomial", "max_abs_error_vs_exact")),
            "generation": d["generated_token_ids"] == manifest["exact_token_ids"],
            "no_intermediate_decryptions": d["evaluation_decryptions"] == 0,
        }
        if "m3" in references:
            checks["same_operations"] = all(d[k] == references["m3"][k] for k in M3_KEYS)
        if candidate:
            calls = d["square_arithmetic_calls"]
            checks["square_dispatch"] = calls > 0 and (
                d["square_arithmetic_reused_inputs"] + d["square_arithmetic_cloned_inputs"] == calls)
        return checks
    run(name, command, env, 600, check,
        {**metadata(candidate), "payload": str(M3), "manifest_sha256": sha(M3 / "manifest.json"),
         "program_sha256": sha(M3 / "program.txt")})

def m2(name, candidate):
    env = base_env()
    env.update(read(ROOT / "environment.json"))
    binary = (ROOT if candidate else PREVIOUS) / "build/stage1_mamba2_decode_fideslib"
    env.update(BINARY=str(binary), BINARY_SHA256=sha(binary), CUDA_LAUNCH_BLOCKING="1",
               FAST_PLAINTEXT_UPLOAD="1", GPU_PLAINTEXT_NTT="1", DIRECT_PLAINTEXT_UPLOAD="1",
               MOVE_PLAINTEXT_COEFFICIENTS="0", BORROW_PLAINTEXT_UPLOAD="1",
               AUTOREGRESSIVE_CLIENT_LOOP="0", LAYERS="2", TOKENS="2")
    def check(d):
        p, m = d["parameters"], d["measurements"]
        checks = {
            "native": d["passed"], "layers": p["n_layers_loaded"] == 2, "tokens": p["tokens"] == 2,
            "gate": p["tolerance"] == .05, "error": 0 <= m["max_abs_error"] <= .05,
            "all_outputs": len(m["per_token_decrypt_ok"]) == len(m["per_token_max_abs_error"]) == 2,
            "decrypt": all(m["per_token_decrypt_ok"]),
            "all_errors": all(math.isfinite(x) and 0 <= x <= .05 for x in m["per_token_max_abs_error"]),
            "no_intermediate_decryptions": d["measurement_scope"]["zero_intermediate_decrypts"],
        }
        if "m2" in references:
            checks["same_operations_and_levels"] = all(d[k] == references["m2"][k] for k in M2_KEYS)
        if candidate:
            s = m["square_arithmetic"]
            checks["square_dispatch"] = s["calls"] > 0 and s["reused_inputs"] + s["cloned_inputs"] == s["calls"]
        return checks
    run(name, ["bash", ROOT / "launch_m2.sh", "{output}/native.json", "2", "2"], env, 600, check, metadata(candidate))

def controls(arch, measure):
    samples = []
    for i, candidate in enumerate((False, True, True, False), 1):
        name = f"{arch}-{i}-" + ("candidate" if candidate else "base")
        measure(name, candidate)
        if arch not in references:
            references[arch] = read(ROOT / name / "native.json")
        samples.append(read(ROOT / name / "run.json")["eval_seconds"])
    a, b = mean((samples[0], samples[3])), mean(samples[1:3])
    result = {"samples_abba": samples, "base_mean_seconds": a, "candidate_mean_seconds": b,
              "reduction_percent": 100 * (1 - b / a),
              "pair_reduction_percent": [100 * (1 - samples[1] / samples[0]), 100 * (1 - samples[2] / samples[3])],
              "clears_prefix_threshold": b < .995 * a,
              "scope": "Two fresh processes per variant; prefix evidence only, no full-model speedup claim."}
    save(ROOT / (arch + "-short-comparison.json"), result)
    return result

try:
    subprocess.run(["python3", str(ROOT / "preflight.py")], check=True)
    for arch in ("mamba3", "mamba2"):
        env = base_env()
        env["OMP_NUM_THREADS"] = "4"
        command = ["taskset", "-c", "15-19", ROOT / "build/owned_arithmetic_probe", "{output}/native.json"]
        if arch == "mamba2":
            command.append("--mamba2")
        run("probe-rns-" + arch, command, env, 300, lambda d: {
            "native": d["passed"], "existing_cases": d["cases"] == 96,
            "existing_exact": d["exact_rns_results"], "existing_live": d["live_inputs_unchanged"],
            "square_cases": d["square_cases"] == 36, "square_exact": d["square_exact_rns_results"],
            "square_live": d["square_live_inputs_unchanged"],
            "ownership": d["square_reused_inputs"] == 12 and d["square_cloned_inputs"] == 24,
        })
    comparisons = {"m3": controls("m3", m3), "m2": controls("m2", m2)}
    save(ROOT / "short-comparison.json", comparisons)
    result = {"passed": True, "short": comparisons, "full": {"skipped": True, "reason": "Outside this bounded trial."}}
except Exception:
    result = {"passed": False, "error": traceback.format_exc()}
result["finished_utc"] = stamp()
save(ROOT / "completion.json", result)
print(json.dumps(result), flush=True)
