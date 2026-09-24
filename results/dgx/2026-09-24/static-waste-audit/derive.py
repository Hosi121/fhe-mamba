"""Reproduce static candidate counts using preserved inventories and prior runs."""

import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent

    def read(name):
        return json.loads((here / name).read_text())

    inventory = read("inventory.json")
    m3, m2 = read("m3-native.json"), read("m2-native.json")
    provenance = read("provenance.json")
    checks = {}
    for model in ("m3", "m2"):
        run = read(f"{model}-run.json")
        checks[f"{model}_raw_hash"] = sha(here / f"{model}-native.json") == run["native_sha256"]
        checks[f"{model}_existing_run_passed"] = (
            run["returncode"] == 0 and run["passed"] and all(run["checks"].values())
        )
    checks["source_manifest"] = (
        sha(here / "compiled-sources.json")
        == provenance["source_manifest_sha256"]
        == read("m3-run.json")["source_manifest_sha256"]
    )
    checks["payload_identity"] = (
        provenance["input_program_sha256"] == read("m3-run.json")["program_sha256"]
    )
    linear, routing, cheb = (inventory[k] for k in ("linears", "routing", "chebyshev"))
    stats = m3["operation_stats"]
    checks["static_ct_ct_matches_recorded"] = (
        inventory["model_ctct_from_static_calls"] == m3["ct_ct_mul"]
    )
    checks["static_cheb_calls_match_recorded"] = cheb["calls"] == stats["cheb"]["nodes"]
    checks["static_linears_match_recorded"] = (
        linear["calls"] == stats["linear"]["nodes"] + stats["linear_ref"]["nodes"]
    )
    checks["static_routing_calls_match_recorded"] = routing["calls"] == sum(
        stats[k]["nodes"] for k in ("gather", "scatter", "repeat", "sum")
    )
    checks["static_routing_stages_match_recorded"] = (
        routing["optimized_stages"] == m3["optimized_routing_stages"]
    )
    checks["m3_expected_mode"] = (
        all(
            m3[k]
            for k in (
                "inplace_ops",
                "reuse_dead_inputs",
                "bsgs_routing_stages",
                "borrow_plaintext_upload",
            )
        )
        and not m3["cache_plaintexts"]
        and m3["rotation_decomposition"] == "naf"
    )
    checks["m3_existing_error_gates"] = (
        m3["non_finite"] == m3["evaluation_decryptions"] == 0
        and m3["max_abs_error_vs_exact"] <= m3["exact_tolerance"] == 0.001
        and m3["max_abs_error_vs_polynomial"] <= m3["polynomial_tolerance"] == 0.001
    )
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    additions = {
        "linear_inner": linear["inner_adds"],
        "linear_outer": linear["outer_adds"],
        "routing": routing["accumulator_adds"],
        "cheb_leaves": cheb["owned_leaf_adds"],
        "cheb_recombines": cheb["recombines"],
    }
    sync_calls = {
        "rotation_steps": m3["rotations"],
        "ct_pt_muls": m3["ct_pt_mul"],
        "ct_ct_muls": m3["ct_ct_mul"],
        "identified_accumulator_adds": sum(additions.values()),
        "evaluated_nodes": m3["evaluated_nodes"],
        "single_device_fast_uploads": m3["fast_plaintext_uploads"],
    }
    saved_muls = cheb["ctct"] - cheb["tuned_ctct"]
    cache = m2["measurements"]["pt_cache"]
    result = {
        "scope": (
            "Static opportunities, not measured savings; existing completed full runs "
            "supply counts and phase ceilings."
        ),
        "checks": checks,
        "m3_accumulator_adds": additions,
        "m3_outer_clone_candidates": 2 * sum(additions.values()),
        "m2_joint_multiply_result_copy_candidates_minimum": m2["phase_operation_counts"][
            "joint_selective_gates"
        ]["ct_ct_mul"],
        "m3_explicit_sync_call_components": sync_calls,
        "m3_explicit_sync_calls_minimum": sum(sync_calls.values()),
        "m3_repeated_weight_diagonal_identities": linear["diagonal_encodes"]
        - linear["unique_diagonal_identities"],
        "weight_reuse_caveat": (
            "Identity count ignores level/scale compatibility; it is an upper opportunity "
            "count, not proven cache hits."
        ),
        "illustrative_full_weight_rns_cache": {
            "level": 21,
            "residues": m3["depth"] + 1 - 21,
            "single_copy_gib": linear["unique_diagonal_identities"]
            * m3["ring_dimension"]
            * 8
            * (m3["depth"] + 1 - 21)
            / 2**30,
            "scope": (
                "One assumed encoding level, no metadata; CPU and GPU residency can duplicate this."
            ),
        },
        "m3_existing_profile": {
            "eval_seconds": m3["eval_seconds"],
            **{
                key: {"seconds": m3[key], "percent_of_eval": 100 * m3[key] / m3["eval_seconds"]}
                for key in (
                    "bootstrap_seconds",
                    "host_encoding_seconds",
                    "plaintext_upload_seconds",
                    "mask_preparation_seconds",
                )
            },
            "caveat": (
                "Nested counters; do not add these percentages or treat them as removable overhead."
            ),
        },
        "m2_cache": {
            key: cache[key]
            for key in (
                "budget_gib",
                "bytes_cached_gib",
                "entries_cached",
                "replicated_cache_hits",
                "replicated_cache_misses",
            )
        },
        "m2_cache_caveat": (
            "Budget estimates one RNS array per entry; uploaded entries can retain a CPU "
            "and GPU array. RSS savings are unmeasured."
        ),
        "deprioritized": {
            "m2_composite_rotation_steps": m2["operation_counts"]["rotations_composite_steps"],
            "m3_cheb_zero_scalar_adds": cheb["zero_scalar_adds"],
            "m3_ps_ct_ct_savings": saved_muls,
            "m3_ps_model_ct_ct_percent": 100 * saved_muls / m3["ct_ct_mul"],
            "m3_ps_scalar_savings": cheb["scalar_multiplies"] - cheb["tuned_scalar"],
            "ps_caveat": (
                "Same symbolic multiply depth and current coefficient cutoff; numerical "
                "parity and runtime untested."
            ),
        },
    }
    (here / "findings.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checks_passed": len(checks),
                "clone_candidates": result["m3_outer_clone_candidates"],
                "sync_call_lower_bound": result["m3_explicit_sync_calls_minimum"],
            }
        )
    )


if __name__ == "__main__":
    main()
