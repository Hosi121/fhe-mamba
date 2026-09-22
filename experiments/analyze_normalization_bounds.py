#!/usr/bin/env python3
"""Derive necessary polynomial degrees and summarize recorded RMS refresh costs.

This is an offline analysis, not a benchmark or a new approximation certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path

from manage_dgx_build import payload_sha256


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def necessary_degree(lo: float, hi: float, tolerance: float, precision: int) -> int:
    """Chebyshev extrapolation bound; Decimal evaluates the analytic formula.

    If |sqrt(v)*p(v)-1| <= e, r(v)=v*p(v)^2-1 is bounded by delta=2e+e^2
    on [lo, hi], but r(0)=-1. For m=2*degree(p)+1, Chebyshev extremality
    requires delta*cosh(m*acosh((hi+lo)/(hi-lo))) >= 1.
    """
    if not 0 < lo < hi or not 0 < tolerance < 1:
        raise ValueError("positive interval and tolerance in (0,1) required")
    with localcontext() as context:
        context.prec = precision
        lower, upper, error = map(Decimal.from_float, (lo, hi, tolerance))
        delta = 2 * error + error * error
        if delta >= 1:
            return 0
        c = (upper + lower) / (upper - lower)

        def acosh(x: Decimal) -> Decimal:
            return (x + (x * x - 1).sqrt()).ln()

        threshold = (acosh(1 / delta) / acosh(c) - 1) / 2
        return max(0, int(threshold.to_integral_value(rounding=ROUND_CEILING)))


def multiplication_depth(degree: int) -> int:
    return (max(1, degree) - 1).bit_length()


def analyze(native_path: Path, payload: Path, bundle_path: Path) -> dict:
    native = json.loads(native_path.read_text())
    bundle = json.loads(bundle_path.read_text())
    chain_path = payload / "chain.json"
    chain = json.loads(chain_path.read_text())
    payload_hash = payload_sha256(payload)
    assert payload_hash == native["input_payload_sha256"]
    assert chain["normalization_bundle_sha256"] == sha256(bundle_path)
    recipes = {}
    source_paths = [native_path, bundle_path, chain_path]
    for index, name in enumerate(chain["layer_dirs"]):
        meta_path = payload / name / "meta.json"
        source_paths.append(meta_path)
        meta = json.loads(meta_path.read_text())
        assert meta["normalization_bundle_sha256"] == sha256(bundle_path)
        for site in ("rms_invsqrt", "gated_rms_invsqrt"):
            recipes[index, site] = meta["polys"][site]
    recipes[chain["n_layers"], "rms_invsqrt"] = chain["final_norm_poly"]

    operators = []
    for operator in bundle["operators"]:
        recipe = operator["recipe"]
        assert recipes[operator["layer"], operator["site"]] == recipe
        assert recipe["kind"] == "scaled-goldschmidt-invsqrt-v1"
        assert recipe["final_recomputed_newton"] is True
        assert recipe["seed"] > 0
        assert all(a > 0 and b > 0 for a, b in recipe["coefficients"])
        stages = len(recipe["coefficients"])
        degree = (3 ** (stages + 1) - 1) // 2
        minimum = necessary_degree(recipe["lo"], recipe["hi"], recipe["tolerance"], 80)
        assert minimum == necessary_degree(recipe["lo"], recipe["hi"], recipe["tolerance"], 120)
        assert degree >= minimum
        operators.append(
            {
                "layer": operator["layer"],
                "site": operator["site"],
                "interval": [recipe["lo"], recipe["hi"]],
                "condition_ratio": recipe["hi"] / recipe["lo"],
                "relative_error_target": recipe["tolerance"],
                "scheduled_stages": stages,
                "fixed_recipe_degree": degree,
                "fixed_recipe_minimum_ct_ct_depth": multiplication_depth(degree),
                "balanced_recipe_ct_ct_depth": 2 * stages,
                "any_uniform_polynomial_minimum_degree": minimum,
                "any_uniform_polynomial_minimum_ct_ct_depth": multiplication_depth(minimum),
            }
        )
    assert len(operators) == len(recipes)

    stages = defaultdict(lambda: {"events": 0, "physical_calls": 0, "seconds": 0.0})
    for event in native["measurements"]["bootstrap_events"]:
        if ".scheduled_norm." not in event["checkpoint"]:
            continue
        stage = event["checkpoint"].rsplit(".", 1)[1]
        group = stages[stage]
        group["events"] += 1
        group["physical_calls"] += event["physical_bootstraps"]
        group["seconds"] += event["seconds"]
    refresh_seconds = sum(value["seconds"] for value in stages.values())
    total = native["timing"]["eval_seconds"]
    scenarios = []
    for name, saved in (
        ("all_internal_normalization_refresh_cost_zero", refresh_seconds),
        ("stage_11_internal_refresh_cost_zero", stages.get("11", {}).get("seconds", 0)),
    ):
        scenarios.append(
            {
                "assumption": name,
                "seconds_saved": saved,
                "remaining_eval_seconds": total - saved,
                "reduction_fraction": saved / total,
                "accuracy_or_feasibility_proved": False,
            }
        )
    return {
        "analysis_kind": "normalization-necessary-bounds-not-a-benchmark",
        "script_sha256": sha256(Path(__file__)),
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "payload_sha256": payload_hash,
        "operator_count": len(operators),
        "degree_proof": (
            "d_next=3*d+1 from a*y-b*v*y^3, starting d=0; n scheduled steps plus one repair"
        ),
        "uniform_approximation_bound": {
            "assumption": "real polynomial p(v), all v in [lo,hi], |sqrt(v)*p(v)-1| <= epsilon",
            "residual": (
                "r(v)=v*p(v)^2-1; degree<=2*d+1; r(0)=-1; sup|r|<=delta=2*epsilon+epsilon^2"
            ),
            "inequality": "1 <= delta*cosh((2*d+1)*acosh((hi+lo)/(hi-lo)))",
            "minimum_degree": "ceil((acosh(1/delta)/acosh((hi+lo)/(hi-lo))-1)/2)",
            "numerical_evaluation": (
                "80/120-digit Decimal evaluations agree on the integer; "
                "not an outward interval proof"
            ),
            "sufficiency_claimed": False,
        },
        "scope": {
            "encrypted_execution": False,
            "new_approximation_constructed": False,
            "includes_ckks_error": False,
            "includes_scalar_rescaling_or_refresh_depth": False,
            "bootstrap_minimum_proved": False,
            "global_domain_membership_proved": False,
            "bound_applies_to": (
                "polynomials in the variance; unrestricted ordinary binary products "
                "and free public scalar operations"
            ),
        },
        "operators": operators,
        "recorded_internal_refreshes": {
            "includes_final_normalization": True,
            "by_stage": dict(stages),
            "physical_calls": sum(value["physical_calls"] for value in stages.values()),
            "seconds": refresh_seconds,
        },
        "conditional_timing_scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.native, args.payload, args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("operator_count", "recorded_internal_refreshes")}))


if __name__ == "__main__":
    main()
