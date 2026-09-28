"""Shared recurrence and refresh comparisons for CPU CKKS recipes."""

from __future__ import annotations

import time

import numpy as np

from fhemamba.ckks_probes import CkksPrimitives

from .ckks import context_config, make_context, validate_recipe
from .ckks_curves import Curve, max_error


class RefreshProbe(CkksPrimitives):
    def __init__(self, cc, keys, recipe):
        super().__init__(
            cc,
            keys,
            batch=recipe["context"]["batch"],
            coefficient_floor=recipe["coefficient_floor"],
        )
        self.depth = recipe["context"]["depth"]
        self.towers = recipe["bootstrap"]["forced_input_towers"]
        self.bound = recipe["b_norm"]
        self.bootstrap_seconds = []

    def boot(self, ct):
        start = time.perf_counter()
        result = self.cc.EvalBootstrap(ct)
        self.bootstrap_seconds.append(time.perf_counter() - start)
        return result

    def descend(self, ct):
        # Shallow bootstrap may be a level-preserving no-op. Force exhaustion
        # with a noiseless tower drop before measuring genuine refresh noise.
        return (
            self.cc.Compress(ct, self.towers)
            if int(ct.GetLevel()) < self.depth + 1 - self.towers
            else ct
        )

    def refresh(self, ct, bound=None):
        bound = self.bound if bound is None else bound
        return self.cc.EvalMult(self.boot(self.cc.EvalMult(self.descend(ct), 1.0 / bound)), bound)


def growth_inputs(recipe):
    rng = np.random.default_rng(recipe["seed"])
    batch, steps, clip = recipe["context"]["batch"], recipe["n_steps"], recipe["state_clip"]
    state = np.clip(rng.standard_normal(batch), -clip, clip)
    decay = rng.uniform(*recipe["decay_range"], batch)
    dt = np.log(rng.uniform(*recipe["decay_range"], (steps, batch)))
    updates = np.sqrt(1.0 - decay**2) * np.clip(rng.standard_normal((steps, batch)), -clip, clip)
    scale = 1.0

    def peak(scale):
        value, maximum = state.copy(), float(np.max(np.abs(state)))
        for update in updates:
            value = value * decay + scale * update
            maximum = max(maximum, float(np.max(np.abs(value))))
        return maximum

    if (
        not 0 < clip < 0.98 * recipe["b_norm"]
        or not 0 < recipe["decay_range"][0] <= recipe["decay_range"][1] < 1
    ):
        raise ValueError("growth needs 0 < decay < 1 and state_clip < 0.98 * b_norm")
    while peak(scale) >= 0.98 * recipe["b_norm"]:
        scale *= 0.85
    return (state, decay, dt, updates * scale), scale


def run_arm(he, spec, inputs, polynomial, steps):
    state, decay, dt, updates = inputs
    mode, cc, bound = spec["mode"], he.cc, he.bound
    if mode not in {"recurrence", "meta", "compensated"}:
        raise ValueError(f"unknown growth mode: {mode}")
    extra = {}
    if spec.get("polynomial"):
        extra = {
            "poly_fit_max_abs_err_vs_exp": max_error(polynomial.plain(dt), np.exp(dt)),
            "poly": {
                "base_degree": polynomial.degree,
                "squarings": polynomial.squarings,
                "base_interval": polynomial.base_interval,
            },
        }
    ct, plain = he.encrypt(state), state.copy()
    if mode == "compensated":
        lo = he.encrypt(np.zeros(he.batch))
        extra.update(s_bits=spec["s_bits"], b_lo=spec["b_lo"], errors_hi_only=[])
    if mode == "meta":
        extra["amp_bits"] = spec["amp_bits"]
    errors, levels, notes = [], [], []
    start = time.perf_counter()
    for k in range(steps):
        if mode == "meta":
            amp = 2.0 ** spec["amp_bits"]
            x = cc.EvalMult(he.descend(ct), 1.0 / bound)
            y1 = he.boot(x)
            residual = cc.EvalMult(cc.EvalSub(x, y1), amp)
            maximum = float(np.max(np.abs(he.decrypt(residual))))
            y2 = he.boot(he.descend(residual))
            ct = cc.EvalMult(cc.EvalAdd(y1, cc.EvalMult(y2, 1.0 / amp)), bound)
            notes.append({"step": k + 1, "max_abs_r_amp": maximum, "in_range": maximum <= 1.0})
        elif mode == "compensated":
            value = cc.EvalAdd(cc.EvalMult(ct, he.encrypt(decay)), he.encrypt(updates[k]))
            hi = he.refresh(value)
            residual = cc.EvalSub(value, hi)
            lo = cc.EvalAdd(
                cc.EvalMult(lo, he.encrypt(decay)), cc.EvalMult(residual, 2.0 ** spec["s_bits"])
            )
            maximum = float(np.max(np.abs(he.decrypt(lo))))
            lo = he.refresh(lo, spec["b_lo"])
            ct, plain = hi, plain * decay + updates[k]
            notes.append(
                {
                    "step": k + 1,
                    "max_abs_h_lo_pre_refresh": maximum,
                    "lo_in_range": maximum <= spec["b_lo"],
                }
            )
        else:
            if spec.get("decay"):
                factor = (
                    polynomial.encrypted(he, he.encrypt(dt[k]))
                    if spec.get("polynomial")
                    else he.encrypt(decay)
                )
                ct = cc.EvalMult(ct, factor)
                plain = plain * (polynomial.plain(dt[k]) if spec.get("polynomial") else decay)
            if spec.get("update"):
                ct, plain = cc.EvalAdd(ct, he.encrypt(updates[k])), plain + updates[k]
            if spec["refresh"]:
                ct = he.refresh(ct)
        if mode != "recurrence" or spec["refresh"]:
            plain = (plain * (1.0 / bound)) * bound
        got = he.decrypt(ct)
        if mode == "compensated":
            got = got + 2.0 ** -spec["s_bits"] * he.decrypt(lo)
            extra["errors_hi_only"].append(max_error(he.decrypt(ct), plain))
        errors.append(max_error(got, plain))
        levels.append(int(ct.GetLevel()))
        print(
            f"{spec['name']} step {k + 1}: level {levels[-1]}, error {errors[-1]:.3e}", flush=True
        )
    return {
        "errors": errors,
        "levels": levels,
        "seconds": round(time.perf_counter() - start, 3),
        "notes": notes,
        **extra,
    }


def fit_growth(errors):
    k, values = (
        np.arange(1, len(errors) + 1, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )
    slope, intercept = np.polyfit(k, values, 1)
    residual = values - (slope * k + intercept)
    return {
        "per_step_slope": float(slope),
        "intercept": float(intercept),
        "linear_r2": 1.0
        - float(np.sum(residual**2)) / max(float(np.sum((values - values.mean()) ** 2)), 1e-300),
        "mean_step_delta": float((values[-1] - values[0]) / (len(values) - 1)),
    }


def horizon(fit, target):
    return (
        None
        if fit["per_step_slope"] <= 0
        else float((target - fit["intercept"]) / fit["per_step_slope"])
    )


def summarize(arms, specs, target):
    horizon_key = "projected_tokens_to_5e-2" if target == 0.05 else "projected_tokens_to_target"
    for arm in arms.values():
        arm["fit"] = fit_growth(arm["errors"])
        arm[horizon_key] = horizon(arm["fit"], target)
    components, dominant, dominant_arm, verdict, refresh_verdict = {}, None, None, None, None
    # Attribution requires matched baseline arms. A partial run is still useful
    # evidence, but cannot make a complete attribution claim.
    if all(key in specs and specs[key]["name"] in arms for key in "ABCDE"):
        baseline = {key: arms[specs[key]["name"]] for key in "ABCDE"}
        slope = {key: arm["fit"]["per_step_slope"] for key, arm in baseline.items()}
        components = {
            "refresh_noise": slope["A"],
            "ctct_mult_noise": slope["B"],
            "poly_eval_noise": slope["D"] - slope["C"],
        }
        dominant = max(components, key=components.get)
        dominant_arm = specs[
            {"refresh_noise": "A", "ctct_mult_noise": "B", "poly_eval_noise": "D"}[dominant]
        ]["name"]
        realistic = baseline["E"]
        verdict = (
            f"{dominant} dominates the per-token growth ({components[dominant]:.3e}/step vs"
            f" refresh {components['refresh_noise']:.3e},"
            f" ct-ct mult {components['ctct_mult_noise']:.3e},"
            f" poly-eval delta {components['poly_eval_noise']:.3e});"
            f" projected token horizon at {target:g} for the dominant arm"
            f" ({dominant_arm}): {horizon(arms[dominant_arm]['fit'], target)} tokens;"
            f" realistic arm E: {realistic['fit']['per_step_slope']:.3e}/step,"
            f" horizon {realistic[horizon_key]} tokens"
        )
    baseline_name = specs.get("A", {}).get("name")
    strategies = [s for s in specs.values() if s["mode"] != "recurrence" and s["name"] in arms]
    if baseline_name in arms and strategies:
        baseline_error = float(np.mean(arms[baseline_name]["errors"]))
        parts = [f"plain refresh A mean err {baseline_error:.3e}"]
        for spec in strategies:
            mean = float(np.mean(arms[spec["name"]]["errors"]))
            parts.append(
                f"{spec['label']} mean err {mean:.3e}"
                f" ({baseline_error / max(mean, 1e-300):.1f}x vs A)"
            )
        refresh_verdict = "; ".join(parts)
    return {
        "component_per_step_slopes": components,
        "dominant_component": dominant,
        "dominant_arm": dominant_arm,
        "verdict": verdict,
        "refresh_strategy_verdict": refresh_verdict,
    }


def run_growth(recipe, selected, previous=None):
    validate_recipe(recipe)
    if recipe["n_steps"] < 2:
        raise ValueError("growth fits require at least two steps")
    inputs, scale = growth_inputs(recipe)
    polynomial = Curve(**recipe["decay_polynomial"], coefficient_floor=recipe["coefficient_floor"])
    specs = recipe["arms"]
    cc, keys = make_context(recipe)
    he = RefreshProbe(cc, keys, recipe)
    bootstrap = recipe["bootstrap"]
    cc.EvalBootstrapSetup(
        bootstrap["level_budget"], [0, 0], he.batch, bootstrap["correction_factor"]
    )
    cc.EvalBootstrapKeyGen(keys.secretKey, he.batch)
    arms = dict(previous or {})
    for key in selected:
        arms[specs[key]["name"]] = run_arm(he, specs[key], inputs, polynomial, recipe["n_steps"])
    arms = {spec["name"]: arms[spec["name"]] for spec in specs.values() if spec["name"] in arms}
    return {
        "config": {
            **context_config(recipe),
            **{
                key: recipe[key]
                for key in ("n_steps", "b_norm", "state_clip", "decay_range", "error_target")
            },
            "arm_e_update_scale": scale,
            "bootstrap": {"slots": he.batch, **bootstrap},
            "reference_measurement": "~+4e-3/token on dgx encrypted Mamba-2 decode",
        },
        "arms": arms,
        **summarize(arms, specs, recipe["error_target"]),
        "mean_bootstrap_eval_seconds": round(float(np.mean(he.bootstrap_seconds)), 3)
        if he.bootstrap_seconds
        else None,
        "n_bootstraps": len(he.bootstrap_seconds),
        "arms_run_this_invocation": selected,
    }
