"""Run recipe-defined local CKKS feasibility studies (toy security only)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from fhemamba.benchmarks.io import file_sha256, read_object, write_json


def validate_recipe(recipe):
    """Reject misspelled fields before fitting curves or allocating a context."""

    def fields(value, required, optional=""):
        required, allowed = set(required.split()), set((required + " " + optional).split())
        if not isinstance(value, dict) or required - value.keys() or value.keys() - allowed:
            raise ValueError(f"expected fields {sorted(required)}; optional {optional!r}")

    suite = recipe.get("suite")
    if recipe.get("schema_version") != 1 or suite not in {"primitives", "growth"}:
        raise ValueError("expected a schema_version 1 primitives or growth recipe")
    common = "schema_version suite context coefficient_floor seed notes "
    fields(
        recipe,
        common
        + (
            "n_inputs layout_block cases chain"
            if suite == "primitives"
            else "n_steps b_norm state_clip decay_range error_target "
            "bootstrap decay_polynomial arms"
        ),
    )
    context = recipe["context"]
    fields(context, "ring_dimension batch depth scale_bits first_mod_bits")
    if any(type(value) is not int or value <= 0 for value in context.values()):
        raise ValueError("context fields must be positive integers")
    if context["batch"] * 2 != context["ring_dimension"] or context["batch"] & (
        context["batch"] - 1
    ):
        raise ValueError("CPU probes require full packing and a power-of-two ring dimension")
    if type(recipe["seed"]) is not int or recipe["seed"] < 0 or recipe["coefficient_floor"] < 0:
        raise ValueError("seed and coefficient_floor must be non-negative")
    if suite == "primitives":
        fields(
            recipe["chain"],
            "input_interval glue level_budget correction_factor refresh_error_limit "
            "fallback_scale_bits fallback_correction_factor",
        )
        fields(recipe["chain"]["glue"], "a scale shift")
        return
    fields(recipe["bootstrap"], "level_budget correction_factor forced_input_towers")
    if not 1 <= recipe["bootstrap"]["forced_input_towers"] <= context["depth"]:
        raise ValueError("forced_input_towers must be between 1 and depth")
    names = []
    for spec in recipe["arms"].values():
        mode = spec.get("mode")
        if mode == "recurrence":
            fields(spec, "name mode decay refresh", "polynomial update")
            if any(
                type(spec[key]) is not bool
                for key in ("decay", "refresh", "polynomial", "update")
                if key in spec
            ):
                raise ValueError("recurrence switches must be booleans")
            if spec.get("polynomial") and not spec["decay"]:
                raise ValueError("polynomial decay requires decay=true")
        elif mode in {"meta", "compensated"}:
            fields(spec, "name mode label " + ("amp_bits" if mode == "meta" else "s_bits b_lo"))
        else:
            raise ValueError(f"unknown growth mode: {mode}")
        names.append(spec["name"])
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("growth arm names must be nonempty and unique")


def context_config(recipe):
    c = recipe["context"]
    return {
        "ring_dim": c["ring_dimension"],
        "batch": c["batch"],
        "depth": c["depth"],
        "scaling_mod_size": c["scale_bits"],
        "first_mod_size": c["first_mod_bits"],
        "scaling_technique": "FLEXIBLEAUTO",
        "secret_key_dist": "uniform-ternary",
        "security": "not-set (toy)",
        "seed": recipe["seed"],
    }


def make_context(recipe, *, rotations=(), scale_bits=None):
    from fhemamba.ckks_probes import build_toy_context

    context = dict(recipe["context"])
    if scale_bits is not None:
        context["scale_bits"] = scale_bits
    return build_toy_context(**context, rotations=rotations)


def run_primitive(he, curve, rng, n):
    import numpy as np

    from .ckks_curves import max_error

    lo, hi = curve.interval
    inverse = curve.kind in {"rsqrt", "sq_rsqrt"}
    x = np.exp(rng.uniform(np.log(lo), np.log(hi), n)) if inverse else rng.uniform(lo, hi, n)
    if inverse:
        x[0], x[1] = lo, hi
    start = time.perf_counter()
    ct = he.encrypt(x, fill=1.0 if inverse else 0.0)
    level = int(ct.GetLevel())
    out = curve.encrypted(he, ct)
    seconds = time.perf_counter() - start
    plain = curve.plain(x)
    return {
        "name": curve.name,
        "n_inputs": n,
        "level_in": level,
        "level_out": int(out.GetLevel()),
        "levels_consumed": int(out.GetLevel()) - level,
        "max_abs_err_ckks_vs_plain64": max_error(he.decrypt(out, n), plain),
        "max_abs_err_plain64_vs_exact": max_error(plain, curve.exact(x)),
        "seconds": round(seconds, 3),
        **curve.metadata(),
    }


def run_layout(he, rng, block):
    import numpy as np

    from fhemamba.ckks_probes import ceil_log2

    from .ckks_curves import max_error

    cc, batch = he.cc, he.batch
    data = rng.uniform(-1.0, 1.0, batch)
    sums = data.reshape(-1, block).sum(axis=1)
    start = time.perf_counter()
    ct = he.encrypt(data)
    level = int(ct.GetLevel())

    def doubling(value, stride, count):
        for k in range(ceil_log2(count)):
            value = cc.EvalAdd(value, cc.EvalRotate(value, stride << k))
        return value

    def measurement(value, expected, stride=1):
        return {
            "level_out": int(value.GetLevel()),
            "max_abs_err": max_error(he.decrypt(value)[::stride], expected),
        }

    reduced = doubling(ct, 1, block)
    reduction = measurement(reduced, sums, block)
    mask = np.zeros(batch)
    mask[::block] = 1.0
    bcast = doubling(he.mask_mult(reduced, mask), -1, block)
    broadcast = measurement(bcast, np.repeat(sums, block))
    base = np.zeros(batch)
    base[:block] = rng.uniform(-1.0, 1.0, block)
    rep = doubling(he.encrypt(base), -block, batch // block)
    replication = measurement(rep, np.tile(base[:block], batch // block))
    return {
        "name": "state_layout_rotate_ops",
        "block": block,
        "level_in": level,
        "reduce": reduction,
        "broadcast": broadcast,
        "stride_replication": replication,
        "seconds": round(time.perf_counter() - start, 3),
    }


def run_chain(he, recipe, curves, x, *, label, correction):
    import numpy as np
    import torch
    from torch.nn import functional as F  # noqa: N812

    from fhemamba.ckks_probes import cheb_ps_depth

    from .ckks_curves import max_error

    chain, cc = recipe["chain"], he.cc
    glue = chain["glue"]
    depth, batch = recipe["context"]["depth"], he.batch
    stages = []
    ct, plain = he.encrypt(x), x
    tail = curves["rsqrt"]
    needed = 1 + cheb_ps_depth(tail.degree) + 2 * tail.iterations + 2

    def error(value, reference):
        try:
            result = max_error(he.decrypt(value, len(x)), reference)
            return result if np.isfinite(result) else None
        except RuntimeError as exc:
            print(f"decode failed: {exc}", flush=True)
            return None

    for key, transform in (
        ("silu", None),
        ("softplus", None),
        ("exp", (glue["a"], 0.0)),
        ("rsqrt", (glue["scale"], glue["shift"])),
    ):
        curve = curves[key]
        if key == "rsqrt":
            level = int(ct.GetLevel())
            bootstrap = {
                "performed": False,
                "levels_needed_estimate": needed,
                "levels_remaining": depth - level,
            }
            if depth - level < needed:
                start = time.perf_counter()
                cc.EvalBootstrapSetup(chain["level_budget"], [0, 0], batch, correction)
                setup = time.perf_counter() - start
                start = time.perf_counter()
                cc.EvalBootstrapKeyGen(he.keys.secretKey, batch)
                keygen = time.perf_counter() - start
                start = time.perf_counter()
                ct = cc.EvalBootstrap(ct)
                elapsed = time.perf_counter() - start
                err = error(ct, plain)
                bootstrap = {
                    "performed": True,
                    "levels_needed_estimate": needed,
                    "level_before": level,
                    "level_after": int(ct.GetLevel()),
                    "setup_seconds": round(setup, 3),
                    "keygen_seconds": round(keygen, 3),
                    "eval_seconds": round(elapsed, 3),
                    "slots": batch,
                    "level_budget": chain["level_budget"],
                    "correction_factor": correction,
                    "max_abs_err_after_bootstrap": err,
                    "decode_failed": err is None,
                }
        start = time.perf_counter()
        ct = curve.encrypted(he, ct, glue=transform)
        plain = curve.plain(plain, glue=transform)
        elapsed = time.perf_counter() - start
        stages.append(
            {
                "name": f"silu_deg{curve.degree}" if key == "silu" else curve.name,
                "level": int(ct.GetLevel()),
                "max_abs_err_ckks_vs_plain64": error(ct, plain),
                "seconds": round(elapsed, 3),
            }
        )
        print(f"{label}: {stages[-1]}", flush=True)
    exact = 1.0 / np.sqrt(
        glue["scale"] * np.exp(glue["a"] * F.softplus(F.silu(torch.from_numpy(x))).numpy())
        + glue["shift"]
    )
    return {
        "label": label,
        "input_interval": chain["input_interval"],
        "glue": glue,
        "stages": stages,
        "bootstrap": bootstrap,
        "final_level": int(ct.GetLevel()),
        "final_max_abs_err_ckks_vs_plain64": stages[-1]["max_abs_err_ckks_vs_plain64"],
        "final_max_abs_err_vs_exact_chain": error(ct, exact),
    }


def run_primitives(recipe):
    import numpy as np

    from fhemamba.ckks_probes import CkksPrimitives, ceil_log2

    from .ckks_curves import Curve

    validate_recipe(recipe)
    batch, block = recipe["context"]["batch"], recipe["layout_block"]
    if block < 1 or block > batch or block & (block - 1) or batch % block:
        raise ValueError("layout block must be a power of two dividing batch")
    if not 2 <= recipe["n_inputs"] <= batch:
        raise ValueError("n_inputs must be between 2 and batch")
    curves = {
        name: Curve(**case, coefficient_floor=recipe["coefficient_floor"])
        for name, case in recipe["cases"].items()
    }
    if any(
        name not in curves or curves[name].kind != name
        for name in ("silu", "softplus", "exp", "rsqrt")
    ):
        raise ValueError("primitive chain requires silu, softplus, exp and rsqrt cases")
    rotations = (
        [1 << k for k in range(ceil_log2(block))]
        + [-(1 << k) for k in range(ceil_log2(block))]
        + [-(block << k) for k in range(ceil_log2(batch // block))]
    )
    rng = np.random.default_rng(recipe["seed"])
    start = time.perf_counter()
    cc, keys = make_context(recipe, rotations=rotations)
    setup = time.perf_counter() - start
    he = CkksPrimitives(cc, keys, batch=batch, coefficient_floor=recipe["coefficient_floor"])
    primitives = []
    for curve in curves.values():
        primitives.append(run_primitive(he, curve, rng, recipe["n_inputs"]))
        print(primitives[-1], flush=True)
    primitives.append(run_layout(he, rng, block))
    chain = recipe["chain"]
    x = rng.uniform(*chain["input_interval"], recipe["n_inputs"])
    scale = recipe["context"]["scale_bits"]
    torture = run_chain(
        he,
        recipe,
        curves,
        x,
        label=f"pinned scale-{scale} context",
        correction=chain["correction_factor"],
    )
    bootstrap, fallback = torture["bootstrap"], None
    err = bootstrap.get("max_abs_err_after_bootstrap")
    if bootstrap["performed"] and (
        bootstrap["decode_failed"] or err is None or err > chain["refresh_error_limit"]
    ):
        scale = chain["fallback_scale_bits"]
        cc, keys = make_context(recipe, scale_bits=scale)
        he = CkksPrimitives(cc, keys, batch=batch, coefficient_floor=recipe["coefficient_floor"])
        fallback = run_chain(
            he,
            recipe,
            curves,
            x,
            label=f"sibling scale-{scale} context",
            correction=chain["fallback_correction_factor"],
        )
    return {
        "config": {
            **context_config(recipe),
            "n_inputs": recipe["n_inputs"],
            "kernel": "native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp",
            "context_setup_seconds": round(setup, 3),
        },
        "primitives": primitives,
        "torture": torture,
        f"torture_fallback_scale{chain['fallback_scale_bits']}": fallback,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh result JSON path")
    parser.add_argument("--arms", help="growth arm IDs, comma separated; default: all recipe arms")
    parser.add_argument(
        "--resume", type=Path, help="merge growth arms from a matching recipe result"
    )
    args = parser.parse_args(argv)
    recipe = read_object(args.recipe)
    try:
        validate_recipe(recipe)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error("output already exists; choose a fresh path")
    identity = file_sha256(args.recipe)
    previous = read_object(args.resume) if args.resume else {}
    if args.resume and previous.get("recipe_sha256") != identity:
        parser.error("resume recipe hash differs; results cannot be merged")
    started = time.perf_counter()
    if recipe["suite"] == "primitives":
        if args.arms is not None or args.resume:
            parser.error("--arms and --resume apply only to growth studies")
        payload = run_primitives(recipe)
    else:
        from .ckks_growth import run_growth

        selected = (
            list(recipe["arms"])
            if args.arms is None
            else [a.strip().upper() for a in args.arms.split(",") if a.strip()]
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or set(selected) - recipe["arms"].keys()
        ):
            parser.error("--arms must contain distinct IDs from the recipe")
        payload = run_growth(recipe, selected, previous.get("arms", {}))
    payload.update(
        recipe_sha256=identity,
        total_seconds=round(time.perf_counter() - started, 3),
        notes=recipe["notes"],
    )
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
