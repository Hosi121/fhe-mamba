"""Exact-rational Bernstein bounds for exported polynomial initializers.

This certifies a real-arithmetic polynomial on a specified public interval.
It does not bound floating-point/CKKS evaluation errors or establish that a
private activation belongs to the interval. Binary64 inputs are interpreted
as their exact dyadic rationals, so no numerical root finder is trusted.
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise


def polynomial_product(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    result = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            result[i + j] += left * right
    return result


def chebyshev_unit_power(coefficients: list[float]) -> list[Fraction]:
    """Power coefficients of sum c[k] T_k(2t-1), for t in [0,1]."""
    if not coefficients or not all(math.isfinite(c) for c in coefficients):
        raise ValueError("finite, non-empty Chebyshev coefficients are required")
    result = [Fraction(0)] * len(coefficients)
    previous, current = [Fraction(1)], [Fraction(-1), Fraction(2)]
    result[0] = Fraction(coefficients[0])
    for degree in range(1, len(coefficients)):
        for i, coefficient in enumerate(current):
            result[i] += Fraction(coefficients[degree]) * coefficient
        following = polynomial_product(current, [Fraction(-2), Fraction(4)])
        for i, coefficient in enumerate(previous):
            following[i] -= coefficient
        previous, current = current, following
    return result


def power_to_bernstein(power: list[Fraction]) -> list[Fraction]:
    degree = len(power) - 1
    return [
        sum(
            (power[k] * Fraction(math.comb(i, k), math.comb(degree, k)) for k in range(i + 1)),
            Fraction(0),
        )
        for i in range(degree + 1)
    ]


def subdivide_midpoint(coefficients: list[Fraction]) -> tuple[list[Fraction], list[Fraction]]:
    row = list(coefficients)
    left, right = [row[0]], [row[-1]]
    while len(row) > 1:
        row = [(a + b) / 2 for a, b in pairwise(row)]
        left.append(row[0])
        right.append(row[-1])
    return left, list(reversed(right))


def certify_bounds(
    power: list[Fraction],
    *,
    lower: Fraction | None = None,
    upper: Fraction | None = None,
    max_depth: int = 16,
    max_nodes: int = 4096,
) -> dict:
    """Prove bounds by convex hulls; undecided subdivisions fail closed."""
    if not power or max_depth < 0 or max_nodes < 1:
        raise ValueError("non-empty polynomial and valid subdivision limits are required")
    if lower is None and upper is None:
        raise ValueError("at least one bound is required")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower exceeds upper")
    queue = [(power_to_bernstein(power), 0)]
    nodes, deepest, leaves = 0, 0, 0
    while queue:
        if nodes >= max_nodes:
            return {"certified": False, "reason": "node-limit", "nodes": nodes}
        coefficients, depth = queue.pop()
        nodes += 1
        deepest = max(deepest, depth)
        if (lower is None or min(coefficients) >= lower) and (
            upper is None or max(coefficients) <= upper
        ):
            leaves += 1
            continue
        # Endpoints of a Bernstein segment are exact polynomial evaluations.
        if any(
            (lower is not None and value < lower) or (upper is not None and value > upper)
            for value in (coefficients[0], coefficients[-1])
        ):
            return {
                "certified": False,
                "reason": "exact-endpoint-counterexample",
                "nodes": nodes,
                "depth": depth,
            }
        if depth >= max_depth:
            return {"certified": False, "reason": "depth-limit", "nodes": nodes}
        left, right = subdivide_midpoint(coefficients)
        queue.extend(((right, depth + 1), (left, depth + 1)))
    return {
        "certified": True,
        "reason": "rational-bernstein-convex-hulls",
        "nodes": nodes,
        "leaves": leaves,
        "depth": deepest,
    }


def certify_chebyshev_bounds(
    coefficients: list[float],
    *,
    lower: Fraction,
    upper: Fraction,
    max_depth: int = 16,
    max_nodes: int = 4096,
) -> dict:
    """Exact Bernstein bounds with shared integer denominators.

    This is the same convex-hull argument as ``certify_bounds``. Keeping a
    common denominator avoids millions of Fraction reductions when checking
    high-degree series. Every float coefficient is its exact binary rational;
    neither conversion nor subdivision uses floating-point arithmetic.
    """
    if (
        not coefficients
        or not all(math.isfinite(c) for c in coefficients)
        or lower > upper
        or max_depth < 0
        or max_nodes < 1
    ):
        raise ValueError("finite coefficients, ordered bounds and valid limits are required")
    rational = [Fraction(c) for c in coefficients]
    denominator = math.lcm(*(c.denominator for c in rational))
    scaled = [c.numerator * (denominator // c.denominator) for c in rational]
    power = [0] * len(scaled)
    power[0] = scaled[0]
    previous, current = [1], [-1, 2]
    for degree in range(1, len(scaled)):
        for i, value in enumerate(current):
            power[i] += scaled[degree] * value
        following = [0] * (len(current) + 1)
        for i, value in enumerate(current):
            following[i] -= 2 * value
            following[i + 1] += 4 * value
        for i, value in enumerate(previous):
            following[i] -= value
        previous, current = current, following
    degree = len(power) - 1
    divisors = [math.comb(degree, k) for k in range(degree + 1)]
    common = math.lcm(*divisors)
    weighted = [v * (common // d) for v, d in zip(power, divisors, strict=True)]
    bernstein = [
        sum(weighted[k] * math.comb(i, k) for k in range(i + 1)) for i in range(degree + 1)
    ]
    queue = [(bernstein, denominator * common, 0)]
    nodes, deepest, leaves = 0, 0, 0
    while queue:
        if nodes >= max_nodes:
            return {"certified": False, "reason": "node-limit", "nodes": nodes}
        values, divisor, depth = queue.pop()
        nodes += 1
        deepest = max(deepest, depth)

        def inside(value, divisor=divisor):
            return (
                lower.numerator * divisor <= value * lower.denominator
                and value * upper.denominator <= upper.numerator * divisor
            )

        if inside(min(values)) and inside(max(values)):
            leaves += 1
            continue
        if not inside(values[0]) or not inside(values[-1]):
            return {
                "certified": False,
                "reason": "exact-endpoint-counterexample",
                "nodes": nodes,
                "depth": depth,
            }
        if depth >= max_depth:
            return {"certified": False, "reason": "depth-limit", "nodes": nodes}
        row = values
        left, right = [row[0] << degree], [row[-1] << degree]
        for step in range(1, degree + 1):
            row = [a + b for a, b in pairwise(row)]
            left.append(row[0] << (degree - step))
            right.append(row[-1] << (degree - step))
        next_divisor = divisor << degree
        queue.extend(
            ((list(reversed(right)), next_divisor, depth + 1), (left, next_divisor, depth + 1))
        )
    return {
        "certified": True,
        "reason": "integer-bernstein-convex-hulls",
        "nodes": nodes,
        "leaves": leaves,
        "depth": deepest,
    }


def certify_dissipative_gate(dissipation_root: list[float], equilibrium_root: list[float]) -> dict:
    """Certify a=1-p^2, b=p^2*q^2 on one shared Chebyshev domain.

    |p|<=1 gives 0<=a<=1, while |q|<=sum|q_k| gives
    0<=b<=(1-a)*K with K=(sum|q_k|)^2. Factoring the small
    dissipation, instead of subtracting an approximation to unit decay,
    avoids imposing a uniform forgetting floor on long-memory heads.
    """
    if not equilibrium_root or not all(math.isfinite(c) for c in equilibrium_root):
        raise ValueError("finite non-empty equilibrium coefficients are required")
    bound = sum((abs(Fraction(c)) for c in equilibrium_root), Fraction(0)) ** 2
    gain = float(bound)
    if not math.isfinite(gain):
        raise ValueError("state gain overflows binary64")
    if Fraction(gain) < bound:
        gain = math.nextafter(gain, math.inf)
    contraction = certify_chebyshev_bounds(dissipation_root, lower=Fraction(-1), upper=Fraction(1))
    return {
        "certified": contraction["certified"],
        "gain": gain,
        "dissipation_root_in_unit_interval": contraction,
        "equilibrium_bound": "exact Chebyshev coefficient l1 norm, squared and rounded upward",
        "claim": "Conditional exact-arithmetic invariant |state| <= gain * input_bound.",
    }


def certify_newton_initializer(spec: dict) -> dict:
    """Certify y0>=0 and v*y0^2<=3 on the declared variance interval.

    One exact Newton step then gives 0 <= sqrt(v)*y1 <= 1. All subsequent
    steps remain non-expansive. This is a range certificate, not an accuracy
    certificate: y0=0, for example, is safe but never converges to rsqrt(v).
    """
    kind = spec.get("kind")
    if kind not in {"poly-newton", "sq-poly-newton"}:
        raise ValueError("only exported polynomial Newton initializers are supported")
    lo, hi = float(spec["lo"]), float(spec["hi"])
    damping = float(spec["damping"])
    if not all(math.isfinite(v) for v in (lo, hi, damping)) or not 0 <= lo < hi:
        raise ValueError("finite nonnegative ordered variance bounds are required")
    if damping <= 0 or spec["iterations"] < 1:
        raise ValueError("positive damping and at least one Newton iteration are required")
    guess = chebyshev_unit_power(spec["coeffs"])
    if kind == "sq-poly-newton":
        guess = polynomial_product(guess, guess)
        positive = {"certified": True, "reason": "positive-damping-times-square"}
    else:
        positive = certify_bounds(guess, lower=Fraction(0))
    guess = [Fraction(damping) * coefficient for coefficient in guess]
    basin = polynomial_product(
        [Fraction(lo), Fraction(hi) - Fraction(lo)], polynomial_product(guess, guess)
    )
    basin_result = certify_bounds(basin, upper=Fraction(3))
    return {
        "certified": positive["certified"] and basin_result["certified"],
        "interval": [lo, hi],
        "nonnegative_initializer": positive,
        "newton_basin": basin_result,
        "claim": "In exact arithmetic on this interval, Newton normalization is non-expansive.",
    }


def certify_affine_state_invariant(
    decay_chebyshev: list[float], write_chebyshev: list[float], gain: float
) -> dict:
    """Certify a common-domain recurrence s'=a(z)s+b(z)v, |v|<=V.

    If 0<=a<=1 and |b|<=gain*(1-a), then |s|<=gain*V is invariant,
    including a=1,b=0. Both coefficient lists use the same Chebyshev
    variable on [-1,1]. This checks rounded polynomial coefficients, not
    CKKS evaluation error or domain membership of z.
    """
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    decay = chebyshev_unit_power(decay_chebyshev)
    write = chebyshev_unit_power(write_chebyshev)
    contraction = certify_bounds(decay, lower=Fraction(0), upper=Fraction(1))
    gain_q = Fraction(gain)
    room = [gain_q] + [Fraction(0)] * (max(len(decay), len(write)) - 1)
    for i, coefficient in enumerate(decay):
        room[i] -= gain_q * coefficient
    write_bounds = []
    for sign in (-1, 1):
        inequality = list(room)
        for i, coefficient in enumerate(write):
            inequality[i] += sign * coefficient
        write_bounds.append(certify_bounds(inequality, lower=Fraction(0)))
    return {
        "certified": contraction["certified"] and all(b["certified"] for b in write_bounds),
        "gain": gain,
        "decay_in_unit_interval": contraction,
        "write_dominated_by_dissipation": write_bounds,
        "claim": "Conditional exact-arithmetic invariant |state| <= gain * input_bound.",
    }


def certify_bernstein_state_invariant(decay: list[float], write: list[float], gain: float) -> dict:
    """A coefficientwise certificate for a *shared* Bernstein basis on [0,1].

    Each basis function is nonnegative and their sum is one. Thus the linear
    inequalities 0<=a_i<=1 and |b_i|<=gain*(1-a_i) also hold for the
    interpolated gates. This includes perfect memory a_i=1 only with b_i=0.
    It does not certify a subsequently rounded change of polynomial basis.
    """
    if (
        not decay
        or len(decay) != len(write)
        or not all(math.isfinite(v) for v in [*decay, *write, gain])
        or gain <= 0
    ):
        raise ValueError("equal finite Bernstein coefficient lists and positive gain required")
    bound = Fraction(gain)
    failed = [
        i
        for i, (a, b) in enumerate(zip(decay, write, strict=True))
        if not 0 <= Fraction(a) <= 1 or abs(Fraction(b)) > bound * (1 - Fraction(a))
    ]
    return {
        "certified": not failed,
        "reason": "exact-rational-bernstein-coefficient-inequalities",
        "failed_coefficients": failed,
        "degree": len(decay) - 1,
        "gain": gain,
        "claim": (
            "On the shared domain, |state| <= gain * input_bound is invariant in exact arithmetic."
        ),
    }
