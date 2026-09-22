"""Public, interval-certified scaled Goldschmidt inverse square roots.

This is an adaptation of known scaled Newton/Goldschmidt iteration (including
THOR, Appendix C), not a new approximation family. Certificates concern exact
real evaluation of the stored binary64 coefficients, conditional on the input
interval. They do not include CKKS noise or floating-point rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from torch import Tensor


def _outward(lo: Fraction, hi: Fraction, bits: int) -> tuple[Fraction, Fraction]:
    scale = 1 << bits
    lower, upper = lo * scale, hi * scale
    return (
        Fraction(lower.numerator // lower.denominator, scale),
        Fraction(-(-upper.numerator // upper.denominator), scale),
    )


def _image(
    lo: Fraction, hi: Fraction, a: Fraction, b: Fraction, bits: int
) -> tuple[Fraction, Fraction]:
    """Range of u*(a-b*u)^2; positivity removes the second critical point."""
    if not 0 < lo <= hi or not b > 0 or not a - b * hi > 0:
        raise ValueError("positive interval and update factor are required")
    values = [u * (a - b * u) ** 2 for u in (lo, hi)]
    if lo <= a / (3 * b) <= hi:
        values.append(4 * a**3 / (27 * b))
    return _outward(min(values), max(values), bits)


@dataclass(frozen=True)
class ScheduledInvSqrt:
    """Polynomial oracle with public coefficients and one final Newton repair.

    Track u=v*y^2 in parallel with y. Recompute this identity from the original
    v for the final Newton update: a coupled residual alone can hide roundoff.
    The runtime contains only addition/multiplication; no clipping or branching
    on the encrypted input is required. The oracle uses float64 and casts back.
    """

    lo: float
    hi: float
    seed: float
    coefficients: tuple[tuple[float, float], ...]
    tolerance: float = 1e-7

    def __post_init__(self):
        scalars = [self.lo, self.hi, self.seed, self.tolerance]
        scalars.extend(value for pair in self.coefficients for value in pair)
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("finite schedule parameters are required")
        if not 0 < self.lo < self.hi or not self.seed > 0 or not 0 < self.tolerance < 1:
            raise ValueError("positive interval, seed and tolerance below one are required")
        if not self.coefficients or any(len(pair) != 2 for pair in self.coefficients):
            raise ValueError("at least one coefficient pair is required")

    @property
    def interval(self) -> tuple[float, float]:
        return self.lo, self.hi

    def __call__(self, x: Tensor) -> Tensor:
        v = x.double()
        # Separate public scalar products preserve the exact seed^2 identity.
        u = v * self.seed * self.seed
        y = self.seed
        for i, (a, b) in enumerate(self.coefficients):
            factor = a - b * u
            y = y * factor
            if i + 1 < len(self.coefficients):
                u = u * (factor * factor)
        return (y * (1.5 - 0.5 * (v * (y * y)))).to(x.dtype)

    def recipe(self) -> dict:
        return {
            "kind": "scaled-goldschmidt-invsqrt-v1",
            "lo": self.lo,
            "hi": self.hi,
            "seed": self.seed,
            "coefficients": [list(pair) for pair in self.coefficients],
            "tolerance": self.tolerance,
            "final_recomputed_newton": True,
            "oracle_workspace_dtype": "float64",
        }

    def evaluate_balanced(self, x: Tensor) -> Tensor:
        """Same real polynomial with a two-ct-product-depth cubic update.

        y*(a-b*v*y*y) = a*y + ((-b*v)*y)*(y*y). The two inner
        products run in parallel. Recompute from v at every step so a stale
        coupled residual cannot hide an error in y. Scalar products still
        need their own physical CKKS scale/level accounting.
        """
        v = x.double()
        a, b = self.coefficients[0]
        y = (a - b * (v * self.seed * self.seed)) * self.seed
        for a, b in (*self.coefficients[1:], (1.5, 0.5)):
            y = a * y + ((-b * v) * y) * (y * y)
        return y.to(x.dtype)

    def balanced_cost(self) -> dict:
        n = len(self.coefficients)
        return {
            "ct_ct_products": 3 * n,
            "ct_ct_critical_depth": 2 * n,
            "public_scalar_products": 2 * n + 4,
            "public_scalar_additions": 1,
            "ct_ct_additions": n,
            "includes_encoding_rescale_bootstrap_cost": False,
        }

    def evaluate_weighted(self, x: Tensor) -> Tensor:
        """Hoist a public coefficient out of the coupled critical path.

        r_i = -B_i*v*y_i^2, B_0=b_0, B_(i+1)=B_i*float(b_(i+1)/b_i).
        Scale r in parallel with squaring the factor. Recertification must
        account for the rounded ratios: B_i is not exactly the stored b_i.
        """
        v = x.double()
        a, b = self.coefficients[0]
        r = v * self.seed * self.seed * (-b)
        factor = a + r
        y = self.seed * factor
        for a, next_b in self.coefficients[1:]:
            r = (r * (next_b / b)) * (factor * factor)
            factor = a + r
            y = y * factor
            b = next_b
        # (v*y)*y keeps each intermediate's true magnitude smaller than
        # v*(y*y) or the cubic y correction on a wide mixed SIMD batch.
        residual = (v * y) * y
        return (y * (1.5 - 0.5 * residual)).to(x.dtype)

    @classmethod
    def from_recipe(cls, recipe: dict) -> ScheduledInvSqrt:
        if (
            recipe.get("kind") != "scaled-goldschmidt-invsqrt-v1"
            or recipe.get("final_recomputed_newton") is not True
            or recipe.get("oracle_workspace_dtype") != "float64"
        ):
            raise ValueError("unsupported inverse-square-root recipe")
        return cls(
            recipe["lo"],
            recipe["hi"],
            recipe["seed"],
            tuple(tuple(pair) for pair in recipe["coefficients"]),
            recipe["tolerance"],
        )

    def abstract_cost(self) -> dict:
        """Literal DAG counts per SIMD lane; scalar rescaling is not free in CKKS."""
        n = len(self.coefficients)
        return {
            "accelerated_stages": n,
            "final_recomputed_newton_stages": 1,
            "ct_ct_products": 3 * n,
            "ct_ct_critical_depth": 3 if n == 1 else 2 * n + 2,
            "public_scalar_products": n + 4,
            "public_scalar_additions": n + 1,
            "same_coefficients_recomputed_ct_ct_depth": 3 * n,
            "same_coefficients_balanced_ct_ct_depth": 2 * n,
            "includes_encoding_rescale_bootstrap_cost": False,
        }


def certify_schedule(schedule: ScheduledInvSqrt, *, bits: int = 128) -> dict:
    """Recompute rational enclosures from the actual coefficients, not samples."""
    if not 32 <= bits <= 512:
        raise ValueError("certificate precision must be between 32 and 512 bits")
    seed = Fraction(schedule.seed)
    lo, hi = _outward(Fraction(schedule.lo) * seed**2, Fraction(schedule.hi) * seed**2, bits)
    trace = []

    def record(stage):
        trace.append({"stage": stage, "u_lower": str(lo), "u_upper": str(hi)})

    record("constant-seed")
    reason = None
    if not 0 < lo <= hi <= 1:
        reason = "initial squared normalization factor is outside (0,1]"
    else:
        for i, (a, b) in enumerate((*schedule.coefficients, (1.5, 0.5))):
            try:
                lo, hi = _image(lo, hi, Fraction(a), Fraction(b), bits)
            except ValueError as exc:
                reason = str(exc)
                break
            record(i)
            if not 0 < lo <= hi <= 1:
                reason = "squared normalization factor leaves (0,1]"
                break
        if reason is None and lo < (1 - Fraction(schedule.tolerance)) ** 2:
            reason = "relative-error target is not reached"
    return {
        "certified": reason is None,
        "reason": reason,
        "arithmetic": "exact-rational extrema with outward dyadic enclosures",
        "enclosure_bits": bits,
        "relative_error_bound": schedule.tolerance if reason is None else None,
        "nonexpansive_in_exact_arithmetic": reason is None,
        "includes_execution_roundoff": False,
        "trace": trace,
    }


def certify_weighted_schedule(schedule: ScheduledInvSqrt, *, bits: int = 128) -> dict:
    """Verify the polynomial actually induced by rounded public b ratios."""
    effective_b = Fraction(schedule.coefficients[0][1])
    effective = [(schedule.coefficients[0][0], effective_b)]
    for (a, b), (_, previous_b) in zip(
        schedule.coefficients[1:], schedule.coefficients, strict=False
    ):
        effective_b *= Fraction(b / previous_b)
        effective.append((a, effective_b))
    equivalent = ScheduledInvSqrt(
        schedule.lo, schedule.hi, schedule.seed, tuple(effective), schedule.tolerance
    )
    certificate = certify_schedule(equivalent, bits=bits)
    certificate["rounded_ratio_effective_coefficients"] = [str(b) for _, b in effective]
    return certificate


def plan_invsqrt(
    lo: float,
    hi: float,
    *,
    tolerance: float = 1e-7,
    cushion: float = 1 / 16,
    max_stages: int = 64,
) -> ScheduledInvSqrt:
    """Choose all coefficients offline from a public variance interval.

    For t=sqrt(u) in [l,r], equalize endpoint values of a*t-b*t^3 and
    scale its interior maximum to one. A public lower-end cushion limits
    cancellation at the upper endpoint; it is not an optimality claim.
    """
    if not 0 < cushion < 1 or not max_stages >= 1:
        raise ValueError("cushion in (0,1) and positive max_stages are required")
    # Validate the public domain before sqrt/division.
    ScheduledInvSqrt(lo, hi, 1.0, ((1.5, 0.5),), tolerance)
    seed = 1 / math.sqrt(hi)
    while Fraction(hi) * Fraction(seed) ** 2 > 1:
        seed = math.nextafter(seed, 0.0)
    bits = 128
    lower, upper = _outward(
        Fraction(lo) * Fraction(seed) ** 2, Fraction(hi) * Fraction(seed) ** 2, bits
    )
    coefficients = []
    for _ in range(max_stages):
        left, right = math.sqrt(float(lower)), math.sqrt(float(upper))
        left = max(left, right * cushion)
        s = left * left + left * right + right * right
        b = 3 * math.sqrt(3) / (2 * s**1.5)
        a = b * s
        # Make the actual rounded polynomial nonexpansive, then verify exactly.
        a, b = a * (1 - 1e-12), b * (1 - 1e-12)
        lower, upper = _image(lower, upper, Fraction(a), Fraction(b), bits)
        if not 0 < lower <= upper <= 1:
            raise ValueError("rounded schedule failed its nonexpansion check")
        coefficients.append((a, b))
        final_lower, _ = _image(lower, upper, Fraction(3, 2), Fraction(1, 2), bits)
        if final_lower >= (1 - Fraction(tolerance)) ** 2:
            schedule = ScheduledInvSqrt(lo, hi, seed, tuple(coefficients), tolerance)
            if not certify_schedule(schedule)["certified"]:
                raise ValueError("constructed schedule failed independent recertification")
            return schedule
    raise ValueError("iteration budget exhausted before reaching the error target")


def fixed_newton_cost(schedule: ScheduledInvSqrt, *, max_stages: int = 256) -> dict:
    """Constant-seed Newton baseline on the SAME interval and error target."""
    lower = Fraction(schedule.lo) * Fraction(schedule.seed) ** 2
    upper = Fraction(schedule.hi) * Fraction(schedule.seed) ** 2
    if not 0 < lower <= upper <= 1:
        raise ValueError("constant seed must start in (0,1]")
    for steps in range(1, max_stages + 1):
        lower, upper = _image(lower, upper, Fraction(3, 2), Fraction(1, 2), 128)
        if lower >= (1 - Fraction(schedule.tolerance)) ** 2:
            return {
                "stages": steps,
                "ct_ct_products": 3 * (steps - 1),
                "ct_ct_critical_depth": 2 * (steps - 1),
                "factored_unbalanced_ct_ct_depth": 3 * (steps - 1),
                "same_domain_seed_and_tolerance": True,
                "includes_encoding_rescale_bootstrap_cost": False,
            }
    raise ValueError("fixed-Newton baseline exhausted its iteration budget")
