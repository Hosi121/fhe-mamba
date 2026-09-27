#!/usr/bin/env python3
"""Derive an even refresh seed with an exact rational difference bound.

The input JSON supplies coefficient_literals (OpenFHE c[0]/2 convention) and K.
This offline bound covers real polynomial arithmetic, not CKKS rounding/noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path


def multiply_x(coefficients):
    """Multiply a Chebyshev series by x, with an unhalved constant term."""
    out = [Fraction(0)] * (len(coefficients) + 1)
    out[1] = coefficients[0]
    for i in range(1, len(coefficients)):
        out[i - 1] += coefficients[i] / 2
        out[i + 1] += coefficients[i] / 2
    return out


def translate(coefficients, shift):
    """Exact Chebyshev coefficients of P(x + shift)."""
    out = [Fraction(0)] * len(coefficients)

    def accumulate(basis, weight):
        for i, value in enumerate(basis):
            out[i] += value * weight

    previous, current = [Fraction(1)], [shift, Fraction(1)]
    accumulate(previous, coefficients[0])
    accumulate(current, coefficients[1])
    for n in range(2, len(coefficients)):
        following = multiply_x(current)
        for i, value in enumerate(current):
            following[i] += shift * value
        following = [2 * value for value in following]
        for i, value in enumerate(previous):
            following[i] -= value
        accumulate(following, coefficients[n])
        previous, current = current, following
    return out


def clenshaw(coefficients, x):
    """Independent evaluation recurrence, also entirely rational."""
    b1 = b2 = Fraction(0)
    for coefficient in reversed(coefficients[1:]):
        b1, b2 = coefficient + 2 * x * b1 - b2, b1
    return coefficients[0] + x * b1 - b2


def derive(coefficients: list[float], k: int) -> dict:
    if not coefficients or len(coefficients) < 3 or len(coefficients) % 2 != 1:
        raise ValueError("an even degree of at least two is required")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("K must be a positive integer")
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("coefficients must be finite")
    baseline = [Fraction(value) for value in coefficients]
    baseline[0] /= 2
    shift = Fraction(1, 4 * k)
    shifted = translate(baseline, shift)
    if translate(shifted, -shift) != baseline:
        raise ArithmeticError("translation inverse failed")
    candidate = [float(2 * shifted[0]), *[float(value) for value in shifted[2::2]]]
    exact_candidate = [Fraction(value) for value in candidate]
    exact_candidate[0] /= 2
    expanded = [Fraction(0)] * len(baseline)
    expanded[::2] = exact_candidate
    restored = translate(expanded, -shift)
    bound = sum(abs(a - b) for a, b in zip(restored, baseline, strict=True))
    upper = math.nextafter(float(bound), math.inf)
    if Fraction(upper) < bound:
        raise ArithmeticError("bound rounded down")
    for point in [
        Fraction(-1),
        Fraction(0),
        Fraction(1),
        shift,
        -shift,
        Fraction(-1, 2),
        Fraction(1, 2),
    ]:
        argument = 2 * (point - shift) ** 2 - 1
        if clenshaw(exact_candidate, argument) != clenshaw(restored, point):
            raise ArithmeticError("even-polynomial composition failed")
    return {
        "baseline_degree": len(baseline) - 1,
        "candidate_degree": len(candidate) - 1,
        "shift": str(shift),
        "coefficients": candidate,
        "coefficients_hex": [value.hex() for value in candidate],
        "difference_bound_exact": str(bound),
        "difference_bound_upper": upper,
        "scope": (
            "Uniform real-polynomial difference on [-1,1], including coefficient rounding; "
            "excludes CKKS and floating evaluation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    inputs = json.loads(raw)
    result = derive([float(value) for value in inputs["coefficient_literals"]], inputs["K"])
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
