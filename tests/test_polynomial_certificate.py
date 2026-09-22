from fractions import Fraction

import numpy as np
import pytest

from fhemamba.polynomial_certificate import (
    certify_affine_state_invariant,
    certify_bernstein_state_invariant,
    certify_bounds,
    certify_newton_initializer,
    chebyshev_unit_power,
    power_to_bernstein,
)


def test_joint_bernstein_gates_prove_invariance_including_perfect_memory():
    decay, write = [1.0, 0.75, 0.0], [0.0, 0.5, -2.0]
    assert certify_bernstein_state_invariant(decay, write, 2.0)["certified"]
    assert not certify_bernstein_state_invariant(decay, [1e-30, 0.5, -2.0], 2.0)["certified"]
    assert not certify_bernstein_state_invariant(decay, write, 1.999)["certified"]
    assert not certify_bernstein_state_invariant([1, 1.001, 0], [0, 0, 0], 2)["certified"]


def test_joint_bernstein_certificate_rejects_mismatched_basis_degrees():
    with pytest.raises(ValueError, match="equal finite Bernstein"):
        certify_bernstein_state_invariant([1, 0], [0], 1)
    with pytest.raises(ValueError, match="equal finite Bernstein"):
        certify_bernstein_state_invariant([float("nan")], [0], 1)


def test_chebyshev_change_of_basis():
    coefficients = [0.1, -0.2, 0.5, 0.7, 0.1]
    power = chebyshev_unit_power(coefficients)
    t = np.linspace(0, 1, 31)
    assert np.allclose(
        np.polynomial.polynomial.polyval(t, [float(x) for x in power]),
        np.polynomial.chebyshev.chebval(2 * t - 1, coefficients),
        atol=1e-13,
    )
    assert power_to_bernstein([Fraction(1), Fraction(-2), Fraction(1)]) == [1, 0, 0]


def test_bernstein_subdivision_proves_nonnegative_curvature():
    # (t-1/2)^2 has a negative middle Bernstein coefficient on the full
    # interval, but midpoint subdivision proves its true non-negativity.
    power = [Fraction(1, 4), Fraction(-1), Fraction(1)]
    assert certify_bounds(power, lower=Fraction(0))["certified"]
    assert not certify_bounds(power, lower=Fraction(0), max_depth=0)["certified"]
    assert not certify_bounds(power, upper=Fraction(1, 5))["certified"]


@pytest.mark.parametrize("kind", ["poly-newton", "sq-poly-newton"])
def test_newton_basin_rejects_an_unsafe_guess(kind):
    spec = {"kind": kind, "lo": 1.0, "hi": 2.0, "damping": 1.0, "iterations": 1, "coeffs": [1.0]}
    assert certify_newton_initializer(spec)["certified"]
    spec["coeffs"] = [2.0]
    assert not certify_newton_initializer(spec)["certified"]
    spec["coeffs"] = [-1.0]
    assert certify_newton_initializer(spec)["certified"] == (kind == "sq-poly-newton")


@pytest.mark.parametrize("power", [0.25, 0.5])
def test_positive_binomial_seed_has_a_certified_wide_basin(power):
    from fhemamba.ops import positive_binomial_seed

    seed = positive_binomial_seed(100.0, 15, power=power)
    spec = {
        "kind": "sq-poly-newton" if power == 0.25 else "poly-newton",
        "lo": seed.lo,
        "hi": seed.hi,
        "coeffs": list(seed.coeffs),
        "damping": 0.85 if power == 0.25 else 0.9,
        "iterations": 8,
    }
    assert certify_newton_initializer(spec)["certified"]


def test_joint_gate_certificate_handles_perfect_memory_and_reset():
    # a(z)=(1+z)/2 and b(z)=(1-z)/2: includes a=0 and a=1.
    assert certify_affine_state_invariant([0.5, 0.5], [0.5, -0.5], 1.0)["certified"]
    assert certify_affine_state_invariant([0.5, 0.5], [-0.5, 0.5], 1.0)["certified"]


def test_joint_gate_certificate_rejects_leak_at_unit_decay():
    # A bounded decay alone is insufficient: positive write at a=1 accumulates.
    assert not certify_affine_state_invariant([0.5, 0.5], [0.1], 100.0)["certified"]
    assert not certify_affine_state_invariant([1.000001], [0.0], 1.0)["certified"]
