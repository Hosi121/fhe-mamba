from dataclasses import replace
from fractions import Fraction

import pytest
import torch

from fhemamba.normalization import (
    ScheduledInvSqrt,
    certify_schedule,
    certify_weighted_schedule,
    fixed_newton_cost,
    plan_invsqrt,
)


@pytest.mark.parametrize("hi", [3.35, 466.16, 16935.0, 233000.0])
def test_full_epsilon_domain_accuracy_and_certificate(hi):
    schedule = plan_invsqrt(1e-5, hi)
    certificate = certify_schedule(schedule)
    assert certificate["certified"]
    grid = torch.logspace(-5, torch.log10(torch.tensor(hi)).item(), 8193, dtype=torch.float64)
    grid[0], grid[-1] = schedule.lo, schedule.hi
    factor = schedule(grid) * grid.sqrt()
    assert float((factor - 1).abs().max()) < schedule.tolerance + 2e-13
    assert float(factor.max()) <= 1 + 2e-13
    assert (
        schedule.abstract_cost()["ct_ct_critical_depth"]
        < fixed_newton_cost(schedule)["ct_ct_critical_depth"]
    )
    # Independently check sampled exact-rational trajectories against every
    # published enclosure, including the final recomputed Newton step.
    for value in (schedule.lo, schedule.hi, (schedule.lo + schedule.hi) / 2):
        u = Fraction(value) * Fraction(schedule.seed) ** 2
        for i, (a, b) in enumerate(schedule.coefficients):
            # Check a single step from a rational point inside each recorded
            # interval; avoid exponentially large exact full-polynomial powers.
            before = certificate["trace"][i]
            if i:
                u = (Fraction(before["u_lower"]) + Fraction(before["u_upper"])) / 2
            after = certificate["trace"][i + 1]
            image = u * (Fraction(a) - Fraction(b) * u) ** 2
            assert Fraction(after["u_lower"]) <= image <= Fraction(after["u_upper"])
    restored = ScheduledInvSqrt.from_recipe(schedule.recipe())
    assert restored == schedule
    assert torch.equal(restored(grid), schedule(grid))


def test_coupled_and_recomputed_paths_and_float32():
    schedule = plan_invsqrt(1e-5, 233000)
    v = torch.logspace(-5, 5.36, 1025, dtype=torch.float64)
    direct = torch.full_like(v, schedule.seed)
    for a, b in schedule.coefficients:
        direct = direct * (a - b * v * direct * direct)
    direct = direct * (1.5 - 0.5 * v * direct * direct)
    assert torch.allclose(schedule(v), direct, rtol=3e-13, atol=0)
    assert torch.allclose(schedule.evaluate_balanced(v), direct, rtol=3e-13, atol=0)
    assert torch.allclose(schedule.evaluate_weighted(v), direct, rtol=3e-13, atol=0)
    assert certify_weighted_schedule(schedule)["certified"]
    vf = v.float()
    assert float((schedule(vf).double() * vf.double().sqrt() - 1).abs().max()) < 2e-7
    # Outside-domain behavior remains polynomial, with no silent clipping.
    assert not torch.equal(schedule(torch.tensor([1e9])), schedule(torch.tensor([schedule.hi])))


def test_recomputed_final_step_corrects_coupled_identity_drift():
    # A stale u=1 would report convergence despite a 0.1% error in y.
    v = torch.logspace(-5, 5, 100, dtype=torch.float64)
    y = v.rsqrt() * 1.001
    stale_update = y * (1.5 - 0.5 * torch.ones_like(v))
    repaired = y * (1.5 - 0.5 * v * y * y)
    assert float((stale_update * v.sqrt() - 1).abs().max()) > 0.00099
    assert float((repaired * v.sqrt() - 1).abs().max()) < 1.51e-6


def test_invalid_and_tampered_schedules_fail_closed():
    for lo, hi in ((0, 1), (-1, 1), (1, 1), (1, float("inf"))):
        with pytest.raises(ValueError, match="required"):
            plan_invsqrt(lo, hi)
    with pytest.raises(ValueError, match="budget"):
        plan_invsqrt(1e-5, 233000, max_stages=1)
    with pytest.raises(ValueError, match="cushion"):
        plan_invsqrt(1e-5, 233000, cushion=0)
    schedule = plan_invsqrt(1e-5, 466)
    for changed in (
        replace(schedule, seed=1),
        replace(schedule, coefficients=((1.5, 0.5),)),
        replace(schedule, coefficients=((3.0, 0.5),)),
        replace(schedule, coefficients=((1.0, 2.0),)),
    ):
        assert not certify_schedule(changed)["certified"]
    bad_recipe = schedule.recipe() | {"final_recomputed_newton": False}
    with pytest.raises(ValueError, match="unsupported"):
        ScheduledInvSqrt.from_recipe(bad_recipe)
    with pytest.raises(ValueError, match="precision"):
        certify_schedule(schedule, bits=4)
    # Endpoint-only checks would miss this interior maximum of 32/27.
    interior_escape = ScheduledInvSqrt(0.1, 1.0, 1.0, ((2.0, 1.0),))
    assert not certify_schedule(interior_escape)["certified"]


class _CostTracer:
    """Execute the oracle with a symbolic encrypted value, counting its DAG."""

    def __init__(self, counts, depth=0):
        self.counts, self.depth = counts, depth
        self.dtype = None

    def double(self):
        return self

    def to(self, dtype):
        return self

    def __mul__(self, other):
        if isinstance(other, _CostTracer):
            self.counts["ct_ct_products"] += 1
            return _CostTracer(self.counts, max(self.depth, other.depth) + 1)
        self.counts["public_scalar_products"] += 1
        return _CostTracer(self.counts, self.depth)

    __rmul__ = __mul__

    def __rsub__(self, other):
        self.counts["public_scalar_additions"] += 1
        return _CostTracer(self.counts, self.depth)

    def __add__(self, other):
        assert isinstance(other, _CostTracer)
        self.counts["ct_ct_additions"] += 1
        return _CostTracer(self.counts, max(self.depth, other.depth))


@pytest.mark.parametrize("stages", [1, 2, 7, 16])
def test_cost_ledger_matches_executed_polynomial_dag(stages):
    schedule = ScheduledInvSqrt(1e-5, 466, 0.04, ((1.5, 0.5),) * stages)
    counts = dict.fromkeys(
        ("ct_ct_products", "public_scalar_products", "public_scalar_additions"), 0
    )
    output = schedule(_CostTracer(counts))
    expected = schedule.abstract_cost()
    assert all(count == expected[name] for name, count in counts.items())
    assert output.depth == expected["ct_ct_critical_depth"]


@pytest.mark.parametrize("stages", [1, 2, 7, 16])
def test_balanced_cost_ledger_matches_executed_polynomial_dag(stages):
    schedule = ScheduledInvSqrt(1e-5, 466, 0.04, ((1.5, 0.5),) * stages)
    counts = dict.fromkeys(
        ("ct_ct_products", "public_scalar_products", "public_scalar_additions", "ct_ct_additions"),
        0,
    )
    output = schedule.evaluate_balanced(_CostTracer(counts))
    expected = schedule.balanced_cost()
    assert all(count == expected[name] for name, count in counts.items())
    assert output.depth == expected["ct_ct_critical_depth"]


@pytest.mark.parametrize("hi", [3.35, 466.16, 58265.9, 233000.0])
def test_weighted_ratios_certify_actual_polynomial(hi):
    schedule = plan_invsqrt(1e-5, hi)
    certificate = certify_weighted_schedule(schedule)
    assert certificate["certified"]
    assert len(certificate["rounded_ratio_effective_coefficients"]) == len(schedule.coefficients)
    values = torch.logspace(-5, torch.log10(torch.tensor(hi)).item(), 1025, dtype=torch.float64)
    assert float((schedule.evaluate_weighted(values) * values.sqrt() - 1).abs().max()) < 1e-7
