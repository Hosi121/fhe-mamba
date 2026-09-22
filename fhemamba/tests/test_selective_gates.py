from fractions import Fraction

import numpy as np
import pytest
import torch
from fhemamba.polynomial_certificate import (
    certify_bounds,
    certify_chebyshev_bounds,
    certify_dissipative_gate,
    chebyshev_unit_power,
)
from fhemamba.selective_gates import DissipativeGate, fit_gate_roots


def test_recurrence_recorder_delegates_joint_gate_and_records_once():
    from fhemamba.m1_payload import _RecurrenceRecordingOps
    from fhemamba.ops import Exact

    class JointOps(Exact):
        def __init__(self):
            self.seen = []

        def checkpoint(self, value, site):
            self.seen.append(site)
            return value

        def mamba2_gates(self, z, a_cont, layer, time_step_limit):
            return z.square(), self.checkpoint(1 - z.square(), (layer, "decay_output"))

    base = JointOps()
    recorder = _RecurrenceRecordingOps(base)
    z = torch.tensor([[[0.25, 0.5]]])
    write, decay = recorder.mamba2_gates(z, -torch.ones(2), 3, (0, float("inf")))
    assert torch.equal(write, z.square())
    assert torch.equal(decay, 1 - z.square())
    assert base.seen == [(3, "decay_output")]
    assert len(recorder.records[(3, "decay_output")]) == 1
    assert torch.equal(recorder.records[(3, "decay_output")][0], decay[0, 0])


def test_integer_certificate_matches_fraction_oracle_on_subdivision_and_rejection():
    rng = np.random.default_rng(982)
    cases = [[0.5, 0, 0.5], [0.5, -0.5], [1.0, 2**-52], [1.0], [0.0]]
    cases.extend(rng.normal(size=degree + 1).tolist() for degree in (3, 7, 16, 31))
    for coefficients in cases:
        for lower, upper in ((0, 1), (-3, 3), (-1, 1)):
            for max_depth in (0, 8):
                for max_nodes in (1, 512):
                    kwargs = {
                        "lower": Fraction(lower),
                        "upper": Fraction(upper),
                        "max_depth": max_depth,
                        "max_nodes": max_nodes,
                    }
                    expected = certify_bounds(chebyshev_unit_power(coefficients), **kwargs)
                    actual = certify_chebyshev_bounds(coefficients, **kwargs)
                    assert actual["certified"] == expected["certified"]
                    assert actual["nodes"] == expected["nodes"]
                    if not expected["certified"]:
                        assert actual["reason"] == expected["reason"]


def test_integer_certificate_handles_high_degree_without_numerical_basis_conversion():
    # T_128 spans [-1,1]; adding an exact binary epsilon breaks the upper bound.
    coefficients = [0.0] * 128 + [1.0]
    assert certify_chebyshev_bounds(coefficients, lower=Fraction(-2), upper=Fraction(2))[
        "certified"
    ]
    coefficients[0] = 2**-52
    assert not certify_chebyshev_bounds(coefficients, lower=Fraction(-1), upper=Fraction(1))[
        "certified"
    ]


def test_shared_factor_certificate_handles_perfect_memory_and_reset():
    for root in ([0.0], [1.0], [0.5, 0.5]):
        certificate = certify_dissipative_gate(root, [2.0, -0.25])
        assert certificate["certified"]
        assert Fraction(certificate["gain"]) >= Fraction(9, 4) ** 2
    assert not certify_dissipative_gate([1.0, 1e-12], [2.0])["certified"]


def test_public_gate_fits_preserve_euler_write_and_bound_adversarial_states():
    lo, hi, rates = np.array([-16.0] * 3), np.array([12.0] * 3), np.array([0.25, 3.0, 60.0])
    p, q = fit_gate_roots(
        lo,
        hi,
        rates,
        dissipation_degree=192,
        equilibrium_degree=128,
        inward_margin=2e-6,
    )
    certificates = [certify_dissipative_gate(p[:, h].tolist(), q[:, h].tolist()) for h in range(3)]
    assert all(c["certified"] for c in certificates)
    gate = DissipativeGate(lo, hi, p, q)
    z = torch.linspace(-16, 12, 1025, dtype=torch.float64)[:, None].expand(-1, 3)
    write, decay = gate(z)
    target_write = torch.nn.functional.softplus(z)
    target_decay = torch.exp(-target_write * torch.from_numpy(rates))
    assert float((write - target_write).abs().max()) < 1e-4
    assert float((decay - target_decay).abs().max()) < 5e-6
    gains = torch.tensor([c["gain"] for c in certificates], dtype=torch.float64)
    assert bool(((decay >= 0) & (decay <= 1)).all())
    assert bool((write <= (1 - decay) * gains + 1e-14).all())
    state = gains.clone()
    for k in range(4096):
        index = (k * 751) % len(z)
        state = decay[index] * state + write[index] * (-1 if k % 7 else 1)
    assert bool((state.abs() <= gains).all())
    # Clenshaw remains a polynomial outside the domain; it must not clamp.
    outside_write, _ = gate(torch.tensor([[80.0] * 3], dtype=torch.float64))
    assert bool((outside_write > write[-1]).all())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_joint_gate_inputs_fail_closed(bad):
    with pytest.raises(ValueError, match="finite"):
        certify_dissipative_gate([0.5], [bad])
    with pytest.raises(ValueError, match="finite"):
        DissipativeGate(np.array([0]), np.array([1]), np.array([[bad]]), np.array([[1]]))
    with pytest.raises(ValueError, match="finite"):
        fit_gate_roots(
            np.array([0]),
            np.array([1]),
            np.array([bad]),
            dissipation_degree=16,
            equilibrium_degree=16,
            inward_margin=1e-6,
        )
