import math

import torch
from fhemamba.phase_algebra import shear_rotate


def test_shears_match_polynomial_matrix_and_preserve_area():
    angles = torch.linspace(-1.9, 1.9, 39, dtype=torch.float64)
    pair = torch.stack((torch.sin(angles), torch.cos(angles)), -1)
    # Independent expanded matrix, including the cubic off-diagonal term.
    diagonal = 1 - angles.square() / 2
    upper = -angles + angles.pow(3) / 4
    expected = torch.stack(
        (diagonal * pair[:, 0] + upper * pair[:, 1], angles * pair[:, 0] + diagonal * pair[:, 1]),
        -1,
    )
    torch.testing.assert_close(shear_rotate(pair, angles), expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(diagonal.square() - upper * angles, torch.ones_like(angles))


def test_constant_angle_preserves_modified_energy_but_changes_phase():
    angle = torch.tensor(0.5, dtype=torch.float64)
    k = 1 - angle.square() / 4
    pair = torch.tensor([1.0, 0.0], dtype=torch.float64)
    for _ in range(1024):
        pair = shear_rotate(pair, angle)
        energy = pair[0].square() + k * pair[1].square()
        torch.testing.assert_close(
            energy, torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=0
        )
        assert 1 - 1e-12 <= float(pair.norm()) <= float(k.rsqrt()) + 1e-12
    assert (2 * math.asin(0.5 / 2) - 0.5) * 1024 > 5


def test_changing_small_angles_can_grow_despite_unit_determinants():
    identity = torch.eye(2, dtype=torch.float64)
    a = shear_rotate(identity, torch.tensor(0.5)).T
    b = shear_rotate(identity, torch.tensor(0.1, dtype=torch.float64)).T
    period = torch.linalg.matrix_power(b, 16) @ torch.linalg.matrix_power(a, 3)
    torch.testing.assert_close(torch.linalg.det(period), torch.tensor(1.0, dtype=torch.float64))
    assert float(torch.trace(period)) < -2
    assert float(torch.linalg.eigvals(period).abs().max()) > 1.018
