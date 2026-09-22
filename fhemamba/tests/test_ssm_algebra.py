import math

import pytest
import torch
from fhemamba.ssm_algebra import (
    DeferredMamba2State,
    complex_trapezoid_scan,
    cubic_phase,
    rsqrt_residual_step,
)


@pytest.mark.parametrize("window", [1, 2, 7, 32])
def test_deferred_matches_dense_with_zero_decay_and_partial_flush(window):
    gen = torch.Generator().manual_seed(17)
    state = torch.randn(3, 5, 7, generator=gen, dtype=torch.float64)
    deferred = DeferredMamba2State(state, window)
    for step in range(35):
        decay = torch.rand(3, generator=gen, dtype=torch.float64)
        if step in (3, 19):
            decay[1] = 0  # An inverse-of-prefix-product implementation would fail.
        u = torch.randn(3, 5, generator=gen, dtype=torch.float64)
        b = torch.randn(3, 7, generator=gen, dtype=torch.float64)
        c = torch.randn(3, 7, generator=gen, dtype=torch.float64)
        state = decay[:, None, None] * state + torch.einsum("hp,hn->hpn", u, b)
        expected = torch.einsum("hpn,hn->hp", state, c)
        torch.testing.assert_close(deferred.step(decay, u, b, c), expected)
        torch.testing.assert_close(deferred.materialize(), state)
        assert len(deferred.pending) < window
    assert deferred.flushes == 35 // window
    deferred.flush()
    torch.testing.assert_close(deferred.base, state)
    previous_flushes = deferred.flushes
    deferred.flush()
    assert deferred.flushes == previous_flushes


def test_deferred_rejects_invalid_shapes_and_window():
    with pytest.raises(ValueError, match="shape"):
        DeferredMamba2State(torch.zeros(3, 4), 2)
    for window in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            DeferredMamba2State(torch.zeros(3, 4, 5), window)
    deferred = DeferredMamba2State(torch.zeros(3, 4, 5), 2)
    with pytest.raises(ValueError, match="shapes"):
        deferred.step(torch.ones(3), torch.ones(4), torch.ones(3, 5), torch.ones(3, 5))


def test_complex_trapezoid_matches_independent_real_two_by_two_rotation():
    gen = torch.Generator().manual_seed(19)
    time, heads, channels, size = 13, 2, 3, 4
    decay, dt, mixing = torch.rand(3, time, heads, generator=gen, dtype=torch.float64)
    angle = torch.randn(time, heads, size, generator=gen, dtype=torch.float64)
    x = torch.randn(time, heads, channels, generator=gen, dtype=torch.float64)
    b = torch.randn(time, heads, size, 2, generator=gen, dtype=torch.float64)
    c = torch.randn(time, heads, size, 2, generator=gen, dtype=torch.float64)
    outputs, final = complex_trapezoid_scan(
        decay,
        dt,
        mixing,
        torch.exp(1j * angle),
        x,
        torch.view_as_complex(b),
        torch.view_as_complex(c),
    )
    state = torch.zeros(heads, channels, size, 2, dtype=torch.float64)
    previous = torch.zeros_like(state)
    expected = []
    for t in range(time):
        rotation = torch.stack(
            [torch.cos(angle[t]), -torch.sin(angle[t]), torch.sin(angle[t]), torch.cos(angle[t])],
            dim=-1,
        ).reshape(heads, size, 2, 2)
        current = torch.einsum("hp,hnr->hpnr", x[t], b[t])
        rotate_input = state + ((1 - mixing[t]) * dt[t])[:, None, None, None] * previous
        state = (
            decay[t, :, None, None, None] * torch.einsum("hnij,hpnj->hpni", rotation, rotate_input)
            + (mixing[t] * dt[t])[:, None, None, None] * current
        )
        expected.append(torch.einsum("hpnr,hnr->hp", state, c[t]))
        previous = current
    torch.testing.assert_close(outputs, torch.stack(expected))
    torch.testing.assert_close(torch.view_as_real(final), state)
    with pytest.raises(ValueError, match="shapes"):
        complex_trapezoid_scan(decay, dt, mixing, angle, x, b, c)


def test_cubic_phase_radial_identity_and_memory_loss():
    angle = torch.linspace(-math.sqrt(3), math.sqrt(3), 1001, dtype=torch.float64)
    norm_squared = cubic_phase(angle).abs().square()
    torch.testing.assert_close(norm_squared, 1 - angle**4 / 12 + angle**6 / 36)
    assert float(norm_squared.max()) <= 1 + 1e-14
    # Contractivity alone does not preserve the model's long-term memory.
    assert float(cubic_phase(torch.tensor(0.5)).abs() ** 1024) < 0.1
    assert float(cubic_phase(torch.tensor(2.0)).abs()) > 1


def test_newton_residual_identity_including_outside_basin():
    variance = torch.tensor([0.01, 0.5, 1.0, 9.0], dtype=torch.float64)
    guess = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    residual = 1 - variance * guess.square()
    refined = guess * (1.5 - 0.5 * variance * guess.square())
    torch.testing.assert_close(1 - variance * refined.square(), rsqrt_residual_step(residual))
    assert refined[-1] < 0  # More Newton steps cannot be an unconditional repair.
