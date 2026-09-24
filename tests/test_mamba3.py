"""Stateful Mamba-3 and shared lowering contracts, without GPU dependencies."""

from dataclasses import fields

import numpy as np
import pytest
import torch

from fhemamba.architectures import init_mixer_state, mixer_architecture, mixer_forward_dispatch
from fhemamba.mamba3 import Mamba3Mixer, Mamba3State, init_mamba3_state, mamba3_step
from fhemamba.ops import ChebPoly, Exact, RangeRecorder
from fhemamba.packed_program import Calibration, PackedProgram
from fhemamba.tensor_ops import TensorOps, nonlinear_function, rms_normalize


@pytest.mark.parametrize("rotary_fraction", [0.5, 1.0])
@pytest.mark.parametrize("out_norm", [False, True])
def test_prefill_decode_and_architecture_state(rotary_fraction, out_norm):
    torch.manual_seed(32)
    mixer = Mamba3Mixer(
        8,
        d_state=8,
        headdim=8,
        rope_fraction=rotary_fraction,
        is_outproj_norm=out_norm,
        dtype=torch.float64,
    )
    x = torch.randn(2, 11, 8, dtype=torch.float64)
    state = init_mixer_state(mixer, batch_size=2)
    assert isinstance(state, Mamba3State)
    assert mixer_architecture(mixer) == "mamba3"
    whole = mixer(x)
    pieces = [mixer_forward_dispatch(mixer)(mixer, x[:, :3], state=state)]
    pieces.extend(mixer(x[:, i : i + 1], state=state) for i in range(3, 11))
    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, atol=1e-12, rtol=1e-12)
    assert state.ssm.abs().max() > 0
    assert state.angle.abs().max() > 0
    assert not hasattr(state, "conv")


def test_lagged_write_changes_output_and_state():
    torch.manual_seed(3)
    mixer = Mamba3Mixer(8, d_state=8, headdim=8, dtype=torch.float64)
    x = torch.randn(1, 2, 8, dtype=torch.float64)
    _, carry = mamba3_step(mixer, x[:, 0], init_mamba3_state(mixer), TensorOps())
    without_lag = Mamba3State(carry.angle, carry.ssm, carry.k * 0, carry.v)
    actual, new = mamba3_step(mixer, x[:, 1], carry, TensorOps())
    altered, different = mamba3_step(mixer, x[:, 1], without_lag, TensorOps())
    assert (actual - altered).abs().max() > 1e-5
    assert (new.ssm - different.ssm).abs().max() > 1e-5


def test_a_floor_and_new_nonlinearities_are_recorded():
    x = torch.tensor([-10.0, -1.0, 0.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(
        nonlinear_function("negative_a", 0.6)(x),
        torch.tensor([-0.6, -0.6, -1.0, -3.0], dtype=torch.float64),
    )
    recorder = RangeRecorder()
    ops = TensorOps(recorder)
    ops.nonlinear(x, "tanh", (0, "m3_angle_tanh"))
    assert recorder.ranges[(0, "m3_angle_tanh")] == (-10.0, 2.0)
    torch.testing.assert_close(TensorOps(Exact()).nonlinear(x, "sin", (0, "m3_sin")), x.sin())


def test_packed_gather_broadcast_reduce_and_linear():
    program = PackedProgram({}, slots=32)
    x = program.input(np.arange(6).reshape(2, 3) / 10)
    y = x[:, 1:2] * x
    np.testing.assert_allclose(y.value, x.value[:, 1:2] * x.value)
    z = program.concatenate((x[:, :2], x[:, 1:3]), axis=-1)
    np.testing.assert_allclose(program.sum_last(z).value, z.value.sum(-1, keepdims=True))
    weights = np.arange(12).reshape(2, 6) / 100
    np.testing.assert_allclose(
        program.linear(x.reshape(1, 6), weights).value, x.value.reshape(1, 6) @ weights.T
    )
    with pytest.raises(ValueError, match="power of two"):
        program.sum_last(x)
    with pytest.raises(ValueError, match="different packed programs"):
        _ = x + PackedProgram({}, slots=32).input(np.zeros((2, 3)))


def test_polynomial_and_packed_full_mixer_agree(tmp_path):
    torch.manual_seed(19)
    mixer = Mamba3Mixer(8, d_state=8, headdim=8, dtype=torch.float64)
    x = (torch.rand(1, 24, 8, dtype=torch.float64) - 0.5) * 0.2
    calibration = Calibration()
    with torch.no_grad():
        mixer(x, ops=calibration)
        poly, _ = calibration.fit(tolerance=1e-5)
        program = PackedProgram(poly.polynomials, slots=128)
        state = init_mamba3_state(mixer)
        packed = Mamba3State(*(program.constant(getattr(state, f.name)) for f in fields(state)))
        for t in range(3):
            expected, state = mamba3_step(mixer, x[:, t], state, poly)
            actual, packed = mamba3_step(mixer, program.input(x[:, t]), packed, program)
            np.testing.assert_allclose(actual.value, expected.numpy(), atol=1e-12, rtol=1e-12)
            for f in fields(state):
                np.testing.assert_allclose(
                    getattr(packed, f.name).value,
                    getattr(state, f.name).numpy(),
                    atol=1e-12,
                    rtol=1e-12,
                )
            program.output(actual, expected)
        program.write(tmp_path / "program.txt")
        assert (tmp_path / "program.txt").read_text().startswith("fhemamba-packed-v1")
        assert program.summary()["operations"]["cheb"] == 30


def test_no_exact_fallback_or_out_of_domain_clamp():
    from fhemamba.packed_program import PolynomialTensorOps

    ops = PolynomialTensorOps({(0, "sin"): ChebPoly((0.0, 1.0), -1.0, 1.0)})
    with pytest.raises(KeyError):
        ops.nonlinear(torch.tensor([0.1]), "cos", (0, "missing"))
    with pytest.raises(ValueError, match="outside frozen"):
        ops.nonlinear(torch.tensor([2.0]), "sin", (0, "sin"))


def test_unsupported_mimo_and_unknown_architecture_rejected():
    mixer = Mamba3Mixer(8, headdim=8)
    mixer.is_mimo = True
    with pytest.raises(NotImplementedError, match="MIMO"):
        init_mixer_state(mixer)
    with pytest.raises(ValueError, match="unsupported mixer"):
        mixer_architecture(torch.nn.Linear(8, 8))


def test_shared_rmsnorm_matches_torch():
    x = torch.randn(2, 16, dtype=torch.float64)
    weight = torch.randn(16, dtype=torch.float64)
    actual = rms_normalize(x, weight, 1e-5, TensorOps(), (0, "norm"))
    torch.testing.assert_close(actual, torch.nn.functional.rms_norm(x, (16,), weight, 1e-5))


@pytest.mark.parametrize("out_norm", [False, True])
def test_factored_state_matches_dense_recurrence_and_carried_state(out_norm):
    from fhemamba.mamba3 import init_factored_mamba3_state

    torch.manual_seed(47)
    mixer = Mamba3Mixer(8, d_state=8, headdim=8, is_outproj_norm=out_norm, dtype=torch.float64)
    dense = init_mamba3_state(mixer, 2)
    factored = init_factored_mamba3_state(mixer, 2)
    # Increase dt to exercise decay and the previous-token trapezoidal write.
    with torch.no_grad():
        mixer.dt_bias.fill_(0.3)
        for x in torch.randn(17, 2, 8, dtype=torch.float64):
            expected, dense = mamba3_step(mixer, x, dense, TensorOps())
            actual, factored = mamba3_step(mixer, x, factored, TensorOps())
            restored = sum(
                v[..., None] * k[..., None, :] * w[..., None, None]
                for k, v, w in zip(factored.keys, factored.values, factored.weights, strict=True)
            )
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(restored, dense.ssm, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(factored.angle, dense.angle, rtol=0, atol=0)
