"""Stateful Mamba-3 and shared lowering contracts, without GPU dependencies."""

from dataclasses import fields

import numpy as np
import pytest
import torch

from fhemamba.architectures import init_mixer_state, mixer_architecture, mixer_forward_dispatch
from fhemamba.mamba3 import Mamba3Mixer, Mamba3State, init_mamba3_state, mamba3_step
from fhemamba.ops import ChebPoly, Exact, RangeRecorder
from fhemamba.packed_program import Calibration, PackedProgram
from fhemamba.tensor_ops import (
    HeadTiledState,
    TensorOps,
    nonlinear_function,
    rms_normalize,
    ssm_update_readout,
)


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


@pytest.mark.parametrize("tiled", [False, True])
def test_polynomial_and_packed_full_mixer_agree(tmp_path, tiled):
    torch.manual_seed(19)
    mixer = Mamba3Mixer(8, d_state=8, headdim=8, dtype=torch.float64)
    x = (torch.rand(1, 24, 8, dtype=torch.float64) - 0.5) * 0.2
    calibration = Calibration()
    with torch.no_grad():
        mixer(x, ops=calibration)
        poly, _ = calibration.fit(tolerance=1e-5)
        program = PackedProgram(poly.polynomials, slots=64 if tiled else 128)
        state = init_mamba3_state(mixer)
        packed = Mamba3State(
            *(
                program.recurrent_state(state.ssm)
                if tiled and f.name == "ssm"
                else program.constant(getattr(state, f.name))
                for f in fields(state)
            )
        )
        for t in range(3):
            expected, state = mamba3_step(mixer, x[:, t], state, poly)
            actual, packed = mamba3_step(mixer, program.input(x[:, t]), packed, program)
            np.testing.assert_allclose(actual.value, expected.numpy(), atol=1e-12, rtol=1e-12)
            for f in fields(state):
                value = getattr(packed, f.name)
                value = (
                    np.concatenate([block.value for block in value.blocks], axis=1)
                    if isinstance(value, HeadTiledState)
                    else value.value
                )
                np.testing.assert_allclose(
                    value,
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


@pytest.mark.parametrize("factored", [False, True])
@pytest.mark.parametrize("rotary_fraction", [0.5, 1.0])
@torch.no_grad()
def test_phase_composition_matches_accumulated_angles_across_many_turns(factored, rotary_fraction):
    from fhemamba.mamba3 import init_factored_mamba3_state

    torch.manual_seed(51)
    mixer = Mamba3Mixer(
        8,
        d_state=8,
        headdim=8,
        rope_fraction=rotary_fraction,
        is_outproj_norm=True,
        dtype=torch.float64,
    )
    mixer.dt_bias.fill_(2.0)
    initialize = init_factored_mamba3_state if factored else init_mamba3_state
    angle = initialize(mixer, batch_size=2)
    phase = initialize(mixer, batch_size=2, rotary="phasor")
    for x in torch.randn(96, 2, 8, dtype=torch.float64):
        expected, angle = mamba3_step(mixer, x, angle, TensorOps())
        actual, phase = mamba3_step(mixer, x, phase, TensorOps())
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        torch.testing.assert_close(phase.angle.cosine, angle.angle.cos(), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(phase.angle.sine, angle.angle.sin(), atol=1e-12, rtol=1e-12)
    assert angle.angle.abs().max() > 2 * torch.pi


@torch.no_grad()
def test_packed_phase_composition_with_tiled_state_matches_polynomial_reference():
    from fhemamba.mamba3 import RotaryPhase

    torch.manual_seed(23)
    mixer = Mamba3Mixer(8, d_state=8, headdim=8, dtype=torch.float64)
    inputs = torch.randn(20, 1, 8, dtype=torch.float64) * 0.1
    calibration = Calibration()
    state = init_mamba3_state(mixer, rotary="phasor")
    for x in inputs:
        _, state = mamba3_step(mixer, x, state, calibration)
    poly, recipes = calibration.fit(tolerance=1e-5)
    assert "0:m3_delta_cos" in recipes
    assert "0:m3_cos" not in recipes
    program = PackedProgram(poly.polynomials, slots=64)
    state = init_mamba3_state(mixer, rotary="phasor")
    packed = Mamba3State(
        RotaryPhase(program.constant(state.angle.cosine), program.constant(state.angle.sine)),
        program.recurrent_state(state.ssm),
        program.constant(state.k),
        program.constant(state.v),
    )
    for x in inputs[:8]:
        expected, state = mamba3_step(mixer, x, state, poly)
        actual, packed = mamba3_step(mixer, program.input(x), packed, program)
        np.testing.assert_allclose(actual.value, expected.numpy(), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            packed.angle.cosine.value, state.angle.cosine.numpy(), atol=1e-12
        )
        np.testing.assert_allclose(packed.angle.sine.value, state.angle.sine.numpy(), atol=1e-12)


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


def test_tiled_recurrence_keeps_all_heads_and_matches_outer_product_oracle():
    generator = torch.Generator().manual_seed(73)

    def random(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    state = random(2, 5, 4, 8)
    tiled = HeadTiledState((state[:, :2], state[:, 2:4], state[:, 4:]))
    for _ in range(21):
        decay = torch.sigmoid(random(2, 5))
        writes = [(random(2, 5, 8), random(2, 5, 4), random(2, 5)) for _ in range(2)]
        query = random(2, 5, 8)
        state = state * decay[..., None, None]
        for key, value, weight in writes:
            state += value[..., None] * key[..., None, :] * weight[..., None, None]
        tiled, output = ssm_update_readout(tiled, decay, writes, query, TensorOps())
        assert tiled.shape == (2, 5, 4, 8)
        assert len(tiled.blocks) == 3
        torch.testing.assert_close(torch.cat(tiled.blocks, dim=1), state, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(output, (state * query[..., None, :]).sum(-1))


def test_packed_recurrent_state_geometry_and_encrypted_initial_state():
    program = PackedProgram({}, slots=64)
    state = program.recurrent_state(np.ones((1, 5, 4, 8)), encrypted=True)
    assert [block.shape[1] for block in state.blocks] == [2, 2, 1]
    assert state.shape == (1, 5, 4, 8)
    assert program.summary()["operations"] == {"input": 3}
    with pytest.raises(ValueError, match="one recurrent state head"):
        program.recurrent_state(np.zeros((1, 1, 16, 8)))
    with pytest.raises(ValueError, match="recurrent state must"):
        program.recurrent_state(np.zeros((1, 5, 8)))
    with pytest.raises(ValueError, match="rank-four"):
        HeadTiledState(())
    with pytest.raises(ValueError, match="matching"):
        HeadTiledState((np.zeros((1, 2, 4, 8)), np.zeros((2, 2, 4, 8))))
