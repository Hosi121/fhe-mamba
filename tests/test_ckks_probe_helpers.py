"""Numerical oracles for shared probe circuits, without optional OpenFHE."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fhemamba.ckks_probes import (
    CkksPrimitives,
    affine,
    build_toy_context,
    cheb_eval64,
    floor_coeffs,
)


class ArrayBackend:
    def MakeCKKSPackedPlaintext(self, values):  # noqa: N802 - external OpenFHE API
        return np.asarray(values)

    def Encrypt(self, key, values):  # noqa: N802
        return values.copy()

    def EvalMult(self, left, right):  # noqa: N802
        return left * right

    def EvalAdd(self, left, right):  # noqa: N802
        return left + right

    def EvalSub(self, left, right):  # noqa: N802
        return left - right


@pytest.mark.parametrize("degree", [0, 1, 7, 8, 15, 16, 31, 32, 79])
def test_shared_chebyshev_matches_independent_numpy_oracle(degree):
    he = CkksPrimitives(ArrayBackend(), SimpleNamespace(publicKey=None), batch=37)
    x = np.linspace(-0.9, 0.9, 37)
    coefficients = np.random.default_rng(degree).normal(size=degree + 1)
    if degree > 2:
        coefficients[2] = 1e-15
    coefficients = floor_coeffs(coefficients)
    actual = he.eval_chebyshev(x, coefficients)
    np.testing.assert_allclose(actual, cheb_eval64(coefficients, x), rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(he.eval_chebyshev(x, [0.0] * 8), np.zeros(37))
    assert affine(2, 6) == (0.5, -2.0)


def test_probe_padding_and_newton_tail_preserve_the_circuit():
    he = CkksPrimitives(ArrayBackend(), SimpleNamespace(publicKey=None), batch=4)
    np.testing.assert_array_equal(
        he.encrypt(np.array([1.0, 2.0]), fill=-0.25), [1, 2, -0.25, -0.25]
    )
    x, y = np.array([1.0, 2.0, 3.0, 4.0]), np.full(4, 0.6)
    expected = y.copy()
    for _ in range(3):
        expected = 1.5 * expected - 0.5 * x * expected**3
    np.testing.assert_allclose(he.newton_refine(y, -0.5 * x, 3), expected)


def test_shared_context_keeps_explicit_geometry_and_toy_security(monkeypatch):
    calls = {}

    class Params:
        def __getattr__(self, name):
            return lambda value: calls.update({name: value})

    context = SimpleNamespace(
        Enable=lambda feature: None,
        KeyGen=lambda: SimpleNamespace(secretKey="secret"),
        EvalMultKeyGen=lambda key: calls.update(mult_key=key),
        EvalRotateKeyGen=lambda key, rotations: calls.update(rotations=rotations),
    )
    backend = SimpleNamespace(
        CCParamsCKKSRNS=Params,
        GenCryptoContext=lambda params: context,
        KeySwitchTechnique=SimpleNamespace(HYBRID="hybrid"),
        ScalingTechnique=SimpleNamespace(FLEXIBLEAUTO="flexible"),
        SecretKeyDist=SimpleNamespace(UNIFORM_TERNARY="ternary"),
        SecurityLevel=SimpleNamespace(HEStd_NotSet="not-set"),
        PKESchemeFeature=SimpleNamespace(
            **dict.fromkeys(["PKE", "KEYSWITCH", "LEVELEDSHE", "ADVANCEDSHE", "FHE"], "feature")
        ),
    )
    monkeypatch.setitem(sys.modules, "openfhe", backend)
    cc, _ = build_toy_context(
        ring_dimension=16384,
        batch=8192,
        depth=40,
        scale_bits=59,
        first_mod_bits=60,
        rotations=[1, -1],
    )
    assert cc is context
    assert calls == {
        "SetSecretKeyDist": "ternary",
        "SetSecurityLevel": "not-set",
        "SetRingDim": 16384,
        "SetScalingTechnique": "flexible",
        "SetFirstModSize": 60,
        "SetScalingModSize": 59,
        "SetKeySwitchTechnique": "hybrid",
        "SetMultiplicativeDepth": 40,
        "SetBatchSize": 8192,
        "mult_key": "secret",
        "rotations": [1, -1],
    }
