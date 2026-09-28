"""Shared CPU CKKS probe arithmetic, independent of optional OpenFHE bindings.

The small-ring context below is explicitly a feasibility probe, not classical-128.
"""

from __future__ import annotations

import numpy as np

CHEB_COEFF_FLOOR = 1e-12


def floor_coeffs(coeffs) -> list[float]:
    """Apply the kernel's coefficient floor so ct and replica share terms."""
    return [0.0 if abs(float(c)) < CHEB_COEFF_FLOOR else float(c) for c in coeffs]


def affine(lo: float, hi: float) -> tuple[float, float]:
    """(a, b) with u = a*x + b mapping [lo, hi] -> [-1, 1] (kernel affine())."""
    return 2.0 / (hi - lo), -(lo + hi) / (hi - lo)


def cheb_eval64(coeffs: list[float], u: np.ndarray) -> np.ndarray:
    return np.polynomial.chebyshev.chebval(u, np.asarray(coeffs, dtype=np.float64))


def ceil_log2(value: int) -> int:
    log = 0
    while (1 << log) < value:
        log += 1
    return log


def cheb_baby_size(degree: int) -> int:
    levels = max(1, ceil_log2(degree + 1))
    return 1 << ((levels + 1) // 2)


def cheb_ps_depth(degree: int) -> int:
    """Port of the kernel's level ledger for the PS recursion (estimate only)."""
    m = cheb_baby_size(degree)
    t_level: dict[int, int] = {}

    def level_of(i: int) -> int:
        if i <= 1:
            return 0
        if i in t_level:
            return t_level[i]
        if i % 2 == 0:
            level = level_of(i // 2) + 1
        else:
            level = max(level_of((i + 1) // 2), level_of(i // 2)) + 1
        t_level[i] = level
        return level

    def rec(n: int) -> int:
        if n == 0:
            return 0
        if n < m:
            return max(level_of(i) for i in range(1, n + 1)) + 1
        k = m
        while 2 * k - 1 < n:
            k *= 2
        giant = max(level_of(k), rec(n - k)) + 1
        return max(rec(k - 1), giant)

    return rec(degree)


class CkksPrimitives:
    """Thin wrapper mirroring the kernel's elementary helpers."""

    def __init__(
        self, cc, keys, *, batch: int, coefficient_floor: float = CHEB_COEFF_FLOOR
    ) -> None:
        self.batch = batch
        self.coefficient_floor = coefficient_floor
        self.cc = cc
        self.keys = keys
        # Kernel parity: constant ciphertexts come from a scaled ones ct.
        self.ones_ct = self.encrypt(np.ones(self.batch))

    def encrypt(self, values: np.ndarray, fill: float = 0.0):
        # Unused tail slots get an in-range fill so no slot ever drives a
        # polynomial outside its fit domain (CKKS has no clamp).
        vec = np.full(self.batch, fill, dtype=np.float64)
        vec[: len(values)] = values
        pt = self.cc.MakeCKKSPackedPlaintext([float(v) for v in vec])
        return self.cc.Encrypt(self.keys.publicKey, pt)

    def decrypt(self, ct, n: int | None = None) -> np.ndarray:
        pt = self.cc.Decrypt(ct, self.keys.secretKey)
        pt.SetLength(self.batch)
        return np.asarray(pt.GetRealPackedValue(), dtype=np.float64)[:n]

    def mask_mult(self, ct, mask: np.ndarray):
        pt = self.cc.MakeCKKSPackedPlaintext([float(v) for v in mask])
        return self.cc.EvalMult(ct, pt)

    def const_ct(self, value: float):
        return self.cc.EvalMult(self.ones_ct, float(value))

    def affine_ct(self, ct, a: float, b: float):
        """u = a*ct + b: the kernel's mask-mult + add-const fold (1 level)."""
        return self.cc.EvalAdd(self.cc.EvalMult(ct, float(a)), float(b))

    # -- Chebyshev Paterson-Stockmeyer (port of the kernel's eval_chebyshev) --
    def eval_chebyshev(self, u, coeffs: list[float]):
        degree = len(coeffs) - 1
        if degree < 1:
            return self.const_ct(coeffs[0] if coeffs else 0.0)
        cc = self.cc
        m = cheb_baby_size(degree)
        t_cache = {1: u}

        def get_t(i: int):
            if i in t_cache:
                return t_cache[i]
            if i % 2 == 0:  # T_{2i} = 2*T_i^2 - 1
                half = get_t(i // 2)
                square = cc.EvalMult(half, half)
                value = cc.EvalSub(cc.EvalAdd(square, square), 1.0)
            else:  # T_{2i+1} = 2*T_{i+1}*T_i - T_1
                product = cc.EvalMult(get_t((i + 1) // 2), get_t(i // 2))
                value = cc.EvalSub(cc.EvalAdd(product, product), u)
            t_cache[i] = value
            return value

        def rec(c: list[float]):
            n = len(c) - 1
            if n < m:
                acc = None
                for i in range(1, n + 1):
                    if abs(c[i]) < self.coefficient_floor:
                        continue
                    term = cc.EvalMult(get_t(i), c[i])
                    acc = term if acc is None else cc.EvalAdd(acc, term)
                if acc is None:
                    return self.const_ct(c[0])
                if abs(c[0]) >= self.coefficient_floor:
                    acc = cc.EvalAdd(acc, c[0])
                return acc
            k = m
            while 2 * k - 1 < n:
                k *= 2
            btil = [2.0 * c[k + j] for j in range(n - k + 1)]
            btil[0] = c[k]
            aprime = list(c[:k])
            for i in range(k + 1, n + 1):
                aprime[2 * k - i] -= c[i]
            giant = cc.EvalMult(get_t(k), rec(btil))
            return cc.EvalAdd(rec(aprime), giant)

        return rec(list(coeffs))

    # -- Newton inverse-sqrt refinement (port of the kernel's newton_refine) --
    def newton_refine(self, y, v_neg_half, iterations: int):
        cc = self.cc
        for _ in range(iterations):
            y_squared = cc.EvalMult(y, y)
            vy = cc.EvalMult(v_neg_half, y)
            product = cc.EvalMult(vy, y_squared)
            y = cc.EvalAdd(cc.EvalMult(y, 1.5), product)
        return y


def build_toy_context(*, ring_dimension, batch, depth, scale_bits, first_mod_bits, rotations=()):
    from openfhe import (
        CCParamsCKKSRNS,
        GenCryptoContext,
        KeySwitchTechnique,
        PKESchemeFeature,
        ScalingTechnique,
        SecretKeyDist,
        SecurityLevel,
    )

    params = CCParamsCKKSRNS()
    params.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
    params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)  # TOY, see docstring
    params.SetRingDim(ring_dimension)
    params.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    params.SetFirstModSize(first_mod_bits)
    params.SetScalingModSize(scale_bits)
    params.SetKeySwitchTechnique(KeySwitchTechnique.HYBRID)
    params.SetMultiplicativeDepth(depth)
    params.SetBatchSize(batch)
    cc = GenCryptoContext(params)
    for feature in (
        PKESchemeFeature.PKE,
        PKESchemeFeature.KEYSWITCH,
        PKESchemeFeature.LEVELEDSHE,
        PKESchemeFeature.ADVANCEDSHE,
        PKESchemeFeature.FHE,
    ):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    if rotations:
        cc.EvalRotateKeyGen(keys.secretKey, list(rotations))
    return cc, keys
