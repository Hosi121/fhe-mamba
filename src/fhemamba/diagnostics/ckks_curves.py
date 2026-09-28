"""Parameterized CPU probe circuits, with an independent float64 replica."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from fhemamba.ckks_probes import affine, cheb_eval64, floor_coeffs
from fhemamba.ops import fit_chebyshev, fit_squared_exp


class Curve:
    def __init__(
        self, *, name, kind, interval, degree, iterations=0, damping=1.0, coefficient_floor=1e-12
    ):
        self.name, self.kind = name, kind
        self.interval, self.degree = interval, degree
        self.iterations, self.damping = iterations, damping
        lo, hi = interval
        if type(degree) is not int or type(iterations) is not int:
            raise ValueError("degree and iterations must be integers")
        if degree < 1 or lo >= hi or kind not in {"silu", "softplus", "exp", "rsqrt", "sq_rsqrt"}:
            raise ValueError("invalid curve kind, degree or interval")
        if kind in {"rsqrt", "sq_rsqrt"} and (lo <= 0 or iterations < 0 or damping <= 0):
            raise ValueError("inverse-square-root curves require positive inputs and damping")
        if kind not in {"rsqrt", "sq_rsqrt"} and (iterations or damping != 1.0):
            raise ValueError("iterations and damping apply only to inverse-square-root curves")
        self.squarings = 1 if kind == "softplus" else 0
        if kind == "exp":
            if hi != 0:
                raise ValueError("squared-exp interval must end at zero")
            exp = fit_squared_exp(lo, degree)
            fit, self.squarings = exp.base, exp.squarings
        else:
            function = {
                "silu": F.silu,
                "softplus": lambda x: torch.sqrt(F.softplus(x)),
                "rsqrt": torch.rsqrt,
                "sq_rsqrt": lambda x: x.abs().clamp(min=1e-30).pow(-0.25),
            }[kind]
            fit = fit_chebyshev(function, lo, hi, degree)
        self.base_interval = [fit.lo, fit.hi]
        self.a, self.b = affine(fit.lo, fit.hi)
        if kind == "exp":
            self.a /= 2.0**self.squarings
        coeffs = [damping * c for c in fit.coeffs] if kind == "rsqrt" else fit.coeffs
        self.coeffs = floor_coeffs(coeffs, coefficient_floor)

    def encrypted(self, he, ct, *, glue=None):
        scale, shift = (1.0, 0.0) if glue is None else glue
        u = he.affine_ct(ct, self.a * scale, self.a * shift + self.b)
        y = he.eval_chebyshev(u, self.coeffs)
        for _ in range(self.squarings):
            y = he.cc.EvalMult(y, y)
        if self.kind == "sq_rsqrt":
            y = he.cc.EvalMult(he.cc.EvalMult(y, y), self.damping)
        if self.kind in {"rsqrt", "sq_rsqrt"}:
            # Keep scalar multiplication alone for standalone probes. The chain
            # folds its input transform into both polynomial and Newton operands.
            v = (
                he.cc.EvalMult(ct, -0.5)
                if glue is None
                else he.affine_ct(ct, -0.5 * scale, -0.5 * shift)
            )
            y = he.newton_refine(y, v, self.iterations)
        return y

    def plain(self, x, *, glue=None):
        scale, shift = (1.0, 0.0) if glue is None else glue
        # Preserve the original replica's grouping: exp folds its input scale;
        # Newton first forms v, then maps v to the fitted interval.
        v = x if glue is None else scale * x + shift
        u = (
            (self.a * scale) * x + self.b
            if self.kind == "exp" and shift == 0
            else self.a * v + self.b
        )
        y = cheb_eval64(self.coeffs, u)
        if self.squarings:
            y = y ** (2**self.squarings)
        if self.kind == "sq_rsqrt":
            y = self.damping * y * y
        for _ in range(self.iterations):
            y = 1.5 * y + (-0.5 * v * y) * (y * y)
        return y

    def exact(self, x):
        if self.kind == "exp":
            return np.exp(x)
        if self.kind in {"rsqrt", "sq_rsqrt"}:
            return 1.0 / np.sqrt(x)
        return {"silu": F.silu, "softplus": F.softplus}[self.kind](torch.from_numpy(x)).numpy()

    def metadata(self):
        result = {"degree": self.degree, "interval": self.interval}
        if self.kind == "exp":
            result["squarings"] = self.squarings
        if self.kind in {"rsqrt", "sq_rsqrt"}:
            result.update(iterations=self.iterations, damping=self.damping)
        return result


def max_error(actual, expected):
    return float(np.max(np.abs(actual - expected)))
