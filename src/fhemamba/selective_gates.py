"""Polynomial Mamba-2 gates with a shared dissipation factor.

For c>0 and d=softplus(z), the exact Euler-write gates admit
    p=sqrt(1-exp(-c*d)), q=sqrt(d/(1-exp(-c*d))),
    a=1-p*p, b=p*p*q*q.
Only p and q are approximated; evaluation needs no division by private data.
The removable limit of q at d=0 is 1/sqrt(c). Certificates and domain
membership are separate from float/CKKS error, and from model quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


def fit_gate_roots(
    lo: np.ndarray,
    hi: np.ndarray,
    rates: np.ndarray,
    *,
    dissipation_degree: int,
    equilibrium_degree: int,
    inward_margin: float,
    trim_l1: float = 1e-11,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit public functions, independently of activation or quality samples.

    The inward margin is a fitting policy, not a proof; certify the actual
    returned p coefficients before relying on |p|<=1. Trailing coefficient
    trimming changes the polynomial and is included before certification.
    """
    lo, hi, rates = (np.asarray(x, dtype=np.float64) for x in (lo, hi, rates))
    if (
        lo.ndim != 1
        or not lo.size
        or hi.shape != lo.shape
        or rates.shape != lo.shape
        or not all(np.isfinite(x).all() for x in (lo, hi, rates))
        or np.any(lo >= hi)
        or np.any(rates <= 0)
        or min(dissipation_degree, equilibrium_degree) < 1
        or not np.isfinite(inward_margin)
        or inward_margin <= 0
        or not np.isfinite(trim_l1)
        or trim_l1 < 0
    ):
        raise ValueError("finite head domains, positive rates/degrees/margin and trim required")
    result = []
    for root, degree in (("dissipation", dissipation_degree), ("equilibrium", equilibrium_degree)):
        theta = np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1)
        z = (lo + hi)[None, :] / 2 + np.cos(theta)[:, None] * (hi - lo)[None, :] / 2
        dt = np.logaddexp(0, z)
        dissipation = -np.expm1(-rates[None, :] * dt)
        if root == "dissipation":
            values = np.sqrt(dissipation)
        else:
            quotient = np.broadcast_to(1 / rates, dt.shape).copy()
            np.divide(dt, dissipation, out=quotient, where=dissipation != 0)
            values = np.sqrt(quotient)
        coefficients = (2 / (degree + 1)) * np.cos(np.outer(np.arange(degree + 1), theta)) @ values
        coefficients[0] *= 0.5
        if root == "dissipation":
            coefficients /= 1 + inward_margin
        for head in range(lo.size):
            omitted = 0.0
            for k in range(degree, 0, -1):
                candidate = omitted + abs(coefficients[k, head])
                if candidate > trim_l1:
                    break
                omitted = candidate
                coefficients[k, head] = 0
        last = np.flatnonzero(np.any(coefficients != 0, axis=1))[-1]
        result.append(coefficients[: last + 1].copy())
    return result[0], result[1]


@dataclass
class DissipativeGate:
    """Per-head polynomial oracle; no data-dependent clipping or exponentials."""

    lo: np.ndarray
    hi: np.ndarray
    dissipation_root: np.ndarray
    equilibrium_root: np.ndarray
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        for name in ("lo", "hi", "dissipation_root", "equilibrium_root"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64).copy())
        heads = self.lo.size
        if (
            self.lo.shape != (heads,)
            or heads == 0
            or self.hi.shape != (heads,)
            or np.any(self.lo >= self.hi)
            or any(
                x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != heads
                for x in (self.dissipation_root, self.equilibrium_root)
            )
            or not all(
                np.isfinite(x).all()
                for x in (self.lo, self.hi, self.dissipation_root, self.equilibrium_root)
            )
        ):
            raise ValueError("finite head domains and degree-by-head coefficient matrices required")
        for x in (self.lo, self.hi, self.dissipation_root, self.equilibrium_root):
            x.flags.writeable = False

    def __call__(self, z: Tensor) -> tuple[Tensor, Tensor]:
        if z.shape[-1] != self.lo.size:
            raise ValueError("input head count differs from polynomial gate")
        constants = self._cache.get(z.device)
        if constants is None:
            constants = tuple(
                torch.tensor(x, dtype=torch.float64, device=z.device)
                for x in (self.lo, self.hi, self.dissipation_root, self.equilibrium_root)
            )
            self._cache[z.device] = constants
        lo, hi, p_coeff, q_coeff = constants
        t = (2 * z.double() - (lo + hi)) / (hi - lo)

        def clenshaw(coefficients):
            previous, before = torch.zeros_like(t), torch.zeros_like(t)
            for coefficient in coefficients[1:].flip(0):
                previous, before = 2 * t * previous - before + coefficient, previous
            return t * previous - before + coefficients[0]

        p, q = clenshaw(p_coeff), clenshaw(q_coeff)
        dissipation = p.square()
        write, decay = dissipation * q.square(), 1 - dissipation
        return write.to(z.dtype), decay.to(z.dtype)
