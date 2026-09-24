"""Architecture-neutral packed arithmetic program and polynomial calibration.

Each value occupies the leading slots of one ciphertext. Indexing/broadcasts
become public gather maps; no model operation is evaluated with decrypted data.
The extended format supports finite-history language-model sessions, shared
weight tables, non-power-of-two widths and genuine client feedback.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .ops import ChebPoly, fit_chebyshev
from .tensor_ops import TensorOps, nonlinear_function


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


class Calibration(TensorOps):
    def __init__(self):
        super().__init__()
        self.ranges = {}
        self.functions = {}

    def nonlinear(self, x, kind, site, parameter=0.0):
        lo, hi = float(x.min()), float(x.max())
        old = self.ranges.get(site, (lo, hi))
        self.ranges[site] = min(lo, old[0]), max(hi, old[1])
        self.functions[site] = kind, parameter
        return super().nonlinear(x, kind, site, parameter)

    def fit(
        self, *, margin=0.5, tolerance=1e-6, max_degree=255, quadrature=False, kind_tolerances=None
    ):
        polys, metadata = {}, {}
        for site, (lo, hi) in self.ranges.items():
            kind, parameter = self.functions[site]
            target = (kind_tolerances or {}).get(kind, tolerance)
            pad = max(hi - lo, 0.05) * margin
            lo, hi = lo - pad, hi + pad
            if kind == "inv_sqrt":
                lo = max(self.ranges[site][0] * 0.5, 1e-6)
            function = nonlinear_function(kind, parameter)
            grid = torch.linspace(lo, hi, 8193, dtype=torch.float64)
            degree = 7
            while degree <= max_degree:
                if quadrature:
                    # On Chebyshev nodes the columns are orthogonal: a DCT
                    # computes the same discrete least-squares projection.
                    count = max(4 * (degree + 1), 256)
                    k = np.arange(count)
                    t = np.cos(np.pi * (k + 0.5) / count)
                    y = function(torch.from_numpy((t + 1) * (hi - lo) / 2 + lo)).numpy()
                    spectrum = np.fft.fft(np.concatenate((y, y[::-1])))
                    c = (
                        spectrum[: degree + 1] * np.exp(-1j * np.pi * k[: degree + 1] / (2 * count))
                    ).real / count
                    c[0] *= 0.5
                    poly = ChebPoly(tuple(map(float, c)), lo, hi)
                else:
                    poly = fit_chebyshev(function, lo, hi, degree)
                error = float((poly(grid) - function(grid)).abs().max())
                if error <= target:
                    break
                degree = 2 * degree + 1
            else:
                raise ValueError(f"{site}: fit failed within degree {max_degree}")
            polys[site] = poly
            metadata[f"{site[0]}:{site[1]}"] = {
                "kind": kind,
                "parameter": parameter,
                "lo": lo,
                "hi": hi,
                "degree": degree,
                "grid_max_abs_error": error,
                **({"fit_tolerance": target} if kind_tolerances else {}),
                "grid_points": len(grid),
                "interval_certified": False,
                "coefficients": list(poly.coeffs),
            }
        return PolynomialTensorOps(polys), metadata


class PolynomialTensorOps(TensorOps):
    def __init__(self, polynomials: dict[tuple[int, str], ChebPoly]):
        super().__init__()
        self.polynomials = polynomials

    def nonlinear(self, x, kind, site, parameter=0.0):
        poly = self.polynomials[site]
        if not torch.isfinite(x).all() or torch.any(x < poly.lo) or torch.any(x > poly.hi):
            raise ValueError(f"{site}: input outside frozen polynomial domain")
        return poly(x)


@dataclass
class PackedValue:
    program: PackedProgram
    index: int
    value: np.ndarray

    @property
    def shape(self):
        return self.value.shape

    def reshape(self, *shape):
        return PackedValue(self.program, self.index, self.value.reshape(*shape))

    def __getitem__(self, index):
        mapping = np.arange(self.value.size).reshape(self.shape)[index]
        return self.program.gather(self, mapping)

    def __add__(self, other):
        return self.program.binary(self, other, "add")

    __radd__ = __add__

    def __mul__(self, other):
        return self.program.binary(self, other, "mul")

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return self * -1 + other


class PackedProgram(TensorOps):
    def __init__(self, polynomials, *, slots=1024, bound=64.0, extended=False):
        super().__init__()
        if slots < 2 or slots & (slots - 1) or slots > 32768:
            raise ValueError("slots must be a power of two between 2 and 32768")
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("refresh bound must be finite and positive")
        self.polynomials, self.slots, self.bound = polynomials, slots, bound
        self.nodes = []
        self.outputs = []
        self.extended = extended
        self.linear_weights = {}
        self.node_bounds = []

    def node(self, operation, parents, data, value):
        value = _array(value)
        if value.size < 1 or value.size > self.slots:
            raise ValueError("packed value must fit in one ciphertext")
        if not np.isfinite(value).all() or np.max(np.abs(value)) > self.bound:
            raise ValueError(f"{operation}: fixture exceeds fixed public refresh bound")
        data = np.asarray(data, dtype=np.float64).reshape(-1)
        if not np.isfinite(data).all():
            raise ValueError("program constants must be finite")
        index = len(self.nodes)
        self.node_bounds.append(min(self.bound, max(1.0, float(np.abs(value).max()) * 8)))
        self.nodes.append(
            (
                operation,
                [p.index for p in parents],
                value.size,
                data if self.extended else data.tolist(),
            )
        )
        return PackedValue(self, index, value)

    def input(self, value):
        return self.node("input", [], _array(value), value)

    def constant(self, value):
        return self.node("public", [], _array(value), value)

    def client_input(self, hidden, reference_embedding):
        """Client decrypts final hidden, selects a token, encrypts its embedding.

        reference_embedding is only the compiler's plaintext oracle. It is not
        serialized: the runtime must use its actual decrypted result.
        """
        if not self.extended:
            raise ValueError("client feedback requires the extended packed format")
        return self.node("feedback", [hidden], [], _array(reference_embedding))

    def gather(self, value, mapping):
        mapping = np.asarray(mapping)
        if mapping.size == value.value.size and np.array_equal(
            mapping.ravel(), np.arange(mapping.size)
        ):
            return value.reshape(*mapping.shape)
        return self.node("gather", [value], mapping, value.value.ravel()[mapping])

    def _broadcast(self, value, shape):
        if value.shape == shape:
            return value
        dimensions = [1] * (len(shape) - len(value.shape)) + list(value.shape)
        value = value.reshape(*dimensions)
        # Expand trailing axes first: adjacent singleton dimensions then share
        # a contiguous block, avoiding one rotation/mask per destination slot.
        for axis in reversed(range(len(shape))):
            if dimensions[axis] == shape[axis]:
                continue
            if dimensions[axis] != 1:
                raise ValueError("incompatible broadcast shapes")
            repeat = shape[axis]
            if repeat & (repeat - 1) and not self.extended:
                mapping = np.broadcast_to(np.arange(value.value.size).reshape(value.shape), shape)
                return self.gather(value, mapping)
            outer = math.prod(dimensions[:axis])
            inner = math.prod(dimensions[axis + 1 :])
            dimensions[axis] = repeat
            out = np.broadcast_to(value.value, dimensions).copy()
            value = self.node("repeat", [value], [outer, inner, repeat], out)
        return value

    def binary(self, left, right, operation):
        if isinstance(right, PackedValue) and right.program is not self:
            raise ValueError("cannot combine values from different packed programs")
        b = right.value if isinstance(right, PackedValue) else _array(right)
        shape = np.broadcast_shapes(left.shape, b.shape)
        left = self._broadcast(left, shape)
        fn = np.add if operation == "add" else np.multiply
        if isinstance(right, PackedValue):
            right = self._broadcast(right, shape)
            return self.node(operation, [left, right], [], fn(left.value, right.value))
        b = np.broadcast_to(b, shape)
        return self.node(operation + "p", [left], b, fn(left.value, b))

    def linear(self, x, weight):
        key = id(weight)
        owner = weight
        weight = _array(weight)
        if x.value.size != weight.shape[1]:
            raise ValueError("packed linear currently requires batch size one")
        if self.extended and key in self.linear_weights:
            return self.node("linear_ref", [x], [self.linear_weights[key][1]], x.value @ weight.T)
        out = self.node("linear", [x], weight, x.value @ weight.T)
        if self.extended:
            self.linear_weights[key] = (owner, out.index)
        return out

    def sum_last(self, x):
        width = x.shape[-1]
        if width & (width - 1):
            if not self.extended:
                raise ValueError("packed reduction width must be a power of two")
            padded = 1 << (width - 1).bit_length()
            shape = (*x.shape[:-1], padded)
            value = np.zeros(shape)
            value[..., :width] = x.value
            destination = np.arange(value.size).reshape(shape)[..., :width]
            x = self.node("scatter", [x], destination, value)
            width = padded
        value = x.value.sum(-1, keepdims=True)
        return self.node("sum", [x], [width], value)

    def concatenate(self, values, axis=-1):
        shape = list(values[0].shape)
        shape[axis] = sum(value.shape[axis] for value in values)
        destination = np.arange(math.prod(shape)).reshape(shape)
        offset, result = 0, None
        for value in values:
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(offset, offset + value.shape[axis])
            dest = destination[tuple(slices)].ravel()
            out = np.zeros(shape)
            out.flat[dest] = value.value.ravel()
            piece = self.node("scatter", [value], dest, out)
            result = piece if result is None else result + piece
            offset += value.shape[axis]
        return result

    def nonlinear(self, x, kind, site, parameter=0.0):
        poly = self.polynomials[site]
        plain = torch.from_numpy(x.value)
        if np.any(x.value < poly.lo) or np.any(x.value > poly.hi):
            raise ValueError(f"{site}: input outside frozen polynomial domain")
        return self.node("cheb", [x], [poly.lo, poly.hi, *poly.coeffs], poly(plain))

    def output(self, value, exact):
        exact = _array(exact).reshape(-1)
        if exact.size != value.value.size or not np.isfinite(exact).all():
            raise ValueError("invalid exact output reference")
        self.outputs.append((value.index, value.value.ravel().tolist(), exact.tolist()))

    def write(self, path: str | Path):
        # A bounded, versioned text format keeps the native executor independent
        # of Python/Torch and of architecture-specific JSON payload readers.
        with Path(path).open("w") as stream:
            stream.write(
                f"fhemamba-packed-v{2 if self.extended else 1} {self.slots} {self.bound:.17g} "
                f"{len(self.nodes)} {len(self.outputs)}\n"
            )
            for index, (op, parents, size, data) in enumerate(self.nodes):
                fields = [op, str(size)]
                if self.extended:
                    fields.append(format(self.node_bounds[index], ".17g"))
                fields.extend([str(len(parents)), *map(str, parents), str(len(data))])
                fields.extend(format(v, ".17g") for v in data)
                stream.write(" ".join(fields) + "\n")
            for index, polynomial, exact in self.outputs:
                fields = [str(index), str(len(polynomial))]
                fields.extend(format(v, ".17g") for v in [*polynomial, *exact])
                stream.write(" ".join(fields) + "\n")

    def summary(self):
        return {
            "format": f"fhemamba-packed-v{2 if self.extended else 1}",
            "slots": self.slots,
            "public_refresh_bound": self.bound,
            "nodes": len(self.nodes),
            "operations": dict(Counter(n[0] for n in self.nodes)),
            "outputs": len(self.outputs),
            "encrypted_benchmark": False,
        }
