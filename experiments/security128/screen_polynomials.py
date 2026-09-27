"""Screen a frozen packed program for direct slim-polynomial applicability."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def parity_split(coefficients):
    """Return A, B with p(x) = A(T_2(x)) + x B(T_2(x)); ordinary Chebyshev c0."""
    even, odd = coefficients[::2], coefficients[1::2]
    branch = [0.0] * len(odd)
    tail = 0.0
    for index in range(len(odd) - 1, -1, -1):
        tail = odd[index] - tail
        branch[index] = tail if index == 0 else 2 * tail
    return even, branch


def inspect_parity(rows):
    """Floating-point algebra screen only; not encrypted accuracy or a cost model."""
    import math

    def value(coefficients, x):
        current = previous = 0.0
        for coefficient in coefficients[:0:-1]:
            current, previous = coefficient + 2 * x * current - previous, current
        return coefficients[0] + x * current - previous if coefficients else 0.0

    points = sorted({-1.0, 0.0, 1.0, *(math.cos(math.pi * j / 128) for j in range(129))})
    maximum_error = maximum_relative_error = maximum_growth = 0.0
    both_odd = 0
    for row in rows:
        coefficients = row["coefficients"]
        even, branch = parity_split(coefficients)
        both_odd += (len(even) - 1) % 2 == 1 and (len(branch) - 1) % 2 == 1
        errors, reference = [], []
        for x in points:
            z = 2 * x * x - 1
            expected = value(coefficients, x)
            actual = value(even, z) + x * value(branch, z)
            reference.append(abs(expected))
            errors.append(abs(expected - actual))
        maximum_error = max(maximum_error, max(errors))
        maximum_relative_error = max(maximum_relative_error, max(errors) / max(1.0, max(reference)))
        norm = sum(map(abs, coefficients))
        if norm:
            maximum_growth = max(
                maximum_growth, (sum(map(abs, even)) + sum(map(abs, branch))) / norm
            )
    return {
        "scope": (
            "One exact algebraic split, checked in float64 at 131 or fewer normalized points; "
            "no coefficient dropping, FHE test, interval certificate or speed claim"
        ),
        "nodes": len(rows),
        "points_per_node": len(points),
        "max_abs_sample_error": maximum_error,
        "max_scale_relative_sample_error": maximum_relative_error,
        "max_branch_coefficient_l1_growth": maximum_growth,
        "both_branches_still_odd_degree": both_odd,
    }


def screen(program, check_parity=False):
    digest = hashlib.sha256()
    degrees, widths, shapes = Counter(), Counter(), Counter()
    exact_even = significant_even = power_two = 0
    polynomials = []
    with program.open("rb") as stream:
        header = next(stream)
        digest.update(header)
        fields = header.split()
        if fields[0] not in (b"fhemamba-packed-v1", b"fhemamba-packed-v2"):
            raise ValueError("not a packed program")
        count = int(fields[3])
        version_two = fields[0].endswith(b"v2")
        for index, line in enumerate(stream):
            digest.update(line)
            if index >= count or not line.startswith(b"cheb "):
                continue
            fields = line.split()
            parent_index = 3 if version_two else 2
            parents = int(fields[parent_index])
            data = list(map(float, fields[parent_index + parents + 2 :]))
            coefficients = data[2:]
            if check_parity:
                polynomials.append({"coefficients": coefficients})
            degree, width = len(coefficients) - 1, int(fields[1])
            degrees[degree] += 1
            widths[width] += 1
            shapes[width, degree] += 1
            while degree and coefficients[degree] == 0:
                degree -= 1
            exact_even += degree % 2 == 0
            power_two += degree > 0 and degree & (degree - 1) == 0
            while degree and abs(coefficients[degree]) < 1e-14:
                degree -= 1
            significant_even += degree % 2 == 0
    result = {
        "scope": "Static metadata; includes dead nodes; no coefficient changes or speed estimate",
        "program_sha256": digest.hexdigest(),
        "polynomial_nodes": sum(degrees.values()),
        "declared_degrees": dict(sorted(degrees.items())),
        "widths": dict(sorted(widths.items())),
        "even_declared_degree": sum(n for d, n in degrees.items() if d % 2 == 0),
        "exact_even_degree": exact_even,
        "significant_even_degree_1e14_screen": significant_even,
        "sos_direct_power_of_two_degree": power_two,
        "shapes": [{"width": w, "degree": d, "nodes": n} for (w, d), n in sorted(shapes.items())],
    }
    if check_parity:
        result["parity_split"] = inspect_parity(polynomials)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parity", action="store_true", help="check a Chebyshev parity adaptation in float64"
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(screen(args.program, args.parity), indent=2) + "\n")
