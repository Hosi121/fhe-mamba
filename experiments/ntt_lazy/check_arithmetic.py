#!/usr/bin/env python3
"""Check the redundant-residue CT butterfly with unbounded integer arithmetic."""

import json
import random


def check():
    rng = random.Random(20260928)
    word = 1 << 64
    cases = 0
    for p in [3, 17, (1 << 59) - 55, (1 << 60) - 93, (1 << 62) - 57]:
        for _ in range(20000):
            a, b, w = rng.randrange(4 * p), rng.randrange(4 * p), rng.randrange(1, p)
            a0 = a - 2 * p if a >= 2 * p else a
            quotient = (b * (w * word // p)) // word
            t = (b * w - quotient * p) % word
            c, d = a0 + t, a0 + 2 * p - t
            assert 0 <= t < 2 * p
            assert 0 <= c < 4 * p
            assert 0 <= d < 4 * p
            assert c % p == (a + b * w) % p
            assert d % p == (a - b * w) % p
            cases += 1
    return {
        "passed": True,
        "cases": cases,
        "scope": "integer identity and range, not GPU validation",
    }


if __name__ == "__main__":
    print(json.dumps(check()))
