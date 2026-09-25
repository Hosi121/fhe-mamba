"""Deterministic NTT primes for a 59-bit scale, two-limb RNS experiment."""
import math
from pathlib import Path

ROOT = Path(__file__).parent
used = set()

def prime(n):
    if n < 2:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def take(bits, count):
    n = (int(2**bits) - 2) // 131072 * 131072 + 1
    out = []
    while len(out) < count:
        if n not in used and prime(n):
            used.add(n)
            out.append(n)
        n -= 131072
    return out

def composite_pairs(bits, count):
    out = []
    for _ in range(count):
        p = take(bits / 2, 1)[0]
        target = ((1 << bits) // p - 1) // 131072 * 131072 + 1
        distance = 0
        while True:
            choices = (target,) if distance == 0 else (target-distance*131072, target+distance*131072)
            candidates = [q for q in choices if 2 < q < 2**31 and q not in used and prime(q)]
            if candidates:
                q = min(candidates, key=lambda q: abs(p*q-(1 << bits)))
                used.add(q)
                out.extend((p,q))
                break
            distance += 1
    return out

lines = ['#pragma once', '#include <vector>', '#include <cstdint>',
         'template<class W> struct CompositePrimes;']
for bits in (32, 64):
    used.clear()
    limbs = 2 if bits == 32 else 1
    q = take(62 / limbs, limbs)
    q += composite_pairs(59, 21) if bits == 32 else take(59, 21)
    q += composite_pairs(55, 12) if bits == 32 else take(55, 12)
    aux = take(bits - 1, math.ceil(len(q) / 3))
    lines += [f'template<> struct CompositePrimes<uint{bits}_t> {{',
              f'static constexpr int limbs = {limbs};',
              f'static std::vector<uint{bits}_t> q() {{ return {{{",".join(map(str,q))}}}; }}',
              f'static std::vector<uint{bits}_t> p() {{ return {{{",".join(map(str,aux))}}}; }}', '};']
(ROOT/'cheddar/composite_params.hpp').write_text('\n'.join(lines)+'\n')
