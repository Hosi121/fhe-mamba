"""Check the ownership permutation independently of CUDA modular arithmetic."""

import json
import random


def check(block, seed):
    rng = random.Random(seed)
    q = (1 << 61) - 1
    original = [rng.randrange(q) for _ in range(2 * block)]
    twiddles = [rng.randrange(q) for _ in range(block)]
    expected = original.copy()
    mask_psi = 2 * block - 32
    shift = 4
    m = 16
    while m:
        for tid in range(block):
            a = (tid & (m - 1)) | ((tid & ~(m - 1)) << 1)
            b = a + m
            x, y = expected[a], expected[b] * twiddles[(tid & mask_psi) >> shift] % q
            expected[a], expected[b] = (x + y) % q, (x - y) % q
        m >>= 1
        shift -= 1
        mask_psi |= mask_psi >> 1
    positions = [((tid & ~31) << 1) + (tid & 31) for tid in range(block)]
    lo = [original[p] for p in positions]
    hi = [original[p + 32] for p in positions]
    mask_psi = 2 * block - 32
    shift = 4
    m = 16
    while m:
        outgoing = [lo[t] if t & m else hi[t] for t in range(block)]
        for tid in range(block):
            partner = tid ^ m
            assert tid // 32 == partner // 32
            if tid & m:
                lo[tid] = outgoing[partner]
            else:
                hi[tid] = outgoing[partner]
            x, y = lo[tid], hi[tid] * twiddles[(tid & mask_psi) >> shift] % q
            lo[tid], hi[tid] = (x + y) % q, (x - y) % q
        m >>= 1
        shift -= 1
        mask_psi |= mask_psi >> 1
    actual = [x for pair in zip(lo, hi, strict=True) for x in pair]
    assert actual == expected, (block, seed)


if __name__ == "__main__":
    for block in [32, 64, 128, 256, 512, 1024]:
        for seed in range(32):
            check(block, seed)
    print(json.dumps({"passed": True, "cases": 192, "scope": "warp-tail lane mapping"}))
