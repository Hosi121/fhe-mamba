# Necessary bounds for normalization polynomials

This offline analysis uses the frozen generation payload and the completed
42.9-minute periodic-coefficient baseline. It was computed locally while the
subring generation ran on Spark. The calculator checks the payload
digest, bundle digest, and all 49 embedded normalization recipes before
producing [the numerical report](../../results/dgx/2026-09-22/subring-gates/normalization-bounds.json).

## A bound that allows a different approximation

Let a real polynomial `p(v)` approximate the inverse square root over the
entire public variance interval `[lo, hi]`, with the existing relative-error
target `e = 1e-7`:

```text
|sqrt(v) p(v) - 1| <= e.
```

If `p` has degree `d`, the residual `r(v) = v p(v)^2 - 1` has degree at most
`m = 2d+1`. Its magnitude on the interval is at most `delta = 2e+e^2`, while
`r(0) = -1`. Map the interval to `[-1,1]`; zero maps to
`-(hi+lo)/(hi-lo)`, outside it.

For a polynomial bounded by one on `[-1,1]`, the degree-`m` Chebyshev
polynomial gives the largest possible magnitude at any fixed real point
outside the interval. To see why, suppose another polynomial exceeded it.
Subtract a scaled Chebyshev polynomial chosen to match at that exterior
point. At the `m+1` alternating Chebyshev extrema, the difference must
alternate sign, giving `m` interior roots and one exterior root. That is
impossible for a nonzero degree-`m` polynomial.

Applying this property to `r/delta` gives the necessary inequality

```text
1 <= delta cosh((2d+1) acosh((hi+lo)/(hi-lo)))

d >= ceil((acosh(1/delta) / acosh((hi+lo)/(hi-lo)) - 1) / 2).
```

This constrains **any polynomial in the variance** satisfying the full
interval and precision contract, including a replacement for the current
recipe. It does not construct a polynomial at that degree. Since ordinary
binary multiplication depth `D` permits degree at most `2^D`, the necessary
degree also implies `D >= ceil(log2(d))`.

The formula is evaluated at 80 and 120 decimal digits using the stored
binary64 endpoints and tolerance; all integer bounds agree. An independent
binary64 calculation uses the stable identity
`acosh((hi+lo)/(hi-lo)) = 2 atanh(sqrt(lo/hi))`. For all 49 operators it agrees
on the integer and checks that the previous degree fails the inequality.
This is a high-precision evaluation of an analytic bound, not an outward
interval certificate for transcendental arithmetic.

## A stronger bound when the current polynomial is fixed

Each current update is `y <- a*y - b*v*y^3`, starting with a nonzero public
constant seed. Its degree obeys `d_next = 3*d + 1`. All stored `b` values
are positive, so the leading term does not cancel. With `n` scheduled
updates and the final Newton repair, the exact polynomial degree is
`(3^(n+1)-1)/2`. The balanced implementation uses `2n` dependent ciphertext
products, excluding public coefficient scaling and rescaling.

| Scheduled updates | Operators | Current balanced depth | Exact current polynomial: necessary depth | Any polynomial with the same interval/error: necessary depth |
| ---: | ---: | ---: | ---: | ---: |
| 9 | 1 | 18 | 15 | 11 |
| 10 | 2 | 20 | 17 | 12 |
| 11 | 2 | 22 | 19 | 13 |
| 12 | 13 | 24 | 20 | 14–16 |
| 13 | 17 | 26 | 22 | 16–17 |
| 14 | 11 | 28 | 23 | 17–18 |
| 15 | 3 | 30 | 25 | 19 |

The widest interval is approximately `[1e-5, 58265.9]`. A polynomial with
relative error at most `1e-7` over that interval needs degree at least
**307,582**, hence at least **19 dependent binary multiplication stages**.
The current exact polynomial has degree **21,523,360**, giving a stronger
**25-stage** necessary bound if that polynomial is preserved. Its balanced
evaluation uses 30 stages. Neither gap proves a circuit achieving the bound;
a dense polynomial at the degree floor could have a much worse product count.

These are arithmetic depth bounds measured from the variance input. They
exclude CKKS noise, scalar rescaling, precision repair, and the cost of
forming the variance. Bootstrapping preserves the intended value and
replenishes physical level capacity, so its required count also depends on
the level schedule and all live operands. In particular, refreshing `y`
does not refresh the original variance branch. The earlier
[integration failure](2026-09-21-normalization-integration.md#keeping-the-variance-branch-live)
demonstrates why a depth bound alone cannot certify removing a refresh.

## Measured budget for the next experiment

The baseline's recorded internal normalization refreshes, including the
dedicated final norm, are:

| Checkpoint stage | Logical events | Physical bootstraps | Recorded seconds |
| --- | ---: | ---: | ---: |
| 5 | 240 | 480 | 237.177 |
| 11 | 230 | 460 | 227.276 |
| Total | 470 | 940 | 464.454 |

Each event here uses two physical bootstraps for precision repair. These
timers already sit inside the normalization phase timers; adding them again
would double-count time. The block/gated subtotal is 920 physical calls;
the dedicated final norm adds 20.

If the stage-11 refresh cost alone disappeared and every other cost stayed
fixed, the 2573.366-second baseline would fall to **2346.090 seconds
(39.10 minutes, 8.83% less)**. Making every internal normalization refresh
free would yield **35.15 minutes (18.05% less)**. These are conditional
budgets, not measured speedups or achievable runtime guarantees.

The next bounded experiment should target one fewer refresh while preserving
the original interval, `1e-7` certificate, and immutable variance identity.
First derive the level and noise schedule; then check a separate encrypted
normalization probe before changing the full model. An alternative
compositional approximation needs a new certificate. Reducing the existing
headroom or disabling the second precision-repair bootstrap alone is not
supported by this calculation or by the earlier failed integration runs.

## Reproduce

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python experiments/analyze_normalization_bounds.py \
  --native results/dgx/2026-09-22/periodic-gates/m2_chain_periodic-client-generation_l24_t5.json \
  --payload runs/client-generation-20260922/payload \
  --bundle config/mamba2-130m-normalization-20260921.json \
  --output runs/normalization-bounds.json
```

The report binds source artifacts and script hashes. It is an analysis
artifact and must not be treated as an encrypted benchmark result.
