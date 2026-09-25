# Arithmetic structure and kernel opportunities

The next small experiment should preserve square identity through the shared
arithmetic interface. The larger opportunity is to schedule independent
polynomials in shared SIMD lanes and keep their layouts through refresh. These
are proposals supported by a new static inventory, **not measured speedups**.
No inference implementation, cryptographic parameter, approximation polynomial,
error gate, or DGX executable changed for this study.

The later [refresh scheduling study](2026-09-25-packed-frontiers.md) implements
and measures polynomial grouping, persistent windows and ready-node scheduling.
It selects scheduling alone; the optimistic product count below is not a
polynomial latency gain.

The analysis uses main `88870f9`, the frozen 12-layer/five-evaluation Mamba-3
program, and the adopted 949.257737-second DGX Spark result. Its counts reproduce
the measured 50,780 rotations outside refresh and 12,005 ciphertext products.
[Source, counts and derivations](../../results/cpu/2026-09-25/math-kernel-audit/README.md)
are retained. Mamba-2 source paths were inspected for transfer opportunities;
these Mamba-3 counts are not Mamba-2 measurements.

## What must change for a large gain

Subtracting each operation's nested refresh timer gives:

| Existing category | Seconds | If this whole category alone were twice as fast |
| --- | ---: | ---: |
| Refresh, including its packing and correction work | 515.043 | 27.13% shorter evaluation |
| Gather/scatter/repeat/sum, outside refresh | 188.096 | 9.91% shorter |
| Linear maps, outside refresh | 131.224 | 6.91% shorter |
| Polynomial nodes, outside refresh | 100.558 | 5.30% shorter |

These are conditional Amdahl scenarios, not forecasts. The 169.527 seconds of
host encoding are nested within these categories and must not be added again.
Even eliminating every other operation while leaving refresh unchanged would
cap speedup at **1.843×**. This is a limit of that restricted optimization
strategy, not a cryptographic lower bound. A twofold overall improvement must
also reduce refresh time, through fewer refreshes or faster backend operations.

## 1. Preserve squares and use ring identities inside the kernel

The current Mamba-3 evaluator calls general multiplication for both
`mul(basis(i/2), basis(i/2))` and DAG nodes with identical parents. Cloning the
two operands before dispatch hides their identity from the backend.

The static inventory finds **4,141 basis squares + 245 DAG squares = 4,386**,
or 36.53% of the 12,005 ciphertext multiplications outside bootstrap internals.
For each RNS coefficient, the unrelinearized product is

\[
(c_0+c_1s)(d_0+d_1s)
=c_0d_0+(c_0d_1+c_1d_0)s+c_1d_1s^2.
\]

A general product uses four modular multiplications; a square needs three:
`c0*c0`, `c0*c1`, `c1*c1`, doubling the middle term. The relevant product kernel
also reads two input words instead of four. Relinearization and its transforms
remain necessary, so neither 36.53% nor 25% is a whole-inference saving.

This is an existing reachable specialization, not an imagined kernel:

`EvalSquareInPlace → Ciphertext::square → mult(*this) → binomialMult(square=true)
→ binomialSquare_`.

Some older square implementations in `Ciphertext.cpp` are disabled, but the
active general routine forwards the alias flag to `LimbPartition::binomialMult`,
which dispatches the dedicated kernel. The inspected implementation is bound
by [backend source hashes](../../results/cpu/2026-09-25/math-kernel-audit/backend-sources.json).

Preserve the square operation **before cloning** in the shared arithmetic
interface. Mamba-2's Chebyshev and normalization code has related patterns;
changing only Mamba-3's helper would miss them. First compare exact RNS words,
metadata and unmodified live inputs at degree one and two, then compare a
carried-state prefix. No new key or approximation is required.

For the remaining general products, a separate kernel candidate is the
degree-one Karatsuba identity, in each ring modulo `q`:

\[
u=c_0d_0,\quad v=c_1d_1,\quad
w=(c_0+c_1)(d_0+d_1)-u-v.
\]

This uses three modular products and four modular additions/subtractions,
instead of four products and one addition. Canonical modular sums keep the
existing multiplication input ranges; no CKKS rescale boundary moves. The
extra additions and register dependencies might cost more than the removed
product, so compare the actual GB10 kernel with the existing Barrett path.
Keep the cheaper dedicated square path. A successful backend change could
also affect bootstrap internals and both models, but that reach is unmeasured.

## 2. Share the rotation computation graph

NAF minimizes the signed-digit weight of an individual offset. It does not
minimize the work for a **set** of rotations of one ciphertext. Existing linear
baby rotations and routing transforms independently replay common NAF prefixes.
A prefix trie computes each such intermediate once. Sibling edges can then
share the same ModUp/decomposition using the backend's vector rotation API.

| Static quantity, excluding refresh | Current | Proposed count |
| --- | ---: | ---: |
| Rotation/key-switch outputs, sharing identical prefixes | 50,780 | 50,180 |
| ModUp source preparations, also hoisting sibling edges | 50,780 | 46,700 |

These count reductions are **600 rotations (1.18%)** and **4,080 preparations
(8.03%)**, respectively. They overlap and must not be added as saved rotations.
The analysis conservatively leaves giant rotations and rotation-sum branches
separate. Every edge remains an existing signed-power-of-two key, and every
output retains its original NAF sequence.

The scalar GPU `EvalFastRotationPrecompute` returns null; replacing a scalar
call's name does not create reuse. The vector overload reaches
`rotate_hoisted`, which actually shares ModUp. Its output allocation currently
copies inputs, and its streams need the existing lifetime guarantees. Test
exact RNS parity and peak live storage before assigning a time saving.

Full fused linear transforms are a larger, separate option. Mamba-2 already
uses that path for all 120 measured output projections. It is not a new
optimization there. Applying it to Mamba-3 changes direct-key requirements,
scale handling and possibly rounding; its full-model error gates still need
validation. Historical B300 Mamba-2 results cannot predict a GB10 Mamba-3 gain.

## 3. Evaluate compatible nonlinear functions together

There are two distinct forms of sharing.

**Same encrypted argument:** all 60 sin/cos pairs share the parent, shape and
normalization interval. The existing recursive Chebyshev evaluator rebuilds
their bases separately. Sharing only the requested basis union removes
**465 ciphertext multiplications** on the static graph, without changing the
polynomial coefficients. Cached bases must be keyed by the actual input
version, level, scale and interval: a refresh can change the input ciphertext.

**Independent arguments in different lanes:** let the packed argument `u`
contain blocks `u1, ..., um`. In block `j`, encode coefficients `a[j,k]` for its
existing polynomial. Componentwise Chebyshev evaluation gives

\[
\sum_k A_k\odot T_k(u)
=\bigl(p_1(u_1),\ldots,p_m(u_m)\bigr),
\]

where `A_k` holds each block's public coefficient. Nonlinear work is shared
across the SIMD lanes, even when the functions or intervals differ. Each lane
retains its existing polynomial; this is not replacing functions by a new
approximation. Encoding, alignment and rounding still change in FHE execution.

The analyzer assigns every node the maximum number of polynomial nodes on an
ancestor path. Polynomial nodes at the same rank form an antichain: none
depends on another. Grouping by that rank, client-feedback epoch and exact
polynomial degree gives the following **optimistic circuit inventory**:

| Quantity | Separate evaluation | Grouped polynomial cores |
| --- | ---: | ---: |
| Polynomial evaluations/groups | 761 | 408 |
| Ciphertext products inside polynomial cores | 9,991 | 5,980 |

There are 171 groups with multiple nodes; the largest has 10 nodes. Every group
fits in 32,768 slots; maximum occupied width is **3,188 slots**. The 4,011
removed polynomial-core products amount to 40.15% of those products, or 33.41%
of current model ciphertext products. **These are not net operation or latency
savings:** input-level compatibility, packing/unpacking, public-vector encoding,
changed refresh placement, liveness and error are not modeled. The same-input
465-product opportunity overlaps this candidate.

Mamba-2's joint selective-gate evaluator already demonstrates vector-coefficient
polynomials and periodic encoding in this repository. Generalize that mechanism
into a shared evaluator and scheduler rather than reimplementing it independently
for Mamba-3. The smallest useful trial is one ready group from one layer with
all packing costs included, followed by the unchanged carried-state gate.

## 4. Keep layouts symbolic and plan refresh on ready groups

The current compiler materializes each broadcast, gather and concatenation.
The 188.096-second routing category makes this worth a separate compiler pass.
For a coordinate permutation or gather `P`,

\[
P(x\odot y)=(Px)\odot(Py),\qquad P(x+y)=Px+Py.
\]

Track the logical-to-physical slot mapping through elementwise operations;
materialize only when layouts disagree, a reduction needs it, or an externally
observable value requires it. Compatible selectors can be composed, and public
linear maps can absorb permutations. Reductions are not componentwise: a sum
cannot be moved using these identities. Inactive slots and polynomial constant
terms must retain the existing padding correction.

This inventory found **no duplicate routing node with the exact same parent
and map**, so a simple node cache does not solve the problem. Algebraic movement
of layouts, with re-evaluated depth and error, is the relevant change.

The refresh scheduler currently scans already-produced values in program order.
A ready-node scheduler can instead bring independent compatible branches to
their refresh points together. Model this jointly with nonlinear packing:
packing once and retaining the layout can avoid paying two sets of routing
costs. Optimize a cost that includes level-dependent arithmetic, physical
bootstrap events, preparation and peak live buffers, rather than logical
refresh count alone. This is a proposed scheduling objective, not an optimality
claim or an executable schedule.

The current run has 363 refresh events, 726 physical bootstraps and a maximum
group of eight despite a cap of sixteen. Raising the cap alone is unsupported.
Dependencies and compatible levels, not just empty slots, determine which
values can share a refresh. Removing the second precision-recovery pass would
be a different numerical change and is not assumed here.

## Backend research boundaries

Shoup twiddles, two-dimensional NTT, fused transforms, limb batching and lazy
base-conversion reductions already exist. The inspected current source, rather
than the older paper's feature list, determines what is missing. The
[FIDESlib paper, Sections III-F2–F6](https://arxiv.org/html/2507.04775v1#S3.SS6)
explains these established building blocks; their existence alone does not
measure the current patched GB10 backend.

An additional research candidate is integrating rescale into double-hoisted
linear transforms. [Min, Lee and Song, Algorithm 2](https://eprint.iacr.org/2025/429.pdf)
derive a saving of `beta * n2 * (L - Lprime)` word-sized transforms, with `beta`
decomposition digits, `n2` giant steps and `L - Lprime` removed limbs. The current
backend already has conditional ModDown hoisting; this newer transformation
must be compared against that implementation, not credited with all hoisting
savings. Its rounding and output-level schedule differ, so exact arithmetic
identities alone do not establish model parity.

FP16/BF16 model weights do not imply FP16/BF16 ciphertext arithmetic: the
current residues use approximately 59/60-bit primes. Tensor-core use would need
an exact digit decomposition, accumulation bounds, carry/reduction handling and
conversion-cost accounting. It is a backend research project, not a dtype switch.
Likewise, a new NTT tile or radix needs a current kernel profile; the older
short-prefix trace predates several adopted copy reductions.

The proposed order is: square dispatch, a Karatsuba kernel comparison, one
joint nonlinear group, then a layout/refresh scheduler. The first two provide
small bounded tests; the latter two target the work required for a substantial
end-to-end reduction. None of these proposals has been adopted by this study.
