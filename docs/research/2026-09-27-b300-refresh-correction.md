# Classical-128 refresh correction merging

The twelve-layer Mamba-3 SISO 187M comparison falls from **235.839 to
210.326 seconds (10.82%)**, or **52.58 seconds per generated token**.
Both runs use the same classical-128 context, frozen program and absolute
`0.001` error gates. The candidate is adopted as the opt-in
`--merge-refresh-correction` path. Each arm is one fresh process; this is a
matched comparison, not a statistical significance claim.

## Mechanism

The two-pass refresh returns two components. Previously, extracting a batch
member rotated and masked both separately, then added them. The candidate
combines the components once with exact modular integer multiplication:

```text
combined = 4096 * first + second
output = extract(combined, mask / 4096)
```

`multIntScalar` preserves the CKKS scale and level. It is not the ordinary
encoded scalar product, which would spend another level. Shared inputs are
cloned before mutation. Context, key, level, scale degree, finite scale factor,
slots and Q-format must agree; incompatible inputs retain the original path.
The [probe and implementation guide](../../experiments/refresh_correction/README.md)
describe these preconditions and the normalized module gate.

## Qualification

The real probe checks six cases against a host modular-integer oracle, five
metadata rejection cases, six refresh layouts/levels, eighteen extractions
and eighteen following products. Its maximum normalized error is
`6.39382e-7`, below `1e-6`. The separate maximum absolute error for the large
scaled products is `0.193804`; that is not the model's absolute-error metric.
The full model retains its absolute `0.001` gates.

The complex probe checks the shared integer primitive (six residue cases and
five metadata rejection cases, maximum error `4.03412e-12`). The existing
S2C-first backend only accepts real packing, so this is not a Mamba-2 refresh
qualification. An initial complex probe was rejected by that backend guard.
An initial real probe crashed because its oracle read the unpopulated
`RawCipherText::moduli`; the corrected oracle reads the public Q prefix.
These failed attempts remain identifiable in the evidence and are excluded
from timing comparisons.

## Matched model comparison

| Measurement | Control | Merged correction |
| --- | ---: | ---: |
| Prefix evaluation | 14.040815 s | 13.367793 s |
| Full evaluation | 235.839497 s | 210.325738 s |
| Full refresh time, included above | 86.212448 s | 61.782071 s |
| Physical bootstrap calls | 484 | 484 |
| Ciphertext–ciphertext products | 11,540 | 11,540 |
| Ciphertext–plaintext products | 94,361 | 93,026 |
| Ordinary rotations | 59,489 | 55,426 |
| Maximum error vs exact reference | 4.59050e-5 | 4.64204e-5 |
| Maximum error vs polynomial reference | 3.84740e-6 | 3.59906e-6 |

The full candidate merges 242 correction pairs with zero fallbacks. Generated
IDs remain `[315,279,1614,315]`, with no non-finite values and no evaluator
decryptions. Five encrypted evaluations produce four generated tokens; dividing
evaluation time by four includes prompt work. Setup/key generation is separate
(141.84 / 144.04 seconds), as are client checks outside the evaluation timer.
The complete process takes 398.29 / 374.83 seconds.

The [parameter audit](2026-09-27-b300-classical128.md) is unchanged:
N=131,072, 65,536 slots, uniform ternary secret, Gaussian sigma 3.19,
HYBRID digits 4, depth 44, scaling 59 and first modulus 60. Actual QP is
3,376 bits versus the 3,523-bit classical guideline bound. Both runs use one
B300, the same CPU affinity and four OpenMP threads, two preparation workers,
2,048 cached plaintexts, S2C-first refresh and the existing even seed.
This does not add a separate-process client/server protocol.

## Reproduction and evidence

The [curated records](../../results/b300/2026-09-27/refresh-correction/README.md)
retain source/build identities, measured commands, probe results, full/prefix
JSONs and failed probe records. Disposable orchestration scripts are removed;
use the reusable [benchmark tools](../experiments.md).

```bash
python -m fhemamba.benchmarks compare \
  --baseline results/b300/2026-09-27/refresh-correction/full-correction-control-r3/native.json \
  --candidate results/b300/2026-09-27/refresh-correction/full-correction-candidate-r3/native.json \
  --contract config/packed-classical128-comparison.json
```

Both arms must use the same frozen program and client-head hashes. Add only
`--merge-refresh-correction` to the recorded control invocation. Rebuild the
packed executable; the pinned backend archives also include the previously
qualified refresh/rotation changes. This local result does not establish a
speedup for other rings, models or GPUs.
