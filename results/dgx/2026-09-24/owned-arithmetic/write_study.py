"""Describe only verified controls and completed model runs."""
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
ROOT = REPO / 'results/dgx/2026-09-24/owned-arithmetic'
read = lambda p: json.loads(p.read_text())
assert read(ROOT / 'postprocess-completion.json')['passed']
assert read(ROOT / 'postflight.json')['passed']
comparison = read(ROOT / 'comparison.json')
short, full = comparison['short'], comparison['full']
decisions = comparison['adoption']

prefix_rows = '\n'.join(
    f"| {arch} | {', '.join(f'{x:.6f}' for x in short[key]['samples_in_abba_order'])} | "
    f"{short[key]['base_mean_seconds']:.6f} → {short[key]['candidate_mean_seconds']:.6f} | "
    f"{short[key]['reduction_percent']:.2f}% |"
    for arch, key in [('Mamba-3', 'm3'), ('Mamba-2', 'm2')])
full_rows = []
details = []
for arch, key in [('Mamba-3', 'm3'), ('Mamba-2', 'm2')]:
    result = full[key]
    if result.get('skipped'):
        full_rows.append(f'| {arch} | — | — | — | Skipped after prefix |')
        continue
    full_rows.append(f"| {arch} | {result['base_seconds']:.6f} | {result['candidate_seconds']:.6f} | "
                     f"{result['reduction_percent']:.2f}% | {'Adopt' if decisions[key] else 'Do not adopt'} |")
    base = read(ROOT / f'{key}-full-base/native.json')
    candidate = read(ROOT / f'{key}-full-candidate/native.json')
    if key == 'm3':
        detail = f'''### Mamba-3

The 12-layer, five-evaluation request retains IDs `[315,279,1614,315]`,
`The capital of the state of`. The candidate uses
**{candidate['owned_arithmetic_calls']:,} ownership-aware additions**, reuses
**{candidate['owned_arithmetic_reused_inputs']:,} input buffers** and performs
**{candidate['owned_arithmetic_cloned_inputs']:,} defensive helper clones**.
The addition count matches the preceding static inventory exactly. The other
existing scratch and DAG-final-use paths remain separate counters.

| Full-run measurement | Baseline | Candidate |
| --- | --- | --- |
| Evaluation seconds | {base['eval_seconds']:.6f} | {candidate['eval_seconds']:.6f} |
| Host encoding seconds | {base['host_encoding_seconds']:.6f} | {candidate['host_encoding_seconds']:.6f} |
| Upload seconds | {base['plaintext_upload_seconds']:.6f} | {candidate['plaintext_upload_seconds']:.6f} |
| Bootstrap seconds | {base['bootstrap_seconds']:.6f} | {candidate['bootstrap_seconds']:.6f} |
| Peak RSS GiB | {base['peak_rss_gib']:.6f} | {candidate['peak_rss_gib']:.6f} |
| Maximum exact error | {base['max_abs_error_vs_exact']:.9g} | {candidate['max_abs_error_vs_exact']:.9g} |
| Maximum polynomial error | {base['max_abs_error_vs_polynomial']:.9g} | {candidate['max_abs_error_vs_polynomial']:.9g} |

Both runs retain {candidate['bootstraps']:,} bootstraps,
{candidate['rotations']-candidate['refresh_rotations']:,} model rotations excluding
refresh, {candidate['host_encodes']:,} encodes and
{candidate['plaintext_cache_hits']:,} cache hits. Counts, refresh schedule,
level/depth parameters, weights and both 0.001 gates are unchanged.
'''
    else:
        a, b = base['measurements'], candidate['measurements']
        stats = b['owned_arithmetic']
        detail = f'''### Mamba-2

The 24-layer, five-evaluation request retains IDs `[273,253,4687,273]`,
`The capital of the Republic of`. The helper executes
**{stats['calls']:,} calls**, reusing {stats['reused_inputs']:,} already-private
operands and adding {stats['cloned_inputs']:,} defensive helper clones.
Thus {stats['calls']:,} backend result copies are removed. The two input
protection clones at each original normalization/joint-gate call remain.
This count covers additions, subtractions and normalization as well as the
8,280 joint multiplications identified by the earlier static lower bound.

| Full-run measurement | Baseline | Candidate |
| --- | --- | --- |
| Evaluation seconds | {base['timing']['eval_seconds']:.6f} | {candidate['timing']['eval_seconds']:.6f} |
| Bootstrap seconds | {base['timing']['bootstrap_eval_seconds']:.6f} | {candidate['timing']['bootstrap_eval_seconds']:.6f} |
| Peak RSS GiB | {a['peak_rss_gib']:.6f} | {b['peak_rss_gib']:.6f} |
| Maximum polynomial error | {a['max_abs_error']:.9g} | {b['max_abs_error']:.9g} |

All operation counts, per-token counts, CKKS levels, selected IDs and the 0.05
polynomial gate are unchanged. Intermediate diagnostic decryptions remain zero.
The 0.56% prefix difference is small; a single full pair does not establish
statistical significance or a universal improvement.
'''
    details.append(detail)

budget = read(ROOT / 'budget.json')
confirmation = comparison['m2_confirmation']
text = f'''# Reusing private ciphertext operands in both models

The [static audit](2026-09-24-static-waste-audit.md) identified copies of
already-private arithmetic buffers. This experiment implements a shared
ownership-aware operation, verifies exact encrypted arithmetic and compares
the previous and modified executables on both trained model paths.

| Complete workload | Baseline seconds | Candidate seconds | Reduction | Decision |
| --- | --- | --- | --- | --- |
{chr(10).join(full_rows)}

Each complete mode has one fresh-process/key sample, measured baseline then
candidate after the short interleaved controls. This is native evaluation time;
setup, input preparation and transfer are additional. The models have different
sizes, weights and error contracts, so this table does not rank architectures.

[Raw runs, failed probes, source archives, hashes and comparison script](../../results/dgx/2026-09-24/owned-arithmetic/).

## Mechanism

`owned_arithmetic.hpp` consumes scratch handles by value. A uniquely owned
input is reused; a live alias is cloned before level alignment can mutate it.
If both inputs reference the same dying object, detaching the first leaves
the second safe to consume. The completion callback runs before release of
the right input so queued GPU reads cannot outlive its storage.
`fideslib_owned_arithmetic.hpp` shares add/subtract/multiply dispatch while each
model keeps its existing alignment and synchronization policy.

Mamba-3 passes final-use transform, replicated-linear and Chebyshev accumulator
terms to this helper under the existing `--inplace-ops` option. Shared rotation
babies and polynomial bases keep their original protection. This removes the
two input clones previously made by each accumulator addition. Mamba-2's
normalization and joint-gate adapters already cloned both inputs; the helper
reuses that private storage instead of requesting another result allocation
and copy. General Mamba-2 aligned operators remain available to their other
callers. No new tuning switch is introduced.

The pinned backend's allocating and in-place entry points execute the same
GPU arithmetic. Addition order, polynomial coefficients, input protection,
level alignment, moduli and refresh arithmetic are unchanged. Synchronization
needed for lifetime safety is retained; it is not removed to improve a timer.

## Matched controls

Both models run baseline/candidate/candidate/baseline in separate processes.
The baseline is the previously built executable, revalidated by hash; the
candidate has its own immutable source archive and binary. Compiler, backend
commit and local patches, installed libraries/headers, payload and options are
fixed. Only seven native source/test/build paths differ between snapshots.

| Prefix | ABBA samples (s) | Baseline → candidate mean (s) | Reduction |
| --- | --- | --- | --- |
{prefix_rows}

Because the first Mamba-2 difference was small, four reversed controls were
declared before either full pair completed and run after them. Their BAAB
samples are `{confirmation['samples_in_baab_order']}` seconds. That block gives
{confirmation['base_mean_seconds']:.6f} → {confirmation['candidate_mean_seconds']:.6f}
seconds ({confirmation['reduction_percent']:.2f}%). Across all four samples per
mode, means are {confirmation['combined_base_mean_seconds']:.6f} →
{confirmation['combined_candidate_mean_seconds']:.6f} seconds
({confirmation['combined_reduction_percent']:.2f}%). Final Mamba-2 adoption
requires improvement in this reversed block, the combined mean and the full
pair; the decision is **{'adopt' if decisions['m2'] else 'do not adopt'}**.

The predefined full-run qualification is a prefix mean reduction exceeding
0.5%. Mamba-3 uses CPUs 15–19, four OpenMP threads, borrowed upload, coefficient
moves, NAF, final-use reuse, compact weights, optimized routing and the existing
64-entry mask cache. Mamba-2 uses CPUs 0–19, eight cache workers and
`CUDA_LAUNCH_BLOCKING=1`; borrowed upload and GPU NTT are on, coefficient moves
stay off. Their existing architecture-specific configurations are not unified
by changing cryptographic or numerical requirements.

{chr(10).join(details)}

Phase timers can overlap or nest and are not a disjoint decomposition. Raw
maximum RSS is process-wide. No kernel bandwidth or isolated allocation-time
claim is inferred from the model timers.

## Correctness, failed checks and reproducibility

The native exactness probe checks 96 cases in each configured context: real
Mamba-3 and complex Mamba-2, four equal/unequal level pairs, scalar-scaled or
unscaled inputs, add/subtract/multiply and four alias patterns. All **192 cases
match every RNS output coefficient and metadata exactly**, and all original
live inputs remain unchanged. Each probe exercises 96 consumed and 96 cloned
inputs. The encrypted synthetic carried-state regression also passes.

The first probe build failed on mutable FIDESlib handle-reference requirements;
the corrected build uses named mutable handles. The first GPU probe then
incorrectly required repeated decrypted values to be bit-identical. The pinned
OpenFHE REAL CKKS decoder adds Gaussian noise on each decoding; repeated decoding
of the same ciphertext confirmed the difference. The corrected acceptance check
compares the ciphertext itself, with zero tolerance, and records repeat-decoding
as a diagnostic. No decoder behavior or model error gate was relaxed. Both
failed revisions, logs and source manifests are preserved alongside the passing
revision. These were probe/harness failures, not hidden model speed samples.

The initial full Mamba-3 candidate passed numerical and selected-token gates
in 947.589282 seconds against a 978.468744-second baseline. Its coverage check
failed: 91,716 consumed additions versus 91,956 predicted. The implementation
had left the non-BSGS radix-routing accumulator on the old protected-copy path.
A source inventory confirms precisely 240 additions there. That call site now
uses the same consuming helper; the final synthetic, prefix and complete
Mamba-3 comparisons were rerun on a separately frozen executable. The initial
measurements and failed coverage check are preserved under
`initial-routing-coverage/`. They are not substituted for the final comparison.
Mamba-2 and the shared helper/probe source and binaries did not change.

The Python suite passes **286 tests at 87.60% coverage** and includes all
**18 C++ contracts**. The ownership contract checks storage identity, aliased
live-outs through mutating alignment, two dying aliases, identical operands,
right-input lifetime through completion, and null rejection before effects.

Per-file manifests bind both compiled snapshots. Every native run records its
binary, source manifest, input, command, environment and raw-output hash. The
postflight check rereads Mamba-2's complete payload and all frozen sources,
dependencies and binaries after GPU timing; Mamba-3 inputs were also hashed
before each run. The portable comparator checks identical architecture-specific
parameters, operations and levels, then reproduces decisions and timings.
Reporting decodes actual selected IDs and validates generation artifacts.
Collection and postprocessing hooks save durable completion records and submit
a local desktop notification. The starting Git commit alone is not treated as
the identity of the uncommitted measured candidate.

## Separate encoder follow-up inventory

While GPU binaries stayed frozen, a local static mask scan confirmed 4,488
unique public-weight diagonals across 22,440 encodes. Saving one inverse-FFT
complex-double vector per identity would require 2.1914 GiB, versus 52.5938 GiB
for expanded RNS at illustrative level 21. This is a storage calculation, not
an implemented cache or a speed result.

An extracted Encode range scan also replaces per-component logarithms with a
single logarithm after finding the largest absolute component. It preserves
scaled bytes and the range result in 6,356 finite-input local cases, including
neighbors of powers and subnormals. A local Intel x86-64 microbenchmark gives
about 3.51× for that isolated scan. A separate
[actual-encoder CPU study](2026-09-25-encoding-range.md) then checks
167,772,160 integer coefficient words and metadata exactly and measures
2.31–3.92% shorter full plaintext construction across four fixtures. Neither
local measurement establishes DGX or model performance. The production
model/library code does not include this prototype. See
[the detailed follow-up and all raw local samples](../../results/dgx/2026-09-24/owned-arithmetic/next-encoding-candidates.md)
for proposed exact-RNS gates and unsupported cases.

The subsequent [preparation design study](2026-09-25-preparation-design.md)
also follows the actual diagonal access order: five identical 4,488-entry
cycles. Below the full working-set size, ordinary LRU has zero identity hits;
a fixed 2-GiB subset can reuse 16,384 preparations. These are static counts,
not a new cache or speed result. That study records the CPU metadata,
registry-concurrency and buffer-retirement constraints for the next experiment.

Scope remains DGX Spark GB10, one frozen prompt, inline client and
`security=not-set`. This does not validate other prompts, long sessions or a
production security setting. The finite campaign ledger includes the failed
initial probe and every timed control. Persistent encoded-weight caching and
asynchronous preparation remain separate, unimplemented candidates.

Recorded DGX native process wall time is {budget['used_seconds']:.3f} seconds within
the {budget['review_seconds']/3600:g}-hour internal review interval.
'''
(REPO / 'docs/research/2026-09-24-owned-arithmetic.md').write_text(text)
print('Wrote ownership study from verified artifacts.')
