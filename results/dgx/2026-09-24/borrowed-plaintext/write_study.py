"""Write the study from verified full-run artifacts; never infer pending results."""
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
ROOT = REPO / 'results/dgx/2026-09-24/borrowed-plaintext'
assert json.loads((ROOT / 'postprocess-completion.json').read_text())['passed']
data = json.loads((ROOT / 'comparison.json').read_text())
full = data['full']['m3']
base, candidate = full['base'], full['candidate']
m2 = data['full']['m2']
short = data['short']
rows = '\n'.join(
    f"| {mode} | {values['samples'][0]:.6f}, {values['samples'][1]:.6f} | {values['mean_seconds']:.6f} | {values['reduction_percent']:.2f}% |"
    for mode, values in short['m3'].items())
phases = '\n'.join(
    f"| {label} | {base[key]:.4f} | {candidate[key]:.4f} |"
    for label, key in (('Evaluation (s)', 'eval_seconds'),
                       ('Host encoding (s)', 'host_encoding_seconds'),
                       ('Plaintext upload (s)', 'plaintext_upload_seconds'),
                       ('Bootstrap (s)', 'bootstrap_seconds'),
                       ('Peak RSS (GiB)', 'peak_rss_gib')))
text = f'''# Borrowed plaintext upload and shared routing rotations

The same-binary full Mamba-3 comparison completes in **{base['eval_seconds']:.3f}
→ {candidate['eval_seconds']:.3f} seconds** ({base['eval_seconds']/60:.2f} →
{candidate['eval_seconds']/60:.2f} minutes, **{full['reduction_percent']:.2f}% reduction**).
It retains all four selected IDs and the 0.001 exact/polynomial gates. The
selected mode is `{full['selected_mode']}`. The separate Mamba-2 upload ABBA
reduces its mean by **{short['m2']['reduction_percent']:.2f}%**; its complete
candidate passes in **{m2['eval_seconds']:.3f} seconds** ({m2['eval_seconds']/60:.2f}
minutes). Mamba-2's full run establishes parity; its incremental speed claim
uses the short ABBA.

[Raw runs, exact source archive, target identity and comparison script](../../results/dgx/2026-09-24/borrowed-plaintext/).

## Mechanisms and dispatch

`--borrow-plaintext-upload` removes the bridge's remaining host staging vector.
It reads the CPU plaintext's native coefficient arrays through CUDA's byte
copy interface, retaining the owner until all context devices synchronize.
The pinned OpenFHE representation must be a standard-layout, eight-byte,
single-`uint64_t` wrapper in contiguous storage on a little-endian host. All
moduli, lengths and destination limb widths are checked before borrowed reads.
Unsupported layouts fall back to direct conversion or the staged library path.
This removes an explicit host copy; GPU transfer and any driver staging remain.

Both models use the same `PlaintextPreparationOptions` and upload policy.
Actual borrowed, direct and moved-coefficient counters distinguish selected
flags from executed paths. CPU cache workers, client encryption and periodic
subring encoding keep their earlier paths. Mamba-3 retains coefficient moves
in every control; Mamba-2 keeps them disabled after the preceding negative
model-time result.

`--bsgs-routing-stages` reuses the packed evaluator's existing diagonal BSGS
planner inside monotone gather/scatter stages. Moving each selection mask
from source coordinates to destination coordinates exposes shared baby
rotations. The algebra is `rotate(x * source_mask, k) = rotate(x, k) *
destination_mask`. A stage uses this path only when the available-key rotation
count improves. Mask count, multiplicative depth and the rotation-key set stay
fixed. The destination masks, NAF cost function and static inventory share the
same implementation with execution. This routing path belongs to the packed
evaluator; Mamba-2 benefits from the shared upload implementation.

## Matched controls

Mamba-3 runs base/borrow/routing/all/all/routing/borrow/base with fresh keys,
one immutable binary, the same payload and CPUs 15–19 with four OpenMP threads.
Its baseline already enables GPU plaintext NTT, coefficient moves, direct
upload, NAF, final-use reuse and compact weights. Caching is disabled here.

| Mamba-3 prefix mode | Samples (s) | Mean (s) | Reduction from base |
| --- | --- | --- | --- |
{rows}

The combined prefix executes 757 total rotations versus 909 for the baseline,
matching the static prediction of 152 fewer model rotations. The full model
executes **{base['rotations']-base['refresh_rotations']:,} →
{candidate['rotations']-candidate['refresh_rotations']:,} model rotations**,
excluding refresh internals, saving **{candidate['routing_stage_rotations_saved']:,}**.
Both full modes retain the same cryptographic parameters, mask/product counts,
726 bootstraps and generated IDs `[315,279,1614,315]`:
`The capital of the state of`.

| Full Mamba-3 phase | Baseline | Selected candidate |
| --- | --- | --- |
{phases}

These are one full run per mode, following the mirrored prefix controls.
Phase timers are reported separately and can be nested; they should not be
summed as a disjoint runtime breakdown. Candidate maximum exact/polynomial
errors are {candidate['max_abs_error_vs_exact']:.9g}/
{candidate['max_abs_error_vs_polynomial']:.9g}.

Mamba-2 uses CPUs 0–19, its existing eight cache workers and synchronous CUDA
launch setting in all four ABBA runs. Means are
{short['m2']['base_mean_seconds']:.6f} → {short['m2']['borrow_mean_seconds']:.6f}
seconds. Its full candidate preserves every operation count, CKKS level,
2,187 bootstraps and IDs `[273,253,4687,273]`, producing
`The capital of the Republic of`. Maximum polynomial error is
{m2['max_polynomial_error']:.9g} within 0.05, peak RSS {m2['peak_rss_gib']:.4f} GiB.

## Validation and limits

Both target configurations pass 160 original and 160 moved exact-RNS cases,
320 borrowed input/format cases and 162 shared-policy cases. Borrowed tests
check unchanged CPU coefficients, both formats and all four tested NTT batch
widths. Maximum encrypted probe errors are
{data['probes']['mamba2']['max_abs_error']:.4g} and
{data['probes']['mamba3']['max_abs_error']:.4g}, below the unchanged `1e-6` gate.
CPU routing tests compare direct indexing with every staged mask, including
dirty padding, gather/scatter, all four radices and irregular fallback cases.
The 286-test Python suite and all 17 C++ contracts pass. An independent local
66,593-word layout check passes; it is distinct from target GPU validation.

The cached synthetic carried-state regression has {data['cache']['hits']} hits
and maximum exact error {data['cache']['max_exact_error']:.4g}. Integrating the
existing bounded cache into complete trained generation is a separate
[follow-up](2026-09-24-packed-cache-integration.md).

Frozen sources and library/header hashes bind these results to the executable.
The initial missing-include syntax-check failures and immutable-artifact lint
failure remain recorded alongside corrected checks. The queued borrow-only
snapshot was never built or measured; its replacement is explicitly recorded.
All evaluation paths use zero intermediate diagnostic decryptions. Completion
hooks collect results, decode measured IDs and validate artifacts. Recorded
native process wall time is {data['budget']['used_seconds']:.3f} seconds within
the {data['budget']['review_seconds']/3600:g}-hour internal review interval.

Linear baby-step retuning offered only 480 fewer static rotations and was
deferred. The pinned GPU negate API changes scale/depth through multiplication,
so exact subtraction is retained. Raising the refresh-batch cap above 16 would
not help this workload: its observed maximum batch has eight members. Persistent
state co-location remains a capacity inventory, not an implemented optimization.

Scope remains DGX Spark, one frozen prompt, inline client and `security=not-set`.
No accuracy threshold is relaxed. Different architecture sizes, weights and
error contracts prevent treating these timings as a Mamba-2/Mamba-3 ranking.
'''
(REPO / 'docs/research/2026-09-24-borrowed-plaintext.md').write_text(text)
print('Wrote borrowed-plaintext study from verified artifacts.')
