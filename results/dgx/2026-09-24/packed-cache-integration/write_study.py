"""Describe the bounded-cache decision from the completed control campaign."""
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
ROOT = REPO / 'results/dgx/2026-09-24/packed-cache-integration'
assert json.loads((ROOT / 'postprocess-completion.json').read_text())['passed']
data = json.loads((ROOT / 'comparison.json').read_text())
short, full = data['short'], data['full']
if 'cache' in full:
    base, cached = full['base'], full['cache']
    outcome = f"""The full pair completes in **{base['eval_seconds']:.3f} →
{cached['eval_seconds']:.3f} seconds** ({base['eval_seconds']/60:.2f} →
{cached['eval_seconds']/60:.2f} minutes, {full['reduction_percent']:.2f}% reduction).
The final recommendation {'enables' if full['adopt_cache'] else 'leaves off'}
the cache for this workload. Both runs generate IDs `[315,279,1614,315]`,
`The capital of the state of`, under the same 0.001 exact/polynomial gates.

| Full-model property | Uncached | Cached |
| --- | --- | --- |
"""
    for label, key in (('Evaluation seconds', 'eval_seconds'),
                       ('Host encoding seconds', 'host_encoding_seconds'),
                       ('Plaintext upload seconds', 'plaintext_upload_seconds'),
                       ('Host encodes', 'host_encodes'), ('Cache hits', 'plaintext_cache_hits'),
                       ('Peak RSS GiB', 'peak_rss_gib'),
                       ('Maximum exact error', 'max_abs_error_vs_exact'),
                       ('Maximum polynomial error', 'max_abs_error_vs_polynomial')):
        outcome += f"| {label} | {base[key]:.9g} | {cached[key]:.9g} |\n"
else:
    outcome = f"Full follow-up skipped: {full['reason']} No new full-cache timing is claimed."
text = f'''# Integrating the bounded plaintext cache

The existing 64-entry encoded-mask cache is tested together with the selected
borrowed-upload/routing options. Its prefix ABBA mean is
**{short['base_mean_seconds']:.6f} → {short['cache_mean_seconds']:.6f} seconds**
({short['reduction_percent']:.2f}% reduction). This is the existing cache under
the latest composition of optimizations; no native code or binary changes
between the [preceding campaign](2026-09-24-borrowed-plaintext.md) and this one.

[Raw controls, frozen source/binary identities and reproducible comparison](../../results/dgx/2026-09-24/packed-cache-integration/).

## Result and adoption

{outcome}

Prefix runs use uncached/cached/cached/uncached order with a fresh process and
key each time. The cache starts empty. The samples in that order are
`{short['samples_in_abba_order']}` seconds. The binary, complete source archive,
payload hash, CPUs 15–19, four OpenMP threads and all other flags are fixed.
The selected preceding mode is `{short['selected_mode']}`.

The uncached full baseline is the preceding campaign's completed candidate.
A Mamba-2 validation lies between that run and the cached full run, so the
single full pair is reported separately from the interleaved prefix controls.
It is not a repeated full-model statistical estimate.

## Contracts and provenance

The cache is bounded by 64 entries and keys exact coefficient bits and level
within one fixed context/slot count. Only degree-one multiplication plaintexts
are admitted; additive plaintexts keep their separate degree alignment. The
existing bypass policy avoids filling it with dense model weight masks.
Validation checks actual cache hits and entry bounds,
unaltered operation counts, rotations, depth, refresh schedule, key set,
model weights, selected IDs and both numerical gates. Each run uses zero
intermediate diagnostic decryptions. Decoded text comes from actual client
output IDs. CPU cache contracts and target cached-state tests are preserved
in the predecessor studies; this campaign adds composition and full-model
evidence without rebuilding the native evaluator.

`baseline-prefix/` and `baseline-full/` retain their original bytes. The
comparator verifies their hashes, source archive, binary, payload, environment,
commands and candidate bindings, then reproduces the selection and timings.
The copied Mamba-2 launcher is unused by these native commands; the common
measurement helper records its hash as a provenance dependency.

Native process wall totals {data['budget']['used_seconds']:.3f} seconds within
the one-hour internal review interval. Collection, comparison and token-report
hooks save durable completion records and submit a desktop notification.
Scope remains a DGX Spark feasibility run on one prompt with an inline client
and `security=not-set`. This does not validate long sessions or other prompts.
'''
(REPO / 'docs/research/2026-09-24-packed-cache-integration.md').write_text(text)
print('Wrote packed-cache-integration study from verified artifacts.')
