"""Link completed ownership measurements without relabeling historical runs."""
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
ROOT = REPO / 'results/dgx/2026-09-24/owned-arithmetic'
read = lambda p: json.loads(p.read_text())
comparison = read(ROOT / 'comparison.json')
assert read(ROOT / 'postprocess-completion.json')['passed']
full, short = comparison['full'], comparison['short']
for key in ('m3', 'm2'):
    if not full[key].get('skipped'):
        full[key]['adopt'] = comparison['adoption'][key]

def insert(path, marker, text):
    content = path.read_text()
    if text in content:
        return
    assert content.count(marker) == 1, (path, marker)
    path.write_text(content.replace(marker, marker + text + '\n', 1))

def replace(path, old, new):
    content = path.read_text()
    if new in content:
        return
    assert content.count(old) == 1, (path, old)
    path.write_text(content.replace(old, new, 1))

summary = '; '.join(
    f"{label} {full[key]['base_seconds']/60:.2f} → {full[key]['candidate_seconds']/60:.2f} min "
    f"({full[key]['reduction_percent']:.2f}%, {'adopt' if full[key]['adopt'] else 'not adopted'})"
    for label, key in [('Mamba-3', 'm3'), ('Mamba-2', 'm2')] if not full[key].get('skipped'))
insert(REPO / 'docs/research/README.md',
       '## Complete generation and optimization\n\n| Study | Focus |\n| --- | --- |\n',
       '| [2026-09-24 — Shared ciphertext ownership](2026-09-24-owned-arithmetic.md) | ' + summary + '; exact-RNS gates and unchanged model contracts |')
insert(REPO / 'results/README.md', '| Location | Contents |\n| --- | --- |\n',
       '| [`dgx/2026-09-24/owned-arithmetic/`](dgx/2026-09-24/owned-arithmetic/) | Shared scratch ownership: 192 exact-RNS cases, both prefix ABBA controls and qualified fresh full pairs; failed probe revisions retained |')
insert(REPO / 'docs/evidence.md', '| Claim | Value | Class | Source / action |\n|---|---|---|---|\n',
       '| Shared ciphertext ownership | ' + summary + ' | Exact-RNS, CPU contracts, ABBA prefixes, one fresh full pair per model | [Study](research/2026-09-24-owned-arithmetic.md), [artifacts](../results/dgx/2026-09-24/owned-arithmetic/); separate frozen executables with matching dependencies and inputs, unchanged gates; no statistical significance claim |')
insert(REPO / 'docs/research/2026-09-24-static-waste-audit.md',
       '# Static waste audit: remaining ciphertext copies and repeated preparation\n',
       '\nFollow-up: the first two ownership candidates are implemented and measured in\n'
       'the [shared ciphertext ownership study](2026-09-24-owned-arithmetic.md).\n'
       'The inventory below describes the pre-implementation snapshot; its raw\n'
       'measurements remain unchanged.\n')

if full['m3'].get('adopt'):
    result = full['m3']
    candidate = read(ROOT / 'm3-full-candidate/native.json')
    latest = f'''The latest shared-ownership implementation completes in **{result['candidate_seconds']/60:.2f}
minutes**. A fresh full baseline/candidate pair measures
**{result['base_seconds']/60:.2f} → {result['candidate_seconds']/60:.2f} minutes
({result['reduction_percent']:.2f}% reduction)**; the separate prefix ABBA measures
{short['m3']['reduction_percent']:.2f}%. It reuses {candidate['owned_arithmetic_reused_inputs']:,} temporary input buffers
and preserves all four generated IDs, operation counts and both 0.001 gates.
Maximum exact/polynomial errors are `{candidate['max_abs_error_vs_exact']:.7g}` /
`{candidate['max_abs_error_vs_polynomial']:.7g}`. Each full mode was measured once.
See the [ownership study](LINKresearch/2026-09-24-owned-arithmetic.md).

The preceding upload/routing comparison saves 9,120 model rotations
(59,900 → 50,780); the selected configuration also retains the 64-entry cache.
See the [upload/routing study](LINKresearch/2026-09-24-borrowed-plaintext.md)
and [cache integration](LINKresearch/2026-09-24-packed-cache-integration.md).
'''
    for path, prefix in [(REPO / 'README.md', 'docs/'), (REPO / 'docs/mamba3.md', '')]:
        content = path.read_text()
        if latest.replace('LINK', prefix) in content:
            continue
        start = content.index('The latest selected complete Mamba-3 configuration takes')
        ending = f'and [cache integration]({prefix}research/2026-09-24-packed-cache-integration.md).\n'
        stop = content.index(ending, start) + len(ending)
        replace(path, content[start:stop], latest.replace('LINK', prefix))
    insert(REPO / 'docs/mamba3.md',
           '[microkernel study](research/2026-09-24-mamba3-microkernels.md) for scope and profiles.\n',
           '\nThe same `--inplace-ops` option now consumes final-use accumulator terms\n'
           'through the shared ownership helper. Aliases detach before alignment and\n'
           'right-input storage stays live through GPU completion. The\n'
           '[ownership comparison](research/2026-09-24-owned-arithmetic.md) records\n'
           'exact-RNS checks and a fresh full baseline/candidate pair.\n')
    path = REPO / 'docs/backlog.md'
    content = path.read_text()
    line = next(line for line in content.splitlines() if line.startswith('| PBI-M3-001 |'))
    old_time = line[line.index('completion **')+len('completion **'):line.index(' minutes**')]
    new_line = line.replace(old_time + ' minutes**', f"{result['candidate_seconds']/60:.2f} minutes**", 1)
    start = new_line.index('exact error ')
    stop = new_line.index(' at unchanged', start)
    new_line = new_line[:start] + f"exact error {candidate['max_abs_error_vs_exact']:.5g}" + new_line[stop:]
    if '[Shared scratch ownership]' not in new_line:
        new_line = new_line.replace('Frozen sources,', '[Shared scratch ownership](research/2026-09-24-owned-arithmetic.md) also passes both model paths. Frozen sources,')
    replace(path, line, new_line)

if full['m2'].get('adopt'):
    result = full['m2']
    candidate = read(ROOT / 'm2-full-candidate/native.json')['measurements']
    old = '| [Borrowed plaintext upload](docs/research/2026-09-24-borrowed-plaintext.md) | **32.94 min** | **0.010484** |'
    new = old.replace('**', '') + '\n' + (
        f"| [Shared ciphertext ownership](docs/research/2026-09-24-owned-arithmetic.md) | **{result['candidate_seconds']/60:.2f} min** | **{candidate['max_abs_error']:.6f}** |")
    replace(REPO / 'README.md', old, new)
    for path in [REPO / 'README.md', REPO / 'results/README.md']:
        replace(path, 'borrowed-plaintext/m2-full-borrow/generation.json',
                'owned-arithmetic/m2-full-candidate/generation.json')
    marker = '| Borrowed plaintext upload | [`borrowed-plaintext/m2-full-borrow/`](dgx/2026-09-24/borrowed-plaintext/m2-full-borrow/) | [Borrowed upload and routing](../docs/research/2026-09-24-borrowed-plaintext.md) |\n'
    insert(REPO / 'results/README.md', marker,
           '| Shared ciphertext ownership | [`owned-arithmetic/m2-full-candidate/`](dgx/2026-09-24/owned-arithmetic/m2-full-candidate/) | [Ownership comparison](../docs/research/2026-09-24-owned-arithmetic.md) |')
    description = f'''The additional ownership change removes {candidate['owned_arithmetic']['calls']:,} backend
result copies in normalization and joint gates. A fresh full pair measures
**{result['base_seconds']/60:.2f} → {result['candidate_seconds']/60:.2f} minutes ({result['reduction_percent']:.2f}%)**;
the prefix ABBA difference is {short['m2']['reduction_percent']:.2f}%. Counts, selected IDs and the
0.05 polynomial gate are unchanged. These small differences and one full run
per mode do not establish statistical significance. See the
[ownership study](LINKresearch/2026-09-24-owned-arithmetic.md).
'''
    insert(REPO / 'README.md', 'Coefficient moves remain off for Mamba-2 after their non-improving control.\n',
           '\n' + description.replace('LINK', 'docs/'))
    insert(REPO / 'docs/reproducing.md',
           '[upload/routing study](research/2026-09-24-borrowed-plaintext.md).\n',
           '\n' + description.replace('LINK', ''))
(SOURCE / 'recommended-settings.json').write_text(json.dumps({
    'm3_owned_arithmetic': full['m3'].get('adopt', False),
    'm2_owned_arithmetic': full['m2'].get('adopt', False),
    'new_runtime_flags': [],
    'scope': 'Adoption applies to the measured model paths, after matching full numerical and token gates.'
}, indent=2) + '\n')
print(summary)
