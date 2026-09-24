"""Preserve completed cache integration controls and the exact reused binary identity."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
DEST = REPO / 'results/dgx/2026-09-24/packed-cache-integration'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert json.loads((SOURCE / 'postprocess-completion.json').read_text())['passed']
assert (DEST.parent / 'borrowed-plaintext/provenance.json').exists()
assert not DEST.exists(), 'Refuse to overwrite a curated evidence directory.'
for path in sorted(SOURCE.rglob('*')):
    relative = path.relative_to(SOURCE)
    if not path.is_file() or '__pycache__' in relative.parts or path.name in {'status.json', 'watcher.log', 'curate.log'}:
        continue
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert sha(path) == sha(target)
with (SOURCE / 'curated-compare.log').open('w') as log:
    subprocess.run([sys.executable, str(DEST / 'compare.py')], check=True,
                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
assert sha(DEST / 'comparison.json') == sha(SOURCE / 'comparison.json')
(DEST / 'README.md').write_text('''# Plaintext cache integration

This campaign reuses the immutable executable from `../borrowed-plaintext/`.
It checks the existing 64-entry mask cache with the preceding selected upload
and routing options. No native rebuild or source change occurs between these
campaigns. Prefix controls run uncached/cached/cached/uncached, each in a fresh
process. A cached full run is attempted only if that prefix comparison improves.

`baseline-prefix/` and `baseline-full/` are byte-preserved artifacts from the
preceding campaign. Its full candidate is the uncached full baseline. The
cached full run follows an intervening Mamba-2 validation, so the single full
pair is reported separately from the interleaved prefix controls.

Recompute source/binary/input bindings, actual cache dispatch, unchanged gates,
operation counts, selected IDs and timing comparisons with `python3 compare.py`.
`comparison.json` includes the final adoption decision. The copied Mamba-2
launcher is an unused provenance dependency of the common run helper; every
native command in this campaign runs the packed Mamba-3 evaluator.

See the [study](../../../../docs/research/2026-09-24-packed-cache-integration.md).
This is a DGX Spark feasibility workload, one prompt, inline client and
`security=not-set`; it does not establish a model architecture ranking.
''')
(DEST / 'provenance.json').write_text(json.dumps({
    'base_commit': '70bf711cf4968bd2c818bc4637382acd8afe5d4c',
    'scope': 'Raw records preserve original bytes and predecessor source/binary identities. No native rebuild; one full run per cache mode when qualified.',
    'artifacts_sha256': {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
}, indent=2) + '\n')
print(DEST)
