"""Promote completed, checked evidence without changing raw measurement bytes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
DEST = REPO / 'results/dgx/2026-09-24/borrowed-plaintext'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


assert json.loads((SOURCE / 'postprocess-completion.json').read_text())['passed']
assert not DEST.exists(), 'Refuse to overwrite a curated evidence directory.'
skip_dirs = {'build', 'source', 'syntax-deps', 'cpu-build', '__pycache__'}
skip_files = {'status.json', 'watcher.log', 'native_layout', 'curate.log'}
for path in sorted(SOURCE.rglob('*')):
    relative = path.relative_to(SOURCE)
    if not path.is_file() or set(relative.parts) & skip_dirs or path.name in skip_files:
        continue
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert sha(path) == sha(target)
with (SOURCE / 'curated-compare.log').open('w') as log:
    subprocess.run([sys.executable, str(DEST / 'compare.py')], check=True,
                   cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
assert sha(DEST / 'comparison.json') == sha(SOURCE / 'comparison.json')
(DEST / 'README.md').write_text('''# Borrowed plaintext upload and routing stages

The shared upload bridge borrows validated OpenFHE coefficient arrays through
the synchronized CUDA transfer, avoiding its remaining host staging vector.
Packed routing stages reuse the existing baby-step/giant-step evaluator when
its measured-key rotation count is lower. No error gate, key set or mask depth
is changed. All paths remain explicit options.

`comparison.json` is derived by `python3 compare.py` from mirrored separate
and combined controls, both exact-RNS probes, a same-binary full Mamba-3 pair
and Mamba-2 ABBA plus full parity. The latter's full run is a completion check;
its incremental speed claim uses ABBA. `compiled-sources.tar.gz` is the exact
compiled snapshot; `target-provenance.json` binds binaries, dependencies and
the OpenFHE layout headers. `validation-sources.json` separately binds the
later lint-exclusion configuration; all measured native sources are unchanged.

`superseded-borrow-only/` is an explicitly unbuilt waiting-queue revision.
Syntax-check failures from missing local include paths and the initial
immutable-artifact formatting failure are retained alongside corrected checks.
No borrowed GPU run failed those local harness checks. `layout.json` is a
local 66,593-word byte-layout check, separate from target GPU evidence.

The completion and postprocess hooks collect outputs, decode actual selected
IDs, verify source/binary/input hashes and save durable success/failure records.
`notification.json` records submission of the local desktop notification.

See the [study](../../../../docs/research/2026-09-24-borrowed-plaintext.md)
for results and scope: DGX Spark, one frozen prompt, inline client and
`security=not-set`, unchanged architecture-specific numerical gates.
''')
save(DEST / 'provenance.json', {
    'base_commit': '70bf711cf4968bd2c818bc4637382acd8afe5d4c',
    'scope': 'Raw measured files are byte-preserved; validation-only configuration is separate. Generated comparison was independently reproduced after curation.',
    'artifacts_sha256': {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
})
print(DEST)
