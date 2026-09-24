"""Promote checked evidence, retaining raw failures and source revisions."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[1]
DEST = REPO / 'results/dgx/2026-09-24/owned-arithmetic'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert read(SOURCE / 'postprocess-completion.json')['passed']
assert read(SOURCE / 'postflight.json')['passed']
assert read(SOURCE / 'm2-confirmation.json')['passed']
assert not DEST.exists(), 'Refuse to overwrite curated evidence.'
skip_dirs = {'build', 'source', 'build-final', 'source-final', 'cpu-build', '__pycache__'}
skip_files = {'status.json', 'watcher.log', 'watcher-rns.log', 'test_owned_arithmetic',
              'encoding_inventory', 'encoding_log_probe', 'routing_coverage_inventory',
              'openfhe-ckkspackedencoding.cpp', 'curate.log', 'finalize.log',
              'target_provenance.py', 'finalize-pid.json'}
for path in sorted(SOURCE.rglob('*')):
    relative = path.relative_to(SOURCE)
    if not path.is_file() or set(relative.parts) & skip_dirs or path.name in skip_files:
        continue
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert sha(path) == sha(target)
with (SOURCE / 'curated-compare.log').open('w') as log:
    subprocess.run([sys.executable, str(DEST / 'compare.py')], check=True, cwd=REPO,
                   stdout=log, stderr=subprocess.STDOUT)
assert sha(DEST / 'comparison.json') == sha(SOURCE / 'comparison.json')
(DEST / 'README.md').write_text('''# Shared ownership-aware ciphertext arithmetic

The shared helper consumes private scratch operands and detaches live aliases
before either input can be changed by level alignment. Completion precedes
release of the right operand. Mamba-3 accumulators reuse final-use inputs;
Mamba-2 normalization and joint gates retain their existing input protection
and avoid the backend's additional result copy. Arithmetic order is unchanged.

`comparison.json` is reproduced by `python3 compare.py`: two 96-case exact-RNS
probes, a carried-state regression, baseline/candidate/candidate/baseline
prefixes and qualified full baseline/candidate pairs. Each full mode has one
sample with fresh keys. These are separate frozen executables, with identical
dependency hashes, inputs, architecture-specific settings and numerical gates.
No claim of statistical significance follows from a single full pair.
The small Mamba-2 prefix effect additionally receives four reversed BAAB
controls after the full pairs. Their separate and combined results, including
the final adoption recommendation, are in `m2-confirmation.json`.

`compiled-sources.tar.gz` and its manifest contain the initial measured candidate;
`baseline/` preserves the prior compiled snapshot and identity. The base Git
commit labels the starting tree, not the uncommitted candidate. `postflight.json`
rechecks Mamba-2 payload bytes, dependencies, binaries and sources after the
campaign; Mamba-3 payload hashes were also verified before every native run.
`compiled-final-sources.tar.gz` and `target-final-provenance.json` bind the final
Mamba-3 revision. Mamba-2 and the helper probe retain their original binaries;
their relevant source files are identical in both candidate snapshots.
`source.diff` is the initial partial tracked diff; the source archives are
authoritative, including later probe/CMake fixes and untracked new headers.

The initial probe compile errors are retained in `failed-probe-build/`.
The first encrypted probe incorrectly required repeated decoded values to be
bit-identical; it failed because the pinned REAL CKKS decoder injects Gaussian
noise. `failed-decoded-bit-check/` and `probe-mamba3/` preserve that revision
and failure. The corrected `probe-rns-*` checks every output coefficient,
metadata and unmodified live input exactly. The decoder and numerical gates
were not changed. The complex-mode repeat-decode diagnostic passes; the real
mode diagnostic differs, as expected from the inspected decoder source.

`initial-routing-coverage/` preserves an initial full Mamba-3 completion whose
numerical/token gates passed, but which reached only 91,716 of the predicted
91,956 additions. The non-BSGS radix-routing path still used the old helper.
Its 240 additions are confirmed by `routing_coverage_inventory.cpp`. The final
revision consumes those terms too and repeats synthetic/prefix/full Mamba-3
checks. Initial controls and the failed coverage assertion remain unchanged;
all process time stays in the ledger. No error threshold was relaxed.

Collection and reporting hooks retain success/failure records, decode actual
client-selected IDs, validate artifacts and submit a local desktop notice.
The [study](../../../../docs/research/2026-09-24-owned-arithmetic.md) gives the
adoption decisions and limits: DGX Spark, one frozen prompt, inline client,
`security=not-set`, Mamba-3 gates 0.001 and Mamba-2 polynomial gate 0.05.
''')
manifest = {'base_commit': read(SOURCE / 'compiled-sources.json')['base_commit'],
            'scope': 'Raw measurements byte-preserved, including failed controls. Derived comparison independently reproduced after curation. Candidate source identity is its archive and per-file hashes.',
            'artifacts_sha256': {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}}
(DEST / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(DEST)
