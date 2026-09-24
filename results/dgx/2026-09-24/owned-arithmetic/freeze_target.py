"""Verify existing baseline/dependencies and bind final candidate binaries."""
import json
from pathlib import Path
import subprocess
from runlib import ROOT, sha, save

previous = ROOT.parent / 'borrowed-plaintext-20260924'
old = json.loads((previous / 'target-provenance.json').read_text())
for path, value in old['source_and_binary_sha256'].items():
    assert sha(path) == value, path
manifest = json.loads((ROOT / 'compiled-sources.json').read_text())
assert all(sha(ROOT / 'source' / p) == value for p, value in manifest['files'].items())
backend = '/home/kataiwa/fhemamba/spark/source-2a70798e869944af'
def command(*args):
    return subprocess.check_output(args, text=True).strip()
import hashlib
assert hashlib.sha256(command('git', '-C', backend, 'diff').encode()).hexdigest() == old['backend_diff_sha256']
assert command('g++', '--version') == old['compiler']
assert command('/usr/local/cuda-13.0/bin/nvcc', '--version') == old['cuda']
assert command('nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader') == old['hardware']
assert command('git', '-C', backend, 'rev-parse', 'HEAD') == old['backend_commit']
baseline = ROOT / 'baseline'; baseline.mkdir(exist_ok=True)
import shutil
for name in ('compiled-sources.json', 'compiled-sources.tar.gz', 'target-provenance.json'):
    shutil.copyfile(previous / name, baseline / name)
save(ROOT / 'target-provenance.json', {
    'baseline_target_provenance_sha256': sha(baseline / 'target-provenance.json'),
    'baseline_source_manifest_sha256': sha(baseline / 'compiled-sources.json'),
    'candidate_source_manifest_sha256': sha(ROOT / 'compiled-sources.json'),
    'candidate_binaries_sha256': {name: sha(ROOT / 'build' / name) for name in
        ('packed_fideslib', 'stage1_mamba2_decode_fideslib', 'owned_arithmetic_probe')},
    'dependencies_verified': old['source_and_binary_sha256'],
    'hardware': old['hardware'], 'compiler': old['compiler'], 'cuda': old['cuda'],
    'backend_commit': old['backend_commit'], 'backend_diff_sha256': old['backend_diff_sha256'],
    'compile_commands': json.loads((ROOT / 'build/compile_commands.json').read_text()),
    'base_commit': manifest['base_commit'], 'scope': 'Separate immutable base/candidate source and binary comparisons; reused baseline build verified byte-for-byte.',
})
