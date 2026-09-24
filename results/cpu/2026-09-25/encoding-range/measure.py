"""Interleave immutable local OpenFHE libraries after exact coefficient parity."""
import datetime
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import time

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def save(path, data): path.write_text(json.dumps(data, indent=2) + '\n')

base, candidate = (read(ROOT / (mode + '-correctness.json')) for mode in ('base','candidate'))
assert base['passed'] and candidate['passed'] and base['coefficient_words'] == candidate['coefficient_words'] == 167772160
assert base['cases'] == candidate['cases'] == 120
assert read(ROOT / 'base-correctness-run.json')['binary_sha256'] == sha(ROOT / 'encoding_probe')
assert read(ROOT / 'candidate-correctness-run.json')['binary_sha256'] == sha(ROOT / 'encoding_probe')
libraries = {mode: {p.name: sha(p) for p in sorted((ROOT / ('lib-' + mode)).glob('*.so.1.4.2'))}
             for mode in ('base', 'candidate')}
assert libraries['base'].keys() == libraries['candidate'].keys()
assert {name for name in libraries['base'] if libraries['base'][name] != libraries['candidate'][name]} == {'libOPENFHEpke.so.1.4.2'}
identity = {'libraries_sha256': libraries, 'probe_sha256': sha(ROOT / 'encoding_probe'),
    'probe_source_sha256': sha(ROOT / 'encoding_probe.cpp'), 'baseline_encoding_source_sha256': sha(ROOT / 'encoding-base.cpp'),
    'candidate_encoding_source_sha256': sha(ROOT / 'encoding-candidate.cpp'), 'patch_sha256': sha(ROOT / 'reduce-encode-logarithms.patch'),
    'reference_coefficients_sha256': sha(ROOT / 'coefficients.bin'), 'reference_bytes': (ROOT / 'coefficients.bin').stat().st_size,
    'compiler': subprocess.check_output(['g++', '--version'], text=True),
    'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT / 'source'), 'rev-parse', 'HEAD'], text=True).strip(),
    'submodules': subprocess.check_output(['git', '-C', str(ROOT / 'source'), 'submodule', 'status'], text=True),
    'compile_commands': read(ROOT / 'build-base/compile_commands.json'),
    'scope': 'Local upstream CPU encoder proof, same Encode source as pinned DGX dependency; target FIDESlib integration patches and GPU inference are not exercised.'}
save(ROOT / 'identity.json', identity)
samples = []
core = min(os.sched_getaffinity(0))
for i, mode in enumerate(('base','candidate','candidate','base','base','candidate','candidate','base'), 1):
    output = ROOT / f'bench-{i}-{mode}.json'
    assert not output.exists()
    env = dict(os.environ, OMP_NUM_THREADS='4', LD_LIBRARY_PATH=str(ROOT / ('lib-' + mode)))
    bindings = subprocess.check_output(['ldd', str(ROOT / 'encoding_probe')], env=env, text=True)
    for name in libraries[mode]:
        assert str(ROOT / ('lib-' + mode)) in next(line for line in bindings.splitlines() if name.split('.so')[0] in line)
    command = ['taskset', '-c', str(core), str(ROOT / 'encoding_probe'), 'bench', 'unused', str(output)]
    started = time.monotonic()
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    record = {'mode': mode, 'command': command, 'returncode': result.returncode,
        'wall_seconds': time.monotonic() - started, 'stdout': result.stdout, 'stderr': result.stderr,
        'ldd': bindings, 'environment': {k: env[k] for k in ('OMP_NUM_THREADS','LD_LIBRARY_PATH')},
        'identity_sha256': sha(ROOT / 'identity.json')}
    save(ROOT / f'bench-{i}-{mode}-run.json', record)
    assert result.returncode == 0, record
    native = read(output)
    assert native['passed'] and len(native['benchmarks']) == 4
    record['native_sha256'] = sha(output)
    save(ROOT / f'bench-{i}-{mode}-run.json', record)
    samples.append({'mode': mode, 'benchmarks': native['benchmarks']})

rows = []
for complex in (False, True):
    for pattern in (1, 2):
        by_mode = {mode: [next(v['seconds'] for v in sample['benchmarks']
                                if v['complex'] == complex and v['pattern'] == pattern)
                          for sample in samples if sample['mode'] == mode]
                   for mode in ('base','candidate')}
        a, b = mean(by_mode['base']), mean(by_mode['candidate'])
        rows.append({'complex': complex, 'pattern': pattern, 'samples': by_mode,
                     'base_mean_seconds': a, 'candidate_mean_seconds': b,
                     'reduction_percent': 100 * (1 - b / a)})
save(ROOT / 'comparison.json', {'passed': True, 'rows': rows, 'exact_cases': 120,
    'coefficient_words_compared': 167772160, 'identity_sha256': sha(ROOT / 'identity.json'),
    'scope': 'OpenFHE MakeCKKSPackedPlaintext with identity roots, 32768 slots, level 21 and degree 1. Input creation, context setup, and plaintext release are outside the Encode construction timer. CPU only; no model speed claim.',
    'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
print(json.dumps(rows), flush=True)
