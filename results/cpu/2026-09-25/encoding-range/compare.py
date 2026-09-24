"""Reproduce the local encoder comparison without rebuilding the dependency."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
identity = read(ROOT / 'identity.json')
assert sha(ROOT / 'encoding_probe.cpp') == identity['probe_source_sha256']
assert sha(ROOT / 'encoding-base.cpp') == identity['baseline_encoding_source_sha256']
assert sha(ROOT / 'encoding-candidate.cpp') == identity['candidate_encoding_source_sha256']
assert sha(ROOT / 'reduce-encode-logarithms.patch') == identity['patch_sha256']
base, candidate = (read(ROOT / (mode + '-correctness.json')) for mode in ('base', 'candidate'))
for d in (base, candidate):
    assert d['passed'] and d['cases'] == 120 and d['encoded_cases'] == 96
    assert d['expected_scaling_failures'] == 24 and d['coefficient_words'] == 167772160
assert base['all_words'] == candidate['all_words']
assert base['all_words'] * 8 == identity['reference_bytes']
for mode in ('base', 'candidate'):
    run = read(ROOT / (mode + '-correctness-run.json'))
    assert run['returncode'] == 0 and run['binary_sha256'] == identity['probe_sha256']
    assert run['source_sha256'] == identity['probe_source_sha256']
samples = []
for i, mode in enumerate(('base','candidate','candidate','base','base','candidate','candidate','base'), 1):
    path = ROOT / f'bench-{i}-{mode}.json'
    run = read(ROOT / f'bench-{i}-{mode}-run.json')
    assert run['returncode'] == 0 and run['native_sha256'] == sha(path)
    assert run['identity_sha256'] == sha(ROOT / 'identity.json')
    native = read(path)
    assert native['passed']
    samples.append({'mode': mode, 'benchmarks': native['benchmarks']})
rows = []
for complex in (False, True):
    for pattern in (1, 2):
        by_mode = {mode: [next(v['seconds'] for v in sample['benchmarks']
                            if v['complex'] == complex and v['pattern'] == pattern)
                         for sample in samples if sample['mode'] == mode]
                   for mode in ('base', 'candidate')}
        a, b = mean(by_mode['base']), mean(by_mode['candidate'])
        rows.append({'complex': complex, 'pattern': pattern, 'samples': by_mode,
                     'base_mean_seconds': a, 'candidate_mean_seconds': b,
                     'reduction_percent': 100 * (1 - b / a)})
result = read(ROOT / 'comparison.json')
assert result['rows'] == rows and result['identity_sha256'] == sha(ROOT / 'identity.json')
print(json.dumps({'passed': True, 'cases': 120, 'coefficients_compared': 167772160, 'rows': rows}))
