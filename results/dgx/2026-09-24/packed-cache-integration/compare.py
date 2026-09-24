"""Check cached model runs against their immutable uncached inputs and binary."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
import tarfile

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


manifest = read('compiled-sources.json')['files']
with tarfile.open(ROOT / 'compiled-sources.tar.gz') as archive:
    assert set(archive.getnames()) == set(manifest)
    for member in archive:
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == manifest[member.name]
target = read('target-provenance.json')
assert all(manifest[path] == value for path, value in target['native_sources_sha256'].items())
binaries = {Path(path).name: value for path, value in target['source_and_binary_sha256'].items()}
previous = ROOT.parent / 'borrowed-plaintext'
if not previous.exists():
    previous = ROOT.parent / 'borrowed-plaintext-20260924'
for name in ('compiled-sources.json', 'compiled-sources.tar.gz', 'target-provenance.json'):
    assert sha(name) == hashlib.sha256((previous / name).read_bytes()).hexdigest()
assert json.loads((previous / 'completion.json').read_text())['passed']

KEYS = (
    'nodes', 'evaluated_nodes', 'bootstraps', 'ct_ct_mul', 'ct_pt_mul',
    'logical_refreshes', 'planned_logical_refreshes', 'refresh_batches',
    'largest_refresh_batch', 'bootstrap_passes', 'ring_dimension', 'slots',
    'depth', 'scale_bits', 'refresh_policy', 'batch_refresh',
    'public_weight_count', 'public_weight_bytes', 'bf16_weight_count',
    'rotations', 'refresh_rotations', 'bsgs_routing_stages',
    'routing_stage_rotations_saved', 'optimized_routing_stages',
    'reuse_dead_inputs', 'lifetime_clones_eliminated', 'compact_weights',
    'linear_method', 'routing_method', 'rotation_decomposition',
    'gpu_plaintext_ntt', 'plaintext_ntt_batch', 'move_plaintext_coefficients',
    'borrow_plaintext_upload', 'direct_plaintext_upload', 'fast_plaintext_upload',
    'profile_evaluation', 'inplace_ops', 'plaintext_cache_capacity',
    'security', 'encrypted', 'backend', 'scope',
    'client_output_decrypt_count', 'generated_token_ids',
)
SUMMARY = (
    'eval_seconds', 'host_encoding_seconds', 'plaintext_upload_seconds',
    'bootstrap_seconds', 'host_encodes', 'gpu_ntt_encodes',
    'plaintext_cache_hits', 'plaintext_cache_misses', 'plaintext_cache_bypasses',
    'plaintext_cache_evictions', 'plaintext_cache_entries',
    'max_abs_error_vs_exact', 'max_abs_error_vs_polynomial', 'peak_rss_gib',
)


def checked(name, cached, baseline=None):
    run, data = read(name + '/run.json'), read(name + '/native.json')
    assert run['passed'] and data['passed'] and all(run['checks'].values())
    assert run['returncode'] == 0 and not run['timed_out']
    assert run['native_sha256'] == sha(name + '/native.json')
    assert run['source_manifest_sha256'] == sha('compiled-sources.json')
    assert run['binary_sha256'] == binaries['packed_fideslib']
    assert run['command'][:3] == ['taskset', '-c', '15-19']
    assert run['environment']['OMP_NUM_THREADS'] == '4'
    assert 'CUDA_LAUNCH_BLOCKING' not in run['environment']
    assert ('--cache-plaintexts' in run['command']) == cached
    assert data['cache_plaintexts'] == cached
    assert data['evaluation_decryptions'] == data['non_finite'] == 0
    assert data['polynomial_tolerance'] == data['exact_tolerance'] == .001
    for field in ('max_abs_error_vs_polynomial', 'max_abs_error_vs_exact'):
        assert math.isfinite(data[field]) and 0 <= data[field] <= .001
    assert data['plaintext_cache_capacity'] == 64
    if cached:
        assert data['plaintext_cache_hits'] > 0
        assert 0 < data['plaintext_cache_entries'] <= 64
    else:
        assert data['plaintext_cache_hits'] == data['plaintext_cache_entries'] == 0
    assert data['moved_coefficient_encodes'] == data['gpu_ntt_encodes']
    if data['borrow_plaintext_upload']:
        assert data['borrowed_plaintext_uploads'] == data['direct_plaintext_uploads']
    if baseline:
        assert run['controller_sha256'] == sha('runlib.py')
        assert run['launcher_sha256'] == sha('launch_m2.sh')
        old_run, old = read(baseline + '/run.json'), read(baseline + '/native.json')
        for field in KEYS:
            assert data[field] == old[field], (name, field)
        for field in ('payload', 'manifest_sha256', 'program_sha256', 'binary_sha256', 'environment'):
            assert run[field] == old_run[field], (name, field)
        assert run['baseline_native_sha256'] == sha(baseline + '/native.json')
        assert run['baseline_run_sha256'] == sha(baseline + '/run.json')
        expected_command = old_run['command'][:]
        expected_command[5] = run['command'][5]
        if cached:
            expected_command.append('--cache-plaintexts')
        assert run['command'] == expected_command
        for operation, values in old['operation_stats'].items():
            for field in ('nodes', 'bootstraps'):
                assert data['operation_stats'][operation][field] == values[field]
    return run, data


checked('baseline-prefix', False)
checked('baseline-full', False)
rows = [checked(f'prefix-{i}-' + ('cache' if cached else 'base'), cached, 'baseline-prefix')
        for i, cached in enumerate((False, True, True, False), 1)]
samples = [data['eval_seconds'] for run, data in rows]
base, candidate = mean((samples[0], samples[3])), mean(samples[1:3])
selected = json.loads((previous / 'selection.json').read_text())['m3_mode']
short = {'samples_in_abba_order': samples, 'base_mean_seconds': base,
         'cache_mean_seconds': candidate, 'reduction_percent': 100 * (1 - candidate / base),
         'selected_mode': selected}
assert short == read('short-comparison.json')
if candidate < base:
    run, data = checked('full-cache', True, 'baseline-full')
    baseline = read('baseline-full/native.json')
    assert data['generated_token_ids'] == [315, 279, 1614, 315]
    full = {'base': {key: baseline[key] for key in SUMMARY},
            'cache': {key: data[key] for key in SUMMARY},
            'reduction_percent': 100 * (1 - data['eval_seconds'] / baseline['eval_seconds']),
            'adopt_cache': data['eval_seconds'] < baseline['eval_seconds']}
else:
    assert not (ROOT / 'full-cache').exists()
    full = {**read('full-skipped.json'), 'adopt_cache': False}
assert read('completion.json')['passed']
result = {'short': short, 'full': full, 'budget': read('budget.json'),
          'scope': 'Same immutable binary and payload, fresh-process prefix ABBA. The full uncached run is the preceding campaign candidate; the cached full run follows the intervening Mamba-2 validation. One full run per mode, unchanged gates, counts and selected IDs.'}
(ROOT / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
