"""Independent verification of frozen sources, exactness and timing pairs."""
import hashlib
import json
from pathlib import Path
from statistics import mean
import tarfile

ROOT = Path(__file__).resolve().parent
def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

target = read(ROOT / 'target-provenance.json')
assert sha(ROOT / 'compiled-sources.json') == target['candidate_source_manifest_sha256']
assert sha(ROOT / 'baseline/compiled-sources.json') == target['baseline_source_manifest_sha256']
assert sha(ROOT / 'baseline/target-provenance.json') == target['baseline_target_provenance_sha256']
old_target = read(ROOT / 'baseline/target-provenance.json')
assert target['dependencies_verified'] == old_target['source_and_binary_sha256']
final_target = read(ROOT / 'target-final-provenance.json')
assert final_target['previous_target_sha256'] == sha(ROOT / 'target-provenance.json')
assert final_target['source_manifest_sha256'] == sha(ROOT / 'compiled-final-sources.json')

candidate_sources = read(ROOT / 'compiled-sources.json')['files']
baseline_sources = read(ROOT / 'baseline/compiled-sources.json')['files']
final_sources = read(ROOT / 'compiled-final-sources.json')['files']
assert {name for name in final_sources if final_sources[name] != candidate_sources.get(name)} == {
    'native/fideslib_stage0/src/packed_fideslib.cpp'}
with tarfile.open(ROOT / 'compiled-final-sources.tar.gz') as archive:
    actual = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
              for member in archive.getmembers() if member.isfile()}
assert actual == final_sources
for directory, expected in ((ROOT, candidate_sources), (ROOT / 'baseline', baseline_sources)):
    with tarfile.open(directory / 'compiled-sources.tar.gz') as archive:
        actual = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                  for member in archive.getmembers() if member.isfile()}
    assert actual == expected, str(directory)
assert not set(baseline_sources) - set(candidate_sources)
changes = {name for name in candidate_sources if candidate_sources[name] != baseline_sources.get(name)}
assert changes == {
    'native/fideslib_stage0/CMakeLists.txt',
    'native/fideslib_stage0/include/owned_arithmetic.hpp',
    'native/fideslib_stage0/src/fideslib_owned_arithmetic.hpp',
    'native/fideslib_stage0/src/owned_arithmetic_probe.cpp',
    'native/fideslib_stage0/src/packed_fideslib.cpp',
    'native/fideslib_stage0/src/stage1_mamba2_decode_fideslib.cpp',
    'native/fideslib_stage0/tests/test_owned_arithmetic.cpp',
}, changes

def checked(name, candidate, executable):
    directory = ROOT / name
    record, native = read(directory / 'run.json'), read(directory / 'native.json')
    assert record['passed'] and record['returncode'] == 0 and not record['timed_out']
    assert all(record['checks'].values()) and native['passed']
    assert sha(directory / 'native.json') == record['native_sha256']
    assert record['controller_sha256'] == sha(ROOT / 'runlib.py')
    assert record['launcher_sha256'] == sha(ROOT / 'launch_m2.sh')
    manifest = (final_target['source_manifest_sha256'] if candidate and executable == 'packed_fideslib'
                else target['candidate_source_manifest_sha256' if candidate else 'baseline_source_manifest_sha256'])
    assert record['source_manifest_sha256'] == manifest
    binary = (target['candidate_binaries_sha256'][executable] if candidate else
              old_target['source_and_binary_sha256'][
                  '/home/kataiwa/fhemamba/borrowed-plaintext-20260924/build/' + executable])
    if candidate and executable == 'packed_fideslib': binary = final_target['binary_sha256']
    assert record['binary_sha256'] == binary
    return record, native

for architecture in ('mamba3', 'mamba2'):
    r, d = checked('probe-rns-' + architecture, True, 'owned_arithmetic_probe')
    assert d['cases'] == 96 and d['exact_rns_results'] and d['live_inputs_unchanged']
    assert d['reused_inputs'] == d['cloned_inputs'] == 96
checked('m3-synthetic-candidate', True, 'packed_fideslib')

def compare_conditions(a, b, architecture):
    ra, na = a; rb, nb = b
    ea, eb = ra['environment'].copy(), rb['environment'].copy()
    for env in (ea, eb):
        env.pop('BINARY', None); env.pop('BINARY_SHA256', None)
    assert ea == eb
    ca, cb = ra['command'][:], rb['command'][:]
    if architecture == 'm3':
        ca[3] = cb[3] = 'binary'; ca[5] = cb[5] = 'output'
        assert ra['program_sha256'] == rb['program_sha256']
        keys = ('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul','rotations',
                'refresh_rotations','logical_refreshes','refresh_batches','ring_dimension',
                'slots','depth','scale_bits','bootstrap_passes','refresh_policy','batch_refresh',
                'public_weight_count','public_weight_bytes','host_encodes','plaintext_cache_hits',
                'plaintext_cache_misses','optimized_routing_stages','routing_stage_rotations_saved')
        for d in (na, nb):
            assert d['polynomial_tolerance'] == d['exact_tolerance'] == .001
            assert d['non_finite'] == d['evaluation_decryptions'] == 0
    else:
        ca[2] = cb[2] = 'output'
        keys = ('parameters','ckks_levels','operation_counts','operation_counts_by_token','phase_operation_counts')
        for d in (na, nb):
            assert d['parameters']['tolerance'] == .05
            assert d['measurement_scope']['zero_intermediate_decrypts']
    assert ca == cb
    assert all(na[k] == nb[k] for k in keys)

result = {'exact_rns_cases': 192, 'source_changes': sorted(changes), 'short': {}, 'full': {}}
for architecture, executable in (('m3', 'packed_fideslib'), ('m2', 'stage1_mamba2_decode_fideslib')):
    rows = [checked(f'{architecture}-{i}-' + ('candidate' if candidate else 'base'), candidate, executable)
            for i, candidate in enumerate((False, True, True, False), 1)]
    for row in rows[1:]: compare_conditions(rows[0], row, architecture)
    samples = [row[0]['eval_seconds'] for row in rows]
    base, candidate = mean((samples[0], samples[3])), mean(samples[1:3])
    result['short'][architecture] = {'samples_in_abba_order': samples, 'base_mean_seconds': base,
        'candidate_mean_seconds': candidate, 'reduction_percent': 100 * (1 - candidate / base),
        'qualifies_full_pair': candidate < .995 * base}
    assert result['short'][architecture] == read(ROOT / (architecture + '-short-comparison.json'))
    if candidate >= .995 * base:
        result['full'][architecture] = {'skipped': True}
        continue
    a = checked(architecture + '-full-base', False, executable)
    b = checked(architecture + '-full-candidate', True, executable)
    compare_conditions(a, b, architecture)
    if architecture == 'm3':
        assert a[1]['generated_token_ids'] == b[1]['generated_token_ids'] == [315, 279, 1614, 315]
        assert b[1]['owned_arithmetic_calls'] == 91956
    else:
        assert a[1]['measurements']['autoregressive_selected_ids'] == b[1]['measurements']['autoregressive_selected_ids'] == [273, 253, 4687, 273]
    base, candidate = a[0]['eval_seconds'], b[0]['eval_seconds']
    result['full'][architecture] = {'base_seconds': base, 'candidate_seconds': candidate,
        'reduction_percent': 100 * (1 - candidate / base), 'adopt': candidate < base}
    assert result['full'][architecture] == read(ROOT / 'full-comparison.json')[architecture]
assert read(ROOT / 'completion.json')['passed']
if (ROOT / 'm2-confirmation.json').exists():
    confirmation = read(ROOT / 'm2-confirmation.json')
    assert confirmation['passed']
    rows = [checked(f'm2-confirm-{i}-' + ('candidate' if candidate else 'base'), candidate,
                    'stage1_mamba2_decode_fideslib')
            for i, candidate in enumerate((True, False, False, True), 1)]
    original = checked('m2-1-base', False, 'stage1_mamba2_decode_fideslib')
    for row in rows: compare_conditions(original, row, 'm2')
    samples = [row[0]['eval_seconds'] for row in rows]
    assert samples == confirmation['samples_in_baab_order']
    base, candidate = mean(samples[1:3]), mean((samples[0], samples[3]))
    assert base == confirmation['base_mean_seconds']
    assert candidate == confirmation['candidate_mean_seconds']
    assert 100 * (1 - candidate / base) == confirmation['reduction_percent']
    first = result['short']['m2']['samples_in_abba_order']
    all_base = [first[0], first[3], samples[1], samples[2]]
    all_candidate = [first[1], first[2], samples[0], samples[3]]
    assert all_base == confirmation['combined_base_samples']
    assert all_candidate == confirmation['combined_candidate_samples']
    assert mean(all_base) == confirmation['combined_base_mean_seconds']
    assert mean(all_candidate) == confirmation['combined_candidate_mean_seconds']
    assert 100 * (1 - mean(all_candidate) / mean(all_base)) == confirmation['combined_reduction_percent']
    assert confirmation['supports_adoption'] == (candidate < base and mean(all_candidate) < mean(all_base)
                                                and result['full']['m2']['adopt'])
    result['m2_confirmation'] = confirmation
result['adoption'] = {'m3': result['full']['m3'].get('adopt', False),
                      'm2': result.get('m2_confirmation', {}).get('supports_adoption')}
(ROOT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
