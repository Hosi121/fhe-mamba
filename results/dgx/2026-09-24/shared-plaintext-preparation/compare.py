"""Regenerate comparisons from immutable backend outputs and measured identities."""
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
def checked(name):
    directory = Path(name)
    record_name = 'run-corrected.json' if (ROOT/directory/'run-corrected.json').exists() else 'run.json'
    run = read(directory / record_name)
    native = read(directory / 'native.json')
    assert run['passed'] and native['passed'] and run['returncode'] == 0 and not run['timed_out'], name
    assert run['native_sha256'] == sha(directory/'native.json'), name
    assert all(run['checks'].values()), name
    return run, native

def m2_group(names):
    samples = [checked(name) for name in names]
    first_run, first = samples[0]
    keys = ('parameters','ckks_levels','operation_counts','operation_counts_by_token','phase_operation_counts')
    for run, row in samples:
        assert run['binary_sha256'] == first_run['binary_sha256'] == row['binary_sha256']
        assert run['source_manifest_sha256'] == sha(Path('compiled-sources.json'))
        assert run['environment']['INPUT_CHAIN_SHA256'] == read('target-provenance.json')['payload_sha256']
        for key in keys: assert row[key] == first[key], (key, run['name'])
        for key in ('executed_bootstrap_count','state_bootstrap_count','paired_state_bootstrap_count',
                    'complex_constant_plaintext_count','per_token_bootstrap_count',
                    'autoregressive_selected_ids','autoregressive_expected_ids'):
            assert row['measurements'][key] == first['measurements'][key], (key,run['name'])
        m = row['measurements']
        assert all(math.isfinite(v) and 0 <= v <= .05 for v in m['per_token_max_abs_error'])
        assert len(m['per_token_decrypt_ok']) == row['parameters']['tokens']
        assert all(m['per_token_decrypt_ok']) and row['measurement_scope']['zero_intermediate_decrypts']
        mode = run['name'][-1]
        e = m['plaintext_encoding']
        assert e['fast_upload'] == (mode != 'a') and e['gpu_ntt'] == (mode == 'c')
        assert (e['eval_fast_uploads'] == 0) == (mode == 'a')
        assert (e['gpu_ntt_encodes'] == 0) if mode != 'c' else True
    groups = {}
    for mode in sorted({run['name'][-1] for run, _ in samples}):
        rows = [(run, row) for run, row in samples if run['name'][-1] == mode]
        groups[mode] = {
            'eval_seconds':[row['timing']['eval_seconds'] for _, row in rows],
            'mean_seconds':mean(row['timing']['eval_seconds'] for _, row in rows),
            'process_wall_seconds':[run['wall_seconds'] for run,_ in rows],
            'setup_seconds':[row['timing']['setup_seconds'] for _,row in rows],
            'max_abs_error':[row['measurements']['max_abs_error'] for _,row in rows],
            'peak_rss_gib':[row['measurements']['peak_rss_gib'] for _,row in rows],
            'joint_gate_seconds':[row['phase_timings']['joint_selective_gates'] for _,row in rows],
            'bootstrap_seconds':[row['timing']['bootstrap_eval_seconds'] for _,row in rows],
            'plaintext_encoding':[row['measurements']['plaintext_encoding'] for _,row in rows],
        }
    baseline = groups['a' if 'a' in groups else 'b']['mean_seconds']
    for data in groups.values():
        data['reduction_percent'] = 100 * (1 - data['mean_seconds'] / baseline)
        data['speedup'] = baseline / data['mean_seconds']
    return {'matched_conditions':True,'groups':groups,'physical_bootstraps':first['measurements']['executed_bootstrap_count'],
            'subring_encodes':first['parameters']['joint_gate_schedule']['subring_encode_calls'],
            'generated_ids':first['measurements']['autoregressive_selected_ids']}

def m3_gate(name):
    run, row = checked(name)
    assert row['non_finite'] == 0 and row['evaluation_decryptions'] == 0
    assert row['exact_tolerance'] == row['polynomial_tolerance'] == .001
    for ref in ('exact','polynomial'):
        error = row['max_abs_error_vs_'+ref]
        assert math.isfinite(error) and 0 <= error < .001
    return run, row

probes = {}
for name in ('mamba2','mamba3'):
    run, row = checked('probe-'+name)
    assert row['exact_rns_cases'] == 160 and row['shared_policy_cases'] == 54
    assert row['max_abs_error'] < row['tolerance'] == 1e-6
    probes[name] = {k:row[k] for k in ('exact_rns_cases','shared_policy_cases','max_abs_error',
                                      'secret_key_distribution','ckks_data_type')}
manifest = read('compiled-sources.json')
with tarfile.open(ROOT/'compiled-sources.tar.gz','r:gz') as archive:
    assert set(archive.getnames()) == set(manifest)
    for member in archive:
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == manifest[member.name]
target = read('target-provenance.json')
assert all(manifest[k] == v for k,v in target['native_sources_sha256'].items())
candidate = read('selection.json')['candidate']
smoke = m2_group([f'smoke-{i}-{mode}' for i,mode in enumerate('abccba',1)])
pressure = m2_group([f'pressure-{i}-{mode}' for i,mode in enumerate('bccb',1)])
assert all(x['eval_gpu_ntt_encodes'] > 0 for x in pressure['groups']['c']['plaintext_encoding'])
full = m2_group(['full-a','full-'+candidate])
assert full['generated_ids'] == [273,253,4687,273]
m3_prefix = [m3_gate('mamba3-prefix-'+str(i)) for i in range(1,5)]
for run, row in m3_prefix:
    assert run['manifest_sha256'] == m3_prefix[0][0]['manifest_sha256']
    assert run['program_sha256'] == m3_prefix[0][0]['program_sha256']
    for key in ('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul','rotations',
                'logical_refreshes','refresh_batches','gpu_ntt_encodes','fast_plaintext_uploads',
                'host_encodes','generated_token_ids','cache_plaintexts','inplace_ops'):
        assert row[key] == m3_prefix[0][1][key], key
old = mean(m3_prefix[i][1]['eval_seconds'] for i in (0,3))
new = mean(m3_prefix[i][1]['eval_seconds'] for i in (1,2))
_, synthetic = m3_gate('mamba3-synthetic')
_, m3_full = m3_gate('mamba3-full')
assert synthetic['cache_plaintexts'] and synthetic['plaintext_cache_hits'] > 0
assert m3_full['generated_token_ids'] == [315,279,1614,315]
result = {'probes':probes, 'mamba2_smoke':smoke,'mamba2_cache_pressure':pressure,'mamba2_full':full,
          'candidate':candidate,'mamba3_prefix':{'old_seconds':old,'shared_seconds':new,
              'reduction_percent':100*(1-new/old),'samples':[r['eval_seconds'] for _,r in m3_prefix]},
          'mamba3_synthetic':{'passed':True,'cache_hits':synthetic['plaintext_cache_hits']},
          'mamba3_full':{k:m3_full[k] for k in ('eval_seconds','max_abs_error_vs_exact',
              'max_abs_error_vs_polynomial','generated_token_ids','bootstraps','peak_rss_gib')},
          'budget':read('budget.json'), 'limitations':'One full fresh-key run per mode, frozen prompts, inline clients, security not-set; no cross-architecture speed claim.'}
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
