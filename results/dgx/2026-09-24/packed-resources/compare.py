"""Verify source/run identities and regenerate this cycle's comparisons."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
import tarfile

ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
read=lambda p:json.loads((ROOT/p).read_text())
sha=lambda p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def checked(name):
    run=read(name+'/run.json');data=read(name+'/native.json')
    assert run['passed'] and data['passed'] and run['returncode']==0 and not run['timed_out'],name
    assert all(run['checks'].values()) and run['native_sha256']==sha(name+'/native.json'),name
    assert run['source_manifest_sha256']==sha('compiled-sources.json')
    return run,data

def m2_errors(row):
    m=row['measurements'];count=row['parameters']['tokens']
    assert len(m['per_token_decrypt_ok'])==len(m['per_token_max_abs_error'])==count
    assert all(m['per_token_decrypt_ok'])
    assert all(math.isfinite(e) and 0<=e<=.05 for e in m['per_token_max_abs_error'])
    assert row['measurement_scope']['zero_intermediate_decrypts']

manifest=read('compiled-sources.json')['files']
with tarfile.open(ROOT/'compiled-sources.tar.gz') as archive:
    assert set(archive.getnames())==set(manifest)
    for member in archive:
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==manifest[member.name]
provenance=read('build-provenance.json')
assert provenance['compiled_source_manifest_sha256']==sha('compiled-sources.json')
target=read('target-provenance.json')
assert all(manifest[path]==value for path,value in target['native_sources_sha256'].items())
assert provenance['compile_commands']==target['compile_commands']
for name in ('packed_fideslib','packed_plaintext_probe','stage1_mamba2_decode_fideslib'):
    values=[v for p,v in target['source_and_binary_sha256'].items() if Path(p).name==name]
    assert values==[provenance['binaries_sha256'][name]],name
previous=ROOT.parent/'shared-plaintext-preparation'
if not previous.exists():previous=ROOT.parent/'mamba2-plaintext-20260924'
assert hashlib.sha256((previous/'target-provenance.json').read_bytes()).hexdigest()==provenance['predecessor_provenance_sha256']
probes={}
for name in ('mamba2','mamba3'):
    run,row=checked('probe-'+name)
    assert run['binary_sha256']==provenance['binaries_sha256']['packed_plaintext_probe']
    assert row['exact_rns_cases']==160 and row['shared_policy_cases']==90 and row['max_abs_error']<1e-6
    timings={}
    for sample in row['samples']:
        if sample['iteration']>=0:
            key=f"{sample['level']}:{row['modes'][sample['mode']]}"
            timings.setdefault(key,[]).append(sample['encode_ms']+sample['upload_ms'])
    assert all(len(v)==8 for v in timings.values())
    probes[name]={'max_abs_error':row['max_abs_error'],
                  'median_preparation_ms':{k:median(v) for k,v in timings.items()}}

def m3_group(names):
    rows=[checked(name) for name in names];first_run,first=rows[0]
    summary={}
    invariants=('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul','logical_refreshes',
                'refresh_batches','bootstrap_passes','ring_dimension','slots','depth','scale_bits',
                'refresh_policy','batch_refresh','gpu_plaintext_ntt','plaintext_ntt_batch','public_weight_count')
    for run,row in rows:
        assert run['binary_sha256']==provenance['binaries_sha256']['packed_fideslib']
        for key in ('manifest_sha256','program_sha256'):assert run[key]==first_run[key],key
        for key in invariants:assert row[key]==first[key],(key,run['name'])
        assert row['polynomial_tolerance']==row['exact_tolerance']==.001
        assert row['evaluation_decryptions']==row['non_finite']==0
        for ref in ('exact','polynomial'):
            assert math.isfinite(row['max_abs_error_vs_'+ref]) and 0<=row['max_abs_error_vs_'+ref]<=.001
        inv=read(Path(run['payload']).name+'-inventory.json')
        use_naf=run['mode'] in ('naf','all')
        use_compact=run['mode'] in ('compact','all')
        use_reuse=run['mode'] in ('reuse','all')
        use_direct=run['mode'] in ('direct','all')
        assert row['rotation_decomposition']==('naf' if use_naf else 'binary')
        assert row['rotations']-row['refresh_rotations']==inv['naf_rotations' if use_naf else 'binary_rotations']
        assert row['compact_weights']==use_compact and row['public_weight_count']==inv['public_weights']
        assert row['public_weight_bytes']==inv['compact_weight_bytes' if use_compact else 'double_weight_bytes']
        assert row['reuse_dead_inputs']==use_reuse and (row['lifetime_clones_eliminated']>0)==use_reuse
        assert row['direct_plaintext_upload']==use_direct and (row['direct_plaintext_uploads']>0)==use_direct
        mode=run['mode'];item=summary.setdefault(mode,{'eval_seconds':[],'wall_seconds':[],'max_exact_error':[],
            'max_polynomial_error':[],'rotations':[],'refresh_rotations':[],'lifetime_clones_eliminated':[],
            'public_weight_bytes':[],'peak_rss_gib':[],'upload_seconds':[]})
        for key,source in [('eval_seconds','eval_seconds'),('max_exact_error','max_abs_error_vs_exact'),
            ('max_polynomial_error','max_abs_error_vs_polynomial'),('rotations','rotations'),('refresh_rotations','refresh_rotations'),
            ('lifetime_clones_eliminated','lifetime_clones_eliminated'),('public_weight_bytes','public_weight_bytes'),
            ('peak_rss_gib','peak_rss_gib'),('upload_seconds','plaintext_upload_seconds')]:item[key].append(row[source])
        item['wall_seconds'].append(run['wall_seconds'])
    baseline=mean(summary['base']['eval_seconds'])
    for row in summary.values():
        row['mean_seconds']=mean(row['eval_seconds']);row['reduction_percent']=100*(1-row['mean_seconds']/baseline)
    return summary

modes=['base','naf','reuse','direct','compact','all']
prefix=m3_group([f'prefix-{i}-{mode}' for i,mode in enumerate(modes+list(reversed(modes)),1)])
full=m3_group(['full-base','full-all'])
for name in ('full-base','full-all'):
    _,data=checked(name);assert data['generated_token_ids']==[315,279,1614,315] and data['client_output_decrypt_count']==4
_,cache=checked('cache-all');assert cache['plaintext_cache_hits']>0
m2={}
first=None
for index,mode in enumerate(('base','direct','direct','base'),1):
    run,row=checked(f'm2-{index}-{mode}')
    m2_errors(row)
    assert run['binary_sha256']==row['binary_sha256']==provenance['binaries_sha256']['stage1_mamba2_decode_fideslib']
    if first is None:first=row
    for key in ('parameters','ckks_levels','operation_counts','operation_counts_by_token','phase_operation_counts'):
        assert row[key]==first[key],key
    assert run['environment']['INPUT_CHAIN_SHA256']==target['payload_sha256']
    e=row['measurements']['plaintext_encoding']
    assert e['direct_upload']==(mode=='direct') and (e['eval_direct_uploads']>0)==(mode=='direct')
    assert e['gpu_ntt'] and e['eval_gpu_ntt_encodes']>0
    m2.setdefault(mode,[]).append(row['timing']['eval_seconds'])
m2_summary={mode:{'samples':values,'mean_seconds':mean(values)} for mode,values in m2.items()}
m2_summary['reduction_percent']=100*(1-mean(m2['direct'])/mean(m2['base']))
m2_full=None
if (ROOT/'m2-full-direct/native.json').exists():
    run,row=checked('m2-full-direct')
    m2_errors(row)
    assert row['measurements']['autoregressive_selected_ids']==[273,253,4687,273]
    assert run['binary_sha256']==row['binary_sha256']==provenance['binaries_sha256']['stage1_mamba2_decode_fideslib']
    assert run['environment']['INPUT_CHAIN_SHA256']==target['payload_sha256']
    old=json.loads((previous/'full-c/native.json').read_text())
    for key in ('parameters','ckks_levels','operation_counts','operation_counts_by_token','phase_operation_counts'):
        assert row[key]==old[key],('full parity',key)
    assert row['measurements']['executed_bootstrap_count']==old['measurements']['executed_bootstrap_count']
    m2_full={'eval_seconds':row['timing']['eval_seconds'],'max_abs_error':row['measurements']['max_abs_error'],
             'peak_rss_gib':row['measurements']['peak_rss_gib'],'previous_full_parameters_counts_levels_match':True,
             'scope':'Single full direct-upload validation; the paired incremental comparison is the two-layer ABBA.'}
state=read('cpu-state.json');assert state['passed'] and state['prefill_bit_identical'] and state['decode_bit_identical']
assert state['source_sha256']['after']==manifest['src/fhemamba/reference.py']
assert read('inventory.json')==read('lm-full-inventory.json')
assert read('completion.json')['passed']
if m2_full is None:assert read('m2-full-skipped.json')['reason']
result={'probes':probes,'mamba3_prefix':prefix,'mamba3_full':full,'mamba2_control':m2_summary,'mamba2_full':m2_full,
        'inventory':read('lm-full-inventory.json'),'cpu_state':{k:state[k] for k in ('before','after','prefill_bit_identical','decode_bit_identical')},
        'budget':read('budget.json'),'limitations':'One frozen prompt; full M3 one fresh-key run per mode; same errors and inline client/security-not-set scope. Static occupancy does not establish packing speed.'}
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
