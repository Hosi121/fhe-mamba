"""Recompute model comparisons and bind them to immutable sources and binaries."""
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
manifest=read('compiled-sources.json')['files']
with tarfile.open(ROOT/'compiled-sources.tar.gz') as archive:
    assert set(archive.getnames())==set(manifest)
    for member in archive:
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==manifest[member.name]
target=read('target-provenance.json')
assert all(manifest[path]==value for path,value in target['native_sources_sha256'].items())
binaries={Path(path).name:value for path,value in target['source_and_binary_sha256'].items()}
previous=ROOT.parent/'packed-resources'
if not previous.exists():previous=ROOT.parent/'packed-resources-20260924'
old_target=json.loads((previous/'target-provenance.json').read_text())
for key in ('hardware','compiler','cuda','backend_commit','backend_status','backend_diff_sha256','governor','payload_sha256'):
    assert target[key]==old_target[key],key
for path,value in target['source_and_binary_sha256'].items():
    if path in old_target['source_and_binary_sha256']:assert value==old_target['source_and_binary_sha256'][path]


def checked(name,binary):
    run=read(name+'/run.json');d=read(name+'/native.json')
    assert run['passed'] and d['passed'] and all(run['checks'].values())
    assert run['returncode']==0 and not run['timed_out']
    assert run['native_sha256']==sha(name+'/native.json')
    assert run['source_manifest_sha256']==sha('compiled-sources.json')
    assert run['binary_sha256']==binaries[binary]
    return run,d


probes={}
for arch in ('mamba2','mamba3'):
    run,d=checked('probe-'+arch,'packed_plaintext_probe')
    assert d['exact_rns_cases']==d['moved_coefficient_cases']==160
    assert d['shared_policy_cases']==126 and d['tolerance']==1e-6 and 0<=d['max_abs_error']<1e-6
    times={}
    for level in (21,26,34):
        pair={}
        for mode,label in ((7,'copy'),(8,'move')):
            rows=[s for s in d['samples'] if s['mode']==mode and s['level']==level and s['iteration']>=0]
            assert len(rows)==8
            pair[label]={'median_encode_ms':median(s['encode_ms'] for s in rows),
                         'median_preparation_ms':median(s['encode_ms']+s['upload_ms'] for s in rows)}
        pair['reduction_percent']=100*(1-pair['move']['median_preparation_ms']/pair['copy']['median_preparation_ms'])
        times[str(level)]=pair
    probes[arch]={'max_abs_error':d['max_abs_error'],'levels':times}

m3_keys=('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul','logical_refreshes',
    'refresh_batches','bootstrap_passes','ring_dimension','slots','depth','scale_bits',
    'refresh_policy','batch_refresh','gpu_plaintext_ntt','plaintext_ntt_batch',
    'public_weight_count','public_weight_bytes','rotations','refresh_rotations',
    'lifetime_clones_eliminated','gpu_ntt_encodes')
m2_keys=('parameters','ckks_levels','operation_counts','operation_counts_by_token','phase_operation_counts')


def model(name,arch,move):
    run,d=checked(name,'packed_fideslib' if arch=='m3' else 'stage1_mamba2_decode_fideslib')
    if arch=='m3':
        assert d['non_finite']==d['evaluation_decryptions']==0
        assert d['polynomial_tolerance']==d['exact_tolerance']==.001
        assert 0<=d['max_abs_error_vs_exact']<=.001 and 0<=d['max_abs_error_vs_polynomial']<=.001
        assert d['move_plaintext_coefficients']==move
        assert d['moved_coefficient_encodes']==(d['gpu_ntt_encodes'] if move else 0)
    else:
        m=d['measurements'];n=d['parameters']['tokens'];e=m['plaintext_encoding']
        assert len(m['per_token_decrypt_ok'])==len(m['per_token_max_abs_error'])==n
        assert all(m['per_token_decrypt_ok'])
        assert all(math.isfinite(x) and 0<=x<=.05 for x in m['per_token_max_abs_error'])
        assert d['measurement_scope']['zero_intermediate_decrypts']
        assert e['move_coefficients']==move
        assert e['eval_moved_coefficient_encodes']==(e['eval_gpu_ntt_encodes'] if move else 0)
        assert run['environment']['INPUT_CHAIN_SHA256']==target['payload_sha256']
    return run,d


short={}
for arch in ('m3','m2'):
    rows=[model(f'{arch}-{i}-{mode}',arch,mode=='move') for i,mode in enumerate(('base','move','move','base'),1)]
    keys=m3_keys if arch=='m3' else m2_keys
    for run,d in rows:
        for key in keys:assert d[key]==rows[0][1][key],(arch,key)
        if arch=='m3':
            for key in ('manifest_sha256','program_sha256'):assert run[key]==rows[0][0][key]
    times=[d['eval_seconds'] if arch=='m3' else d['timing']['eval_seconds'] for run,d in rows]
    base=mean((times[0],times[3]));candidate=mean(times[1:3])
    short[arch]={'samples_in_abba_order':times,'base_mean_seconds':base,'move_mean_seconds':candidate,
                 'reduction_percent':100*(1-candidate/base)}
recorded=read('short-comparison.json')
for arch in short:
    assert all(recorded[arch][key]==value for key,value in short[arch].items())
cache_run,cache=model('cache-move','m3',True)
assert cache['plaintext_cache_hits']>0
full={}
for arch in ('m3','m2'):
    name=arch+'-full-move'
    if not (ROOT/name/'native.json').exists():
        assert (ROOT/(arch+'-full-skipped.json')).exists()
        full[arch]={'skipped':read(arch+'-full-skipped.json')['reason']};continue
    run,d=model(name,arch,True)
    old=json.loads((previous/('full-all' if arch=='m3' else 'm2-full-direct')/'native.json').read_text())
    for key in (m3_keys if arch=='m3' else m2_keys):assert old[key]==d[key],(arch,'full parity',key)
    if arch=='m3':
        assert d['generated_token_ids']==[315,279,1614,315]
        full[arch]={'eval_seconds':d['eval_seconds'],'max_exact_error':d['max_abs_error_vs_exact'],
                    'max_polynomial_error':d['max_abs_error_vs_polynomial'],'peak_rss_gib':d['peak_rss_gib'],
                    'moved_coefficient_encodes':d['moved_coefficient_encodes']}
    else:
        m=d['measurements'];assert m['autoregressive_selected_ids']==[273,253,4687,273] and m['autoregressive_tokens_match']
        full[arch]={'eval_seconds':d['timing']['eval_seconds'],'max_polynomial_error':m['max_abs_error'],
                    'peak_rss_gib':m['peak_rss_gib'],'moved_coefficient_encodes':m['plaintext_encoding']['eval_moved_coefficient_encodes']}
assert read('completion.json')['passed']
result={'probes':probes,'short':short,'full':full,'cache':{'hits':cache['plaintext_cache_hits'],
        'max_exact_error':cache['max_abs_error_vs_exact'],'max_polynomial_error':cache['max_abs_error_vs_polynomial']},
        'budget':read('budget.json'),'scope':'Same-binary short ABBA establishes incremental timings; single full candidates check completion/parity.'}
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
