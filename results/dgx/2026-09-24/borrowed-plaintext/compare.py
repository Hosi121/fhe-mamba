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
    assert run['controller_sha256']==sha('runlib.py')
    assert run['launcher_sha256']==sha('launch_m2.sh')
    assert run['binary_sha256']==binaries[binary]
    return run,d


probes={}
for arch in ('mamba2','mamba3'):
    run,d=checked('probe-'+arch,'packed_plaintext_probe')
    assert d['exact_rns_cases']==d['moved_coefficient_cases']==160
    assert d['shared_policy_cases']==162 and d['borrowed_rns_cases']==320 and d['tolerance']==1e-6 and 0<=d['max_abs_error']<1e-6
    times={}
    for level in (21,26,34):
        pair={}
        for mode,label in ((8,'copy'),(9,'borrow')):
            rows=[s for s in d['samples'] if s['mode']==mode and s['level']==level and s['iteration']>=0]
            assert len(rows)==8
            pair[label]={'median_encode_ms':median(s['encode_ms'] for s in rows),
                         'median_preparation_ms':median(s['encode_ms']+s['upload_ms'] for s in rows)}
        pair['reduction_percent']=100*(1-pair['borrow']['median_preparation_ms']/pair['copy']['median_preparation_ms'])
        times[str(level)]=pair
    probes[arch]={'max_abs_error':d['max_abs_error'],'levels':times}

m3_keys=('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul','logical_refreshes',
    'refresh_batches','bootstrap_passes','ring_dimension','slots','depth','scale_bits',
    'refresh_policy','batch_refresh','gpu_plaintext_ntt','plaintext_ntt_batch',
    'public_weight_count','public_weight_bytes','gpu_ntt_encodes')
m2_keys=('parameters','ckks_levels','operation_counts','operation_counts_by_token','phase_operation_counts')
modes={'base':(False,False),'borrow':(True,False),'routing':(False,True),'all':(True,True)}


def m3(name,mode,payload='lm-layer1-prefix'):
    run,d=checked(name,'packed_fideslib');borrow,routing=modes[mode]
    assert run['mode']==mode
    assert ('--borrow-plaintext-upload' in run['command'])==borrow
    assert ('--bsgs-routing-stages' in run['command'])==routing
    assert d['non_finite']==d['evaluation_decryptions']==0
    assert d['polynomial_tolerance']==d['exact_tolerance']==.001
    assert 0<=d['max_abs_error_vs_exact']<=.001 and 0<=d['max_abs_error_vs_polynomial']<=.001
    assert d['borrow_plaintext_upload']==borrow and d['bsgs_routing_stages']==routing
    assert d['borrowed_plaintext_uploads']==(d['direct_plaintext_uploads'] if borrow else 0)
    assert d['moved_coefficient_encodes']==d['gpu_ntt_encodes']
    inv=read(payload+('-routing' if routing else '-base')+'-inventory.json')
    base=read(payload+'-base-inventory.json')
    assert d['rotations']-d['refresh_rotations']==inv['naf_rotations']
    assert d['routing_stage_rotations_saved']==base['naf_rotations']-inv['naf_rotations']
    return run,d


def m2(name,borrow):
    run,d=checked(name,'stage1_mamba2_decode_fideslib')
    m=d['measurements'];n=d['parameters']['tokens'];e=m['plaintext_encoding']
    assert len(m['per_token_decrypt_ok'])==len(m['per_token_max_abs_error'])==n
    assert all(m['per_token_decrypt_ok'])
    assert all(math.isfinite(x) and 0<=x<=.05 for x in m['per_token_max_abs_error'])
    assert d['measurement_scope']['zero_intermediate_decrypts']
    assert e['borrow_upload']==borrow and not e['move_coefficients']
    assert e['eval_borrowed_uploads']==(e['eval_fast_uploads'] if borrow else 0)
    assert e['eval_moved_coefficient_encodes']==0
    assert run['environment']['INPUT_CHAIN_SHA256']==target['payload_sha256']
    assert run['environment']['BORROW_PLAINTEXT_UPLOAD']==str(int(borrow))
    return run,d


short={};groups={};first=None
for i,mode in enumerate(('base','borrow','routing','all','all','routing','borrow','base'),1):
    run,d=m3(f'm3-{i}-{mode}',mode)
    if first is None:first=(run,d)
    for key in m3_keys:assert d[key]==first[1][key],('m3',mode,key)
    for key in ('manifest_sha256','program_sha256'):assert run[key]==first[0][key]
    groups.setdefault(mode,[]).append(d['eval_seconds'])
short['m3']={mode:{'samples':values,'mean_seconds':mean(values)} for mode,values in groups.items()}
for mode in short['m3']:
    short['m3'][mode]['reduction_percent']=100*(1-short['m3'][mode]['mean_seconds']/short['m3']['base']['mean_seconds'])
rows=[m2(f'm2-{i}-{mode}',mode=='borrow') for i,mode in enumerate(('base','borrow','borrow','base'),1)]
for run,d in rows:
    for key in m2_keys:assert d[key]==rows[0][1][key],('m2',key)
times=[d['timing']['eval_seconds'] for run,d in rows]
base=mean((times[0],times[3]));candidate=mean(times[1:3])
short['m2']={'samples_in_abba_order':times,'base_mean_seconds':base,'borrow_mean_seconds':candidate,
             'reduction_percent':100*(1-candidate/base)}
assert short==read('short-comparison.json')
cache_run,cache=m3('cache-all','all','payload');assert cache['plaintext_cache_hits']>0
full={}
selected=read('selection.json')['m3_mode']
assert selected==min(short['m3'],key=lambda mode:short['m3'][mode]['mean_seconds'])
if selected!='base':
    rows=[m3('m3-full-base','base','lm-full'),m3('m3-full-candidate',selected,'lm-full')]
    summaries={}
    for label,(run,d) in zip(('base','candidate'),rows):
        for key in m3_keys:assert d[key]==rows[0][1][key],('m3','full parity',key)
        for key in ('manifest_sha256','program_sha256'):assert run[key]==rows[0][0][key]
        assert d['generated_token_ids']==[315,279,1614,315]
        summaries[label]={key:d[key] for key in ('eval_seconds','host_encoding_seconds','plaintext_upload_seconds',
            'bootstrap_seconds','rotations','refresh_rotations','routing_stage_rotations_saved',
            'optimized_routing_stages','gpu_ntt_encodes','borrowed_plaintext_uploads',
            'max_abs_error_vs_exact','max_abs_error_vs_polynomial','peak_rss_gib')}
        summaries[label]['wall_seconds']=run['wall_seconds']
    summaries['selected_mode']=selected
    summaries['reduction_percent']=100*(1-summaries['candidate']['eval_seconds']/summaries['base']['eval_seconds'])
    full['m3']=summaries
else:
    full['m3']=read('m3-full-skipped.json')
if (ROOT/'m2-full-borrow/native.json').exists():
    run,d=m2('m2-full-borrow',True)
    old=json.loads((previous/'m2-full-direct/native.json').read_text())
    for key in m2_keys:assert old[key]==d[key],('m2','full parity',key)
    m=d['measurements'];assert m['autoregressive_selected_ids']==[273,253,4687,273] and m['autoregressive_tokens_match']
    full['m2']={'eval_seconds':d['timing']['eval_seconds'],'max_polynomial_error':m['max_abs_error'],
                'peak_rss_gib':m['peak_rss_gib'],'borrowed_uploads':m['plaintext_encoding']['eval_borrowed_uploads']}
else:full['m2']=read('m2-full-skipped.json')
assert read('completion.json')['passed']
result={'probes':probes,'short':short,'full':full,'cache':{'hits':cache['plaintext_cache_hits'],
        'max_exact_error':cache['max_abs_error_vs_exact'],'max_polynomial_error':cache['max_abs_error_vs_polynomial']},
        'budget':read('budget.json'),'scope':'Separate and combined mirrored M3 controls, matched full M3 base/candidate. M2 keeps coefficient moves disabled; its incremental timing claim uses ABBA and the single full run checks parity.'}
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
