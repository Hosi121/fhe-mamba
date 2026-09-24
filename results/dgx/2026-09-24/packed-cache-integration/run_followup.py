"""Reuse the preceding executable and qualify its cached full-model path."""
import json
from pathlib import Path
from statistics import mean
import shutil
import time
import traceback
from runlib import ROOT, sha, save, stamp, base_env, run

PREVIOUS=ROOT.parent/'borrowed-plaintext-20260924'
KEYS=('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul','logical_refreshes',
      'refresh_batches','bootstrap_passes','ring_dimension','slots','depth','scale_bits',
      'refresh_policy','batch_refresh','public_weight_count','public_weight_bytes',
      'rotations','refresh_rotations','bsgs_routing_stages','routing_stage_rotations_saved')


def measure(name,template,cached,full=False):
    original=json.loads((template/'run.json').read_text())
    baseline=json.loads((template/'native.json').read_text())
    assert original['passed'] and original['native_sha256']==sha(template/'native.json')
    command=original['command'][:]
    assert command[:3]==['taskset','-c','15-19']
    assert sha(command[3])==original['binary_sha256']
    payload=Path(original['payload'])
    assert sha(payload/'manifest.json')==original['manifest_sha256']
    assert sha(payload/'program.txt')==original['program_sha256']
    manifest=json.loads((payload/'manifest.json').read_text())
    assert all(sha(payload/key)==value for key,value in manifest['files_sha256'].items())
    command[5]='{output}/native.json'
    assert '--cache-plaintexts' not in command
    if cached:command.append('--cache-plaintexts')
    env=base_env();env['OMP_NUM_THREADS']='4'
    def checks(d):
        return {'native':d['passed'],'finite':d['non_finite']==0,
            'unchanged_gates':d['polynomial_tolerance']==d['exact_tolerance']==.001,
            'poly':0<=d['max_abs_error_vs_polynomial']<=.001,
            'exact':0<=d['max_abs_error_vs_exact']<=.001,
            'cache_mode':d['cache_plaintexts']==cached,
            'cache_dispatch':d['plaintext_cache_hits']>0 if cached else d['plaintext_cache_hits']==0,
            'same_parameters_and_operations':all(d[key]==baseline[key] for key in KEYS),
            'no_intermediate_decryptions':d['evaluation_decryptions']==0,
            'generation':not full or d['generated_token_ids']==[315,279,1614,315]}
    run(name,command,env,1800 if full else 300,checks,
        {'cache':cached,'payload':str(payload),'manifest_sha256':original['manifest_sha256'],
         'program_sha256':original['program_sha256'],'baseline_native_sha256':sha(template/'native.json'),
         'baseline_run_sha256':sha(template/'run.json')})


try:
    start=time.monotonic()
    save(ROOT/'status.json',{'status':'waiting_for_previous','predecessor':str(PREVIOUS),'started_utc':stamp()})
    while not (PREVIOUS/'completion.json').exists():
        if time.monotonic()-start>10800:raise RuntimeError('predecessor review interval reached')
        time.sleep(30)
    assert json.loads((PREVIOUS/'completion.json').read_text())['passed']
    selected=json.loads((PREVIOUS/'selection.json').read_text())['m3_mode']
    if selected=='base':raise RuntimeError('No optimized full candidate qualified; review the preceding controls first.')
    for name in ('compiled-sources.json','compiled-sources.tar.gz','target-provenance.json','environment.json','launch_m2.sh'):
        shutil.copyfile(PREVIOUS/name,ROOT/name)
    prefix=next(p.parent for p in sorted(PREVIOUS.glob('m3-*-'+selected+'/native.json')) if 'full' not in p.parent.name)
    for label,source in (('baseline-prefix',prefix),('baseline-full',PREVIOUS/'m3-full-candidate')):
        dest=ROOT/label;dest.mkdir()
        for name in ('run.json','native.json','native.log'):shutil.copyfile(source/name,dest/name)
    for i,cached in enumerate((False,True,True,False),1):
        measure(f'prefix-{i}-'+('cache' if cached else 'base'),prefix,cached)
    samples=[json.loads((ROOT/f'prefix-{i}-{mode}/native.json').read_text())['eval_seconds']
             for i,mode in enumerate(('base','cache','cache','base'),1)]
    base=mean((samples[0],samples[3]));candidate=mean(samples[1:3])
    save(ROOT/'short-comparison.json',{'samples_in_abba_order':samples,'base_mean_seconds':base,
        'cache_mean_seconds':candidate,'reduction_percent':100*(1-candidate/base),'selected_mode':selected})
    if candidate<base:measure('full-cache',PREVIOUS/'m3-full-candidate',True,True)
    else:save(ROOT/'full-skipped.json',{'reason':'Cached short mean did not improve; retain the preceding complete candidate.'})
    completion={'passed':True}
except Exception:
    completion={'passed':False,'error':traceback.format_exc()}
completion['finished_utc']=stamp();save(ROOT/'completion.json',completion)
print(json.dumps(completion),flush=True)
