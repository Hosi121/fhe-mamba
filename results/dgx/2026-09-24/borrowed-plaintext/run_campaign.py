"""Serial exactness, isolated/combined controls, then qualified full validation."""
import json
import math
from statistics import mean
import subprocess
import traceback
import time
from runlib import ROOT, M3, sha, save, stamp, base_env, run

PREVIOUS = ROOT.parent / 'coefficient-ownership-models-20260924'
FALLBACK = ROOT.parent / 'packed-resources-20260924'
MODES = {'base': (False, False), 'borrow': (True, False),
         'routing': (False, True), 'all': (True, True)}
M3_INVARIANTS = ('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul',
    'logical_refreshes','refresh_batches','bootstrap_passes','ring_dimension',
    'slots','depth','scale_bits','refresh_policy','batch_refresh','gpu_plaintext_ntt',
    'plaintext_ntt_batch','public_weight_count','public_weight_bytes','gpu_ntt_encodes')
M2_INVARIANTS = ('parameters','ckks_levels','operation_counts',
                 'operation_counts_by_token','phase_operation_counts')


def inventory(payload, routing):
    path=ROOT/(payload+('-routing' if routing else '-base')+'-inventory.json')
    if not path.exists():
        command=[str(ROOT/'build/packed_resource_inventory'),str(M3/payload/'program.txt')]
        if routing:command.append('--bsgs-routing-stages')
        result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=180)
        path.write_text(result.stdout)
    return json.loads(path.read_text())


def m3(name, mode, payload='lm-layer1-prefix', cache=False):
    borrow,routing=MODES[mode]
    p=M3/payload
    manifest=json.loads((p/'manifest.json').read_text())
    assert all(sha(p/key)==value for key,value in manifest['files_sha256'].items())
    inv=inventory(payload,routing)
    original=inventory(payload,False)
    env=base_env();env['OMP_NUM_THREADS']='4'
    cmd=['taskset','-c','15-19',ROOT/'build/packed_fideslib',p/'program.txt',
         '{output}/native.json','.001','.001','--planned-refresh','--batch-refresh',
         '--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation','--naf-rotations',
         '--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights','--move-plaintext-coefficients']
    if borrow:cmd.append('--borrow-plaintext-upload')
    if routing:cmd.append('--bsgs-routing-stages')
    if cache:cmd.append('--cache-plaintexts')
    if (p/'client_head.f32').exists():cmd.extend(['--client-head',p/'client_head.f32'])
    def check(d):
        return {'native':d['passed'],'finite':d['non_finite']==0,
            'poly':0<=d['max_abs_error_vs_polynomial']<=.001,
            'exact':0<=d['max_abs_error_vs_exact']<=.001,
            'unchanged_gates':d['polynomial_tolerance']==d['exact_tolerance']==.001,
            'no_intermediate_decryptions':d['evaluation_decryptions']==0,
            'borrow_mode':d['borrow_plaintext_upload']==borrow,
            'borrow_dispatch':d['borrowed_plaintext_uploads']==(d['direct_plaintext_uploads'] if borrow else 0),
            'move_dispatch':d['moved_coefficient_encodes']==d['gpu_ntt_encodes'],
            'routing_mode':d['bsgs_routing_stages']==routing,
            'rotation_inventory':d['rotations']-d['refresh_rotations']==inv['naf_rotations'],
            'saved_rotations':d['routing_stage_rotations_saved']==original['naf_rotations']-inv['naf_rotations'],
            'generation':payload!='lm-full' or d['generated_token_ids']==[315,279,1614,315]}
    run(name,cmd,env,1800 if payload=='lm-full' else 300,check,
        {'mode':mode,'manifest_sha256':sha(p/'manifest.json'),'program_sha256':sha(p/'program.txt'),'payload':str(p)})


def m2(name, borrow, layers=2, tokens=2):
    env=base_env(); env.update(json.loads((ROOT/'environment.json').read_text()))
    env.update(BINARY=str(ROOT/'build/stage1_mamba2_decode_fideslib'),
               BINARY_SHA256=sha(ROOT/'build/stage1_mamba2_decode_fideslib'),
               CUDA_LAUNCH_BLOCKING='1',FAST_PLAINTEXT_UPLOAD='1',GPU_PLAINTEXT_NTT='1',
               DIRECT_PLAINTEXT_UPLOAD='1',MOVE_PLAINTEXT_COEFFICIENTS='0',BORROW_PLAINTEXT_UPLOAD=str(int(borrow)),
               AUTOREGRESSIVE_CLIENT_LOOP=str(int(layers==24)),LAYERS=str(layers),TOKENS=str(tokens))
    def check(d):
        p,m=d['parameters'],d['measurements']; e=m['plaintext_encoding']
        return {'native':d['passed'],'layers':p['n_layers_loaded']==layers,'tokens':p['tokens']==tokens,
            'error':0 <= m['max_abs_error'] <= .05,
            'all_outputs':len(m['per_token_decrypt_ok'])==len(m['per_token_max_abs_error'])==tokens,
            'decrypt':all(m['per_token_decrypt_ok']),
            'all_errors':all(math.isfinite(x) and 0<=x<=.05 for x in m['per_token_max_abs_error']),
            'no_intermediate_decryptions':d['measurement_scope']['zero_intermediate_decrypts'],
            'mode':e['borrow_upload']==borrow,
            'dispatch':e['eval_borrowed_uploads']==(e['eval_fast_uploads'] if borrow else 0) and e['eval_moved_coefficient_encodes']==0,
            'gpu_ntt':e['gpu_ntt'],'direct':e['direct_upload'],
            'subring':p['joint_gate_schedule']['subring_encoding'],
            'generation':layers!=24 or m['autoregressive_selected_ids']==[273,253,4687,273] and m['autoregressive_tokens_match']}
    run(name,['bash',ROOT/'launch_m2.sh','{output}/native.json',layers,tokens],env,
        2700 if layers==24 else 300,check)


def compare():
    summaries={}
    order=('base','borrow','routing','all','all','routing','borrow','base')
    first=None
    for i,mode in enumerate(order,1):
        d=json.loads((ROOT/f'm3-{i}-{mode}/native.json').read_text())
        if first is None:first=d
        for key in M3_INVARIANTS:assert d[key]==first[key],('m3',mode,key)
        summaries.setdefault(mode,[]).append(d['eval_seconds'])
    m3_summary={mode:{'samples':values,'mean_seconds':mean(values)} for mode,values in summaries.items()}
    for mode in m3_summary:
        m3_summary[mode]['reduction_percent']=100*(1-m3_summary[mode]['mean_seconds']/m3_summary['base']['mean_seconds'])
    rows=[json.loads((ROOT/f'm2-{i}-{mode}/native.json').read_text()) for i,mode in enumerate(('base','borrow','borrow','base'),1)]
    for d in rows:
        for key in M2_INVARIANTS:assert d[key]==rows[0][key],('m2',key)
    times=[d['timing']['eval_seconds'] for d in rows]
    base=mean((times[0],times[3]));candidate=mean(times[1:3])
    return {'m3':m3_summary,'m2':{'samples_in_abba_order':times,'base_mean_seconds':base,
        'borrow_mean_seconds':candidate,'reduction_percent':100*(1-candidate/base)}}


try:
    started=time.monotonic()
    save(ROOT/'status.json',{'status':'waiting_for_previous','predecessor':str(PREVIOUS),'started_utc':stamp()})
    while not (PREVIOUS/'completion.json').exists():
        if time.monotonic()-started>7800:raise RuntimeError('predecessor review interval reached')
        time.sleep(30)
    assert json.loads((PREVIOUS/'completion.json').read_text())['passed']
    with (ROOT/'build.log').open('w') as log:
        subprocess.run(['bash',str(ROOT/'build.sh')],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    expected=json.loads((ROOT/'compiled-sources.json').read_text())['files']
    assert all(sha(ROOT/'source'/p)==value for p,value in expected.items())
    for arch in ('mamba2','mamba3'):
        env=base_env();env['OMP_NUM_THREADS']='4'
        cmd=['taskset','-c','15-19',ROOT/'build/packed_plaintext_probe','{output}/native.json']
        if arch=='mamba2':cmd.append('--mamba2')
        run('probe-'+arch,cmd,env,300,lambda d:{'native':d['passed'],
            'exact_rns':d['exact_rns_cases']==160,'moved_rns':d['moved_coefficient_cases']==160,
            'shared_policy':d['shared_policy_cases']==162,'borrowed_rns':d['borrowed_rns_cases']==320,'error':d['max_abs_error']<1e-6})
    for i,mode in enumerate(('base','borrow','routing','all','all','routing','borrow','base'),1):m3(f'm3-{i}-{mode}',mode)
    for i,borrow in enumerate((False,True,True,False),1):m2(f'm2-{i}-'+('borrow' if borrow else 'base'),borrow)
    comparisons=compare();save(ROOT/'short-comparison.json',comparisons)
    m3('cache-all','all','payload',True)
    selected=min(comparisons['m3'],key=lambda mode:comparisons['m3'][mode]['mean_seconds'])
    save(ROOT/'selection.json',{'m3_mode':selected,'criterion':'Lowest mirrored short mean; full same-binary comparison decides the full-runtime claim.'})
    if selected!='base':
        m3('m3-full-base','base','lm-full')
        m3('m3-full-candidate',selected,'lm-full')
        a=json.loads((ROOT/'m3-full-base/native.json').read_text())
        b=json.loads((ROOT/'m3-full-candidate/native.json').read_text())
        for key in M3_INVARIANTS:assert a[key]==b[key],('m3','full parity',key)
    else:save(ROOT/'m3-full-skipped.json',{'reason':'No candidate improves the short mean.'})
    if comparisons['m2']['reduction_percent']>0:
        m2('m2-full-borrow',True,24,5)
        old=json.loads((FALLBACK/'m2-full-direct/native.json').read_text())
        new=json.loads((ROOT/'m2-full-borrow/native.json').read_text())
        for key in M2_INVARIANTS:assert old[key]==new[key],('m2','full parity',key)
    else:save(ROOT/'m2-full-skipped.json',{'reason':'No improvement in the mirrored short mean.'})
    completion={'passed':True}
except Exception:
    completion={'passed':False,'error':traceback.format_exc()}
completion['finished_utc']=stamp()
save(ROOT/'completion.json',completion)
print(json.dumps(completion),flush=True)
