"""Wait for the frozen predecessor, then run this bounded optimization cycle."""
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import time
import traceback
from runlib import ROOT, M3, sha, save, stamp, base_env, run

PREVIOUS=ROOT.parent/'mamba2-plaintext-20260924'
MODES={'base':[], 'naf':['--naf-rotations'], 'reuse':['--reuse-dead-inputs'],
       'direct':['--direct-plaintext-upload'], 'compact':['--compact-weights'],
       'all':['--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights']}

def inventory(payload):
    out=ROOT/(payload+'-inventory.json')
    if not out.exists():
        result=subprocess.run([str(ROOT/'build/packed_resource_inventory'),str(M3/payload/'program.txt')],check=True,capture_output=True,text=True,timeout=180)
        out.write_text(result.stdout)
    return json.loads(out.read_text())

def m3(name, mode, payload='lm-layer1-prefix', cache=False):
    p=M3/payload
    manifest=json.loads((p/'manifest.json').read_text())
    for fname,value in manifest['files_sha256'].items():
        assert sha(p/fname)==value, fname
    inv=inventory(payload)
    env=base_env();env['OMP_NUM_THREADS']='4'
    cmd=['taskset','-c','15-19',ROOT/'build/packed_fideslib',p/'program.txt','{output}/native.json','.001','.001',
         '--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation',*MODES[mode]]
    if (p/'client_head.f32').exists():cmd.extend(['--client-head',p/'client_head.f32'])
    if cache:cmd.append('--cache-plaintexts')
    def check(d):
        return {'native':d['passed'], 'finite':d['non_finite']==0,
                'poly':0<=d['max_abs_error_vs_polynomial']<=.001,'exact':0<=d['max_abs_error_vs_exact']<=.001,
                'generation':payload!='lm-full' or d['generated_token_ids']==[315,279,1614,315],
                'no_intermediate_decryptions':d['evaluation_decryptions']==0,
                'rotation_mode':d['rotation_decomposition']==('naf' if mode in ('naf','all') else 'binary'),
                'rotation_inventory':d['rotations']-d['refresh_rotations']==inv['naf_rotations' if mode in ('naf','all') else 'binary_rotations'],
                'reuse_mode':d['reuse_dead_inputs']==(mode in ('reuse','all')),
                'direct_mode':d['direct_plaintext_upload']==(mode in ('direct','all')),
                'direct_dispatch':mode not in ('direct','all') or d['direct_plaintext_uploads']>0,
                'compact_mode':d['compact_weights']==(mode in ('compact','all')),
                'weight_storage':d['public_weight_bytes']==inv['compact_weight_bytes' if mode in ('compact','all') else 'double_weight_bytes']}
    run(name,cmd,env,2400 if payload=='lm-full' else 300,check,{'mode':mode,'manifest_sha256':sha(p/'manifest.json'),'program_sha256':sha(p/'program.txt'),'payload':str(p)})

def m2(name,direct,layers=2,tokens=2):
    env=base_env();env.update(json.loads((ROOT/'environment.json').read_text()))
    env.update(BINARY=str(ROOT/'build/stage1_mamba2_decode_fideslib'),BINARY_SHA256=sha(ROOT/'build/stage1_mamba2_decode_fideslib'),CUDA_LAUNCH_BLOCKING='1',
               FAST_PLAINTEXT_UPLOAD='1',GPU_PLAINTEXT_NTT='1',DIRECT_PLAINTEXT_UPLOAD=str(int(direct)),JOINT_SUBRING_ENCODING='1',
               AUTOREGRESSIVE_CLIENT_LOOP=str(int(layers==24)),LAYERS=str(layers),TOKENS=str(tokens))
    def check(d):
        p,m=d['parameters'],d['measurements'];e=m['plaintext_encoding']
        return {'native':d['passed'],'layers':p['n_layers_loaded']==layers,'tokens':p['tokens']==tokens,
                'error':0<=m['max_abs_error']<=.05,'decrypt':all(m['per_token_decrypt_ok']),
                'no_intermediate_decryptions':d['measurement_scope']['zero_intermediate_decrypts'],
                'direct_mode':e['direct_upload']==direct,'direct_dispatch':not direct or e['eval_direct_uploads']>0,
                'gpu_ntt':e['gpu_ntt'],'subring':p['joint_gate_schedule']['subring_encoding'],
                'generation':layers!=24 or m['autoregressive_selected_ids']==[273,253,4687,273] and m['autoregressive_tokens_match']}
    run(name,['bash',ROOT/'launch_m2.sh','{output}/native.json',layers,tokens],env,4200 if layers==24 else 300,check)

try:
    started=time.monotonic()
    save(ROOT/'status.json',{'status':'waiting_for_previous','started_utc':stamp()})
    while not (PREVIOUS/'completion.json').exists():
        if time.monotonic()-started>14400:raise RuntimeError('predecessor wait reached review boundary')
        time.sleep(30)
    assert json.loads((PREVIOUS/'completion.json').read_text())['passed'], 'predecessor failed'
    save(ROOT/'status.json',{'status':'building','started_utc':stamp()})
    with (ROOT/'build.log').open('w') as log:
        subprocess.run(['bash',str(ROOT/'build.sh')],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    native_paths=list((ROOT/'source/native').rglob('*'))
    manifest=json.loads((ROOT/'compiled-sources.json').read_text())['files']
    assert all(sha(ROOT/'source'/p)==value for p,value in manifest.items())
    prior=json.loads((PREVIOUS/'target-provenance.json').read_text())
    binaries={p.name:sha(p) for p in (ROOT/'build').iterdir() if p.is_file() and os.access(p,os.X_OK)}
    save(ROOT/'build-provenance.json',{'compiled_source_manifest_sha256':sha(ROOT/'compiled-sources.json'),
          'binaries_sha256':binaries,'compile_commands':json.loads((ROOT/'build/compile_commands.json').read_text()),
          'hardware':prior['hardware'],'compiler':prior['compiler'],'cuda':prior['cuda'],
          'backend_commit':prior['backend_commit'],'predecessor_provenance_sha256':sha(PREVIOUS/'target-provenance.json')})
    for mode in ('mamba2','mamba3'):
        command=[ROOT/'build/packed_plaintext_probe','{output}/native.json']
        if mode=='mamba2':command.append('--mamba2')
        run('probe-'+mode,command,base_env(),300,lambda d:{'native':d['passed'],'exact_rns':d['exact_rns_cases']==160,'shared_policy':d['shared_policy_cases']==90,'error':d['max_abs_error']<1e-6})
    modes=list(MODES)
    for index,mode in enumerate(modes+list(reversed(modes)),1):m3(f'prefix-{index}-{mode}',mode)
    for index,direct in enumerate((False,True,True,False),1):m2(f'm2-{index}-'+('direct' if direct else 'base'),direct)
    means={mode:mean(json.loads(p.read_text())['eval_seconds'] for p in ROOT.glob('prefix-*-'+mode+'/native.json')) for mode in modes}
    m2_means={mode:mean(json.loads(p.read_text())['timing']['eval_seconds'] for p in ROOT.glob('m2-*-'+mode+'/native.json')) for mode in ('base','direct')}
    save(ROOT/'short-comparison.json',{'m3_mean_seconds':means,'m2_mean_seconds':m2_means})
    m3('cache-all','all','payload',True)
    m3('full-base','base','lm-full')
    m3('full-all','all','lm-full')
    if m2_means['direct']<m2_means['base']:m2('m2-full-direct',True,24,5)
    else:save(ROOT/'m2-full-skipped.json',{'reason':'Direct upload did not improve the mirrored two-layer mean; retain the previous full candidate.'})
    save(ROOT/'completion.json',{'passed':True,'finished_utc':stamp()})
except Exception:
    save(ROOT/'completion.json',{'passed':False,'finished_utc':stamp(),'error':traceback.format_exc()})
    raise
