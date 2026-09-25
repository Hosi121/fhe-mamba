"""Build only after the baseline finishes; run serial immutable prefix controls."""
import json, math, subprocess, sys, time, traceback
from statistics import mean
from runlib import ROOT, base_env, run, save, sha, stamp

revision = ROOT / sys.argv[1]
features = sys.argv[2:]
name = revision.name
base = ROOT.parent / 'square-dispatch-20260925/build/packed_fideslib'
source = ROOT.parent / 'mamba3-20260924'
flags = ['--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation',
    '--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights',
    '--move-plaintext-coefficients','--borrow-plaintext-upload','--bsgs-routing-stages','--cache-plaintexts']
try:
    while not (ROOT/'layout-r1/completion.json').exists(): time.sleep(15)
    assert json.loads((ROOT/'completion.json').read_text())['passed']
    old = json.loads((ROOT/'baseline-target-provenance.json').read_text())
    assert sha(base) == old['candidate_binaries_sha256']['packed_fideslib']
    for p,h in old['dependencies_sha256'].items(): assert sha(p)==h,p
    src = revision/'source';src.mkdir()
    import tarfile
    with tarfile.open(revision/'compiled-sources.tar.gz') as archive: archive.extractall(src,filter='data')
    manifest = json.loads((revision/'compiled-sources.json').read_text())
    assert all(sha(src/p)==h for p,h in manifest['files'].items())
    commands=[['cmake','-S',str(src/'native/fideslib_stage0'),'-B',str(revision/'build'),
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH=/home/kataiwa/fhemamba/spark/install-2a70798e869944af;/home/kataiwa/fhe-deps/openfhe-fides',
        '-Dfideslib_DIR=/home/kataiwa/fhemamba/spark/install-2a70798e869944af/share/fideslib/cmake'],
        ['cmake','--build',str(revision/'build'),'--target','packed_fideslib','stage1_mamba2_decode_fideslib','-j','4']]
    for i, command in enumerate(commands):
        with (revision/f'build-{i}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    binary = revision/'build/packed_fideslib'
    save(revision/'build.json',{'commands':commands,'passed':True,
        'source_manifest_sha256':sha(revision/'compiled-sources.json'),
        'binaries_sha256':{n:sha(revision/'build'/n) for n in ['packed_fideslib','stage1_mamba2_decode_fideslib']},
        'compile_commands':json.loads((revision/'build/compile_commands.json').read_text()),
        'compiler':subprocess.check_output(['g++','--version'],text=True),
        'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True)})
    def measure(label,candidate,payload,full=False):
        p=source/payload;m=json.loads((p/'manifest.json').read_text());client=m['schema']=='fhemamba-mamba3-lm-v1'
        assert all(sha(p/f)==h for f,h in m['files_sha256'].items())
        env=base_env();env['OMP_NUM_THREADS']='4'
        command=['taskset','-c','15-19',binary if candidate else base,p/'program.txt','{output}/native.json','.001','.001']+flags
        if candidate:command+=features
        if client:command+=['--client-head',p/'client_head.f32']
        checks=lambda d:{'native':d['passed'],'finite':d['non_finite']==0,
            'errors':all(math.isfinite(d[k]) and 0<=d[k]<=.001 for k in ['max_abs_error_vs_exact','max_abs_error_vs_polynomial']),
            'tolerances':d['exact_tolerance']==d['polynomial_tolerance']==.001,
            'ckks':d['ring_dimension']==65536 and d['depth']==44 and d['scale_bits']==59 and d['bootstrap_passes']==2,
            'no_intermediate_decrypts':d['evaluation_decryptions']==0,
            'generation':not client or d['generated_token_ids']==m['exact_token_ids']}
        run(label,command,env,1800 if full else 600,checks,
            {'candidate':candidate,'features':features if candidate else [],'payload':str(p),
             'manifest_sha256':sha(p/'manifest.json'),'program_sha256':sha(p/'program.txt'),
             'campaign_sha256':sha(__file__),'source_manifest_sha256':sha(revision/'compiled-sources.json' if candidate else ROOT/'baseline-compiled-sources.json')})
        return json.loads((ROOT/label/'native.json').read_text())
    measure(name+'-synthetic',True,'payload')
    measure(name+'-layout-fixture',True,ROOT/'layout-fixture')
    features=['--batch-polynomials','--defer-layout','--frontier-refresh']
    measure(name+'-combined-fixture',True,ROOT/'layout-fixture')
    samples=[]
    configurations=[[],['--frontier-refresh'],['--batch-polynomials','--defer-layout','--frontier-refresh'],
        ['--batch-polynomials','--defer-layout','--frontier-refresh'],['--frontier-refresh'],[]]
    for i,features in enumerate(configurations,1):
        d=measure(f'{name}-prefix-{i}',bool(features),'lm-layer1')
        samples.append(d['eval_seconds'])
    a=mean([samples[0],samples[5]])
    comparisons=[]
    for label,indices,options in [('frontier',[1,4],['--frontier-refresh']),
        ('combined',[2,3],['--batch-polynomials','--defer-layout','--frontier-refresh'])]:
        b=mean(samples[i] for i in indices)
        comparisons.append({'mechanism':label,'baseline_mean_seconds':a,'candidate_mean_seconds':b,
            'reduction_percent':100*(1-b/a),'features':options})
    save(revision/'prefix-comparison.json',{'samples_alccla':samples,'comparisons':comparisons})
    best=max(comparisons,key=lambda v:v['reduction_percent'])
    result={'passed':True,'prefix':comparisons}
    if best['reduction_percent']>.5:
        features=best['features']
        d=measure(name+'-full',True,'lm-full',True)
        result['full']={'eval_seconds':d['eval_seconds'],'features':features}

except Exception:result={'passed':False,'error':traceback.format_exc()}
result['finished_utc']=stamp();save(revision/'completion.json',result);print(result,flush=True)
