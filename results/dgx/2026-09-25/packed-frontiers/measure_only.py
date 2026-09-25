"""One frozen Mamba-3 run; used for full qualification and independent controls."""
import argparse,json,math,time,traceback
from runlib import ROOT, base_env, run, save, sha, stamp
parser=argparse.ArgumentParser()
parser.add_argument('--revision',required=True)
parser.add_argument('--name',required=True)
parser.add_argument('--payload',default='lm-full')
parser.add_argument('--baseline',action='store_true')
parser.add_argument('--wait')
parser.add_argument('features',nargs='*')
a=parser.parse_args()
try:
    if a.wait:
        while not (ROOT/a.wait).exists():time.sleep(15)
    revision=ROOT/a.revision
    old=json.loads((ROOT/'baseline-target-provenance.json').read_text())
    for path,digest in old['dependencies_sha256'].items():assert sha(path)==digest,path
    if a.baseline:
        binary=ROOT.parent/'square-dispatch-20260925/build/packed_fideslib'
        assert sha(binary)==old['candidate_binaries_sha256']['packed_fideslib']
        source_manifest=ROOT/'baseline-compiled-sources.json'
    else:
        binary=revision/'build/packed_fideslib'
        assert sha(binary)==json.loads((revision/'build.json').read_text())['binaries_sha256']['packed_fideslib']
        source_manifest=revision/'compiled-sources.json'
    payload=ROOT.parent/'mamba3-20260924'/a.payload
    manifest=json.loads((payload/'manifest.json').read_text())
    assert all(sha(payload/p)==h for p,h in manifest['files_sha256'].items())
    env=base_env();env['OMP_NUM_THREADS']='4'
    flags=['--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation',
        '--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights',
        '--move-plaintext-coefficients','--borrow-plaintext-upload','--bsgs-routing-stages','--cache-plaintexts']
    command=['taskset','-c','15-19',binary,payload/'program.txt','{output}/native.json','.001','.001']+flags+a.features
    client=manifest['schema']=='fhemamba-mamba3-lm-v1'
    if client:command+=['--client-head',payload/'client_head.f32']
    def checks(d):
        result={'native':d['passed'],'finite':d['non_finite']==0,'tolerances':d['exact_tolerance']==d['polynomial_tolerance']==.001,
            'errors':all(math.isfinite(d[k]) and 0<=d[k]<=.001 for k in ['max_abs_error_vs_exact','max_abs_error_vs_polynomial']),
            'all_outputs':len(d['per_output_errors'])==len(manifest['output_names']) and
                all(math.isfinite(v) and 0<=v<=.001 for row in d['per_output_errors'] for v in row.values()),
            'ckks':d['ring_dimension']==65536 and d['depth']==44 and d['scale_bits']==59 and d['bootstrap_passes']==2 and d['security']=='not-set',
            'no_intermediate_decrypts':d['evaluation_decryptions']==0}
        if client:result['generation']=d['generated_token_ids']==manifest['exact_token_ids'] and d['client_output_decrypt_count']==manifest['generated_tokens']
        return result
    run(a.name,command,env,1800 if a.payload=='lm-full' else 600,checks,
        {'candidate':not a.baseline,'features':a.features,'payload':str(payload),
         'manifest_sha256':sha(payload/'manifest.json'),'program_sha256':sha(payload/'program.txt'),
         'source_manifest_sha256':sha(source_manifest),'campaign_sha256':sha(__file__)})
    result={'passed':True}
except Exception:result={'passed':False,'error':traceback.format_exc()}
result['finished_utc']=stamp();save(ROOT/(a.name+'-completion.json'),result);print(result,flush=True)
if not result['passed']:raise SystemExit(1)
