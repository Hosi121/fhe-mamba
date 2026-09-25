import json, subprocess, traceback
from runlib import ROOT, run, base_env, sha, save, stamp

try:
    old = json.loads((ROOT / 'baseline-target-provenance.json').read_text())
    for path, value in old['dependencies_sha256'].items():
        assert sha(path) == value, path
    binary = ROOT.parent / 'square-dispatch-20260925/build/packed_fideslib'
    assert sha(binary) == old['candidate_binaries_sha256']['packed_fideslib']
    payload = ROOT.parent / 'mamba3-20260924/lm-full'
    manifest = json.loads((payload / 'manifest.json').read_text())
    assert all(sha(payload / p) == h for p, h in manifest['files_sha256'].items())
    env = base_env(); env['OMP_NUM_THREADS'] = '4'
    command = ['taskset','-c','15-19',binary,payload/'program.txt','{output}/native.json','.001','.001',
        '--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation',
        '--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights',
        '--move-plaintext-coefficients','--borrow-plaintext-upload','--bsgs-routing-stages','--cache-plaintexts',
        '--client-head',payload/'client_head.f32']
    run('full-baseline',command,env,1800,lambda d: {
        'native':d['passed'],'finite':d['non_finite']==0,
        'exact_gate':0 <= d['max_abs_error_vs_exact'] <= .001,
        'poly_gate':0 <= d['max_abs_error_vs_polynomial'] <= .001,
        'tolerances':d['polynomial_tolerance']==d['exact_tolerance']==.001,
        'generation':d['generated_token_ids']==manifest['exact_token_ids']==[315,279,1614,315],
        'zero_intermediate_decryptions':d['evaluation_decryptions']==0,
        'expected_squares':d['square_arithmetic_calls']==4386,
    },{'payload':str(payload),'program_sha256':sha(payload/'program.txt'),
        'manifest_sha256':sha(payload/'manifest.json'),'campaign_sha256':sha(__file__)})
    result={'passed':True,'stage':'baseline only'}
except Exception:
    result={'passed':False,'error':traceback.format_exc()}
result['finished_utc']=stamp()
save(ROOT/'completion.json',result)
print(result,flush=True)
