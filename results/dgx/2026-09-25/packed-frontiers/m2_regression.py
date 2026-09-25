"""Shared evaluator extraction: immutable encrypted Mamba-2 prefix regression."""
import json, math, time, traceback
from statistics import mean
from runlib import ROOT, base_env, run, save, sha, stamp
try:
    while not (ROOT/'layout-r2/completion.json').exists():time.sleep(15)
    revision=ROOT/('layout-r2' if (ROOT/'layout-r2/build.json').exists() else 'frontier-r1')
    manifest=revision/'compiled-sources.json'
    candidate=revision/'build/stage1_mamba2_decode_fideslib'
    baseline=ROOT.parent/'square-dispatch-20260925/build/stage1_mamba2_decode_fideslib'
    expected=json.loads((ROOT/'baseline-target-provenance.json').read_text())
    assert sha(baseline)==expected['candidate_binaries_sha256']['stage1_mamba2_decode_fideslib']
    for p,h in expected['dependencies_sha256'].items():assert sha(p)==h,p
    assert sha(candidate)==json.loads((revision/'build.json').read_text())['binaries_sha256'][candidate.name]
    reference=None;samples=[]
    for i,is_candidate in enumerate([False,True,True,False],1):
        name=f'm2-regression-{i}'
        env=base_env();env.update(json.loads((ROOT/'environment.json').read_text()))
        binary=candidate if is_candidate else baseline
        env.update(BINARY=str(binary),BINARY_SHA256=sha(binary),SOURCE_ROOT=str(revision/'source'),
            REPO_COMMIT=('working-tree:'+sha(manifest) if is_candidate else '18496e4e6a4e42c367fc356e99a7c4ab8e935eb0'),
            CUDA_LAUNCH_BLOCKING='1',FAST_PLAINTEXT_UPLOAD='1',GPU_PLAINTEXT_NTT='1',
            DIRECT_PLAINTEXT_UPLOAD='1',MOVE_PLAINTEXT_COEFFICIENTS='0',BORROW_PLAINTEXT_UPLOAD='1',
            AUTOREGRESSIVE_CLIENT_LOOP='0',LAYERS='2',TOKENS='2')
        def checks(d):
            p,m=d['parameters'],d['measurements']
            out={'native':d['passed'],'layers':p['n_layers_loaded']==2,'tokens':p['tokens']==2,
                'gate':p['tolerance']==.05,'error':0<=m['max_abs_error']<=.05,
                'outputs':len(m['per_token_decrypt_ok'])==len(m['per_token_max_abs_error'])==2,
                'decrypt':all(m['per_token_decrypt_ok']),
                'all_errors':all(math.isfinite(x) and 0<=x<=.05 for x in m['per_token_max_abs_error']),
                'no_intermediate_decrypts':d['measurement_scope']['zero_intermediate_decrypts']}
            if reference is not None:
                out['same_operations_and_levels']=all(d[k]==reference[k] for k in
                    ['parameters','ckks_levels','operation_counts','operation_counts_by_token','phase_operation_counts'])
            return out
        run(name,['bash',ROOT/'launch_m2_regression.sh','{output}/native.json','2','2'],env,600,checks,
            {'candidate':is_candidate,'source_manifest_sha256':sha(manifest if is_candidate else ROOT/'baseline-compiled-sources.json'),
             'campaign_sha256':sha(__file__),'launcher_sha256':sha(ROOT/'launch_m2_regression.sh')})
        d=json.loads((ROOT/name/'native.json').read_text());reference=reference or d
        samples.append(d['timing']['eval_seconds'])
    result={'passed':True,'samples_abba':samples,'baseline_mean_seconds':mean([samples[0],samples[3]]),
        'candidate_mean_seconds':mean(samples[1:3]),'scope':'Mamba-2 two layers/two evaluations; shared code regression, no full-model speedup claim'}
except Exception:result={'passed':False,'error':traceback.format_exc()}
result['finished_utc']=stamp();save(ROOT/'m2-completion.json',result);print(result,flush=True)
