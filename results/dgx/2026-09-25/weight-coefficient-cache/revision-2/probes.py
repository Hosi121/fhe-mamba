import json,subprocess,traceback
from runlib import ROOT,read,run,save,sha,stamp
try:
 subprocess.run(['python3',str(ROOT/'preflight.py')],check=True)
 rev=ROOT/read(ROOT/'config.json')['revision']
 for arch in ('mamba3','mamba2'):
  args=['{output}/native.json']
  if arch=='mamba2':args.append('--mamba2')
  args.extend(['--program',str(ROOT.parent/'mamba3-20260924/lm-layer1-prefix/program.txt')])
  run('probe-'+arch,rev/'build/weight_plaintext_probe',args,
      lambda d:{'native':d['passed'],'exact':d['exact_rns_and_metadata'],'immutable':d['immutable_cache'],
                'level_key':d['level_key_checked'],'budget':d['budget_checked'],'cases':d['cases']==60,
                'gpu_cases':d['gpu_cases']==60,'fallback':d['rejected_cases']>0,'hits':d['hit_cases']>0},
      timeout=600,metadata={'controller_sha256':sha(__file__),'architecture':arch})
 result={'passed':True}
except Exception:result={'passed':False,'error':traceback.format_exc()}
result['finished_utc']=stamp();save(ROOT/'probe-completion.json',result);print(json.dumps(result),flush=True)
