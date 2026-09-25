"""Qualify the independently promising scheduler if combinations miss 20%."""
import json,subprocess,time,traceback
from runlib import ROOT, save, stamp
try:
    while not (ROOT/'m2-completion.json').exists():time.sleep(15)
    baseline=json.loads((ROOT/'full-baseline/native.json').read_text())['eval_seconds']
    qualified=[]
    for label in ['frontier-r1-full','layout-r2-full']:
        record=ROOT/label/'run.json'
        if record.exists():
            d=json.loads(record.read_text())
            if d.get('passed') and d['eval_seconds']<=baseline*.8:qualified.append(label)
    if qualified:result={'passed':True,'skipped':'20 percent target reached','qualifying_trials':qualified}
    else:
        subprocess.run(['python3',ROOT/'measure_only.py','--revision','frontier-r1','--name','frontier-only-full',
            '--','--frontier-refresh'],check=True)
        result={'passed':True,'additional_full_trial':'frontier-only-full'}
except Exception:result={'passed':False,'error':traceback.format_exc()}
result['finished_utc']=stamp();save(ROOT/'qualification-completion.json',result);print(result,flush=True)
