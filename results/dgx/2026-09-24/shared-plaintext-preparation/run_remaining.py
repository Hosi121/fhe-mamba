"""Bounded background sequence: pressure control, full comparison, Mamba-3 regression."""
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys
import traceback

root = Path('/home/kataiwa/fhemamba/mamba2-plaintext-20260924')
def stage(name, *extra):
    with (root / (name + '-controller.log')).open('w') as log:
        subprocess.run([sys.executable, str(root/'run_study.py'), name, *extra],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
try:
    assert json.loads((root / 'smoke-completion.json').read_text())['passed']
    with (root / 'target-provenance.json').open('w') as report:
        subprocess.run([sys.executable, str(root/'target_provenance.py')],check=True,stdout=report)
    stage('pressure')
    rows = {mode:[json.loads(p.read_text()) for p in root.glob(f'pressure-*-{mode}/native.json')]
            for mode in ('b','c')}
    assert all(len(v) == 2 and all(r['passed'] for r in v) for v in rows.values())
    assert all(r['measurements']['plaintext_encoding']['eval_gpu_ntt_encodes'] > 0 for r in rows['c'])
    means = {mode:mean(r['timing']['eval_seconds'] for r in value) for mode,value in rows.items()}
    candidate = min(means,key=means.get)
    (root/'selection.json').write_text(json.dumps({'mean_seconds':means,'candidate':candidate,
        'rule':'Faster valid mean under B/C/C/B with nonzero GPU NTT dispatch.'},indent=2)+'\n')
    stage('full','--candidate',candidate)
    stage('mamba3')
    (root/'completion.json').write_text(json.dumps({'passed':True,'candidate':candidate})+'\n')
except Exception:
    (root/'completion.json').write_text(json.dumps({'passed':False,'error':traceback.format_exc()})+'\n')
    raise
