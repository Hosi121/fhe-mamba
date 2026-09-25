import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/kataiwa/fhemamba/structural-four-20260925')
env=dict(os.environ,OMP_NUM_THREADS='4',LD_LIBRARY_PATH='/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64')
env['PATH']='/usr/local/cuda-13.0/bin:'+env['PATH']
status={'state':'waiting','stage':'remaining-controller','started_at_unix':time.time()}
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def run(name,command,allow_failure=False):
    d=root/name;d.mkdir()
    status.update(state='running',stage=name);save(root/'refinement-status.json',status)
    start=time.time()
    digest=hashlib.sha256(Path(command[3]).read_bytes()).hexdigest() if command[0]=='taskset' else None
    with (d/'run.log').open('w') as log:
        p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    save(d/'run.json',{'command':command,'exit_code':p.returncode,'started_at_unix':start,'wall_seconds':time.time()-start,'binary_sha256':digest})
    if p.returncode and not allow_failure:raise RuntimeError((name,p.returncode))
    if (d/'native.json').exists():return json.loads((d/'native.json').read_text())
try:
    save(root/'refinement-status.json',status)
    while json.loads((root/'remaining-status.json').read_text())['state'] in ('running','waiting'):time.sleep(10)
    for logn in (15,16):
        name=f'small-ring-dense-{logn}'
        run(name,['taskset','-c','15-19',str(root/'small-ring-dense-probe'),'--logn',str(logn),'--output',str(root/name/'native.json')],True)
    run('final-native-build',['cmake','--build',str(root/'backend/spark/kernel'),'-j','4'])
    run('final-native-contracts',['ctest','--test-dir',str(root/'backend/spark/kernel'),'--output-on-failure'])
    flags=['--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation','--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights','--move-plaintext-coefficients','--borrow-plaintext-upload','--bsgs-routing-stages','--cache-plaintexts','--frontier-refresh','--s2c-first','--gpu-plaintext-rns']
    for share in (False,True):
        name='basis-refresh-'+('share' if share else 'baseline')
        binary=root/'backend/spark/kernel/packed_fideslib' if share else Path('/home/kataiwa/fhemamba/gpu-rns-20260925/native-build/packed_fideslib')
        data=run(name,['taskset','-c','15-19',str(binary),str(root/'basis-refresh-probe.txt'),str(root/name/'native.json'),'.001','.001']+flags+(['--share-chebyshev'] if share else []))
        assert data['passed'] and data['bootstraps']>0,name
        if share:assert data['shared_basis_invalidations']>0 and data['shared_basis_hits']>0,data
    status['state']='completed'
except Exception as e:
    status.update(state='failed',error=repr(e))
finally:
    status['finished_at_unix']=time.time();save(root/'refinement-status.json',status)
