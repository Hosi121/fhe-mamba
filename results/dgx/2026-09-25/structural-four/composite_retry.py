import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/kataiwa/fhemamba/structural-four-20260925')
env=dict(os.environ,OMP_NUM_THREADS='4',LD_LIBRARY_PATH='/usr/local/cuda-13.0/lib64')
env['PATH']='/usr/local/cuda-13.0/bin:'+env['PATH']
status={'state':'waiting','stage':'refinement','started_at_unix':time.time()}
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def run(name,command,allow_failure=False):
    d=root/name;d.mkdir();status.update(state='running',stage=name);save(root/'composite-status.json',status)
    start=time.time()
    data={'command':command,'started_at_unix':start}
    if command[0]=='taskset':
        data['binary_sha256']=hashlib.sha256(Path(command[3]).read_bytes()).hexdigest()
        data['library_sha256']=hashlib.sha256((root/'cheddar-build/libcheddar.so').read_bytes()).hexdigest()
    with (d/'run.log').open('w') as log:
        p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    data.update(exit_code=p.returncode,wall_seconds=time.time()-start);save(d/'run.json',data)
    if p.returncode and not allow_failure:raise RuntimeError((name,p.returncode))
try:
    save(root/'composite-status.json',status)
    while json.loads((root/'refinement-status.json').read_text())['state'] in ('waiting','running'):time.sleep(10)
    run('cheddar-reconfigure',['cmake','-S',str(root/'cheddar'),'-B',str(root/'cheddar-build')])
    run('cheddar-rebuild',['cmake','--build',str(root/'cheddar-build'),'--target','composite_probe','-j','6'])
    for bits in (32,64):
        name=f'composite-{bits}'
        run(name,['taskset','-c','15-19',str(root/'cheddar-build/composite_probe'),str(bits),str(root/name/'native.json')],True)
    status['state']='completed'
except Exception as error:
    status.update(state='failed',error=repr(error))
finally:
    status['finished_at_unix']=time.time();save(root/'composite-status.json',status)
