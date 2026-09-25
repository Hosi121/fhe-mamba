import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/kataiwa/fhemamba/structural-four-20260925')
env=dict(os.environ,OMP_NUM_THREADS='4',LD_LIBRARY_PATH='/usr/local/cuda-13.0/lib64')
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
status={'state':'running','stage':'final alternative-source qualification','started_at_unix':time.time()}
try:
    save(root/'final-small-status.json',status)
    results={}
    for bits in (32,64):
        name=f'composite-final-{bits}';d=root/name;d.mkdir()
        status['stage']=name;save(root/'final-small-status.json',status)
        binary=root/'cheddar-build/composite_probe'
        command=['taskset','-c','15-19',str(binary),str(bits),str(d/'native.json')]
        record={'command':command,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
                'library_sha256':hashlib.sha256((root/'cheddar-build/libcheddar.so').read_bytes()).hexdigest(),
                'started_at_unix':time.time()}
        with (d/'run.log').open('w') as log:p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        record.update(exit_code=p.returncode,wall_seconds=time.time()-record['started_at_unix']);save(d/'run.json',record)
        results[bits]=json.loads((d/'native.json').read_text())
    assert not results[32]['passed'] and results[64]['passed']
    assert json.loads((root/'composite-stock32.json').read_text())['passed']
    assert json.loads((root/'refinement-status.json').read_text())['state']=='completed'
    for n in (15,16):assert json.loads((root/f'small-ring-dense-{n}/native.json').read_text())['passed']
    selection={'complete':True,'eligible':['hoist-rotations','share-chebyshev'],
        'decisions':{
            'hoist-rotations':'Exact RNS/metadata checks and alternating model prefixes pass; integrate for full comparison.',
            'share-chebyshev':'Frozen-polynomial and forced-refresh invalidation checks pass; integrate for full comparison.',
            '32-bit-rns':'The nominal 59-bit composite-rescale profile fails numerical gates; the same-circuit 64-bit control and upstream 40-bit 32-bit control pass. Reject this profile; do not infer that all 32-bit RNS is impossible or compare failed-output times as speedups.',
            'small-ring':'CPU encrypted ring-switch/refresh/resume passes eight numerical cases; the roughly 55-second boundary outweighs its ordinary-arithmetic saving and cannot retain current GPU refresh speed. Reject this CPU path; a GPU ring switcher is not implemented or benchmarked.'},
        'scope':'One implementation mechanism per candidate. Full comparison includes only the two qualified shared-intermediate paths; model and gates unchanged.'}
    save(root/'small-selection.json',selection)
    status.update(state='completed',stage='all four small trials reviewed',finished_at_unix=time.time());save(root/'final-small-status.json',status)
    with (root/'full-controller.log').open('w') as log:
        subprocess.run(['python3',str(root/'full_controller.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as error:
    status.update(state='failed',error=repr(error),finished_at_unix=time.time());save(root/'final-small-status.json',status)
