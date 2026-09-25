import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/kataiwa/fhemamba/structural-four-20260925')
env = dict(os.environ, OMP_NUM_THREADS='4')
env['PATH'] = '/usr/local/cuda-13.0/bin:' + env['PATH']
env['LD_LIBRARY_PATH'] = '/usr/local/cuda-13.0/lib64'
status = {'state':'waiting', 'stage':'first-pair', 'started_at_unix':time.time()}
def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)
def run(name, command, cwd=None, limit=3600, allow_failure=False):
    d = root/name
    d.mkdir()
    status.update(state='running', stage=name)
    save(root/'remaining-status.json', status)
    started = time.time()
    with (d/'run.log').open('w') as log:
        p = subprocess.run(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
    record = {'command':command,'cwd':str(cwd) if cwd else None,'exit_code':p.returncode,'started_at_unix':started,'wall_seconds':time.time()-started}
    if command[0]=='taskset':
        record['binary_sha256'] = hashlib.sha256(Path(command[3]).read_bytes()).hexdigest()
    save(d/'run.json',record)
    if p.returncode and not allow_failure:
        raise RuntimeError((name,p.returncode))
try:
    save(root/'remaining-status.json',status)
    while True:
        first=json.loads((root/'first-pair-status.json').read_text())
        if first['state'] in ('completed','failed'):
            break
        time.sleep(10)
    for logn in (15,16):
        name=f'small-ring-{logn}'
        run(name,['taskset','-c','15-19',str(root/'small-ring-probe'),'--logn',str(logn),'--output',str(root/name/'native.json')],allow_failure=True)
    deps=root/'cheddar-private-deps'
    deps.mkdir()
    run('cheddar-gmp-download',['apt-get','download','libgmp-dev'],cwd=deps)
    packages=list(deps.glob('*.deb'))
    assert len(packages)==1
    run('cheddar-gmp-extract',['dpkg-deb','-x',str(packages[0]),str(deps/'extracted')])
    save(deps/'package-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in packages})
    include=deps/'extracted/usr/include'
    run('cheddar-configure',['cmake','-S',str(root/'cheddar'),'-B',str(root/'cheddar-build'),'-DBUILD_UNITTEST=OFF','-DBUILD_TESTS=OFF','-DUSE_GMP=ON','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc','-DGMP=/lib/aarch64-linux-gnu/libgmp.so.10',f'-DCMAKE_CXX_FLAGS=-I{include} -I{include}/aarch64-linux-gnu',f'-DCMAKE_CUDA_FLAGS=-I{include} -I{include}/aarch64-linux-gnu'])
    run('cheddar-build-log',['cmake','--build',str(root/'cheddar-build'),'--target','composite_probe','-j','6'])
    for bits in (32,64):
        name=f'composite-{bits}'
        run(name,['taskset','-c','15-19',str(root/'cheddar-build/composite_probe'),str(bits),str(root/name/'native.json')],allow_failure=True)
    status['state']='completed'
except Exception as e:
    status.update(state='failed',error=repr(e))
finally:
    status['finished_at_unix']=time.time()
    save(root/'remaining-status.json',status)
