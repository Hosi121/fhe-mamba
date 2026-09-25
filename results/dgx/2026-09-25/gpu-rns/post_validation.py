"""Run existing plaintext regressions only after the full timing pair finishes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/kataiwa/fhemamba/gpu-rns-20260925')
env = dict(os.environ, OMP_NUM_THREADS='4', LD_LIBRARY_PATH='/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64')
status = {'state': 'waiting', 'stage': 'full comparison'}


def write():
    temporary = root / 'post-validation.tmp'
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(root / 'post-validation.json')


write()
try:
    while True:
        main = json.loads((root / 'status.json').read_text())
        if main['state'] != 'running':
            assert main['state'] == 'completed', main
            break
        time.sleep(20)
    status['state'] = 'running'
    for mamba2 in [False, True]:
        name = 'existing-plaintext-' + ('mamba2' if mamba2 else 'mamba3')
        status['stage'] = name
        write()
        directory = root / name
        directory.mkdir()
        command = ['taskset', '-c', '15-19', str(root / 'native-build/packed_plaintext_probe'), str(directory / 'native.json')]
        if mamba2:
            command += ['--mamba2']
        started = time.time()
        with (directory / 'run.log').open('w') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        record = {'command': command, 'exit_code': process.returncode, 'started_at_unix': started,
                  'wall_seconds': time.time() - started,
                  'environment': {k:env[k] for k in ['OMP_NUM_THREADS','LD_LIBRARY_PATH']},
                  'binary_sha256': hashlib.sha256(Path(command[3]).read_bytes()).hexdigest()}
        (directory/'run.json').write_text(json.dumps(record,indent=2)+'\n')
        assert process.returncode == 0, name
        data = json.loads((directory/'native.json').read_text())
        assert data['passed'] and data['exact_rns_cases'] == 160 and data['shared_policy_cases'] == 162
    prior = json.loads((root/'prior-dependency-verification.json').read_text())
    for filename,digest in prior['files_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest,filename
    (root/'dependency-verification.json').write_text(json.dumps({'passed':True,'files_sha256':prior['files_sha256'],'note':'Canonical dependencies unchanged after both full runs and existing plaintext regressions'},indent=2)+'\n')
    status['state'] = 'completed'
except Exception as error:
    status['state'] = 'failed'
    status['error'] = repr(error)
finally:
    status['finished_at_unix'] = time.time()
    write()
