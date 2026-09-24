"""Serial measured jobs with durable completion records, timeouts and a review ledger."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/kataiwa/fhemamba/packed-cache-integration-20260924')
M3 = Path('/home/kataiwa/fhemamba/mamba3-20260924')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def base_env():
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUDA_LAUNCH_BLOCKING'):
        env.pop(name, None)
    env['LD_LIBRARY_PATH'] = '/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64'
    return env

def run(name, command, env, timeout, validator, metadata=None):
    out = ROOT / name
    out.mkdir()
    budget_path = ROOT / 'budget.json'
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {
        'review_seconds': 3600, 'used_seconds': 0, 'runs': [],
        'scope': 'Internal review interval, not a newly inferred user spending limit.'}
    if budget['used_seconds'] + timeout > budget['review_seconds']:
        raise RuntimeError('review boundary reached')
    command = [str(x).replace('{output}', str(out)) for x in command]
    record = {'name': name, 'command': command, 'started_utc': stamp(),
              'timeout_seconds': timeout, 'allowed_cpus': sorted(os.sched_getaffinity(0)),
              'environment': {k: v for k, v in env.items() if k in json.loads((ROOT / 'environment.json').read_text())
                              or k in ('LD_LIBRARY_PATH', 'CUDA_LAUNCH_BLOCKING', 'OMP_NUM_THREADS',
                                       'FAST_PLAINTEXT_UPLOAD', 'GPU_PLAINTEXT_NTT', 'DIRECT_PLAINTEXT_UPLOAD', 'MOVE_PLAINTEXT_COEFFICIENTS', 'BORROW_PLAINTEXT_UPLOAD')},
              'source_manifest_sha256': sha(ROOT / 'compiled-sources.json'),
              'controller_sha256': sha(Path(__file__)),
              'launcher_sha256': sha(ROOT / 'launch_m2.sh'), **(metadata or {})}
    if command[0] == 'bash':
        record['binary_sha256'] = sha(env['BINARY'])
    elif command[0] == 'taskset':
        record['binary_sha256'] = sha(command[3])
    else:
        record['binary_sha256'] = sha(command[0])
    start = time.monotonic()
    print('START', name, flush=True)
    with (out / 'native.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        record['pid'] = process.pid
        save(out / 'run.json', record)
        save(ROOT / 'status.json', {**record, 'status': 'running'})
        try:
            while process.poll() is None:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0: raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    status_file = Path(f'/proc/{process.pid}/status')
                    if status_file.exists():
                        selected = [line for line in status_file.read_text().splitlines()
                                    if line.startswith(('Name:', 'Threads:', 'Cpus_allowed_list:', 'VmRSS:', 'VmHWM:'))]
                        record['process_sample'] = selected
                    save(ROOT / 'status.json', {**record, 'status': 'running',
                        'elapsed_seconds': time.monotonic() - start, 'updated_utc': stamp()})
            record['timed_out'] = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record['timed_out'] = True
    record.update(returncode=process.returncode, wall_seconds=time.monotonic() - start,
                  ended_utc=stamp(), passed=False)
    budget['used_seconds'] += record['wall_seconds']
    budget['runs'].append({'name': name, 'charged_seconds': record['wall_seconds']})
    save(budget_path, budget)
    try:
        native = json.loads((out / 'native.json').read_text())
        record['native_sha256'] = sha(out / 'native.json')
        record['checks'] = validator(native)
        record['passed'] = process.returncode == 0 and not record['timed_out'] and all(record['checks'].values())
        record['eval_seconds'] = native.get('eval_seconds', native.get('timing', {}).get('eval_seconds'))
    except Exception as error:
        record['validation_error'] = str(error)
    save(out / 'run.json', record)
    save(ROOT / 'status.json', {**record, 'status': 'completed' if record['passed'] else 'failed'})
    print('END', json.dumps(record), flush=True)
    if not record['passed']:
        raise RuntimeError('job failed: ' + name)

