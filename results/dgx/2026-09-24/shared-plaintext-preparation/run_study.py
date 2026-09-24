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

ROOT = Path('/home/kataiwa/fhemamba/mamba2-plaintext-20260924')
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
        'review_seconds': 21600, 'used_seconds': 0, 'runs': [],
        'scope': 'Internal review interval, not a newly inferred user spending limit.'}
    if budget['used_seconds'] + timeout > budget['review_seconds']:
        raise RuntimeError('review boundary reached')
    command = [str(x).replace('{output}', str(out)) for x in command]
    record = {'name': name, 'command': command, 'started_utc': stamp(),
              'timeout_seconds': timeout, 'allowed_cpus': sorted(os.sched_getaffinity(0)),
              'environment': {k: v for k, v in env.items() if k in json.loads((ROOT / 'environment.json').read_text())
                              or k in ('LD_LIBRARY_PATH', 'CUDA_LAUNCH_BLOCKING', 'OMP_NUM_THREADS',
                                       'FAST_PLAINTEXT_UPLOAD', 'GPU_PLAINTEXT_NTT')},
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

def m2(name, mode, layers, tokens):
    env = base_env()
    env.update(json.loads((ROOT / 'environment.json').read_text()))
    env.update(BINARY=str(ROOT / 'build/stage1_mamba2_decode_fideslib'),
               BINARY_SHA256=sha(ROOT / 'build/stage1_mamba2_decode_fideslib'),
               CUDA_LAUNCH_BLOCKING='1', FAST_PLAINTEXT_UPLOAD=str(int(mode != 'a')),
               GPU_PLAINTEXT_NTT=str(int(mode == 'c')), JOINT_SUBRING_ENCODING='1',
               AUTOREGRESSIVE_CLIENT_LOOP=str(int(layers == 24)), LAYERS=str(layers), TOKENS=str(tokens))
    def validate(d):
        p, m = d['parameters'], d['measurements']
        encoding = m['plaintext_encoding']
        return {'native_passed': d['passed'], 'layers': p['n_layers_loaded'] == layers,
                'tokens': p['tokens'] == tokens, 'error': m['max_abs_error'] <= .05,
                'decrypt': all(m['per_token_decrypt_ok']),
                'no_intermediate_decrypts': d['measurement_scope']['zero_intermediate_decrypts'],
                'fast_upload': encoding['fast_upload'] == (mode != 'a'),
                'gpu_ntt': encoding['gpu_ntt'] == (mode == 'c'),
                'subring': p['joint_gate_schedule']['subring_encoding'],
                'tokens_match': layers != 24 or (m['autoregressive_selected_ids'] == [273,253,4687,273]
                                                and m['autoregressive_tokens_match'])}
    run(name, ['bash', ROOT / 'launch_m2.sh', '{output}/native.json', layers, tokens], env,
        4200 if layers == 24 else 240, validate)

def m3(name, payload, old=False, cache=False):
    binary = M3 / 'packed_fideslib-gpu-ntt-final' if old else ROOT / 'build/packed_fideslib'
    payload_dir = M3 / payload
    manifest = json.loads((payload_dir / 'manifest.json').read_text())
    for fname, expected in manifest['files_sha256'].items():
        if sha(payload_dir / fname) != expected:
            raise RuntimeError('payload differs: ' + fname)
    env = base_env()
    env['OMP_NUM_THREADS'] = '4'
    command = ['taskset', '-c', '15-19', binary, payload_dir / 'program.txt', '{output}/native.json',
               '.001', '.001', '--planned-refresh', '--batch-refresh', '--inplace-ops',
               '--gpu-plaintext-ntt', '--profile-evaluation']
    if (payload_dir / 'client_head.f32').exists():
        command += ['--client-head', payload_dir / 'client_head.f32']
    if cache:
        command += ['--cache-plaintexts']
    def validate(d):
        return {'native_passed': d['passed'], 'poly': d['max_abs_error_vs_polynomial'] < .001,
                'exact': d['max_abs_error_vs_exact'] < .001,
                'generation': payload != 'lm-full' or d['generated_token_ids'] == [315,279,1614,315]}
    run(name, command, env, 2400 if payload == 'lm-full' else 240, validate,
        {'manifest_sha256': sha(payload_dir / 'manifest.json'),
         'program_sha256': sha(payload_dir / 'program.txt'), 'payload': str(payload_dir)})

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['probe', 'smoke', 'pressure', 'full', 'mamba3'])
parser.add_argument('--candidate', choices=['b', 'c'], default='c')
parser.add_argument('--start-index', type=int, default=1)
args = parser.parse_args()
if args.stage == 'probe':
    for mode in ('mamba2', 'mamba3'):
        cmd = [ROOT / 'build/packed_plaintext_probe', '{output}/native.json']
        if mode == 'mamba2': cmd += ['--mamba2']
        run('probe-' + mode, cmd, base_env(), 300, lambda d: {'native_passed': d['passed'],
            'rns': d['exact_rns_cases'] == 160, 'shared_policy': d['shared_policy_cases'] == 54})
elif args.stage == 'smoke':
    for index, mode in enumerate('abccba', 1):
        if index < args.start_index: continue
        m2(f'smoke-{index}-{mode}', mode, 1, 2)
elif args.stage == 'pressure':
    for index, mode in enumerate('bccb', 1):
        m2(f'pressure-{index}-{mode}', mode, 2, 2)
elif args.stage == 'full':
    for mode in ('a', args.candidate):
        m2('full-' + mode, mode, 24, 5)
else:
    for index, old in enumerate((True, False, False, True), 1):
        m3(f'mamba3-prefix-{index}', 'lm-layer1-prefix', old=old)
    m3('mamba3-synthetic', 'payload', cache=True)
    m3('mamba3-full', 'lm-full')
save(ROOT / (args.stage + '-completion.json'), {'stage': args.stage, 'passed': True, 'ended_utc': stamp()})
