"""Freeze final sources, verify the small gate, and compare two full DGX runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tarfile
import time

root = Path('/home/kataiwa/fhemamba/s2c-first-20260925')
old = Path('/home/kataiwa/fhemamba/layout-batch-20260925/final-r1/build/packed_fideslib')
old_hash = '82ff964d3af3f6a73498fec319e5ea643c7c8a5fee4dd4c1560194c6bb81eaff'
env = dict(os.environ, OMP_NUM_THREADS='4')
env['PATH'] = '/usr/local/cuda-13.0/bin:' + env['PATH']
env['LD_LIBRARY_PATH'] = '/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64'
status = {'started_at_unix': time.time(), 'state': 'running'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def run(name, command, timeout=1800):
    status['stage'] = name
    write(root / 'final-status.json', status)
    directory = root / name
    if (directory / 'run.json').exists():
        previous = json.loads((directory / 'run.json').read_text())
        assert previous['command'] == command and previous['exit_code'] == 0
        if (directory / 'native.json').exists():
            result = json.loads((directory / 'native.json').read_text())
            assert result['passed']
            return result
        return None
    directory.mkdir()
    start = time.time()
    with (directory / 'run.log').open('w') as log:
        p = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, check=False)
    record = {'command': command, 'exit_code': p.returncode, 'started_at_unix': start,
              'wall_seconds': time.time() - start,
              'environment': {k: env[k] for k in ['OMP_NUM_THREADS', 'LD_LIBRARY_PATH']}}
    if command[:1] == ['taskset']:
        record['binary_sha256'] = sha(command[3])
    write(directory / 'run.json', record)
    if p.returncode:
        raise RuntimeError(f'{name}: exit {p.returncode}')
    if (directory / 'native.json').exists():
        result = json.loads((directory / 'native.json').read_text())
        if not result['passed']:
            raise RuntimeError(f'{name}: accuracy gate failed')
        return result


def infer(name, binary, payload_name, candidate):
    payload = Path('/home/kataiwa/fhemamba/mamba3-20260924') / payload_name
    manifest = json.loads((payload / 'manifest.json').read_text())
    for filename, digest in manifest['files_sha256'].items():
        assert sha(payload / filename) == digest, filename
    flags = ['--planned-refresh', '--batch-refresh', '--inplace-ops', '--gpu-plaintext-ntt',
             '--profile-evaluation', '--naf-rotations', '--reuse-dead-inputs',
             '--direct-plaintext-upload', '--compact-weights', '--move-plaintext-coefficients',
             '--borrow-plaintext-upload', '--bsgs-routing-stages', '--cache-plaintexts',
             '--frontier-refresh', '--client-head', str(payload / 'client_head.f32')]
    if candidate:
        flags += ['--s2c-first']
    result = run(name, ['taskset', '-c', '15-19', str(binary), str(payload / 'program.txt'),
                        str(root / name / 'native.json'), '.001', '.001'] + flags)
    log = (root / name / 'run.log').read_text()
    tokens = [int(line.split()[0].split('=')[1]) for line in log.splitlines()
              if line.startswith('client_token=')]
    expected = [315, 279, 1614, 315] if payload_name == 'lm-full' else [6864, 6864]
    assert tokens == expected, (name, tokens)
    write(root / name / 'parity.json', {'tokens': tokens, 'expected_tokens': expected,
          'program_sha256': sha(payload / 'program.txt'),
          'manifest_sha256': sha(payload / 'manifest.json'), 'passed': True})
    print(name, result['eval_seconds'], flush=True)
    return result


write(root / 'final-status.json', status)
try:
    prefix_status = json.loads((root / 'prefix-status.json').read_text())
    assert prefix_status['state'] == 'completed'
    times = {}
    for mode, indices in [('baseline', [0, 3]), ('s2c', [1, 2])]:
        data = [json.loads((root / f'prefix-{i}-{mode}/native.json').read_text()) for i in indices]
        assert all(d['passed'] for d in data)
        times[mode] = statistics.mean(d['eval_seconds'] for d in data)
    assert times['s2c'] < times['baseline'], times
    assert sha(old) == old_hash
    with tarfile.open(root / 'final-sources.tar.gz') as archive:
        archive.extractall(root, filter='data')
    manifest = json.loads((root / 'final-source-manifest.json').read_text())
    for directory, files in manifest.items():
        for name, digest in files.items():
            assert sha(root / directory / name) == digest, (directory, name)
    shutil.copytree(root / 'final-backend', root / 'source', dirs_exist_ok=True)
    run('final-backend-build', ['cmake', '--build', str(root / 'build'), '--target', 'install', '-j', '6'])
    run('final-native-configure', ['cmake', '-S', str(root / 'final-source/native/fideslib_stage0'),
         '-B', str(root / 'final-native-build'), '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_CXX_COMPILER=/usr/bin/g++',
         f'-DCMAKE_PREFIX_PATH={root}/install;/home/kataiwa/fhe-deps/openfhe-fides',
         f'-Dfideslib_DIR={root}/install/share/fideslib/cmake'])
    run('final-native-build-log', ['cmake', '--build', str(root / 'final-native-build'),
                                 '--target', 'packed_fideslib', '-j', '4'])
    binary = root / 'final-native-build/packed_fideslib'
    write(root / 'final-build.json', {'baseline_binary_sha256': sha(old),
          'candidate_binary_sha256': sha(binary), 'backend_library_sha256': sha(root / 'install/lib/fideslib.a'),
          'source_manifest_sha256': sha(root / 'final-source-manifest.json'),
          'compiler': subprocess.check_output(['g++', '--version'], text=True),
          'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv'], text=True)})
    infer('final-baseline-prefix', old, 'lm-layer1', False)
    infer('final-candidate-prefix', binary, 'lm-layer1', True)
    baseline = infer('final-baseline-full', old, 'lm-full', False)
    candidate = infer('final-candidate-full', binary, 'lm-full', True)
    status['baseline_eval_seconds'] = baseline['eval_seconds']
    status['candidate_eval_seconds'] = candidate['eval_seconds']
    status['reduction_fraction'] = 1 - candidate['eval_seconds'] / baseline['eval_seconds']
    status['state'] = 'completed'
except Exception as e:
    status['state'] = 'failed'
    status['error'] = str(e)
finally:
    status['finished_at_unix'] = time.time()
    write(root / 'final-status.json', status)
