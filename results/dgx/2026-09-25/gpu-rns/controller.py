"""One exact-RNS candidate: small gates precede any full inference."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

root = Path('/home/kataiwa/fhemamba/gpu-rns-20260925')
baseline = Path('/home/kataiwa/fhemamba/s2c-first-20260925/final-native-build/packed_fideslib')
candidate = root / 'native-build/packed_fideslib'
probe = root / 'native-build/packed_rns_probe'
expected_baseline = '4504ee7ba95d7e5420def9dab07da8760c362b4471644735a9c993fcd0b9fec9'
env = dict(os.environ, OMP_NUM_THREADS='4')
env['PATH'] = '/usr/local/cuda-13.0/bin:' + env['PATH']
env['LD_LIBRARY_PATH'] = '/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64'
status = {'state': 'running', 'started_at_unix': time.time()}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(name, command, threads=4, expected_code=0, timeout=1800):
    directory = root / name
    directory.mkdir(exist_ok=False)
    status['stage'] = name
    write(root / 'status.json', status)
    started = time.time()
    local_env = dict(env, OMP_NUM_THREADS=str(threads))
    with (directory / 'run.log').open('w') as log:
        process = subprocess.run(command, env=local_env, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=timeout, check=False)
    record = {'command': command, 'exit_code': process.returncode, 'started_at_unix': started,
              'wall_seconds': time.time() - started, 'environment': {
                  k: local_env[k] for k in ['OMP_NUM_THREADS', 'LD_LIBRARY_PATH']}}
    if command[0] == 'taskset':
        record['binary_sha256'] = sha(command[3])
    write(directory / 'run.json', record)
    if process.returncode != expected_code:
        raise RuntimeError(f'{name}: exit {process.returncode}, expected {expected_code}')
    path = directory / 'native.json'
    if path.exists():
        data = json.loads(path.read_text())
        assert data['passed'], name
        return data
    return None


def infer(name, binary, payload_name, compact):
    payload = Path('/home/kataiwa/fhemamba/mamba3-20260924') / payload_name
    manifest = json.loads((payload / 'manifest.json').read_text())
    for filename, digest in manifest['files_sha256'].items():
        assert sha(payload / filename) == digest, filename
    flags = ['--planned-refresh', '--batch-refresh', '--inplace-ops', '--gpu-plaintext-ntt',
             '--profile-evaluation', '--naf-rotations', '--reuse-dead-inputs',
             '--direct-plaintext-upload', '--compact-weights', '--move-plaintext-coefficients',
             '--borrow-plaintext-upload', '--bsgs-routing-stages', '--cache-plaintexts',
             '--frontier-refresh', '--s2c-first', '--client-head', str(payload / 'client_head.f32')]
    if compact:
        flags += ['--gpu-plaintext-rns']
    result = run(name, ['taskset', '-c', '15-19', str(binary), str(payload / 'program.txt'),
                        str(root / name / 'native.json'), '.001', '.001'] + flags)
    expected = [315, 279, 1614, 315] if payload_name == 'lm-full' else [6864, 6864]
    assert result['generated_token_ids'] == expected
    assert result['bootstrap_passes'] == 2 and result['s2c_first']
    assert result['refresh_ceiling'] == 35 and result['refreshed_level'] == 18
    assert result['evaluation_decryptions'] == result['non_finite'] == 0
    assert result['max_abs_error_vs_exact'] <= .001
    assert result['max_abs_error_vs_polynomial'] <= .001
    if compact:
        assert result['compact_rns_encodes'] > 0
        assert result['compact_rns_encodes'] == result['compact_rns_uploads']
    write(root / name / 'parity.json', {'tokens': expected, 'passed': True,
        'program_sha256': sha(payload / 'program.txt'), 'manifest_sha256': sha(payload / 'manifest.json')})
    print(name, result['eval_seconds'], result['bootstrap_seconds'], flush=True)
    return result


def same_schedule(a, b):
    for key in ['bootstraps', 'logical_refreshes', 'ct_ct_mul', 'ct_pt_mul', 'rotations',
                'evaluated_nodes', 'refresh_batches', 'frontier_deferrals']:
        assert a[key] == b[key], (key, a[key], b[key])


write(root / 'status.json', status)
try:
    assert sha(baseline) == expected_baseline
    manifest = json.loads((root / 'native-source-manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(root / 'native-source' / name) == digest, name
    build = {'baseline_binary_sha256': sha(baseline), 'candidate_binary_sha256': sha(candidate),
             'probe_binary_sha256': sha(probe), 'backend_library_sha256': sha(root / 'install/lib/fideslib.a'),
             'native_source_manifest_sha256': sha(root / 'native-source-manifest.json'),
             'compiler': subprocess.check_output(['g++', '--version'], text=True),
             'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv'], text=True)}
    write(root / 'build.json', build)
    run('legacy-configure', ['cmake', '-S', str(root / 'native-source/native/fideslib_stage0'),
        '-B', str(root / 'legacy-build'), '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=/usr/bin/g++',
        '-DCMAKE_PREFIX_PATH=/home/kataiwa/fhemamba/spark/install-2a70798e869944af;/home/kataiwa/fhe-deps/openfhe-fides',
        '-Dfideslib_DIR=/home/kataiwa/fhemamba/spark/install-2a70798e869944af/share/fideslib/cmake'])
    run('legacy-build-log', ['cmake', '--build', str(root / 'legacy-build'), '--target', 'packed_fideslib', '-j', '4'])
    run('legacy-rejection', [str(root / 'legacy-build/packed_fideslib'), '/dev/null',
        str(root / 'must-not-exist.json'), '.001', '.001', '--gpu-plaintext-rns'], expected_code=2)
    assert not (root / 'must-not-exist.json').exists()
    for threads, mamba2 in [(4, False), (1, False), (4, True)]:
        name = f'probe-t{threads}-' + ('mamba2' if mamba2 else 'mamba3')
        command = ['taskset', '-c', '15-19', str(probe), str(root / name / 'native.json')]
        if mamba2:
            command += ['--mamba2']
        data = run(name, command, threads=threads)
        assert data['exact_rns_cases'] == 240 and data['encrypted_cases'] == 9
    prefixes = {'baseline': [], 'candidate': []}
    for i, mode in enumerate(['baseline', 'candidate', 'candidate', 'baseline']):
        data = infer(f'prefix-{i}-{mode}', baseline if mode == 'baseline' else candidate,
                     'lm-layer1', mode == 'candidate')
        prefixes[mode].append(data)
    same_schedule(prefixes['baseline'][0], prefixes['candidate'][0])
    means = {mode: statistics.mean(d['eval_seconds'] for d in samples) for mode, samples in prefixes.items()}
    assert means['candidate'] < means['baseline'], means
    write(root / 'small-selection.json', {'passed': True, 'prefix_means_seconds': means,
          'exact_rns_cases': 720, 'encrypted_cases': 27, 'same_schedule': True})
    # No builds or other experiment workers overlap these full runs.
    a = infer('full-baseline', baseline, 'lm-full', False)
    b = infer('full-candidate', candidate, 'lm-full', True)
    same_schedule(a, b)
    status['baseline_eval_seconds'] = a['eval_seconds']
    status['candidate_eval_seconds'] = b['eval_seconds']
    status['reduction_fraction'] = 1 - b['eval_seconds'] / a['eval_seconds']
    status['same_schedule'] = True
    status['state'] = 'completed'
except Exception as error:
    status['state'] = 'failed'
    status['error'] = repr(error)
finally:
    status['finished_at_unix'] = time.time()
    write(root / 'status.json', status)
