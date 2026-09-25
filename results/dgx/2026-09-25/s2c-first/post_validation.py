"""After GPU timing, check all native targets and the unpatched API fallback."""
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/home/kataiwa/fhemamba/s2c-first-20260925')
env = dict(os.environ, OMP_NUM_THREADS='4')
env['PATH'] = '/usr/local/cuda-13.0/bin:' + env['PATH']
env['LD_LIBRARY_PATH'] = '/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64'
record = {'state': 'waiting'}


def save():
    p = root / 'post-validation.json'
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(p)


def run(name, command, expected=0):
    record['stage'] = name
    save()
    out = root / name
    out.mkdir()
    start = time.time()
    with (out / 'run.log').open('w') as log:
        p = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=False, timeout=900)
    result = {'command': command, 'exit_code': p.returncode,
              'wall_seconds': time.time() - start, 'expected_exit_code': expected}
    (out / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    assert p.returncode == expected, (name, p.returncode)


save()
try:
    while True:
        status = json.loads((root / 'final-status.json').read_text())
        if status['state'] != 'running':
            break
        time.sleep(15)
    assert status['state'] == 'completed', status
    record['state'] = 'running'
    run('all-native-build', ['cmake', '--build', str(root / 'final-native-build'), '-j', '4'])
    legacy = '/home/kataiwa/fhemamba/spark/install-2a70798e869944af'
    run('legacy-configure', ['cmake', '-S', str(root / 'final-source/native/fideslib_stage0'),
        '-B', str(root / 'legacy-build'), '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_COMPILER=/usr/bin/g++',
        f'-DCMAKE_PREFIX_PATH={legacy};/home/kataiwa/fhe-deps/openfhe-fides',
        f'-Dfideslib_DIR={legacy}/share/fideslib/cmake'])
    run('legacy-build-log', ['cmake', '--build', str(root / 'legacy-build'),
                            '--target', 'packed_fideslib', '-j', '4'])
    run('legacy-rejection', [str(root / 'legacy-build/packed_fideslib'), '/dev/null',
        str(root / 'legacy-rejection/unused.json'), '.001', '.001',
        '--s2c-first', '--planned-refresh', '--batch-refresh'], expected=2)
    assert 'S2C-first requires the optional FIDESlib bootstrap patch' in (
        root / 'legacy-rejection/run.log').read_text()
    assert not (root / 'legacy-rejection/unused.json').exists()
    record['state'] = 'completed'
except Exception as e:
    record['state'] = 'failed'
    record['error'] = str(e)
finally:
    record['finished_at_unix'] = time.time()
    save()
