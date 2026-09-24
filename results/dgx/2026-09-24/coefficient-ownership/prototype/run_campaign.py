"""Build isolated sources, check coefficient moves, and compare paired phases."""
import json
import statistics
import subprocess
import time
import traceback

from runlib import ROOT, base_env, run, save, sha, stamp


def validate(native):
    return {
        'native_passed': native['passed'],
        'exact_rns_cases': native['exact_rns_cases'] == 160,
        'moved_coefficient_cases': native['moved_coefficient_cases'] == 160,
        'shared_policy_cases': native['shared_policy_cases'] == 90,
        'unchanged_error_gate': native['tolerance'] == 1e-6 and native['max_abs_error'] < 1e-6,
    }


try:
    start = time.monotonic()
    with (ROOT / 'build.log').open('w') as log:
        subprocess.run(['bash', str(ROOT / 'build.sh')], check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=900)
    save(ROOT / 'build-completion.json', {'passed': True, 'wall_seconds': time.monotonic()-start})
    summary = {}
    for arch in ('mamba2', 'mamba3'):
        env = base_env()
        env['OMP_NUM_THREADS'] = '4'
        command = ['taskset', '-c', '15-19', str(ROOT / 'build/packed_plaintext_probe'), '{output}/native.json']
        if arch == 'mamba2': command.append('--mamba2')
        run('probe-' + arch, command, env, 300, validate)
        native_path = ROOT / ('probe-' + arch) / 'native.json'
        native = json.loads(native_path.read_text())
        cells = {}
        for level in (21, 26, 34):
            pairs = {}
            for mode, name in ((7, 'copy'), (8, 'move')):
                samples = [s for s in native['samples'] if s['level'] == level
                           and s['mode'] == mode and s['iteration'] >= 0]
                assert len(samples) == 8
                pairs[name] = {
                    'samples': len(samples),
                    'median_encode_ms': statistics.median(s['encode_ms'] for s in samples),
                    'median_upload_ms': statistics.median(s['upload_ms'] for s in samples),
                    'median_preparation_ms': statistics.median(s['encode_ms']+s['upload_ms'] for s in samples),
                    'mean_preparation_ms': statistics.mean(s['encode_ms']+s['upload_ms'] for s in samples),
                }
            pairs['median_reduction_percent'] = 100 * (1 - pairs['move']['median_preparation_ms']/pairs['copy']['median_preparation_ms'])
            cells[str(level)] = pairs
        summary[arch] = {'max_abs_error': native['max_abs_error'], 'native_sha256': sha(native_path), 'levels': cells}
    save(ROOT / 'comparison.json', summary)
    completion = {'passed': True, 'scope': 'Probe only; model defaults and dispatch are unchanged.'}
except Exception:
    completion = {'passed': False, 'error': traceback.format_exc()}
completion['finished_utc'] = stamp()
save(ROOT / 'completion.json', completion)
print(json.dumps(completion), flush=True)
