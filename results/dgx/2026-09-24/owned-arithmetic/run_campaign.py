"""Separate source/binary ownership controls, encrypted probes and full pairs."""
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import traceback

from runlib import ROOT, sha, save, stamp, base_env, run

PREVIOUS = ROOT.parent / 'borrowed-plaintext-20260924'
CACHED = ROOT.parent / 'packed-cache-integration-20260924'
M3_KEYS = ('nodes', 'evaluated_nodes', 'bootstraps', 'ct_ct_mul', 'ct_pt_mul',
    'rotations', 'refresh_rotations', 'logical_refreshes', 'refresh_batches',
    'ring_dimension', 'slots', 'depth', 'scale_bits', 'bootstrap_passes',
    'refresh_policy', 'batch_refresh', 'public_weight_count', 'public_weight_bytes',
    'host_encodes', 'plaintext_cache_hits', 'plaintext_cache_misses',
    'optimized_routing_stages', 'routing_stage_rotations_saved')
M2_KEYS = ('parameters', 'ckks_levels', 'operation_counts', 'operation_counts_by_token',
           'phase_operation_counts')


def source_metadata(candidate):
    path = ROOT if candidate else PREVIOUS
    return {'candidate': candidate, 'source_manifest_sha256': sha(path / 'compiled-sources.json'),
            'source_manifest_path': str(path / 'compiled-sources.json')}


def m3(name, candidate, full=False, synthetic=False):
    template = (PREVIOUS / 'cache-all' if synthetic else
                CACHED / ('full-cache' if full else 'prefix-2-cache'))
    original = json.loads((template / 'run.json').read_text())
    baseline = json.loads((template / 'native.json').read_text())
    assert sha(template / 'native.json') == original['native_sha256']
    command = original['command'][:]
    command[3] = str((ROOT if candidate else PREVIOUS) / 'build/packed_fideslib')
    command[5] = '{output}/native.json'
    payload = Path(original['payload'])
    assert sha(payload / 'program.txt') == original['program_sha256']
    manifest = json.loads((payload / 'manifest.json').read_text())
    assert all(sha(payload / p) == value for p, value in manifest['files_sha256'].items())
    env = base_env(); env['OMP_NUM_THREADS'] = '4'
    def check(d):
        result = {'native': d['passed'], 'finite': d['non_finite'] == 0,
            'poly': 0 <= d['max_abs_error_vs_polynomial'] <= .001,
            'exact': 0 <= d['max_abs_error_vs_exact'] <= .001,
            'gates': d['polynomial_tolerance'] == d['exact_tolerance'] == .001,
            'same_operations': all(d[k] == baseline[k] for k in M3_KEYS),
            'no_intermediate_decryptions': d['evaluation_decryptions'] == 0,
            'generation': not full or d['generated_token_ids'] == [315, 279, 1614, 315]}
        if candidate:
            calls = d['owned_arithmetic_calls']
            result['ownership_dispatch'] = (calls > 0 and d['owned_arithmetic_reused_inputs'] > 0
                and d['owned_arithmetic_reused_inputs'] + d['owned_arithmetic_cloned_inputs'] == 2 * calls)
            if full: result['static_add_inventory'] = calls == 91956
        return result
    run(name, command, env, 1800 if full else 300, check,
        {**source_metadata(candidate), 'payload': str(payload),
         'program_sha256': original['program_sha256'], 'manifest_sha256': sha(payload / 'manifest.json')})


def m2(name, candidate, full=False):
    layers, tokens = (24, 5) if full else (2, 2)
    template = PREVIOUS / ('m2-full-borrow' if full else 'm2-2-borrow')
    original = json.loads((template / 'run.json').read_text())
    baseline = json.loads((template / 'native.json').read_text())
    assert sha(template / 'native.json') == original['native_sha256']
    env = base_env(); env.update(json.loads((ROOT / 'environment.json').read_text()))
    env.update(BINARY=str((ROOT if candidate else PREVIOUS) / 'build/stage1_mamba2_decode_fideslib'),
        CUDA_LAUNCH_BLOCKING='1', FAST_PLAINTEXT_UPLOAD='1', GPU_PLAINTEXT_NTT='1',
        DIRECT_PLAINTEXT_UPLOAD='1', MOVE_PLAINTEXT_COEFFICIENTS='0', BORROW_PLAINTEXT_UPLOAD='1',
        AUTOREGRESSIVE_CLIENT_LOOP=str(int(full)), LAYERS=str(layers), TOKENS=str(tokens))
    env['BINARY_SHA256'] = sha(env['BINARY'])
    def check(d):
        p, m = d['parameters'], d['measurements']
        result = {'native': d['passed'], 'layers': p['n_layers_loaded'] == layers,
            'tokens': p['tokens'] == tokens, 'gate': p['tolerance'] == .05,
            'error': 0 <= m['max_abs_error'] <= .05,
            'all_outputs': len(m['per_token_decrypt_ok']) == len(m['per_token_max_abs_error']) == tokens,
            'decrypt': all(m['per_token_decrypt_ok']),
            'all_errors': all(math.isfinite(x) and 0 <= x <= .05 for x in m['per_token_max_abs_error']),
            'same_operations_and_levels': all(d[k] == baseline[k] for k in M2_KEYS),
            'no_intermediate_decryptions': d['measurement_scope']['zero_intermediate_decrypts'],
            'generation': not full or (m['autoregressive_selected_ids'] == [273, 253, 4687, 273]
                                       and m['autoregressive_tokens_match'])}
        if candidate:
            s = m['owned_arithmetic']
            result['ownership_dispatch'] = s['calls'] > 0 and s['cloned_inputs'] == 0 and s['reused_inputs'] == 2 * s['calls']
        return result
    run(name, ['bash', ROOT / 'launch_m2.sh', '{output}/native.json', layers, tokens], env,
        2700 if full else 300, check, source_metadata(candidate))


def controls(architecture, measure):
    samples = []
    for i, candidate in enumerate((False, True, True, False), 1):
        name = f'{architecture}-{i}-' + ('candidate' if candidate else 'base')
        measure(name, candidate)
        samples.append(json.loads((ROOT / name / 'run.json').read_text())['eval_seconds'])
    base, candidate = mean((samples[0], samples[3])), mean(samples[1:3])
    result = {'samples_in_abba_order': samples, 'base_mean_seconds': base,
              'candidate_mean_seconds': candidate, 'reduction_percent': 100 * (1 - candidate / base),
              'qualifies_full_pair': candidate < .995 * base}
    save(ROOT / (architecture + '-short-comparison.json'), result)
    return result


try:
    subprocess.run(['python3', str(ROOT / 'freeze_target.py')], check=True)
    for arch in ('mamba3', 'mamba2'):
        env = base_env(); env['OMP_NUM_THREADS'] = '4'
        command = ['taskset', '-c', '15-19', ROOT / 'build/owned_arithmetic_probe', '{output}/native.json']
        if arch == 'mamba2': command.append('--mamba2')
        run('probe-rns-' + arch, command, env, 300,
            lambda d: {'native': d['passed'], 'cases': d['cases'] == 96,
                       'exact_rns': d['exact_rns_results'], 'live_outs': d['live_inputs_unchanged']})
    m3('m3-synthetic-candidate', True, synthetic=True)
    comparisons = {'m3': controls('m3', m3), 'm2': controls('m2', m2)}
    save(ROOT / 'short-comparison.json', comparisons)
    full = {}
    for arch, measure in (('m3', m3), ('m2', m2)):
        if not comparisons[arch]['qualifies_full_pair']:
            full[arch] = {'skipped': True, 'reason': 'Less than 0.5% improvement in ABBA prefix.'}
            continue
        measure(arch + '-full-base', False, full=True)
        measure(arch + '-full-candidate', True, full=True)
        a, b = (json.loads((ROOT / (arch + '-full-' + mode) / 'run.json').read_text())
                for mode in ('base', 'candidate'))
        full[arch] = {'base_seconds': a['eval_seconds'], 'candidate_seconds': b['eval_seconds'],
                      'reduction_percent': 100 * (1 - b['eval_seconds'] / a['eval_seconds']),
                      'adopt': b['eval_seconds'] < a['eval_seconds']}
        save(ROOT / 'full-comparison.json', full)
    result = {'passed': True, 'short': comparisons, 'full': full}
except Exception:
    result = {'passed': False, 'error': traceback.format_exc()}
result['finished_utc'] = stamp()
save(ROOT / 'completion.json', result)
print(json.dumps(result), flush=True)
