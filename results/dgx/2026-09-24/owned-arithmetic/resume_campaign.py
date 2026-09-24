"""Validate the corrected Mamba-3 routing coverage, then finish Mamba-2."""
import ast
import json
from pathlib import Path
import subprocess
import traceback

ROOT = Path('/home/kataiwa/fhemamba/owned-arithmetic-20260924')
path = ROOT / 'run_campaign.py'
definitions = []
for node in ast.parse(path.read_text(), filename=str(path)).body:
    if isinstance(node, ast.Try): break
    definitions.append(node)
namespace = {}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
original_run = namespace['run']
def final_run(name, command, env, timeout, validator, metadata=None):
    if name.startswith('m3-') and metadata and metadata.get('candidate'):
        command = command[:]
        command[3] = ROOT / 'build-final/packed_fideslib'
        metadata = {**metadata,
            'source_manifest_sha256': namespace['sha'](ROOT / 'compiled-final-sources.json'),
            'source_manifest_path': str(ROOT / 'compiled-final-sources.json')}
    return original_run(name, command, env, timeout, validator, metadata)
namespace['run'] = final_run
save, stamp, m3, m2, controls = (namespace[name] for name in ('save','stamp','m3','m2','controls'))
read = lambda p: json.loads(p.read_text())
try:
    subprocess.run(['python3', str(ROOT / 'freeze_final_target.py')], check=True)
    m3('m3-synthetic-candidate', True, synthetic=True)
    comparisons = {'m3': controls('m3', m3), 'm2': read(ROOT / 'm2-short-comparison.json')}
    save(ROOT / 'short-comparison.json', comparisons)
    full = {}
    for arch, measure in (('m3', m3), ('m2', m2)):
        if not comparisons[arch]['qualifies_full_pair']:
            full[arch] = {'skipped': True, 'reason': 'Less than 0.5% improvement in prefix.'}
            continue
        measure(arch + '-full-base', False, full=True)
        measure(arch + '-full-candidate', True, full=True)
        a, b = (read(ROOT / (arch + '-full-' + mode) / 'run.json') for mode in ('base','candidate'))
        full[arch] = {'base_seconds': a['eval_seconds'], 'candidate_seconds': b['eval_seconds'],
                     'reduction_percent': 100 * (1 - b['eval_seconds'] / a['eval_seconds']),
                     'adopt': b['eval_seconds'] < a['eval_seconds']}
        save(ROOT / 'full-comparison.json', full)
    result = {'passed': True, 'short': comparisons, 'full': full,
              'revision': 'Includes the 240 remaining private routing accumulations; initial coverage failure preserved.'}
except Exception:
    result = {'passed': False, 'error': traceback.format_exc()}
result['finished_utc'] = stamp()
save(ROOT / 'completion.json', result)
print(json.dumps(result), flush=True)
