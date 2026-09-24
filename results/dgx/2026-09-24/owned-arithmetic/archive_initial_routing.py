"""Preserve the first successful model run with incomplete optimization coverage."""
import json
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve()
record = json.loads((root / 'm3-full-candidate/run.json').read_text())
assert not record['passed']
assert {k for k, v in record['checks'].items() if not v} == {'static_add_inventory'}
native = json.loads((root / 'm3-full-candidate/native.json').read_text())
assert native['passed'] and native['owned_arithmetic_calls'] == 91716
destination = root / 'initial-routing-coverage'
destination.mkdir(exist_ok=False)
paths = list(root.glob('m3-*'))
paths += [root / name for name in ('completion.json', 'completion.local.json',
    'postprocess-completion.json', 'finalization.json', 'short-comparison.json',
    'controller-rns.log', 'watcher-rns.log', 'postprocess-hook.log', 'finalize.log', 'notification.json')]
moved = []
for path in paths:
    if path.exists():
        path.rename(destination / path.name)
        moved.append(path.name)
shutil.copyfile(root / 'budget.json', destination / 'budget.json')
budget = json.loads((root / 'budget.json').read_text())
for run in budget['runs']:
    if run['name'] in moved:
        run['artifact_directory'] = 'initial-routing-coverage/' + run['name']
(root / 'budget.json').write_text(json.dumps(budget, indent=2) + '\n')
(destination / 'reason.json').write_text(json.dumps({
    'model_numerical_and_token_gates_passed': True,
    'failed_gate': 'static_add_inventory', 'expected_calls': 91956, 'observed_calls': 91716,
    'cause': 'The initial implementation missed 240 accumulator additions in radix-routing stages where BSGS is not beneficial. They still called the protecting add helper.',
    'action': 'Consume those private temporary operands through the same validated helper. Rebuild only Mamba-3, repeat its state/prefix/full checks, then continue unchanged Mamba-2.',
    'raw_measurements': 'Moved without changing bytes; original source archive and binaries remain at campaign root. All native process time remains charged.',
    'moved_paths': moved,
}, indent=2) + '\n')
print(destination)
