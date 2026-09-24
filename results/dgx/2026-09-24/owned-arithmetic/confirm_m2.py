"""Four predeclared reversed controls for the small Mamba-2 prefix effect."""
import ast
import json
from pathlib import Path
from statistics import mean
import traceback

ROOT = Path('/home/kataiwa/fhemamba/owned-arithmetic-20260924')
# Reuse the exact measured launcher/validator definitions without entering the
# original campaign's top-level try block or rerunning its completed jobs.
path = ROOT / 'run_campaign.py'
tree = ast.parse(path.read_text(), filename=str(path))
definitions = []
for node in tree.body:
    if isinstance(node, ast.Try):
        break
    definitions.append(node)
namespace = {}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
save, stamp, measure = (namespace[name] for name in ('save', 'stamp', 'm2'))
read = lambda p: json.loads(p.read_text())
assert read(ROOT / 'completion.json')['passed']
try:
    samples = []
    for i, candidate in enumerate((True, False, False, True), 1):
        name = f'm2-confirm-{i}-' + ('candidate' if candidate else 'base')
        measure(name, candidate)
        samples.append(read(ROOT / name / 'run.json')['eval_seconds'])
    first = read(ROOT / 'm2-short-comparison.json')['samples_in_abba_order']
    base, candidate = mean(samples[1:3]), mean((samples[0], samples[3]))
    all_base = [first[0], first[3], samples[1], samples[2]]
    all_candidate = [first[1], first[2], samples[0], samples[3]]
    combined_base, combined_candidate = mean(all_base), mean(all_candidate)
    result = {'passed': True, 'samples_in_baab_order': samples,
              'base_mean_seconds': base, 'candidate_mean_seconds': candidate,
              'reduction_percent': 100 * (1 - candidate / base),
              'combined_base_samples': all_base, 'combined_candidate_samples': all_candidate,
              'combined_base_mean_seconds': combined_base,
              'combined_candidate_mean_seconds': combined_candidate,
              'combined_reduction_percent': 100 * (1 - combined_candidate / combined_base),
              'supports_adoption': candidate < base and combined_candidate < combined_base
                   and read(ROOT / 'full-comparison.json')['m2']['adopt'],
              'scope': 'Reversed short controls after the full pairs, added before full results because the initial 0.56% Mamba-2 prefix difference was small. Every sample preserved; no significance claim.'}
except Exception:
    result = {'passed': False, 'error': traceback.format_exc()}
result['finished_utc'] = stamp()
save(ROOT / 'm2-confirmation.json', result)
print(json.dumps(result), flush=True)
