"""Qualify committed sources, then close with a fresh immutable baseline."""
import json
import subprocess
import sys
import traceback

from runlib import ROOT, save, sha, stamp

revision, *features = sys.argv[1:]
try:
    assert json.loads((ROOT / 'qualification-completion.json').read_text())['passed']
    subprocess.run([sys.executable, str(ROOT / 'build_final.py'), revision], check=True)
    for name, payload, baseline in [
        ('final-synthetic', 'payload', False),
        ('final-candidate-full', 'lm-full', False),
        ('final-baseline-full', 'lm-full', True),
    ]:
        command = [sys.executable, str(ROOT / 'measure_only.py'),
                   '--revision', revision, '--name', name, '--payload', payload]
        command += ['--baseline'] if baseline else ['--', *features]
        subprocess.run(command, check=True)
    baseline = json.loads((ROOT / 'final-baseline-full/native.json').read_text())
    candidate = json.loads((ROOT / 'final-candidate-full/native.json').read_text())
    result = {
        'passed': True,
        'revision': revision,
        'implementation_commit': json.loads(
            (ROOT / revision / 'compiled-sources.json').read_text())['implementation_commit'],
        'features': features,
        'baseline_seconds': baseline['eval_seconds'],
        'candidate_seconds': candidate['eval_seconds'],
        'reduction_percent': 100 * (1 - candidate['eval_seconds'] / baseline['eval_seconds']),
        'target_met': candidate['eval_seconds'] <= .8 * baseline['eval_seconds'],
        'baseline_sha256': sha(ROOT / 'final-baseline-full/native.json'),
        'candidate_sha256': sha(ROOT / 'final-candidate-full/native.json'),
    }
except Exception:
    result = {'passed': False, 'error': traceback.format_exc()}
result['finished_utc'] = stamp()
save(ROOT / 'final-completion.json', result)
print(json.dumps(result), flush=True)
if not result['passed']:
    raise SystemExit(1)
