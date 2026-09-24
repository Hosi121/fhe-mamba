"""Finalize checked GPU work, including portable evidence and documentation."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
REMOTE = '/home/kataiwa/fhemamba/owned-arithmetic-20260924'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def save(path, data): path.write_text(json.dumps(data, indent=2) + '\n')

try:
    started = time.monotonic()
    while not (ROOT / 'postprocess-completion.json').exists():
        if time.monotonic() - started > 15000:
            raise RuntimeError('postprocessing review interval reached')
        time.sleep(10)
    assert read(ROOT / 'postprocess-completion.json')['passed']
    # The initial Mamba-2 effect was small. Run the four declared reversed
    # controls only after both full pairs, then collect before final curation.
    if not (ROOT / 'm2-confirmation.json').exists():
        subprocess.run(['rsync', '-az', '--protect-args', str(ROOT / 'confirm_m2.py'),
                        'dgx:' + REMOTE + '/'], check=True, timeout=60)
        with (ROOT / 'confirmation-controller.log').open('w') as log:
            subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dgx', 'python3', REMOTE + '/confirm_m2.py'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1500)
        names = ['m2-confirmation.json', 'budget.json'] + [
            f'm2-confirm-{i}-' + ('candidate' if candidate else 'base')
            for i, candidate in enumerate((True, False, False, True), 1)]
        subprocess.run(['rsync', '-az', '--protect-args', *['dgx:' + REMOTE + '/' + name for name in names],
                        str(ROOT) + '/'], check=True, timeout=120)
    assert read(ROOT / 'm2-confirmation.json')['passed']
    with (ROOT / 'compare-confirmation.log').open('w') as log:
        subprocess.run([sys.executable, str(ROOT / 'compare.py')], check=True, cwd=REPO,
                       stdout=log, stderr=subprocess.STDOUT, timeout=60)
    with (ROOT / 'postflight.log').open('w') as log:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dgx', 'python3', '-', REMOTE],
                       input=(ROOT / 'postflight.py').read_text(), text=True,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    subprocess.run(['rsync', '-az', '--protect-args', 'dgx:' + REMOTE + '/postflight.json',
                    str(ROOT) + '/'], check=True, timeout=60)
    assert read(ROOT / 'postflight.json')['passed']
    expected = read(ROOT / 'compiled-final-sources.json')['files']
    assert all(sha(REPO / name) == value for name, value in expected.items()), 'Native tree changed during validation'
    cleanup = []
    for name, stem in [('source', 'compiled-sources'), ('source-final', 'compiled-final-sources')]:
        source = ROOT / name
        if source.exists():
            files = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
            assert files == read(ROOT / (stem + '.json'))['files'], 'Unexpected local snapshot content; preserve it'
            shutil.rmtree(source)
            cleanup.append({'path': name + '/', 'replacement': stem + '.tar.gz', 'files_verified': len(files)})
    for name in ('test_owned_arithmetic', 'encoding_inventory', 'encoding_log_probe', 'routing_coverage_inventory'):
        binary = ROOT / name
        if binary.exists():
            cleanup.append({'path': binary.name, 'sha256': sha(binary), 'reason': 'Local standalone validation finished; source and output preserved'})
            binary.unlink()
    save(ROOT / 'cleanup.json', {'removed': cleanup, 'worktrees_created': 0,
         'retained': 'DGX baseline/candidate builds, measured source archives, raw logs and failed controls'})
    destination = REPO / 'results/dgx/2026-09-24/owned-arithmetic'
    for name in ('curate', 'write_study', 'update_guides'):
        if name == 'curate' and destination.exists():
            for path, value in read(destination / 'provenance.json')['artifacts_sha256'].items():
                assert sha(destination / path) == value
            continue
        with (ROOT / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(ROOT / (name + '.py'))], cwd=REPO,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=REPO, text=True).splitlines()
    docs = [REPO / p for p in changed if p.endswith('.md')]
    docs += [REPO / 'docs/research/2026-09-24-owned-arithmetic.md', destination / 'README.md']
    for path in docs:
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            target = match.group(1).split('#')[0]
            if not target or ':' in target or target.startswith('app:'):
                continue
            assert (path.parent / target).exists(), (str(path), target)
    subprocess.run(['git', 'diff', '--check'], cwd=REPO, check=True, timeout=60)
    for path, value in read(destination / 'provenance.json')['artifacts_sha256'].items():
        assert sha(destination / path) == value
    result = {'passed': True, 'comparison': read(ROOT / 'comparison.json'),
              'implementation_adoption_complete': all(read(ROOT / 'comparison.json')['adoption'].values()),
              'evidence': str(destination.relative_to(REPO)),
              'git_stage': 'Local implementation and evidence; no new commit or push'}
except Exception:
    result = {'passed': False, 'error': traceback.format_exc()}
result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
save(ROOT / 'finalization.json', result)
print(json.dumps(result), flush=True)
