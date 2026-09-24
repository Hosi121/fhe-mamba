"""Run evidence and token checks once the collection hook completes."""
import datetime
import json
from pathlib import Path
import subprocess
import time
import traceback

root = Path(__file__).resolve().parent
repo = root.parents[1]
started = time.monotonic()
try:
    while not (root / 'completion.local.json').exists():
        if time.monotonic() - started > 15000:
            raise RuntimeError('collection hook review boundary')
        time.sleep(10)
    assert json.loads((root / 'completion.local.json').read_text())['passed']
    for name, command in (
        ('compare', ['python3', str(root / 'compare.py')]),
        ('reports', [str(repo / '.venv/bin/python'), str(root / 'finish_reports.py')]),
    ):
        with (root / (name + '.log')).open('w') as log:
            subprocess.run(command, cwd=repo, check=True, stdout=log,
                           stderr=subprocess.STDOUT, timeout=300)
    result = {'passed': True}
except Exception:
    result = {'passed': False, 'error': traceback.format_exc()}
result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(root / 'postprocess-completion.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
