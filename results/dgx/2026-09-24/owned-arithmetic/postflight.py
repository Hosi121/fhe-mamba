"""Recheck inputs and frozen executables after GPU timing has finished."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

assert read(ROOT / 'completion.json')['passed']
target = read(ROOT / 'target-provenance.json')
baseline = read(ROOT / 'baseline/target-provenance.json')
environment = read(ROOT / 'environment.json')
payload = Path(environment['INPUT_CHAIN'])
files = {str(p.relative_to(payload)): sha(p) for p in sorted(payload.rglob('*'))
         if p.is_file() and p.suffix in ('.json', '.bin')}
assert files == baseline['payload_files_sha256']
aggregate = hashlib.sha256()
for name, value in files.items():
    aggregate.update(f'{name}\0{value}\n'.encode())
assert aggregate.hexdigest() == baseline['payload_sha256'] == environment['INPUT_CHAIN_SHA256']
for name, value in target['dependencies_verified'].items():
    assert sha(name) == value, name
for name, value in target['candidate_binaries_sha256'].items():
    assert sha(ROOT / 'build' / name) == value, name
sources = read(ROOT / 'compiled-sources.json')['files']
assert all(sha(ROOT / 'source' / name) == value for name, value in sources.items())
final_target = read(ROOT / 'target-final-provenance.json')
assert final_target['previous_target_sha256'] == sha(ROOT / 'target-provenance.json')
assert sha(ROOT / 'build-final/packed_fideslib') == final_target['binary_sha256']
final_sources = read(ROOT / 'compiled-final-sources.json')['files']
assert all(sha(ROOT / 'source-final' / name) == value for name, value in final_sources.items())
backend = '/home/kataiwa/fhemamba/spark/source-2a70798e869944af'
diff = subprocess.check_output(['git', '-C', backend, 'diff'], text=True).strip()
assert hashlib.sha256(diff.encode()).hexdigest() == target['backend_diff_sha256']
governor = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read_text().strip()
assert governor == baseline['governor']
source = Path('/home/kataiwa/fhe-deps/FIDESlib/deps/openfhe-src/src/pke/lib/encoding/ckkspackedencoding.cpp')
result = {'passed': True, 'checked_after_campaign': True,
          'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'mamba2_payload_sha256': aggregate.hexdigest(), 'mamba2_payload_file_count': len(files),
          'candidate_sources': len(sources), 'candidate_binaries': target['candidate_binaries_sha256'],
          'final_candidate_sources': len(final_sources), 'final_mamba3_binary': final_target['binary_sha256'],
          'dependencies_verified': target['dependencies_verified'], 'governor': governor,
          'openfhe_decode_source': {'path': str(source), 'sha256': sha(source),
              'observation': 'REAL CKKS Decode injects Gaussian samples at lines 480 and 482; exact comparison uses ciphertext RNS coefficients instead of repeated decoded bits.'}}
(ROOT / 'postflight.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
