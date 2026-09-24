"""Freeze actual target binaries, dependencies, native sources and Mamba-2 input."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/home/kataiwa/fhemamba/coefficient-ownership-models-20260924')
backend = Path('/home/kataiwa/fhemamba/spark/source-2a70798e869944af')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()
def cmd(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
env = json.loads((root / 'environment.json').read_text())
payload = Path(env['INPUT_CHAIN'])
payload_files = {str(p.relative_to(payload)): sha(p) for p in sorted(payload.rglob('*'))
                 if p.is_file() and p.suffix in ('.json', '.bin')}
h = hashlib.sha256()
for name, value in payload_files.items(): h.update(f'{name}\0{value}\n'.encode())
assert h.hexdigest() == env['INPUT_CHAIN_SHA256']
expected = json.loads((root / 'compiled-sources.json').read_text())['files']
native = {p: sha(root / 'source' / p) for p in expected if p.startswith('native/')}
assert all(expected[p] == value for p, value in native.items())
files = [root / 'build' / p for p in ('packed_fideslib','packed_plaintext_probe','stage1_mamba2_decode_fideslib')]
files += [Path('/home/kataiwa/fhemamba/spark/install-2a70798e869944af/lib/fideslib.a')]
files += list(Path('/home/kataiwa/fhe-deps/openfhe-fides/lib').glob('*.a'))
files += list(Path('/home/kataiwa/fhe-deps/openfhe-fides/lib').glob('*.so.1.4.2'))
files += [backend / p for p in ('api/CryptoContext.cpp','api/Plaintext.cpp','src/CKKS/Plaintext.cu',
                               'src/CKKS/RNSPoly.cpp','src/CKKS/LimbPartition.cu','src/CKKS/Limb.cu',
                               'src/CKKS/openfhe-interface/RawCiphertext.cu')]
result = {'hardware':cmd('nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'),
          'kernel':cmd('uname','-a'), 'compiler':cmd('g++','--version'),
          'cuda':cmd('/usr/local/cuda-13.0/bin/nvcc','--version'),
          'backend_commit':cmd('git','-C',str(backend),'rev-parse','HEAD'),
          'backend_status':cmd('git','-C',str(backend),'status','--short'),
          'backend_diff_sha256':hashlib.sha256(cmd('git','-C',str(backend),'diff').encode()).hexdigest(),
          'source_and_binary_sha256':{str(p):sha(p) for p in files},
          'compile_commands':json.loads((root/'build/compile_commands.json').read_text()),
          'native_sources_sha256':native, 'payload_sha256':h.hexdigest(),
          'payload_files_sha256':payload_files,
          'governor':Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read_text().strip()}
previous = json.loads((root.parent/'packed-resources-20260924/target-provenance.json').read_text())
for key in ('hardware','compiler','cuda','backend_commit','backend_status','backend_diff_sha256','payload_sha256','governor'):
    assert result[key] == previous[key], key
for name, value in result['source_and_binary_sha256'].items():
    if name in previous['source_and_binary_sha256']:
        assert value == previous['source_and_binary_sha256'][name], name
print(json.dumps(result,indent=2))
