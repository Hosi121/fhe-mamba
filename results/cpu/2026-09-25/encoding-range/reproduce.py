"""Fresh CPU-only reproduction: python3 reproduce.py UNUSED_OUTPUT_DIRECTORY."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

EVIDENCE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]).resolve()
ROOT.mkdir(exist_ok=False, parents=True)
COMMIT = 'aa391988d354d4360f390f223a90e0d1b98839d7'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def save(path, data): path.write_text(json.dumps(data, indent=2) + '\n')
def run(command, name, env=None):
    started = time.monotonic()
    with (ROOT / (name + '.log')).open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    record = {'command': command, 'returncode': result.returncode,
              'wall_seconds': time.monotonic() - started}
    if env:
        record['environment'] = {key: env[key] for key in ('OMP_NUM_THREADS','LD_LIBRARY_PATH')}
    if name.endswith('-correctness'):
        record.update(binary_sha256=sha(ROOT / 'encoding_probe'), source_sha256=sha(ROOT / 'encoding_probe.cpp'))
    save(ROOT / (name + '-run.json'), record)
    if result.returncode: raise SystemExit('Failed: ' + name + '; inspect its preserved log')

for name in ('encoding_probe.cpp','reduce-encode-logarithms.patch','measure.py','compare.py'):
    shutil.copyfile(EVIDENCE / name, ROOT / name)
source, build = ROOT / 'source', ROOT / 'build-base'
run(['git','init','-q',str(source)], 'git-init')
run(['git','-C',str(source),'remote','add','origin','https://github.com/openfheorg/openfhe-development'], 'git-remote')
run(['git','-C',str(source),'fetch','--depth','1','origin',COMMIT], 'git-fetch')
run(['git','-C',str(source),'switch','--detach','FETCH_HEAD'], 'git-checkout')
encoding = source / 'src/pke/lib/encoding/ckkspackedencoding.cpp'
shutil.copyfile(encoding, ROOT / 'encoding-base.cpp')
run(['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
     '-DBUILD_UNITTESTS=OFF','-DBUILD_EXAMPLES=OFF','-DBUILD_BENCHMARKS=OFF',
     '-DBUILD_STATIC=OFF','-DBUILD_SHARED=ON','-DNATIVE_SIZE=64','-DWITH_OPENMP=ON',
     '-DWITH_NATIVEOPT=OFF','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'], 'configure-base')
run(['cmake','--build',str(build),'-j','4','--target','OPENFHEpke'], 'build-base')
shutil.copytree(build / 'lib', ROOT / 'lib-base', symlinks=True)
command = ['g++','-O3','-std=c++20','-fopenmp']
command += ['-I' + str(p) for p in (source/'src/core/include',source/'src/pke/include',
                                  source/'src/binfhe/include',build/'src/core',source/'third-party/cereal/include')]
command += [str(ROOT/'encoding_probe.cpp'),'-L'+str(ROOT/'lib-base'),'-lOPENFHEpke',
            '-lOPENFHEcore','-lOPENFHEbinfhe','-Wl,-rpath,'+str(ROOT/'lib-base'),'-o',str(ROOT/'encoding_probe')]
save(ROOT / 'probe-build-command.json', command)
run(command, 'probe-build')
env = dict(os.environ, OMP_NUM_THREADS='4', LD_LIBRARY_PATH=str(ROOT/'lib-base'))
run([str(ROOT/'encoding_probe'),'write',str(ROOT/'coefficients.bin'),str(ROOT/'base-correctness.json')],
    'base-correctness', env)
run(['git','-C',str(source),'apply',str(ROOT/'reduce-encode-logarithms.patch')], 'apply-candidate')
shutil.copyfile(encoding, ROOT / 'encoding-candidate.cpp')
run(['cmake','--build',str(build),'-j','4','--target','OPENFHEpke'], 'build-candidate')
shutil.copytree(build / 'lib', ROOT / 'lib-candidate', symlinks=True)
env['LD_LIBRARY_PATH'] = str(ROOT / 'lib-candidate')
run([str(ROOT/'encoding_probe'),'compare',str(ROOT/'coefficients.bin'),str(ROOT/'candidate-correctness.json')],
    'candidate-correctness', env)
run([sys.executable,str(ROOT/'measure.py')], 'measure')
run([sys.executable,str(ROOT/'compare.py')], 'verify')
print(ROOT / 'comparison.json')
