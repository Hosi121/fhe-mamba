"""Build only after the baseline finishes; run serial immutable prefix controls."""
import json, math, subprocess, sys, time, traceback
from statistics import mean
from runlib import ROOT, base_env, run, save, sha, stamp

revision = ROOT / sys.argv[1]
features = sys.argv[2:]
name = revision.name
base = ROOT.parent / 'square-dispatch-20260925/build/packed_fideslib'
source = ROOT.parent / 'mamba3-20260924'
flags = ['--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation',
    '--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights',
    '--move-plaintext-coefficients','--borrow-plaintext-upload','--bsgs-routing-stages','--cache-plaintexts']
try:
    while not (ROOT/'qualification-completion.json').exists(): time.sleep(15)
    assert json.loads((ROOT/'completion.json').read_text())['passed']
    old = json.loads((ROOT/'baseline-target-provenance.json').read_text())
    assert sha(base) == old['candidate_binaries_sha256']['packed_fideslib']
    for p,h in old['dependencies_sha256'].items(): assert sha(p)==h,p
    src = revision/'source';src.mkdir()
    import tarfile
    with tarfile.open(revision/'compiled-sources.tar.gz') as archive: archive.extractall(src,filter='data')
    manifest = json.loads((revision/'compiled-sources.json').read_text())
    assert all(sha(src/p)==h for p,h in manifest['files'].items())
    commands=[['cmake','-S',str(src/'native/fideslib_stage0'),'-B',str(revision/'build'),
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH=/home/kataiwa/fhemamba/spark/install-2a70798e869944af;/home/kataiwa/fhe-deps/openfhe-fides',
        '-Dfideslib_DIR=/home/kataiwa/fhemamba/spark/install-2a70798e869944af/share/fideslib/cmake'],
        ['cmake','--build',str(revision/'build'),'--target','packed_fideslib','stage1_mamba2_decode_fideslib','-j','4']]
    for i, command in enumerate(commands):
        with (revision/f'build-{i}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    binary = revision/'build/packed_fideslib'
    save(revision/'build.json',{'commands':commands,'passed':True,
        'source_manifest_sha256':sha(revision/'compiled-sources.json'),
        'binaries_sha256':{n:sha(revision/'build'/n) for n in ['packed_fideslib','stage1_mamba2_decode_fideslib']},
        'compile_commands':json.loads((revision/'build/compile_commands.json').read_text()),
        'compiler':subprocess.check_output(['g++','--version'],text=True),
        'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True)})
    result={'passed':True,'stage':'immutable build only'}
except Exception:result={'passed':False,'error':traceback.format_exc()}
result['finished_utc']=stamp();save(revision/'build-completion.json',result);print(result,flush=True)
if not result['passed']:raise SystemExit(1)
