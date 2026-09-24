"""Bind both executables to unchanged libraries, source archives and payloads."""
from pathlib import Path
import hashlib,subprocess
from runlib import ROOT,sha,read,save,stamp
oldroot=ROOT.parent/'owned-arithmetic-20260924'
old=read(oldroot/'target-provenance.json');final=read(oldroot/'target-final-provenance.json')
rev=ROOT/read(ROOT/'config.json')['revision']
for p,h in old['dependencies_verified'].items():assert sha(p)==h,p
base=oldroot/'build-final/packed_fideslib';assert sha(base)==final['binary_sha256']
files=read(rev/'compiled-sources.json')['files']
for p,h in files.items():assert sha(rev/'source'/p)==h,p
def command(*args):return subprocess.check_output(args,text=True).strip()
backend='/home/kataiwa/fhemamba/spark/source-2a70798e869944af'
assert command('git','-C',backend,'rev-parse','HEAD')==old['backend_commit']
assert hashlib.sha256(command('git','-C',backend,'diff').encode()).hexdigest()==old['backend_diff_sha256']
assert command('g++','--version')==old['compiler']
assert command('/usr/local/cuda-13.0/bin/nvcc','--version')==old['cuda']
assert command('nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader')==old['hardware']
payloads={}
for name in ('lm-full','lm-layer1','lm-layer1-prefix','payload'):
 root=ROOT.parent/'mamba3-20260924'/name;manifest=read(root/'manifest.json')
 for p,h in manifest['files_sha256'].items():assert sha(root/p)==h,(name,p)
 payloads[name]={'path':str(root),'manifest_sha256':sha(root/'manifest.json'),'files_sha256':manifest['files_sha256']}
result={k:old[k] for k in ('hardware','compiler','cuda','backend_commit','backend_diff_sha256','dependencies_verified')}
result.update(baseline_binary=str(base),baseline_binary_sha256=sha(base),baseline_source_manifest_sha256=sha(oldroot/'compiled-final-sources.json'),candidate_source_manifest_sha256=sha(rev/'compiled-sources.json'),candidate_binaries_sha256={name:sha(rev/'build'/name) for name in ('packed_fideslib','weight_plaintext_probe')},compile_commands=read(rev/'build/compile_commands.json'),payloads=payloads,verified_utc=stamp())
save(rev/'target-provenance.json',result)
print('verified source, dependencies, executables and four payloads',flush=True)
