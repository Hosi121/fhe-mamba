"""Recheck all measured identities after the final native job; do not overwrite preflight."""
import hashlib,subprocess
from pathlib import Path
from runlib import ROOT,sha,read,save,stamp
rev=ROOT/read(ROOT/'config.json')['revision'];target=read(rev/'target-provenance.json')
assert read(ROOT/'completion.json')['passed']
for p,h in target['dependencies_verified'].items():assert sha(p)==h,p
assert sha(target['baseline_binary'])==target['baseline_binary_sha256']
for p,h in target['candidate_binaries_sha256'].items():assert sha(rev/'build'/p)==h,p
files=read(rev/'compiled-sources.json')['files']
for p,h in files.items():assert sha(rev/'source'/p)==h,p
for info in target['payloads'].values():
 root=Path(info['path']);assert sha(root/'manifest.json')==info['manifest_sha256']
 for p,h in info['files_sha256'].items():assert sha(root/p)==h,p
backend='/home/kataiwa/fhemamba/spark/source-2a70798e869944af'
assert subprocess.check_output(['git','-C',backend,'rev-parse','HEAD'],text=True).strip()==target['backend_commit']
diff=subprocess.check_output(['git','-C',backend,'diff'],text=True).strip()
assert hashlib.sha256(diff.encode()).hexdigest()==target['backend_diff_sha256']
manifest=read(ROOT.parent/'owned-arithmetic-20260924/compiled-final-sources.json')
assert sha(ROOT.parent/'owned-arithmetic-20260924/compiled-final-sources.json')==target['baseline_source_manifest_sha256']
# Preserve model metadata needed to interpret the original raw runs without a checkpoint.
for name,info in target['payloads'].items():
 p=ROOT/'payload-manifests'/name;p.mkdir(parents=True,exist_ok=True)
 import shutil
 shutil.copyfile(Path(info['path'])/'manifest.json',p/'manifest.json')
base=ROOT/'baseline';base.mkdir(exist_ok=True)
for source,name in [('compiled-final-sources.json','compiled-sources.json'),('compiled-final-sources.tar.gz','compiled-sources.tar.gz'),('target-final-provenance.json','target-provenance.json')]:
 shutil.copyfile(ROOT.parent/'owned-arithmetic-20260924'/source,base/name)
save(ROOT/'postflight.json',{'passed':True,'finished_utc':stamp(),'candidate_files':len(files),'baseline_files':len(manifest['files']),
 'baseline_binary_sha256':target['baseline_binary_sha256'],'candidate_binaries_sha256':target['candidate_binaries_sha256'],
 'dependencies_verified':target['dependencies_verified'],'payloads_verified':target['payloads'],
 'backend_commit':target['backend_commit'],'backend_diff_sha256':target['backend_diff_sha256']})
print('postflight identity checks passed',flush=True)
