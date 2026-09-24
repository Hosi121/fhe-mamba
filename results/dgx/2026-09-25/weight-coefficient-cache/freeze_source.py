from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tarfile
root=Path(__file__).resolve().parent
rev=root/f'revision-{int(sys.argv[1])}'
rev.mkdir(exist_ok=False)
paths=set(json.loads(Path('results/dgx/2026-09-24/owned-arithmetic/compiled-final-sources.json').read_text())['files'])
paths.update(['native/fideslib_stage0/include/compact_rns.hpp','native/fideslib_stage0/src/weight_plaintext_probe.cpp','native/fideslib_stage0/tests/test_compact_rns.cpp'])
files={}
for name in sorted(paths):
 p=Path(name);target=rev/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(rev/'compiled-sources.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'files':files,'scope':f'Immutable candidate source snapshot for compact coefficient cache {rev.name}'},indent=2)+'\n')
with tarfile.open(rev/'compiled-sources.tar.gz','w:gz') as t:
 for name in files:t.add(rev/'source'/name,arcname=name)
script=(root/'build.sh').read_text().replace('revision-1',rev.name)
(rev/'build.sh').write_text(script)
(root/'config.json').write_text(json.dumps({'revision':rev.name},indent=2)+'\n')
print(len(files),'source files frozen',rev.name)
