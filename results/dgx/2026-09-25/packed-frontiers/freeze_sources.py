"""Freeze the existing source inventory; never mutate an earlier revision."""
import hashlib,json,subprocess,sys,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
revision=root/sys.argv[1];revision.mkdir()
paths=set(json.loads((root/'frontier-r1/compiled-sources.json').read_text())['files'])
paths.update(subprocess.check_output(['git','ls-files','native/fideslib_stage0'],text=True).splitlines())
paths={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths) if Path(p).is_file()}
for p,h in paths.items():
    committed=subprocess.check_output(['git','show','HEAD:'+p])
    assert hashlib.sha256(committed).hexdigest()==h,p
manifest={'base_commit':'18496e4e6a4e42c367fc356e99a7c4ab8e935eb0',
    'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'files':paths}
(revision/'compiled-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(revision/'compiled-sources.tar.gz','w:gz') as t:
    for p in paths:t.add(p,arcname=p)
print(json.dumps({'revision':str(revision),'files':len(paths)}))
