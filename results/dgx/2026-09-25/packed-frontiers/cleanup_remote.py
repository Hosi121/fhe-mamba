"""Remove this campaign's disposable build trees after final qualification."""
import hashlib,json,shutil,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
assert root==Path('/home/kataiwa/fhemamba/layout-batch-20260925')
assert json.loads((root/'final-completion.json').read_text())['passed']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for name in ['batch-r1','layout-r1','frontier-r1','layout-r2','final-r1']:
 r=root/name
 manifest=json.loads((r/'compiled-sources.json').read_text())
 with tarfile.open(r/'compiled-sources.tar.gz') as t:
  actual={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t if m.isfile()}
 assert actual==manifest['files'],name
 build=json.loads((r/'build.json').read_text())
 for binary,h in build['binaries_sha256'].items():assert sha(r/'build'/binary)==h,(name,binary)
 source=r/'source'
 actual_source={str(p.relative_to(source)):sha(p) for p in source.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts}
 assert actual_source==manifest['files'],name
 shutil.rmtree(source)
 removed=['source/']
 kept=[]
 if name=='final-r1':
  for p in (r/'build').iterdir():
   if p.name in build['binaries_sha256']:
    kept.append(str(p.relative_to(root)))
   else:
    shutil.rmtree(p) if p.is_dir() else p.unlink()
  removed.append('build intermediates (selected executables retained)')
 else:
  shutil.rmtree(r/'build')
  removed.append('build/')
 records.append({'revision':name,'verified_files':len(actual),'removed':removed,'kept':kept})
result={'passed':True,'scope':'Only this campaign. Shared inputs, dependencies and baseline executables are untouched.', 'revisions':records}
(root/'cleanup.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
