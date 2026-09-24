"""Read the serialized campaign's durable status; does not change target state."""
import subprocess
import sys

code = r'''
import json
from pathlib import Path
from statistics import mean
r=Path('/home/kataiwa/fhemamba/packed-resources-20260924')
s=json.loads((r/'status.json').read_text())
print({k:round(s[k],1) if k=='elapsed_seconds' else s[k] for k in ('name','status','elapsed_seconds') if k in s})
p=r/s.get('name','')/'native.log'
if p.exists():
 lines=p.read_text().splitlines()
 selected=[v for v in lines if v.startswith(('node=','client_token=','passed=')) or ' done output_level=' in v]
 print(selected[-1:])
 tokens=[v for v in lines if v.startswith('client_token=') or 'selected_id=' in v]
 if tokens:print('client',tokens)
for prefix in (() if brief else ('prefix','m2')):
 groups={}
 for p in r.glob(prefix+'-*-*/native.json'):
  if not p.parent.name.split('-')[1].isdigit():continue
  d=json.loads(p.read_text())
  groups.setdefault(p.parent.name.rsplit('-',1)[1],[]).append(d.get('eval_seconds',d.get('timing',{}).get('eval_seconds')))
 if groups:print(prefix,{k:{'n':len(v),'mean_s':round(mean(v),4)} for k,v in groups.items()})
for name in (() if brief else ('full-base','full-all','m2-full-direct')):
 p=r/name/'native.json'
 if p.exists():
  d=json.loads(p.read_text());print(name,d['passed'],d.get('eval_seconds',d.get('timing',{}).get('eval_seconds')))
for name in (('completion.json',) if brief else ('completion.json','short-comparison.json')):
 p=r/name
 if p.exists():print(name,p.read_text())
'''
result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','dgx',
                         "python3 - <<'PY'\nbrief="+repr('--brief' in sys.argv)+"\n"+code+"\nPY"], check=True, capture_output=True,
                        text=True,timeout=30)
print(result.stdout, end='')
