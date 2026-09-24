"""Final identity checks and evidence collection after durable completion."""
from pathlib import Path
import json,subprocess,time,traceback
root=Path(__file__).resolve().parent
remote='/home/kataiwa/fhemamba/weight-preparation-20260925'
started=time.monotonic()
try:
 while time.monotonic()-started<15000:
  if (root/'completion.local.json').exists():break
  time.sleep(10)
 else:raise RuntimeError('postprocess review interval reached')
 completion=json.loads((root/'completion.local.json').read_text())
 if completion['passed']:
  result=subprocess.run(['ssh','dgx','python3 '+remote+'/postflight.py'],capture_output=True,text=True,timeout=180)
  (root/'postflight.log').write_text(result.stdout+result.stderr)
  assert result.returncode==0,result.stderr
  subprocess.run(['rsync','-az','--exclude=build/','--exclude=source/','dgx:'+remote+'/',str(root)+'/'],check=True)
 result={'passed':bool(completion['passed'])}
except Exception:result={'passed':False,'error':traceback.format_exc()}
(root/'postprocess-completion.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
