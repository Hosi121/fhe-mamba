import json,subprocess,traceback
from runlib import ROOT,read,save,stamp
try:
 subprocess.run(['python3',str(ROOT/'probes.py')],check=True)
 assert read(ROOT/'probe-completion.json')['passed']
 subprocess.run(['python3',str(ROOT/'campaign.py')],check=True)
except Exception:
 save(ROOT/'completion.json',{'passed':False,'error':traceback.format_exc(),'finished_utc':stamp()})
