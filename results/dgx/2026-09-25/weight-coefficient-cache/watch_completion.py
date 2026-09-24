"""Collect completed GPU work and submit a local Windows desktop notification."""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import time

out = Path(__file__).resolve().parent
remote = '/home/kataiwa/fhemamba/weight-preparation-20260925'
started = time.monotonic()
while time.monotonic() - started < 15000:
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','dgx',
        f'test ! -f {remote}/completion.json || cat {remote}/completion.json'],
        capture_output=True,text=True,timeout=30)
    if result.returncode == 0 and result.stdout.strip():
        completion = json.loads(result.stdout)
        subprocess.run(['rsync','-az','--protect-args','--exclude=build/','--exclude=source/','--exclude=cpu-build/',
                        'dgx:'+remote+'/',str(out)+'/'],check=True)
        (out/'completion.local.json').write_text(json.dumps(completion,indent=2)+'\n')
        command = shutil.which('powershell.exe')
        if command:
            message = 'Public-weight cache validation completed. Results collected.' if completion['passed'] else 'GPU validation stopped. Check the saved result and logs.'
            ps = '''Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$fheNotice = New-Object System.Windows.Forms.NotifyIcon
$fheNotice.Icon = [System.Drawing.SystemIcons]::Information
$fheNotice.Visible = $true
$fheNotice.ShowBalloonTip(10000, 'FHE Mamba', '%s', [System.Windows.Forms.ToolTipIcon]::Info)
Start-Sleep -Seconds 10
$fheNotice.Dispose()
''' % message
            encoded = base64.b64encode(ps.encode('utf-16le')).decode('ascii')
            notice = subprocess.run([command,'-NoProfile','-NonInteractive','-EncodedCommand',encoded],
                                    capture_output=True,text=True,encoding='cp932',errors='replace',timeout=30)
            (out/'notification.json').write_text(json.dumps({'returncode':notice.returncode,
                'stdout':notice.stdout,'stderr':notice.stderr},indent=2)+'\n')
        print(json.dumps(completion),flush=True)
        break
    time.sleep(30)
else:
    raise SystemExit('Completion watcher reached the review interval; GPU jobs were not terminated.')
