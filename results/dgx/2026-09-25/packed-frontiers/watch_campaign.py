"""Collect each serial stage and notify locally after the full campaign."""
import base64,json,shutil,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parent
remote='/home/kataiwa/fhemamba/layout-batch-20260925'
paths=['batch-r1/completion.json','layout-r1/completion.json','frontier-r1/completion.json','layout-r2/completion.json','m2-completion.json']
started=time.monotonic()
for path in paths:
    while time.monotonic()-started<24000:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','dgx',
            f'test ! -f {remote}/{path} || cat {remote}/{path}'],capture_output=True,text=True,timeout=30)
        if result.returncode==0 and result.stdout.strip():
            completion=json.loads(result.stdout)
            subprocess.run(['rsync','-az','--exclude=build/','--exclude=source/','--exclude=cpu-build/',
                'dgx:'+remote+'/',str(out)+'/'],check=True)
            print(json.dumps({'stage':path,'completion':completion}),flush=True)
            break
        time.sleep(30)
    else:raise SystemExit('Review interval reached; GPU jobs left running.')
command=shutil.which('powershell.exe')
if command:
    script='''Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$fheNotice = New-Object System.Windows.Forms.NotifyIcon
$fheNotice.Icon = [System.Drawing.SystemIcons]::Information
$fheNotice.Visible = $true
$fheNotice.ShowBalloonTip(10000, 'FHE Mamba', 'Optimization campaign finished. Results collected.', [System.Windows.Forms.ToolTipIcon]::Info)
Start-Sleep -Seconds 10
$fheNotice.Dispose()
'''
    encoded=base64.b64encode(script.encode('utf-16le')).decode('ascii')
    n=subprocess.run([command,'-NoProfile','-NonInteractive','-EncodedCommand',encoded],capture_output=True,
        text=True,encoding='cp932',errors='replace',timeout=30)
    (out/'campaign-notification.json').write_text(json.dumps({'returncode':n.returncode,'stdout':n.stdout,'stderr':n.stderr},indent=2)+'\n')
