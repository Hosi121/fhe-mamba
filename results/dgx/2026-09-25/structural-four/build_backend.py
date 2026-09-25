import json
import os
import subprocess
import time
from pathlib import Path
root=Path('/home/kataiwa/fhemamba/structural-four-20260925')
command=['bash',str(root/'repo/scripts/build_dgx_spark.sh')]
started=time.time()
env=dict(os.environ,FHEMAMBA_REMOTE_ROOT=str(root/'backend'),BUILD_JOBS='6')
with (root/'backend-build.log').open('w') as log:
 p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
(root/'backend-build.json').write_text(json.dumps({'command':command,'exit_code':p.returncode,'wall_seconds':time.time()-started,'started_at_unix':started},indent=2)+'\n')
