import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
root=Path('/home/kataiwa/fhemamba/structural-four-20260925')
exe=root/'backend/spark/kernel/packed_fideslib'
base=Path('/home/kataiwa/fhemamba/gpu-rns-20260925/native-build/packed_fideslib')
env=dict(os.environ,OMP_NUM_THREADS='4',LD_LIBRARY_PATH='/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64')
env['PATH']='/usr/local/cuda-13.0/bin:'+env['PATH']
status={'state':'running','stage':'build','started_at_unix':time.time()}
def save(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(name,command,limit=1800):
 d=root/name;d.mkdir()
 status['stage']=name;save(root/'first-pair-status.json',status)
 start=time.time()
 with (d/'run.log').open('w') as log:
  p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=limit)
 data={'command':command,'exit_code':p.returncode,'started_at_unix':start,'wall_seconds':time.time()-start,'environment':{k:env[k] for k in ['OMP_NUM_THREADS','LD_LIBRARY_PATH']}}
 if command[0]=='taskset':data['binary_sha256']=sha(Path(command[3]))
 save(d/'run.json',data)
 assert p.returncode==0,(name,p.returncode)
 if (d/'native.json').exists():
  data=json.loads((d/'native.json').read_text());assert data['passed'],name;return data
try:
 assert sha(base)=='bd7ab3c1fca7465a7186bf098c876ae16c199f7b7f19d3ae888731ce2b67e994'
 run('first-pair-configure',['cmake','-S',str(root/'repo/native/fideslib_stage0'),'-B',str(root/'backend/spark/kernel')])
 run('first-pair-build',['cmake','--build',str(root/'backend/spark/kernel'),'--target','packed_fideslib','rotation_batch_probe','test_shared_plans','-j','4'])
 run('first-pair-cpu-contract',['ctest','--test-dir',str(root/'backend/spark/kernel'),'-R','shared_plans','--output-on-failure'])
 for mode in ['mamba3','mamba2']:
  name='rotation-probe-'+mode
  command=['taskset','-c','15-19',str(root/'backend/spark/kernel/rotation_batch_probe'),str(root/name/'native.json')]
  if mode=='mamba2':command+=['--mamba2']
  run(name,command)
 common=['--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation','--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights','--move-plaintext-coefficients','--borrow-plaintext-upload','--bsgs-routing-stages','--cache-plaintexts','--frontier-refresh','--s2c-first','--gpu-plaintext-rns']
 for mode in ['baseline','share']:
  name='basis-probe-'+mode
  binary=base if mode=='baseline' else exe
  flags=common+(['--share-chebyshev'] if mode=='share' else [])
  data=run(name,['taskset','-c','15-19',str(binary),str(root/'basis-probe.txt'),str(root/name/'native.json'),'.001','.001']+flags)
  if mode=='share':assert data['shared_basis_hits']>0
 records=[]
 payload=Path('/home/kataiwa/fhemamba/mamba3-20260924/lm-layer1')
 manifest=json.loads((payload/'manifest.json').read_text())
 for name,digest in manifest['files_sha256'].items():assert sha(payload/name)==digest,name
 for i,mode in enumerate(['baseline','hoist','share','both','both','share','hoist','baseline']):
  name=f'prefix-{i}-{mode}'
  flags=list(common)
  if mode in ['hoist','both']:flags+=['--hoist-rotations']
  if mode in ['share','both']:flags+=['--share-chebyshev']
  data=run(name,['taskset','-c','15-19',str(base if mode=='baseline' else exe),str(payload/'program.txt'),str(root/name/'native.json'),'.001','.001']+flags+['--client-head',str(payload/'client_head.f32')])
  assert data['generated_token_ids']==[6864,6864]
  assert data['evaluation_decryptions']==0 and data['bootstrap_passes']==2 and data['s2c_first']
  if mode in ['hoist','both']:assert data['rotation_sibling_batches']>0
  if mode in ['share','both']:assert data['shared_basis_hits']>0
  records.append({'mode':mode,'eval_seconds':data['eval_seconds'],'bootstrap_seconds':data['bootstrap_seconds'],'rotations':data['rotations'],'ct_ct_mul':data['ct_ct_mul'],'bootstraps':data['bootstraps'],'shared_basis_hits':data.get('shared_basis_hits',0)})
  save(root/'first-pair-samples.json',records)
  print(name,data['eval_seconds'],flush=True)
 status['state']='completed'
except Exception as e:
 status['state']='failed';status['error']=repr(e)
finally:
 status['finished_at_unix']=time.time();save(root/'first-pair-status.json',status)
