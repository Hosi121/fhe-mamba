"""Frozen-source serial jobs with a finite ledger and durable completion records."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,time
ROOT=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path,data):
 path=Path(path); tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(name,binary,args,check,timeout=300,metadata=None):
 config=read(ROOT/'config.json');rev=ROOT/config['revision']
 budget_path=ROOT/'budget.json'
 budget=read(budget_path) if budget_path.exists() else {'review_seconds':10800,'used_seconds':0,'runs':[]}
 if budget['used_seconds']+timeout>budget['review_seconds']:raise RuntimeError('native process review boundary reached')
 out=ROOT/name;out.mkdir(exist_ok=False)
 env=dict(os.environ)
 for key in ('OMP_NUM_THREADS','CUDA_LAUNCH_BLOCKING','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env.pop(key,None)
 env.update(OMP_NUM_THREADS='4',LD_LIBRARY_PATH='/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64')
 command=['taskset','-c','15-19',str(binary)]+[str(x).replace('{output}',str(out)) for x in args]
 record={'name':name,'command':command,'started_utc':stamp(),'timeout_seconds':timeout,
   'binary_sha256':sha(binary),'candidate_source_manifest_sha256':sha(rev/'compiled-sources.json'),
   'revision':config['revision'],'runlib_sha256':sha(__file__),'environment':{k:env[k] for k in ('OMP_NUM_THREADS','LD_LIBRARY_PATH')},**(metadata or {})}
 start=time.monotonic();print('START',name,flush=True)
 with (out/'native.log').open('w') as log:
  p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record['pid']=p.pid;save(out/'run.json',record);save(ROOT/'status.json',{**record,'status':'running'})
  try:
   while p.poll() is None:
    remaining=timeout-(time.monotonic()-start)
    if remaining<=0:raise subprocess.TimeoutExpired(command,timeout)
    try:p.wait(timeout=min(20,remaining))
    except subprocess.TimeoutExpired:
     process=Path(f'/proc/{p.pid}/status')
     if process.exists():record['process_sample']=[line for line in process.read_text().splitlines() if line.startswith(('Name:','Threads:','Cpus_allowed_list:','VmRSS:','VmHWM:'))]
     save(ROOT/'status.json',{**record,'status':'running','elapsed_seconds':time.monotonic()-start})
   record['timed_out']=False
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   record['timed_out']=True
 record.update(returncode=p.returncode,wall_seconds=time.monotonic()-start,ended_utc=stamp(),passed=False)
 budget['used_seconds']+=record['wall_seconds'];budget['runs'].append({'name':name,'charged_seconds':record['wall_seconds']});save(budget_path,budget)
 try:
  native=read(out/'native.json');record['native_sha256']=sha(out/'native.json');record['checks']=check(native)
  record['passed']=p.returncode==0 and not record['timed_out'] and all(record['checks'].values())
  record['eval_seconds']=native.get('eval_seconds')
 except Exception as error:record['validation_error']=str(error)
 save(out/'run.json',record);save(ROOT/'status.json',{**record,'status':'completed' if record['passed'] else 'failed'})
 print('END',name,'passed=',record['passed'],'wall=',record['wall_seconds'],flush=True)
 if not record['passed']:raise RuntimeError('job failed: '+name)
 return read(out/'native.json')
