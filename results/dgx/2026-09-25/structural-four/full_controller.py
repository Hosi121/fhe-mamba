"""Qualified final sources, then serialized ABBA full-model processes."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import time

root=Path('/home/kataiwa/fhemamba/structural-four-20260925')
baseline=Path('/home/kataiwa/fhemamba/gpu-rns-20260925/native-build/packed_fideslib')
candidate=root/'backend/spark/kernel/packed_fideslib'
env=dict(os.environ,OMP_NUM_THREADS='4',LD_LIBRARY_PATH='/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64')
status={'state':'running','stage':'identity','started_at_unix':time.time()}
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(data):
    assert data['passed'] and data['non_finite']==0 and data['evaluation_decryptions']==0
    assert data['exact_tolerance']==data['polynomial_tolerance']==.001
    assert data['generated_token_ids']==[315,279,1614,315]
    assert data['bootstrap_passes']==2 and data['s2c_first'] and data['gpu_plaintext_rns']
    assert (data['ring_dimension'],data['slots'],data['depth'],data['scale_bits'])==(65536,32768,44,59)
    assert (data['refresh_ceiling'],data['refreshed_level'])==(35,18)
    for k in ('max_abs_error_vs_exact','max_abs_error_vs_polynomial'):
        assert math.isfinite(data[k]) and 0<=data[k]<=.001
    assert all(math.isfinite(v) and 0<=v<=.001 for e in data['per_output_errors'] for v in e.values())
try:
    # Explicitly written only after all four small trials have been reviewed.
    selection=json.loads((root/'small-selection.json').read_text())
    assert selection['complete'] and selection['eligible']==['hoist-rotations','share-chebyshev']
    assert json.loads((root/'refinement-status.json').read_text())['state']=='completed'
    assert sha(baseline)=='bd7ab3c1fca7465a7186bf098c876ae16c199f7b7f19d3ae888731ce2b67e994'
    source=root/'repo/native/fideslib_stage0'
    manifest={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    save(root/'native-source-manifest.json',manifest)
    with tarfile.open(root/'native-sources.tar.gz','w:gz') as archive:archive.add(source,arcname='native-source')
    prior=json.loads((root/'prior-source-manifest.json').read_text())['final-backend']
    backend=root/'backend/spark/source-8f75cf9c2329fd3b'
    for name,digest in prior.items():assert sha(backend/name)==digest,name
    save(root/'backend-source-verification.json',{'passed':True,'identical_files':len(prior),'files_sha256':prior})
    deps=json.loads((root/'prior-dependency-verification.json').read_text())['files_sha256']
    for name,digest in deps.items():assert sha(Path(name))==digest,name
    save(root/'dependency-verification-before.json',{'passed':True,'files_sha256':deps})
    payload=Path('/home/kataiwa/fhemamba/mamba3-20260924/lm-full')
    payload_manifest=json.loads((payload/'manifest.json').read_text())
    for name,digest in payload_manifest['files_sha256'].items():assert sha(payload/name)==digest,name
    flags=['--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation','--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights','--move-plaintext-coefficients','--borrow-plaintext-upload','--bsgs-routing-stages','--cache-plaintexts','--frontier-refresh','--s2c-first','--gpu-plaintext-rns']
    records=[]
    for i,mode in enumerate(('baseline','candidate','candidate','baseline')):
        name=f'full-{i}-{mode}';d=root/name;d.mkdir()
        status['stage']=name;save(root/'full-status.json',status)
        binary=baseline if mode=='baseline' else candidate
        command=['taskset','-c','15-19',str(binary),str(payload/'program.txt'),str(d/'native.json'),'.001','.001']+flags+(['--hoist-rotations','--share-chebyshev'] if mode=='candidate' else [])+['--client-head',str(payload/'client_head.f32')]
        digest=sha(binary);started=time.time()
        with (d/'run.log').open('w') as log:
            process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
        save(d/'run.json',{'command':command,'started_at_unix':started,'wall_seconds':time.time()-started,'exit_code':process.returncode,'binary_sha256':digest,'environment':{k:env[k] for k in ('OMP_NUM_THREADS','LD_LIBRARY_PATH')}})
        assert process.returncode==0,name
        data=json.loads((d/'native.json').read_text());validate(data)
        if mode=='candidate':assert data['shared_basis_hits']>0 and data['rotation_sibling_batches']>0
        if records:
            previous=json.loads((root/records[0]['name']/'native.json').read_text())
            for key in ('bootstraps','logical_refreshes','evaluated_nodes','refresh_batches','frontier_deferrals'):
                assert data[key]==previous[key],(name,key,data[key],previous[key])
        save(d/'parity.json',{'passed':True,'manifest_sha256':sha(payload/'manifest.json'),'program_sha256':sha(payload/'program.txt')})
        records.append({'name':name,'mode':mode,'eval_seconds':data['eval_seconds'],'bootstrap_seconds':data['bootstrap_seconds'],'max_abs_error_vs_exact':data['max_abs_error_vs_exact'],'max_abs_error_vs_polynomial':data['max_abs_error_vs_polynomial']})
        save(root/'full-samples.json',records);print(name,data['eval_seconds'],flush=True)
    for name,digest in deps.items():assert sha(Path(name))==digest,name
    save(root/'dependency-verification-after.json',{'passed':True,'files_sha256':deps})
    status['state']='completed'
except Exception as error:
    status.update(state='failed',error=repr(error))
finally:
    status['finished_at_unix']=time.time();save(root/'full-status.json',status)
