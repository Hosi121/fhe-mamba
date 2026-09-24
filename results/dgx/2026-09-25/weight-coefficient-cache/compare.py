"""Reproduce the adoption decision from frozen sources and untouched raw samples."""
from pathlib import Path
import hashlib,json,math
from statistics import mean
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
completion=read(ROOT/'completion.json')
assert completion['passed']
config=read(ROOT/'config.json');revision=config['revision'];rev=ROOT/revision
source=read(rev/'compiled-sources.json');target=read(rev/'target-provenance.json')
assert sha(rev/'compiled-sources.json')==target['candidate_source_manifest_sha256']
base=read(ROOT/'baseline/compiled-sources.json')
assert sha(ROOT/'baseline/compiled-sources.json')==target['baseline_source_manifest_sha256']
changed=[p for p,h in source['files'].items() if base['files'].get(p)!=h]
assert sorted(changed)==sorted(read(ROOT/'source-changes.json'))
assert read(ROOT/'postflight.json')['passed']

def native(name,candidate,probe=False):
 path=ROOT/name;run=read(path/'run.json');data=read(path/'native.json')
 assert run['passed'] and run['returncode']==0 and not run['timed_out'] and all(run['checks'].values()),name
 assert data['passed'] and sha(path/'native.json')==run['native_sha256'],name
 assert run['candidate_source_manifest_sha256']==target['candidate_source_manifest_sha256'],name
 assert run['revision']==revision
 binary=target['candidate_binaries_sha256']['weight_plaintext_probe' if probe else 'packed_fideslib'] if candidate else target['baseline_binary_sha256']
 assert binary==run['binary_sha256'],name
 assert run['environment']=={'OMP_NUM_THREADS':'4','LD_LIBRARY_PATH':'/home/kataiwa/fhe-deps/openfhe-fides/lib:/usr/local/cuda-13.0/lib64'},name
 if not probe:
  assert data['evaluation_decryptions']==0 and data['non_finite']==0
  assert data['polynomial_tolerance']==data['exact_tolerance']==.001
  assert all(math.isfinite(data[k]) and 0<=data[k]<=.001 for k in ('max_abs_error_vs_exact','max_abs_error_vs_polynomial'))
  manifest=read(ROOT/'payload-manifests'/Path(run['payload']).name/'manifest.json')
  assert run['manifest_sha256']==sha(ROOT/'payload-manifests'/Path(run['payload']).name/'manifest.json')
  assert run['program_sha256']==manifest['files_sha256']['program.txt']
  if manifest['schema']=='fhemamba-mamba3-lm-v1':assert data['generated_token_ids']==manifest['exact_token_ids']
 return data,run

probes={}
for arch in ('mamba3','mamba2'):
 d,_=native('probe-'+arch+'-'+revision,True,True)
 assert d['cases']==d['gpu_cases']==60
 assert d['hit_cases']==51 and d['rejected_cases']==9 and d['boundary_cases']==2
 assert d['exact_rns_and_metadata'] and d['immutable_cache'] and d['level_key_checked'] and d['budget_checked']
 probes[arch]={k:d[k] for k in ('cases','hit_cases','rejected_cases','gpu_cases','boundary_cases')}

same=('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul','rotations','refresh_rotations',
      'logical_refreshes','refresh_batches','ring_dimension','slots','depth','scale_bits','bootstrap_passes',
      'refresh_policy','batch_refresh','public_weight_count','public_weight_bytes','plaintext_cache_hits',
      'plaintext_cache_misses','optimized_routing_stages','routing_stage_rotations_saved',
      'owned_arithmetic_calls','owned_arithmetic_reused_inputs','owned_arithmetic_cloned_inputs')
def parity(a,b):
 assert all(a[k]==b[k] for k in same)
 assert a['generated_token_ids']==b['generated_token_ids']
 assert b['host_encodes']+b.get('weight_cache_hits',0)==a['host_encodes']
 assert b['weight_cache_bytes']<=b['weight_cache_capacity_bytes']==2048*1024*1024

short={}
for payload in ('lm-layer1-prefix','lm-layer1'):
 values=[];rows=[]
 for i,candidate in enumerate((False,True,True,False),1):
  d,r=native(f'{payload}-{i}-'+('candidate' if candidate else 'base'),candidate)
  values.append(d['eval_seconds']);rows.append(d)
  if candidate:
   parity(rows[0],d)
   if payload=='lm-layer1-prefix':assert d['weight_cache_entries']==d['weight_cache_hits']==d['weight_cache_misses']==0
   else:assert d['weight_cache_hits']>0
 a,b=mean((values[0],values[3])),mean(values[1:3])
 short[payload]={'samples_abba':values,'base_mean_seconds':a,'candidate_mean_seconds':b,'reduction_percent':100*(1-b/a)}
assert short==read(ROOT/'short-comparison.json')==completion['short']
synth,_=native('synthetic-candidate',True)
assert synth['weight_cache_entries']==synth['weight_cache_hits']==0
full={'skipped':True,'reason':'Repeated-weight prefix reduction did not exceed 0.5%.'}
if short['lm-layer1']['reduction_percent']>.5:
 a,ar=native('full-base',False);b,br=native('full-candidate',True);parity(a,b)
 assert a['generated_token_ids']==b['generated_token_ids']==[315,279,1614,315]
 full={'base_seconds':a['eval_seconds'],'candidate_seconds':b['eval_seconds'],
       'reduction_percent':100*(1-b['eval_seconds']/a['eval_seconds']),'adopt':b['eval_seconds']<a['eval_seconds']}
 assert full==read(ROOT/'full-comparison.json')
 assert b['weight_cache_hits']>0
 assert a['gpu_ntt_encodes']==b['gpu_ntt_encodes'] and a['borrowed_plaintext_uploads']==b['borrowed_plaintext_uploads']
 full['process_wall_seconds']={'baseline':ar['wall_seconds'],'candidate':br['wall_seconds']}
 full['candidate_cache']={k:b[k] for k in b if k.startswith('weight_cache_')}
 full['baseline_host_encoding_seconds']=a['host_encoding_seconds']
 full['candidate_host_encoding_seconds']=b['host_encoding_seconds']
 full['baseline_peak_rss_gib']=a['peak_rss_gib'];full['candidate_peak_rss_gib']=b['peak_rss_gib']
result={'passed':True,'changed_sources':changed,'exact_probes':probes,'short':short,'full':full,
        'adopt':bool(full.get('adopt',False)),'scope':'Frozen prompt and parameters; all samples retained; no statistical significance claim.'}
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result),flush=True)
