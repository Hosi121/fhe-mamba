"""Cold request controls; qualify the full pair only from repeated-weight prefix."""
import math,traceback
from statistics import mean
from runlib import ROOT,read,run,save,sha,stamp
SOURCE=ROOT.parent/'mamba3-20260924'
BASE=ROOT.parent/'owned-arithmetic-20260924/build-final/packed_fideslib'
REV=ROOT/read(ROOT/'config.json')['revision']
FLAGS=['--planned-refresh','--batch-refresh','--inplace-ops','--gpu-plaintext-ntt','--profile-evaluation',
       '--naf-rotations','--reuse-dead-inputs','--direct-plaintext-upload','--compact-weights',
       '--move-plaintext-coefficients','--borrow-plaintext-upload','--bsgs-routing-stages','--cache-plaintexts']
SAME=('nodes','evaluated_nodes','bootstraps','ct_ct_mul','ct_pt_mul','rotations','refresh_rotations',
      'logical_refreshes','refresh_batches','ring_dimension','slots','depth','scale_bits','bootstrap_passes',
      'refresh_policy','batch_refresh','public_weight_count','public_weight_bytes','plaintext_cache_hits',
      'plaintext_cache_misses','optimized_routing_stages','routing_stage_rotations_saved',
      'owned_arithmetic_calls','owned_arithmetic_reused_inputs','owned_arithmetic_cloned_inputs')
references={}
def measure(name,candidate,payload):
 p=SOURCE/payload;m=read(p/'manifest.json');client=m['schema']=='fhemamba-mamba3-lm-v1'
 args=[p/'program.txt','{output}/native.json','.001','.001']+FLAGS
 if client:args+=['--client-head',p/'client_head.f32']
 if candidate:args+=['--weight-cache-mib','2048']
 def check(d):
  checks={'native':d['passed'],'finite':d['non_finite']==0,'gates':d['polynomial_tolerance']==d['exact_tolerance']==.001,
    'errors':all(math.isfinite(d[k]) and 0<=d[k]<=.001 for k in ('max_abs_error_vs_polynomial','max_abs_error_vs_exact')),
    'generation':not client or d['generated_token_ids']==m['exact_token_ids'],
    'no_intermediate_decryptions':d['evaluation_decryptions']==0}
  if payload in references:
   b=references[payload]
   checks['same_operations']=all(d[k]==b[k] for k in SAME)
   checks['same_ids']=d['generated_token_ids']==b['generated_token_ids']
   checks['same_encodes_or_exact_materializations']=d['host_encodes']+d.get('weight_cache_hits',0)==b['host_encodes']
  if candidate:
   checks['byte_budget']=d['weight_cache_bytes']<=d['weight_cache_capacity_bytes']==2048*1024*1024
   checks['dispatch']=(d['weight_cache_hits']>0 if payload in ('lm-layer1','lm-full') else d['weight_cache_entries']==d['weight_cache_hits']==d['weight_cache_misses']==0)
  return checks
 d=run(name,REV/'build/packed_fideslib' if candidate else BASE,args,check,1800 if payload=='lm-full' else 600,
       {'candidate':candidate,'payload':str(p),'manifest_sha256':sha(p/'manifest.json'),
        'program_sha256':m['files_sha256']['program.txt'],'controller_sha256':sha(__file__),
        'baseline_source_manifest_sha256':sha(ROOT.parent/'owned-arithmetic-20260924/compiled-final-sources.json')})
 if not candidate and payload not in references:references[payload]=d
 return d
try:
 assert read(ROOT/'probe-completion.json')['passed']
 assert read(ROOT/'probe-completion.json')['revision']==REV.name
 import subprocess
 subprocess.run(['python3',str(ROOT/'preflight.py')],check=True)
 controls={}
 for payload in ('lm-layer1-prefix','lm-layer1'):
  samples=[]
  for i,candidate in enumerate((False,True,True,False),1):
   d=measure(f'{payload}-{i}-'+('candidate' if candidate else 'base'),candidate,payload)
   samples.append(d['eval_seconds'])
  a,b=mean((samples[0],samples[3])),mean(samples[1:3])
  controls[payload]={'samples_abba':samples,'base_mean_seconds':a,'candidate_mean_seconds':b,'reduction_percent':100*(1-b/a)}
  save(ROOT/'short-comparison.json',controls)
 measure('synthetic-candidate',True,'payload')
 full={'skipped':True,'reason':'Repeated-weight prefix reduction did not exceed 0.5%.'}
 if controls['lm-layer1']['reduction_percent']>.5:
  a=measure('full-base',False,'lm-full');b=measure('full-candidate',True,'lm-full')
  full={'base_seconds':a['eval_seconds'],'candidate_seconds':b['eval_seconds'],
        'reduction_percent':100*(1-b['eval_seconds']/a['eval_seconds']),'adopt':b['eval_seconds']<a['eval_seconds']}
  save(ROOT/'full-comparison.json',full)
 result={'passed':True,'short':controls,'full':full}
except Exception:result={'passed':False,'error':traceback.format_exc()}
result['finished_utc']=stamp();save(ROOT/'completion.json',result);print(result,flush=True)
