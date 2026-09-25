"""Recheck archived identities and numerical gates; never extrapolate prefixes."""
import hashlib,json,math,tarfile
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
archives={}
for manifest in sorted(ROOT.rglob('compiled-sources.json')):
    if 'source' in manifest.parts or 'build' in manifest.parts:continue
    archive=manifest.with_suffix('.tar.gz')
    if not archive.exists():continue
    files=read(manifest)['files']
    with tarfile.open(archive) as t:
        actual={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t if m.isfile()}
    assert actual==files,manifest
    archives[str(manifest.relative_to(ROOT))]={'files':len(files),'sha256':sha(manifest),'archive_sha256':sha(archive)}
base_archive=ROOT/'baseline-compiled-sources.tar.gz'
with tarfile.open(base_archive) as t:
    actual={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t if m.isfile()}
assert actual==read(ROOT/'baseline-compiled-sources.json')['files']
known_sources={sha(ROOT/'baseline-compiled-sources.json')}
known_sources.update(v['sha256'] for v in archives.values())
known_binaries=set(read(ROOT/'baseline-target-provenance.json')['candidate_binaries_sha256'].values())
for build in ROOT.glob('*/build.json'):known_binaries.update(read(build)['binaries_sha256'].values())
samples=[]
m2_reference=None
for run_path in sorted(ROOT.glob('*/run.json')):
    r=read(run_path);npath=run_path.with_name('native.json')
    if 'ended_utc' not in r:continue
    if r['passed']:
        assert r['returncode']==0 and not r['timed_out'] and all(r['checks'].values()),run_path
    assert r['source_manifest_sha256'] in known_sources,run_path
    assert r['binary_sha256'] in known_binaries,run_path
    if npath.exists():assert r['native_sha256']==sha(npath),run_path
    row={'name':r['name'],'passed':r['passed'],'wall_seconds':r['wall_seconds'],
        'binary_sha256':r['binary_sha256'],'source_manifest_sha256':r['source_manifest_sha256']}
    if npath.exists():
        d=read(npath)
        if d.get('schema')=='fhemamba-packed-result-v1':
            payload=Path(r['payload']).name
            manifest_path=(ROOT/'layout-fixture/manifest.json' if payload=='layout-fixture'
                           else ROOT/'payload-manifests'/payload/'manifest.json')
            assert sha(manifest_path)==r['manifest_sha256'],run_path
            manifest=read(manifest_path)
            assert manifest['files_sha256']['program.txt']==r['program_sha256'],run_path
            row.update({k:d[k] for k in ['eval_seconds','bootstraps','logical_refreshes','ct_ct_mul','ct_pt_mul',
                'rotations','host_encoding_seconds','bootstrap_seconds','max_abs_error_vs_exact','max_abs_error_vs_polynomial']})
            if r['passed']:
                assert d['passed'] and d['non_finite']==d['evaluation_decryptions']==0
                assert d['exact_tolerance']==d['polynomial_tolerance']==.001
                assert all(math.isfinite(d[k]) and 0<=d[k]<=.001 for k in ['max_abs_error_vs_exact','max_abs_error_vs_polynomial'])
                assert (d['ring_dimension'],d['depth'],d['scale_bits'],d['bootstrap_passes'])==(65536,44,59,2)
                assert d['security']=='not-set'
                outputs=len(manifest['output_names']) if 'output_names' in manifest else manifest['outputs']
                assert len(d['per_output_errors'])==outputs
                assert all(math.isfinite(v) and 0<=v<=.001 for e in d['per_output_errors'] for v in e.values())
                if manifest['schema']=='fhemamba-mamba3-lm-v1':
                    assert d['generated_token_ids']==manifest['exact_token_ids']
                    assert d['client_output_decrypt_count']==manifest['generated_tokens']
                if 'lm-' in r.get('payload',''):assert d['slots']==32768
                if 'lm-full' in r.get('payload',''):assert d['generated_token_ids']==[315,279,1614,315]
                elif 'lm-layer1' in r.get('payload',''):assert d['generated_token_ids']==[6864,6864]
            row['full_model']='lm-full' in r.get('payload','')
        else:
            row['eval_seconds']=d['timing']['eval_seconds']
            if r['passed']:
                parameters,measurements=d['parameters'],d['measurements']
                assert d['passed'] and parameters['tolerance']==.05
                assert parameters['n_layers_loaded']==parameters['tokens']==2
                assert all(measurements['per_token_decrypt_ok'])
                assert len(measurements['per_token_max_abs_error'])==2
                assert all(math.isfinite(v) and 0<=v<=.05 for v in measurements['per_token_max_abs_error'])
                assert d['measurement_scope']['zero_intermediate_decrypts']
                m2_reference=m2_reference or d
                assert all(d[k]==m2_reference[k] for k in ['parameters','ckks_levels','operation_counts',
                    'operation_counts_by_token','phase_operation_counts'])
    samples.append(row)
full=[v for v in samples if v.get('full_model')]
base=next(v for v in full if v['name']=='full-baseline')['eval_seconds']
for v in full:v['reduction_vs_initial_baseline_percent']=100*(1-v['eval_seconds']/base)
result={'scope':'Measured times; prefix and full workloads remain separate','archives':archives,'samples':samples,
    'full_model':full,'completed_gpu_wall_seconds':sum(v['wall_seconds'] for v in samples)}
if (ROOT/'final-completion.json').exists():
    final=read(ROOT/'final-completion.json')
    assert final['passed']
    control=next(v for v in full if v['name']=='final-baseline-full')
    candidate=next(v for v in full if v['name']=='final-candidate-full')
    assert final['baseline_sha256']==sha(ROOT/'final-baseline-full/native.json')
    assert final['candidate_sha256']==sha(ROOT/'final-candidate-full/native.json')
    assert final['baseline_seconds']==control['eval_seconds'] and final['candidate_seconds']==candidate['eval_seconds']
    assert final['reduction_percent']==100*(1-candidate['eval_seconds']/control['eval_seconds'])
    assert final['target_met']==(candidate['eval_seconds']<=.8*control['eval_seconds'])
    result['final_comparison']=final
(ROOT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'verified_samples':len(samples),'verified_archives':len(archives)+1,'full_model':full},indent=2))
