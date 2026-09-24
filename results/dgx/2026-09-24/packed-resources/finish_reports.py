"""Derive human-readable generated text from raw client-selected token IDs."""
import hashlib
import json
from pathlib import Path
import sys

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
sys.path[:0]=[str(REPO/'src'),str(REPO/'experiments')]
from fhemamba.generation import generation_report
from report_mamba3_generation import report_generation
from transformers import AutoTokenizer

read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
for name in ('full-base','full-all'):
    report=report_generation(REPO/'results/dgx/2026-09-24/mamba3-trained-generation/full-payload',ROOT/name,
                             REPO/'checkpoints/mamba3-siso-187m/tokenizer')
    assert report['passed']
    save(ROOT/name/'generation.json',report)
    print(name,report)
if (ROOT/'m2-full-direct/native.json').exists():
    previous=REPO/'results/dgx/2026-09-24/shared-plaintext-preparation'
    if not previous.exists():previous=REPO/'runs/mamba2-plaintext-20260924'
    target=read(ROOT/'target-provenance.json')
    chain=read(previous/'mamba2-chain.json')
    assert sha(previous/'mamba2-chain.json')==target['payload_files_sha256']['chain.json']
    request=dict(read(REPO/'results/dgx/2026-09-22/subring-gates/generation.json')['parameters'])
    request.update(fast_plaintext_upload=True,gpu_plaintext_ntt=True,direct_plaintext_upload=True)
    directory=ROOT/'m2-full-direct'
    run=read(directory/'run.json');native=read(directory/'native.json')
    assert run['passed'] and run['native_sha256']==sha(directory/'native.json')
    assert run['environment']['INPUT_CHAIN_SHA256']==target['payload_sha256']
    native['input_payload_sha256']=target['payload_sha256']
    native['source_artifact']={'path':'native.json','sha256':sha(directory/'native.json')}
    native['provenance_annotation']='Derived copy; adjacent raw native.json is unchanged. Target provenance independently verifies the input hash.'
    save(directory/'native-with-provenance.json',native)
    tokenizer_path=REPO/'checkpoints/mamba2-130m-hf'
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    assert tokenizer(request['prompt']).input_ids==request['prompt_ids']
    report=generation_report(native,chain,request,tokenizer,payload_sha256=target['payload_sha256'])
    assert report['passed'] and report['measurements']['matches_exact_tokens']
    report['raw_native_sha256']=sha(directory/'native.json')
    report['tokenizer_files_sha256']={p.name:sha(p) for p in tokenizer_path.glob('*token*.json')}
    save(directory/'generation.json',report)
    print('m2-full-direct',report['text'],report['timing']['eval_seconds'])
