"""Bind native results to the frozen inputs and decode actual generated IDs."""
import hashlib
import json
from pathlib import Path
import sys

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
sys.path[:0] = [str(REPO/'src'),str(REPO/'experiments')]
from fhemamba.generation import generation_report
from report_mamba3_generation import report_generation
from transformers import AutoTokenizer

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data): path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

target = read(ROOT/'target-provenance.json')
assert sha(ROOT/'mamba2-chain.json') == target['payload_files_sha256']['chain.json']
chain = read(ROOT/'mamba2-chain.json')
original = read(REPO/'results/dgx/2026-09-22/subring-gates/generation.json')
request = dict(original['parameters'])
tokenizer_path = REPO/'checkpoints/mamba2-130m-hf'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
assert tokenizer(request['prompt']).input_ids == request['prompt_ids']
for name in ('full-a','full-'+read(ROOT/'selection.json')['candidate']):
    run = read(ROOT/name/'run.json')
    raw_path = ROOT/name/'native.json'
    native = read(raw_path)
    assert run['passed'] and run['native_sha256'] == sha(raw_path)
    assert run['binary_sha256'] == native['binary_sha256']
    assert run['environment']['INPUT_CHAIN_SHA256'] == target['payload_sha256']
    native['input_payload_sha256'] = target['payload_sha256']
    native['source_artifact'] = {'path':'native.json','sha256':sha(raw_path)}
    native['provenance_annotation'] = 'Derived copy; the adjacent native.json is byte-preserved. Input digest independently verified in target-provenance.json.'
    save(ROOT/name/'native-with-provenance.json',native)
    e=native['measurements']['plaintext_encoding']
    request.update(fast_plaintext_upload=e['fast_upload'],gpu_plaintext_ntt=e['gpu_ntt'])
    report = generation_report(native,chain,request,tokenizer,payload_sha256=target['payload_sha256'])
    report['raw_native_sha256']=sha(raw_path)
    report['tokenizer_files_sha256']={p.name:sha(p) for p in tokenizer_path.glob('*token*.json')}
    assert report['passed'] and report['measurements']['matches_exact_tokens']
    save(ROOT/name/'generation.json',report)
    print(name,report['text'],report['timing']['eval_seconds'])

m3_report = report_generation(REPO/'results/dgx/2026-09-24/mamba3-trained-generation/full-payload',
                              ROOT/'mamba3-full',REPO/'checkpoints/mamba3-siso-187m/tokenizer')
save(ROOT/'mamba3-full/generation.json',m3_report)
print('mamba3-full',m3_report)
