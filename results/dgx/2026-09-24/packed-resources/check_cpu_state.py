"""Compare the retained storage of the previous view and the owning-state fix."""
import hashlib
import json
from pathlib import Path
import types
import torch
from transformers import Mamba2Config, Mamba2ForCausalLM
import fhemamba.reference as after
from fhemamba.ops import Exact

root = Path(__file__).resolve().parent
path = Path(after.__file__)
source = path.read_text()
old = source.replace('    history = mixer.conv_kernel_size - 1\n    state.conv = full[..., seq_len : seq_len + history].clone()', '    state.conv = full[..., -(mixer.conv_kernel_size - 1) :]')
old = old.replace('state.ssm = states[:, :, -1].clone()', 'state.ssm = states[:, :, -1]')
old = old.replace('state.ssm = carry.clone()', 'state.ssm = carry')
before = types.ModuleType('fhemamba._reference_before_state_fix')
before.__package__ = 'fhemamba'
import sys
sys.modules[before.__name__] = before
exec(compile(old, '<before-state-fix>', 'exec'), before.__dict__)
torch.set_num_threads(2)
torch.manual_seed(19)
config = Mamba2Config(vocab_size=97, hidden_size=32, expand=2, num_heads=4, head_dim=16,
                      state_size=8, n_groups=2, num_hidden_layers=2, conv_kernel=4, chunk_size=8)
model = Mamba2ForCausalLM(config).float().eval()
inputs = torch.randn(2, 65, 32)
results = {}
outputs = {}
for name, module in [('before', before), ('after', after)]:
    states = module.init_states(model, batch_size=2)
    with torch.no_grad():
        prefill = torch.stack([module.mixer2_forward(layer.mixer, inputs[:, :64], Exact(), 0, scan='chunked', state=state) for layer,state in zip(model.backbone.layers,states)])
        results[name] = [{'ssm_storage_bytes':s.ssm.untyped_storage().nbytes(),
                          'ssm_logical_bytes':s.ssm.numel()*s.ssm.element_size(),
                          'conv_storage_bytes':s.conv.untyped_storage().nbytes(),
                          'conv_logical_bytes':s.conv.numel()*s.conv.element_size()} for s in states]
        decode = torch.stack([module.mixer2_forward(layer.mixer, inputs[:, 64:], Exact(), 0, state=state) for layer,state in zip(model.backbone.layers,states)])
        outputs[name] = (prefill, decode)
results['source_sha256'] = {'before':hashlib.sha256(old.encode()).hexdigest(), 'after':hashlib.sha256(source.encode()).hexdigest()}
results['config'] = config.to_dict()
results['prefill_bit_identical'] = torch.equal(outputs['before'][0], outputs['after'][0])
results['decode_bit_identical'] = torch.equal(outputs['before'][1], outputs['after'][1])
results['inference_no_grad'] = True
results['passed'] = results['prefill_bit_identical'] and results['decode_bit_identical'] and all(v['ssm_storage_bytes']==v['ssm_logical_bytes'] and v['conv_storage_bytes']==v['conv_logical_bytes'] for v in results['after'])
(root/'cpu-state.json').write_text(json.dumps(results, indent=2)+'\n')
assert results['passed']
assert all(v['ssm_storage_bytes']==262144 for v in results['before'])
print({k:results[k] for k in ('before','after','prefill_bit_identical','decode_bit_identical','passed')})
