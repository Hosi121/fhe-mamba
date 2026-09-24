"""Bind the one-call-site final revision without changing initial identities."""
import json
from runlib import ROOT, sha, save

old = json.loads((ROOT / 'compiled-sources.json').read_text())
final = json.loads((ROOT / 'compiled-final-sources.json').read_text())
assert final['previous_manifest_sha256'] == sha(ROOT / 'compiled-sources.json')
changed = [p for p in final['files'] if final['files'][p] != old['files'].get(p)]
assert changed == ['native/fideslib_stage0/src/packed_fideslib.cpp']
assert all(sha(ROOT / 'source-final' / p) == value for p, value in final['files'].items())
target = json.loads((ROOT / 'target-provenance.json').read_text())
for p, value in target['dependencies_verified'].items(): assert sha(p) == value
for p, value in target['candidate_binaries_sha256'].items(): assert sha(ROOT / 'build' / p) == value
save(ROOT / 'target-final-provenance.json', {
    'previous_target_sha256': sha(ROOT / 'target-provenance.json'),
    'source_manifest_sha256': sha(ROOT / 'compiled-final-sources.json'),
    'binary_sha256': sha(ROOT / 'build-final/packed_fideslib'),
    'changed_sources': changed,
    'compile_commands': json.loads((ROOT / 'build-final/compile_commands.json').read_text()),
    'scope': 'Only packed_fideslib is rebuilt. Mamba-2 and the exact-RNS helper probe remain their initial immutable binaries and sources.',
})
