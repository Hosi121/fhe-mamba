"""Decode complete-model token IDs from each raw cache comparison artifact."""
import json
from pathlib import Path
import sys

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').exists())
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'experiments')]
from report_mamba3_generation import report_generation

for name in ('baseline-full', 'full-cache'):
    if not (ROOT / name / 'native.json').exists():
        continue
    report = report_generation(
        REPO / 'results/dgx/2026-09-24/mamba3-trained-generation/full-payload',
        ROOT / name, REPO / 'checkpoints/mamba3-siso-187m/tokenizer')
    assert report['passed']
    (ROOT / name / 'generation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(name, report)
