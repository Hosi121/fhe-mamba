"""Check Mamba-3 SISO prefill/decode parity without CUDA or a checkpoint."""

import json

import torch

from fhemamba.architectures import init_mixer_state
from fhemamba.mamba3 import Mamba3Mixer

torch.manual_seed(19)
mixer = Mamba3Mixer(32, d_state=16, headdim=32, dtype=torch.float64).eval()
inputs = torch.randn(1, 4, 32, dtype=torch.float64) * 0.2
with torch.no_grad():
    expected = mixer(inputs)
    carry = init_mixer_state(mixer)
    actual = torch.cat([mixer(inputs[:, t : t + 1], carry) for t in range(4)], dim=1)
error = float((actual - expected).abs().max())
print(
    json.dumps(
        {
            "architecture": "mamba3",
            "encrypted": False,
            "passed": error < 1e-12,
            "max_abs_error": error,
        },
        indent=2,
    )
)
raise SystemExit(0 if error < 1e-12 else 1)
