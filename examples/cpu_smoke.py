"""Exercise Mamba-2 forward and recurrent state locally, without downloads or FHE."""

import json

import torch

from fhemamba._env import block_broken_torchvision
from fhemamba.reference import init_states, model_forward

block_broken_torchvision()

from transformers import Mamba2Config, Mamba2ForCausalLM  # noqa: E402


def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = (
        Mamba2ForCausalLM(
            Mamba2Config(
                vocab_size=97,
                hidden_size=32,
                expand=2,
                num_heads=4,
                head_dim=16,
                state_size=8,
                n_groups=2,
                num_hidden_layers=2,
                conv_kernel=4,
                chunk_size=8,
            )
        )
        .float()
        .eval()
    )
    tokens = torch.randint(0, 97, (1, 17))
    with torch.no_grad():
        official = model(tokens).logits
        reference = model_forward(model, tokens)["logits"]
        states = init_states(model)
        recurrent = torch.cat(
            [
                model_forward(model, tokens[:, i : i + 1], states=states)["logits"]
                for i in range(tokens.shape[1])
            ],
            dim=1,
        )
    errors = {
        "reference_vs_transformers": float((reference - official).abs().max()),
        "recurrent_vs_full_sequence": float((recurrent - reference).abs().max()),
    }
    passed = all(value < 1e-4 for value in errors.values())
    print(
        json.dumps(
            {
                "model": "random two-layer Mamba-2",
                "encrypted": False,
                "max_abs_errors": errors,
                "tolerance": 1e-4,
                "passed": passed,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
