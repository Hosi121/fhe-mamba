import pytest

from fhemamba._env import block_broken_torchvision

block_broken_torchvision()


@pytest.fixture(scope="session")
def model_factory():
    """Fresh deterministic HF models; overrides keep each test's geometry explicit."""
    import torch

    transformers = pytest.importorskip("transformers")

    def build(architecture=2, seed=None, **overrides):
        name = "Mamba" if architecture == 1 else "Mamba2"
        dimensions = {
            "vocab_size": 97,
            "hidden_size": 32,
            "state_size": 8,
            "num_hidden_layers": 2,
            "conv_kernel": 4,
        }
        dimensions.update(
            {"intermediate_size": 64, "time_step_rank": 4, "use_mambapy": False}
            if architecture == 1
            else {"expand": 2, "num_heads": 4, "head_dim": 16, "n_groups": 1, "chunk_size": 8}
        )
        dimensions.update(overrides)
        torch.manual_seed(seed if seed is not None else 7 if architecture == 1 else 19)
        config = getattr(transformers, name + "Config")(**dimensions)
        return getattr(transformers, name + "ForCausalLM")(config).float().eval()

    return build


@pytest.fixture
def tokenizer_factory():
    from types import SimpleNamespace

    import torch

    class Tokenizer:
        def __init__(self, limit=64):
            self.limit = limit

        def __call__(self, text, return_tensors=None):
            ids = torch.tensor([[ord(c) % 90 + 3 for c in text[: self.limit]]])
            return SimpleNamespace(input_ids=ids)

    return Tokenizer
