"""Stateful decode vs full-sequence forward, and greedy parity vs HF generate."""

import pytest
import torch

from fhemamba.generate import generate_greedy
from fhemamba.reference import init_states, model_forward


@pytest.mark.parametrize("architecture", [1, 2])
def test_token_by_token_decode_matches_full_forward(model_factory, architecture) -> None:
    model = model_factory(architecture, **({"n_groups": 2} if architecture == 2 else {}))
    torch.manual_seed(29)
    ids = torch.randint(0, 97, (1, 17))
    full = model_forward(model, ids)["logits"]

    states = init_states(model)
    stepwise = []
    for t in range(ids.shape[1]):
        out = model_forward(model, ids[:, t : t + 1], states=states)
        stepwise.append(out["logits"][:, 0])
    stepwise = torch.stack(stepwise, dim=1)
    diff = float((full - stepwise).abs().max())
    assert diff < 1e-4, f"decode path diverged from full forward by {diff}"


@pytest.mark.parametrize("architecture", [1, 2])
def test_stateful_prefill_then_decode_matches_full_forward(model_factory, architecture) -> None:
    model = model_factory(architecture, **({"n_groups": 2} if architecture == 2 else {}))
    torch.manual_seed(31)
    ids = torch.randint(0, 97, (1, 21))
    full = model_forward(model, ids)["logits"]

    states = init_states(model)
    prefill = model_forward(model, ids[:, :13], scan="chunked", states=states)
    tail = []
    for t in range(13, ids.shape[1]):
        out = model_forward(model, ids[:, t : t + 1], states=states)
        tail.append(out["logits"][:, 0])
    got_last = torch.stack(tail, dim=1)
    assert torch.allclose(full[:, :13], prefill["logits"], atol=1e-4)
    assert torch.allclose(full[:, 13:], got_last, atol=1e-4)


@pytest.mark.parametrize("architecture", [1, 2])
@pytest.mark.parametrize("length", [1, 64, 129])
def test_prefill_state_owns_only_its_logical_storage(model_factory, architecture, length) -> None:
    model = model_factory(architecture, **({"n_groups": 2} if architecture == 2 else {}))
    torch.manual_seed(31)
    ids = torch.randint(0, 97, (1, length + 1))
    states = init_states(model)
    with torch.no_grad():
        model_forward(model, ids[:, :length], scan="chunked", states=states)
        for state in states:
            for value in (state.conv, state.ssm):
                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        decoded = model_forward(model, ids[:, length:], states=states)["logits"]
        full = model_forward(model, ids)["logits"][:, -1:]
    assert torch.allclose(decoded, full, atol=1e-4)


@pytest.mark.parametrize("architecture", [1, 2])
def test_greedy_generation_matches_hf_generate(model_factory, architecture) -> None:
    model = model_factory(architecture, **({"n_groups": 2} if architecture == 2 else {}))
    torch.manual_seed(37)
    ids = torch.randint(0, 97, (1, 9))
    hf_tokens = model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)[
        0, ids.shape[1] :
    ].tolist()
    ours = generate_greedy(model, ids, 6)
    # HF may stop early on the config's eos token; compare the overlap.
    assert len(hf_tokens) >= 3
    assert ours[: len(hf_tokens)] == hf_tokens
