"""Reference forward vs the official transformers implementation."""

import pytest
import torch

from fhemamba.ops import Exact, PolyOps, RangeRecorder
from fhemamba.reference import _affine_scan, chunked_scan, model_forward


@pytest.fixture(scope="module", params=[1, 2], ids=["mamba1", "mamba2"])
def reference_case(request, model_factory):
    architecture = request.param
    model = model_factory(architecture, **({"n_groups": 2} if architecture == 2 else {}))
    torch.manual_seed(11 if architecture == 1 else 23)
    return architecture, model, torch.randint(0, 97, (1, 33))  # Exercise chunk padding.


def test_loop_scan_matches_official_forward(reference_case) -> None:
    architecture, tiny_model, token_ids = reference_case
    with torch.no_grad():
        official = tiny_model(token_ids, output_hidden_states=True)
    ours = model_forward(tiny_model, token_ids, Exact(), scan="loop", output_hidden_states=True)
    assert len(ours["hidden_states"]) == len(official.hidden_states)
    for i, (theirs, mine) in enumerate(
        zip(official.hidden_states, ours["hidden_states"], strict=True)
    ):
        # HF Mamba-2 uses chunked SSD; preserve its distinct absolute-error gate.
        if architecture == 1:
            assert torch.allclose(theirs, mine, atol=1e-5), f"hidden state {i} diverged"
        else:
            diff = float((theirs - mine).abs().max())
            assert diff < 1e-4, f"hidden state {i} diverged by {diff}"
    assert torch.allclose(official.logits, ours["logits"], atol=1e-5 if architecture == 1 else 1e-4)


def test_chunked_scan_matches_loop(reference_case) -> None:
    _, tiny_model, token_ids = reference_case
    loop = model_forward(tiny_model, token_ids, Exact(), scan="loop")
    chunked = model_forward(tiny_model, token_ids, Exact(), scan="chunked")
    assert torch.allclose(loop["logits"], chunked["logits"], atol=1e-4)


def test_chunked_scan_against_direct_recurrence() -> None:
    torch.manual_seed(3)
    decay = torch.rand(2, 5, 50, 4) * 0.9 + 0.05
    update = torch.randn(2, 5, 50, 4)
    state = torch.zeros(2, 5, 4)
    expected = []
    for t in range(50):
        state = decay[:, :, t] * state + update[:, :, t]
        expected.append(state)
    expected = torch.stack(expected, dim=2)
    got = chunked_scan(decay, update, chunk=16)
    assert torch.allclose(got, expected, atol=1e-5)


def test_full_ladder_plumbing_stays_close_at_high_degree(reference_case) -> None:
    """Calibrate -> fit -> substitute every site; logits must stay near exact."""
    architecture, tiny_model, token_ids = reference_case
    recorder = RangeRecorder()
    model_forward(tiny_model, token_ids, recorder, scan="chunked")
    if architecture == 2:
        assert "gated_rms_invsqrt" in recorder.pooled_by_name()
    poly_ops = PolyOps.fit(
        ranges_by_name=recorder.pooled_by_name(),
        enabled=frozenset(recorder.pooled_by_name()),
        degrees=dict.fromkeys(recorder.pooled_by_name(), 31),
    )
    exact = model_forward(tiny_model, token_ids, Exact(), scan="chunked")
    poly = model_forward(tiny_model, token_ids, poly_ops, scan="chunked")
    max_diff = (exact["logits"] - poly["logits"]).abs().max()
    assert float(max_diff) < 0.05, f"poly substitution moved logits by {float(max_diff)}"
    if architecture == 1:
        assert all(rate == 0.0 for rate in poly_ops.violation_summary().values())


def test_affine_scan_keeps_per_head_decay_compact() -> None:
    torch.manual_seed(29)
    decay = torch.rand(1, 4, 1, 9, 1)
    update = torch.randn(1, 4, 16, 9, 8)

    cumulative_decay, scanned = _affine_scan(decay, update)
    state = torch.zeros(1, 4, 16, 8)
    expected = []
    for token in range(update.shape[-2]):
        state = decay[..., token, :] * state + update[..., token, :]
        expected.append(state)

    assert cumulative_decay.shape == decay.shape
    assert torch.allclose(scanned, torch.stack(expected, dim=-2), atol=1e-6)


def test_forward_can_skip_vocabulary_projection(reference_case) -> None:
    _, tiny_model, token_ids = reference_case
    calls = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = tiny_model.lm_head.register_forward_hook(count_call)
    try:
        output = model_forward(
            tiny_model,
            token_ids,
            Exact(),
            scan="chunked",
            output_hidden_states=True,
            output_logits=False,
        )
    finally:
        handle.remove()

    assert calls == 0
    assert "logits" not in output
    assert len(output["hidden_states"]) == len(tiny_model.backbone.layers) + 1
