import json

import numpy as np
import pytest
import torch

from fhemamba.mamba3 import init_mamba3_state, mamba3_step
from fhemamba.mamba3_lm import Mamba3LM
from fhemamba.packed_program import Calibration, PackedProgram
from fhemamba.tensor_ops import TensorOps


def config():
    return {
        "d_model": 8,
        "d_intermediate": 128,
        "n_layer": 2,
        "vocab_size": 32,
        "ssm_cfg": {"layer": "Mamba3", "d_state": 8, "headdim": 8},
        "rms_norm": True,
        "tie_embeddings": True,
    }


def test_complete_lm_includes_mlp_residuals_and_tied_head(tmp_path):
    torch.manual_seed(94)
    model = Mamba3LM(config()).eval()
    with torch.no_grad():
        states = model.initial_states()
        dense = [init_mamba3_state(layer.mixer) for layer in model.backbone.layers]
        for token in [5, 7, 2]:
            x = model.backbone.embedding.weight[token][None]
            actual, states = model.step(x, states)
            for index, layer in enumerate(model.backbone.layers):
                mixed, dense[index] = mamba3_step(
                    layer.mixer, layer.norm(x), dense[index], TensorOps(), index
                )
                x = x + mixed
                value, gate = layer.mlp.fc1(layer.norm2(x)).chunk(2, -1)
                x = x + layer.mlp.fc2(value * torch.nn.functional.silu(gate))
            expected = model.backbone.norm_f(x)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        (tmp_path / "config.json").write_text(json.dumps(config()))
        torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
        loaded = Mamba3LM.from_pretrained(tmp_path)
        assert loaded.lm_head.weight is loaded.backbone.embedding.weight
        assert loaded.generate([5, 7], 3)[0] == model.generate([5, 7], 3)[0]


def test_checkpoint_does_not_silently_ignore_unsupported_architecture():
    cfg = config()
    cfg["ssm_cfg"]["is_mimo"] = True
    with pytest.raises(ValueError, match="SISO"):
        Mamba3LM(cfg)


def test_fast_chebyshev_projection_matches_least_squares():
    calibration = Calibration()
    calibration.nonlinear(torch.linspace(-3, 3, 50), "sigmoid", (0, "sigmoid"))
    old, _ = calibration.fit(tolerance=1e-8)
    new, _ = calibration.fit(tolerance=1e-8, quadrature=True)
    np.testing.assert_allclose(
        old.polynomials[0, "sigmoid"].coeffs,
        new.polynomials[0, "sigmoid"].coeffs,
        atol=1e-13,
        rtol=1e-13,
    )


def test_extended_packing_handles_model_width_and_feedback_without_replay(tmp_path):
    program = PackedProgram({}, slots=2048, bound=1e6, extended=True)
    x = program.input(np.arange(768, dtype=float)[None] / 768)
    np.testing.assert_allclose(program.sum_last(x).value, x.value.sum(-1, keepdims=True))
    np.testing.assert_allclose(program._broadcast(x[:, :1], (1, 3)).value, [[0, 0, 0]])
    weights = np.eye(768)
    first = program.linear(x, weights)
    second = program.linear(x, weights)
    assert program.nodes[second.index][0] == "linear_ref"
    feedback = program.client_input(first, np.full((1, 768), 0.12345678))
    assert program.nodes[feedback.index][0] == "feedback"
    assert len(program.nodes[feedback.index][3]) == 0
    program.output(first, x.value)
    program.write(tmp_path / "program.txt")
    text = (tmp_path / "program.txt").read_text()
    assert text.startswith("fhemamba-packed-v2")
    assert "0.12345678" not in text


def test_generation_prefix_preserves_frozen_nodes_and_feedback(tmp_path):
    from fhemamba.workloads import mamba3_export as exporter

    source, destination = tmp_path / "source", tmp_path / "prefix"
    source.mkdir()
    program = PackedProgram({}, slots=8, extended=True)
    hidden = None
    values = np.arange(16).reshape(4, 4).astype(float)
    for step, value in enumerate(values):
        hidden = program.input(value) if step < 2 else program.client_input(hidden, value)
        program.output(hidden, value)
    program.write(source / "program.txt")
    (source / "client_head.f32").write_bytes(b"public head")
    np.savez(
        source / "fixture.npz",
        prompt_ids=[1, 2],
        exact_token_ids=[3, 4, 5],
        polynomial_token_ids=[3, 4, 5],
        exact_hidden=values,
        polynomial_hidden=values,
    )
    manifest = {
        "schema": "fhemamba-mamba3-lm-v1",
        "generated_tokens": 3,
        "prompt_ids": [1, 2],
        "tokens": 4,
        "exact_token_ids": [3, 4, 5],
        "polynomial_token_ids": [3, 4, 5],
        "reference_text": "entire source",
        "output_names": [f"step{i}" for i in range(4)],
        "program": program.summary(),
        "polynomials": {"sentinel": "unchanged"},
        "files_sha256": {
            name: exporter.digest(source / name)
            for name in ("program.txt", "fixture.npz", "client_head.f32")
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    before = (source / "program.txt").read_text().splitlines()
    result = exporter.prefix_payload(source, destination, 2)
    after = (destination / "program.txt").read_text().splitlines()
    assert after[0].split()[3:] == ["3", "3"]
    assert after[1:4] == before[1:4]
    assert after[4:] == before[5:8]
    assert result["program"]["operations"]["feedback"] == 1
    assert result["generated_tokens"] == 2
    assert result["polynomials"] == manifest["polynomials"]
    assert "reference_text" not in result
    assert exporter.digest(source / "program.txt") == manifest["files_sha256"]["program.txt"]
    with np.load(destination / "fixture.npz") as fixture:
        np.testing.assert_array_equal(fixture["exact_hidden"], values[:3])
        np.testing.assert_array_equal(fixture["prompt_ids"], [1, 2])
        np.testing.assert_array_equal(fixture["exact_token_ids"], [3, 4])
