"""A third model family added through the same public request lifecycle."""

import json
import subprocess
import sys

import pytest
import torch

from fhemamba import inspect_model, load_model
from fhemamba.benchmarks.io import write_json
from fhemamba.cli import main


@pytest.mark.parametrize(
    "geometry",
    [
        {},
        {"hidden_size": 24, "state_size": 16, "num_hidden_layers": 1, "conv_kernel": 1},
        {
            "hidden_size": 48,
            "state_size": 12,
            "num_hidden_layers": 3,
            "conv_kernel": 3,
            "use_bias": True,
            "use_conv_bias": False,
        },
    ],
)
def test_exact_generation_matches_hf_and_resets_state(tmp_path, model_factory, geometry):
    original = model_factory(architecture=1, **geometry)
    original.save_pretrained(tmp_path / "checkpoint")
    model = load_model(tmp_path / "checkpoint")
    assert model.architecture == "mamba1"
    ids = torch.tensor([[5, 7, 11]])
    expected = []
    with torch.no_grad():
        for _ in range(3):
            token = int(original(ids, use_cache=False).logits[0, -1].argmax())
            expected.append(token)
            ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
    for _ in range(2):
        result = model.generate(torch.tensor([5, 7, 11]), max_new_tokens=3)
        assert result.input_ids == [5, 7, 11]
        assert result.generated_ids == expected
        assert result.passed
        assert not result.encrypted
        assert result.report["architecture"] == "mamba1"
        assert result.stop_reason == "length"
    for backend in ("polynomial", "ckks"):
        with pytest.raises(ValueError, match=r"requires model\.prepare"):
            model.generate([5, 7], backend=backend)
    with pytest.raises(ValueError, match="requires explicit profile"):
        model.prepare([5, 7], output=tmp_path / "unsupported")
    assert not (tmp_path / "unsupported").exists()


def test_cli_uses_mamba1_registration_and_sharded_weights(tmp_path, model_factory, capsys):
    original = model_factory(architecture=1)
    checkpoint = tmp_path / "checkpoint"
    original.save_pretrained(checkpoint, max_shard_size="10KB")
    ids = tmp_path / "tokens.json"
    ids.write_text("[5, 7]")
    assert (
        main(
            [
                "generate",
                "--model",
                str(checkpoint),
                "--input-ids",
                str(ids),
                "--max-new-tokens",
                "2",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["passed"]
    assert result["input_ids"] == [5, 7]
    assert result["report"]["architecture"] == "mamba1"
    assert len(result["generated_ids"]) == 2
    support = inspect_model(checkpoint)
    assert support.backends["exact"].status == "implemented"
    assert support.backends["ckks"].status == "requires_preparation"
    assert support.profiles[0].name == "mamba1-experimental"
    assert support.profiles[0].security == "128-classic"


def test_preparation_requires_explicit_profile_before_weights_or_optional_imports(tmp_path):
    write_json(tmp_path / "config.json", {"model_type": "mamba"})
    code = """
import sys
from fhemamba.cli import main
from fhemamba import inspect_model
assert inspect_model(sys.argv[1]).backends['polynomial'].status == 'requires_preparation'
try:
    main(['prepare', '--model', sys.argv[1], '--input-ids', 'absent.json',
          '--output', sys.argv[2]])
except SystemExit as error:
    assert error.code == 2
else:
    raise AssertionError('unsupported preparation succeeded')
assert not {'torch', 'numpy', 'transformers'} & sys.modules.keys()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(tmp_path / "request")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "requires explicit profile" in completed.stderr
    assert not (tmp_path / "request").exists()


def test_unsupported_activation_is_reported_before_loading_weights(tmp_path):
    write_json(tmp_path / "config.json", {"model_type": "mamba", "hidden_act": "relu"})
    support = inspect_model(tmp_path)
    assert support.backends["exact"].status == "unsupported"
    with pytest.raises(ValueError, match="requires SiLU"):
        load_model(tmp_path)


@pytest.mark.parametrize("config", [{"tie_word_embeddings": False}, {"hidden_size": 16385}])
def test_unsupported_fhe_geometry_preserves_cpu_support(tmp_path, config):
    write_json(tmp_path / "config.json", {"model_type": "mamba", **config})
    support = inspect_model(tmp_path)
    assert support.backends["exact"].status == "implemented"
    assert support.backends["polynomial"].status == "unsupported"
    assert support.backends["ckks"].status == "unsupported"
    assert not support.profiles
