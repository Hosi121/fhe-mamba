"""A separately registered integration must work without changes to the core."""

import json
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from fhemamba import GenerationResult, inspect_model, load_model, load_prepared
from fhemamba.benchmarks.io import file_sha256, write_json
from fhemamba.cli import main
from fhemamba.models import ModelRegistration, ModelRegistry, PreparationProfile, register_model
from fhemamba.models.contracts import capabilities, parse_options
from fhemamba.models.registry import registry


@dataclass(frozen=True)
class FixtureOptions:
    offset: int


class FixtureFHE:
    profiles = (PreparationProfile("fixture-profile", "test-only", ("Synthetic test adapter",)),)
    default_profile = "fixture-profile"

    def parse_options(self, value):
        return parse_options(FixtureOptions, value)

    def prepare(
        self, model, checkpoint, identity, ids, length, output, tokenizer, options, profile
    ):
        write_json(output / "coefficients.json", {"offset": options.offset})
        write_json(
            output / "manifest.json",
            {
                "schema": "fixture-request-v1",
                "architecture": "fixture-model",
                "profile": profile.name,
                "checkpoint": {"files_sha256": identity},
                "prompt_ids": ids,
                "generated_tokens": length,
                "coefficient_digest": file_sha256(output / "coefficients.json"),
            },
        )

    def validate_manifest(self, path, manifest):
        if file_sha256(path / "coefficients.json") != manifest["coefficient_digest"]:
            raise ValueError("coefficient digest differs")
        return manifest

    def run(self, prepared, manifest, binary, output, timeout):
        raise AssertionError("this fixture never executes a native process")


class FixtureAdapter:
    architecture = "fixture-model"
    fhe = FixtureFHE()

    def capabilities(self, config):
        return capabilities(self.architecture, "fixture-format", self.fhe)

    def checkpoint_identity(self, checkpoint):
        return {"config.json": file_sha256(checkpoint / "config.json")}

    def load(self, checkpoint):
        return {"vocab_size": 10}

    def vocab_size(self, model):
        return model["vocab_size"]

    def generate_cpu(self, model, ids, length, backend, prepared, manifest):
        offset = (
            1
            if prepared is None
            else json.loads((prepared.path / "coefficients.json").read_text())["offset"]
        )
        tokens = [(ids[-1] + offset * i) % model["vocab_size"] for i in range(1, length + 1)]
        return GenerationResult(ids, tokens, backend, True, False, "length", {})


@pytest.fixture
def integration(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_registrations", dict(registry._registrations))
    monkeypatch.setitem(
        sys.modules, "fixture_integration", SimpleNamespace(ADAPTER=FixtureAdapter())
    )
    entry = ModelRegistration(
        "fixture-model",
        "fixture_integration:ADAPTER",
        lambda config: config.get("model_type") == "fixture-model",
        ("fixture-request-v1",),
    )
    register_model(entry)
    checkpoint = tmp_path / "checkpoint"
    write_json(checkpoint / "config.json", {"model_type": "fixture-model"})
    return checkpoint, entry


def test_registered_integration_uses_shared_lifecycle(integration, tmp_path):
    checkpoint, _ = integration
    model = load_model(checkpoint)
    assert inspect_model(checkpoint).architecture == "fixture-model"
    assert model.generate([2, 3], max_new_tokens=2).generated_ids == [4, 5]
    prepared = model.prepare(
        [2, 3],
        max_new_tokens=2,
        output=tmp_path / "request",
        options=FixtureOptions(offset=2),
    )
    prepared = load_prepared(prepared.path)
    assert prepared.profile == "fixture-profile"
    assert model.generate(
        [2, 3],
        max_new_tokens=2,
        backend="polynomial",
        prepared=prepared,
    ).generated_ids == [5, 7]
    with pytest.raises(ValueError, match="differ from the prepared request"):
        model.generate([3, 2], max_new_tokens=2, backend="polynomial", prepared=prepared)
    write_json(prepared.path / "coefficients.json", {"offset": 3})
    with pytest.raises(ValueError, match="digest differs"):
        load_prepared(prepared.path)


def test_cli_passes_registered_options_without_model_specific_flags(integration, tmp_path, capsys):
    checkpoint, _ = integration
    tokens, options = tmp_path / "tokens.json", tmp_path / "options.json"
    tokens.write_text("[2, 3]")
    write_json(options, {"offset": 3})
    assert (
        main(
            [
                "prepare",
                "--model",
                str(checkpoint),
                "--input-ids",
                str(tokens),
                "--prepare-options",
                str(options),
                "--max-new-tokens",
                "2",
                "--output",
                str(tmp_path / "request"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["profile"] == "fixture-profile"
    assert (
        main(
            [
                "generate",
                "--model",
                str(checkpoint),
                "--prepared",
                str(tmp_path / "request"),
                "--backend",
                "polynomial",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["generated_ids"] == [6, 9]


def test_registration_rejects_collisions_and_ambiguous_detection(integration):
    _, entry = integration
    with pytest.raises(ValueError, match="already registered"):
        register_model(entry)
    with pytest.raises(ValueError, match="schema already registered"):
        register_model(
            ModelRegistration(
                "another", entry.implementation, entry.matches, entry.prepared_schemas
            )
        )
    isolated = ModelRegistry([entry])
    isolated.register(ModelRegistration("another", entry.implementation, entry.matches))
    with pytest.raises(ValueError, match="ambiguous"):
        isolated.detect({"model_type": "fixture-model"})
    with pytest.raises(ValueError, match="no registered model"):
        isolated.detect({"model_type": "unknown"})


@pytest.mark.parametrize("change", ["architecture", "schema", "profile"])
def test_prepared_dispatch_never_guesses_a_model(integration, tmp_path, change):
    model = load_model(integration[0])
    prepared = model.prepare(
        [1], max_new_tokens=1, output=tmp_path / "request", options={"offset": 1}
    )
    manifest = prepared.manifest()
    manifest[change] = "unregistered"
    write_json(prepared.path / "manifest.json", manifest)
    (prepared.path / "request.json").unlink()
    with pytest.raises(ValueError, match=r"differs|registered|unsupported"):
        load_prepared(prepared.path)


@pytest.mark.parametrize("options", [{"offest": 2}, {"offset": 2, "unknown": 3}, [2]])
def test_invalid_options_fail_before_output_creation(integration, tmp_path, options):
    with pytest.raises(ValueError, match=r"options|FixtureOptions"):
        load_model(integration[0]).prepare([1], output=tmp_path / "request", options=options)
    assert not (tmp_path / "request").exists()


def test_inspection_and_unsupported_configuration_need_no_model_dependencies(tmp_path):
    configs = {
        "mamba2": {"model_type": "mamba2"},
        "mamba3": {"ssm_cfg": {"layer": "Mamba3", "is_mimo": True}},
    }
    for name, config in configs.items():
        write_json(tmp_path / name / "config.json", config)
    code = """
import sys
from fhemamba import inspect_model, load_model
from fhemamba.cli import main
support = inspect_model(sys.argv[1])
assert support.backends['exact'].status == 'implemented'
assert support.profiles[0].security == 'not-set'
assert main(['inspect-model', '--model', sys.argv[1], '--json']) == 0
support = inspect_model(sys.argv[2])
assert support.backends['exact'].status == 'unsupported'
assert 'MIMO' in support.backends['exact'].reason
assert not support.profiles
try:
    load_model(sys.argv[2])
except ValueError as error:
    assert 'MIMO' in str(error)
else:
    raise AssertionError('unsupported model was loaded')
assert not {'torch', 'numpy', 'transformers'} & sys.modules.keys()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "mamba2"), str(tmp_path / "mamba3")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
