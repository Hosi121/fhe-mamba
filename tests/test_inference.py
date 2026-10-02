"""Public request lifecycle with real CPU arithmetic and synthetic native reports."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from fhemamba import load_model, load_prepared
from fhemamba.benchmarks.io import file_sha256, write_json
from fhemamba.cli import main
from fhemamba.mamba3_lm import Mamba3LM


class LocalTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        # Decoded text intentionally does not round-trip. Raw IDs must survive.
        return [5, 7] if text != "decoded" else [1]

    def decode(self, ids, **kwargs):
        return "decoded"


@pytest.fixture(scope="module")
def request_case(tmp_path_factory):
    from transformers import AutoTokenizer

    from fhemamba.workloads import mamba3_export as exporter

    root = tmp_path_factory.mktemp("generation")
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    config = {
        "d_model": 8,
        "d_intermediate": 128,
        "n_layer": 2,
        "vocab_size": 32,
        "ssm_cfg": {"layer": "Mamba3", "d_state": 8, "headdim": 8},
        "rms_norm": True,
        "tie_embeddings": True,
    }
    torch.manual_seed(94)
    original = Mamba3LM(config).eval()
    write_json(checkpoint / "config.json", config)
    torch.save(original.state_dict(), checkpoint / "pytorch_model.bin")
    pins = {
        "repo_id": exporter.MODEL_ID,
        "revision": exporter.MODEL_REVISION,
        "files_sha256": {
            name: file_sha256(checkpoint / name) for name in ("config.json", "pytorch_model.bin")
        },
    }
    write_json(root / "config/mamba3-reproduction.json", {"checkpoint": pins})
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    write_json(tokenizer / "tokenizer_config.json", {"test_fixture": True})
    model = load_model(checkpoint)
    # Only checkpoint identity and tokenization are fixtures. Calibration,
    # generation, domain checks, packing and all hashes use production code.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exporter, "repository_root", lambda: root)
        patch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: LocalTokenizer())
        prepared = model.prepare(
            [5, 7], max_new_tokens=2, output=root / "request", tokenizer=tokenizer
        )
        exporter.export(
            checkpoint,
            tokenizer,
            root / "text-export",
            prompt="The capital",
            new_tokens=2,
            state_representation="tiled",
            rotary="phasor",
            calibration_tokens=65,
            normalization_margin=1,
            positive_lower_fraction=0.25,
        )
    return root, model, prepared, original


def test_exact_api_preserves_tokens_and_starts_with_fresh_state(request_case):
    _, model, _, original = request_case
    expected = original.generate([5, 7], 2, factored=False)[0]
    for _ in range(2):
        result = model.generate(torch.tensor([5, 7]), max_new_tokens=2)
        assert result.input_ids == [5, 7]
        assert result.generated_ids == expected
        assert result.passed
        assert not result.encrypted
        assert result.backend == "exact"
        assert result.stop_reason == "length"
        json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("ids", "length"),
    [
        ([], 2),
        ([[5, 7]], 2),
        ([True], 2),
        ([-1], 2),
        ([32], 2),
        ([1.5], 2),
        ([5, 7], 0),
        ([5, 7], True),
        ([5, 7], 1.5),
    ],
)
def test_bad_requests_are_rejected(request_case, ids, length):
    with pytest.raises(ValueError, match=r"input_ids|max_new_tokens"):
        request_case[1].generate(ids, max_new_tokens=length)


def test_preparation_preserves_legacy_payload_and_polynomial_execution(request_case):
    root, model, prepared, _ = request_case
    for name in ("program.txt", "fixture.npz", "client_head.f32"):
        assert file_sha256(prepared.path / name) == file_sha256(root / "text-export" / name)
    restored = load_prepared(prepared.path)
    result = model.generate([5, 7], max_new_tokens=2, backend="polynomial", prepared=restored)
    assert result.passed
    assert not result.encrypted
    assert result.generated_ids == model.generate([5, 7], max_new_tokens=2).generated_ids
    assert result.report["max_abs_error_vs_exact"] <= 0.001
    assert restored.manifest()["prompt_ids"] == [5, 7]


def test_prepared_polynomial_is_evaluated_not_replayed(request_case, tmp_path):
    _, model, prepared, _ = request_case
    path = tmp_path / "changed"
    shutil.copytree(prepared.path, path)
    manifest = prepared.manifest()
    first = next(iter(manifest["polynomials"].values()))
    first.update(lo=1e9, hi=1e9 + 1)
    write_json(path / "manifest.json", manifest)
    (path / "request.json").unlink()  # Also exercise loading a legacy export.
    altered = load_prepared(path)
    with pytest.raises(ValueError, match="outside frozen"):
        model.generate([5, 7], max_new_tokens=2, backend="polynomial", prepared=altered)


@pytest.mark.parametrize(("ids", "length"), [([7, 5], 2), ([5, 7], 3)])
def test_prepared_request_cannot_be_reused_for_other_input(request_case, ids, length):
    _, model, prepared, _ = request_case
    with pytest.raises(ValueError, match="differ from the prepared request"):
        model.generate(ids, max_new_tokens=length, backend="ckks", prepared=prepared)


@pytest.mark.parametrize("name", ["manifest.json", "program.txt"])
def test_loaded_request_detects_mutation_before_execution(request_case, tmp_path, name):
    path = tmp_path / "changed"
    shutil.copytree(request_case[2].path, path)
    prepared = load_prepared(path)
    with (path / name).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match=r"changed|digest differs"):
        prepared.generate(binary=tmp_path / "must-not-run", output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_missing_preparation_never_falls_back_to_exact(request_case):
    model = request_case[1]
    for backend in ("polynomial", "ckks"):
        with pytest.raises(ValueError, match=r"requires model\.prepare"):
            model.generate([5, 7], backend=backend)


@pytest.mark.parametrize(
    "defect", [None, "wrong_tokens", "per_output_error", "missing_result", "truncated_result"]
)
def test_ckks_result_uses_actual_tokens_and_shared_gates(request_case, tmp_path, defect, capsys):
    prepared = request_case[2]
    manifest = prepared.manifest()
    baseline = Path(__file__).resolve().parents[1] / (
        "results/b300/2026-09-28/public-state-reuse/measurements/full-b1/native.json"
    )
    native = json.loads(baseline.read_text())
    native.update(
        generated_token_ids=list(manifest["exact_token_ids"]),
        client_output_decrypt_count=2,
        per_output_errors=[{"exact": 1e-5, "polynomial": 1e-5}] * manifest["tokens"],
    )
    if defect == "wrong_tokens":
        native["generated_token_ids"] = [1, 2]
    elif defect == "per_output_error":
        native["per_output_errors"][-1] = {"exact": 0.01, "polynomial": 1e-5}
    binary = tmp_path / "native"
    contents = "{" if defect == "truncated_result" else json.dumps(native)
    binary.write_text(
        "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
        "assert sys.argv[sys.argv.index('--security') + 1] == '128-classic'\n"
        "assert sys.argv[sys.argv.index('--frontier-live-limit') + 1] == '256'\n"
        "assert '--s2c-first' in sys.argv and '--merge-refresh-correction' in sys.argv\n"
        + ("" if defect == "missing_result" else f"Path(sys.argv[2]).write_text({contents!r})\n")
    )
    binary.chmod(0o755)
    output = tmp_path / "run"
    exit_code = main(
        [
            "generate",
            "--prepared",
            str(prepared.path),
            "--backend",
            "ckks",
            "--binary",
            str(binary),
            "--output",
            str(output),
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert exit_code == (0 if defect is None else 1)
    assert result["passed"] is (defect is None)
    assert result["generated_ids"] == (
        [] if defect in ("missing_result", "truncated_result") else native["generated_token_ids"]
    )
    assert result == json.loads((output / "generation.json").read_text())
    if defect is not None:
        assert result["report"]["error"]


def test_cli_raw_stdin_works_outside_checkout(request_case, tmp_path):
    _, model, _, _ = request_case
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhemamba",
            "generate",
            "--model",
            str(model.checkpoint),
            "--input-ids",
            "-",
            "--max-new-tokens",
            "2",
            "--json",
        ],
        input="[5, 7]",
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["input_ids"] == [5, 7]
    assert result["generated_ids"] == model.generate([5, 7], max_new_tokens=2).generated_ids
    assert result["passed"]
    assert not result["encrypted"]


@pytest.mark.parametrize("source", ["--prompt", "--prompt-file", "--input-ids"])
def test_cli_inputs_have_one_generation_path(request_case, tmp_path, monkeypatch, capsys, source):
    from fhemamba import inference_cli

    monkeypatch.setattr(inference_cli, "_tokenizer", lambda *args: LocalTokenizer())
    value = "The capital"
    if source != "--prompt":
        path = tmp_path / "input"
        path.write_text("[5, 7]" if source == "--input-ids" else value)
        value = str(path)
    code = main(
        [
            "generate",
            "--model",
            str(request_case[1].checkpoint),
            source,
            value,
            "--max-new-tokens",
            "2",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["input_ids"] == [5, 7]
    assert result["generated_text"] == "decoded"


def test_prepare_cli_writes_reloadable_request_and_keeps_stdout_json(
    request_case, tmp_path, monkeypatch, capsys
):
    from transformers import AutoTokenizer

    from fhemamba.workloads import mamba3_export as exporter

    root, model, _, _ = request_case
    monkeypatch.setattr(exporter, "repository_root", lambda: root)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: LocalTokenizer())
    ids = tmp_path / "tokens.json"
    ids.write_text("[5, 7]")
    output = tmp_path / "request"
    code = main(
        [
            "prepare",
            "--model",
            str(model.checkpoint),
            "--input-ids",
            str(ids),
            "--tokenizer",
            str(root / "tokenizer"),
            "--max-new-tokens",
            "2",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert json.loads(captured.out)["input_ids"] == [5, 7]
    assert "compiled_step=" in captured.err
    assert load_prepared(output).manifest()["generated_tokens"] == 2


def test_raw_cli_does_not_need_transformers_for_an_automatic_tokenizer(tmp_path, monkeypatch):
    from argparse import Namespace

    from fhemamba.inference_cli import _tokenizer

    (tmp_path / "tokenizer").mkdir()
    monkeypatch.setitem(sys.modules, "transformers", None)
    args = Namespace(model=tmp_path, tokenizer=None, prompt=None, prompt_file=None, backend="exact")
    assert _tokenizer(args) is None


def test_prepared_request_requires_original_checkpoint(request_case, tmp_path):
    _, model, prepared, _ = request_case
    shutil.copytree(model.checkpoint, tmp_path / "model")
    config_path = tmp_path / "model/config.json"
    config_path.write_text(config_path.read_text() + "\n")
    other = load_model(tmp_path / "model")
    with pytest.raises(ValueError, match="checkpoint differs"):
        other.generate([5, 7], max_new_tokens=2, backend="polynomial", prepared=prepared)
