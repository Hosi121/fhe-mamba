"""Mamba-1 lowering parity, frozen requests and the native client protocol."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fhemamba import load_model, load_prepared
from fhemamba.benchmarks.io import write_json
from fhemamba.cli import main
from fhemamba.mamba1 import Mamba1LM
from fhemamba.models.mamba1 import Mamba1Preparation
from fhemamba.packed_program import Calibration, PackedProgram
from fhemamba.workloads.mamba1_export import calibrate


@pytest.mark.parametrize(
    "geometry",
    [
        {},
        {"conv_kernel": 1, "use_conv_bias": False},
        {"state_size": 12, "hidden_size": 24, "use_bias": True, "num_hidden_layers": 3},
    ],
)
def test_lowering_matches_upstream_with_tiled_selective_state(model_factory, geometry):
    model = model_factory(architecture=1, **geometry)
    lowered = Mamba1LM(model)
    ids = torch.tensor([[5, 7, 11]])
    expected_ids, expected_hidden = [], []
    with torch.no_grad():
        for step in range(5):
            output = model(ids[:, : step + 1], use_cache=False, output_hidden_states=True)
            expected_hidden.append(output.hidden_states[-1][:, -1])
            if step >= 2:
                token = int(output.logits[0, -1].argmax())
                expected_ids.append(token)
                ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
    for _ in range(2):
        generated, trace = lowered.generate([5, 7, 11], 3, slots=128)
        assert generated == expected_ids
        torch.testing.assert_close(
            torch.cat(trace), torch.cat(expected_hidden).double(), atol=2e-6, rtol=2e-6
        )
    assert len(lowered.initial_states(slots=128)[0].ssm) > 1
    assert next(model.parameters()).dtype == torch.float32


def test_packed_lowering_preserves_all_state_and_has_real_feedback(model_factory):
    model = Mamba1LM(model_factory(architecture=1, state_size=12, use_bias=True))
    calibration = Calibration()
    model.generate(list(range(1, 97)), 3, calibration, slots=256)
    poly, _ = calibration.fit(max_degree=1023, quadrature=True, margin=1)
    ids, trace = model.generate([5, 7], 3, poly, slots=256)
    program = PackedProgram(poly.polynomials, slots=256, bound=1e8, extended=True)
    states = model.initial_states(slots=256, program=program)
    assert sum(block.value.size for block in states[0].ssm) == 64 * 12
    hidden = None
    for step, token in enumerate([5, 7, *ids[:-1]]):
        embedding = model.embedding[token][None]
        value = program.input(embedding) if step < 2 else program.client_input(hidden, embedding)
        hidden, states = model.step(value, states, program)
        np.testing.assert_allclose(hidden.value, trace[step].numpy(), atol=1e-9, rtol=1e-9)
    feedback = [node for node in program.nodes if node[0] == "feedback"]
    assert len(feedback) == 2
    assert all(len(node[3]) == 0 for node in feedback)
    assert all(node[2] <= 256 for node in program.nodes)


def test_large_negative_decay_uses_frozen_resquaring_without_clipping(model_factory):
    original = model_factory(architecture=1)
    with torch.no_grad():
        for block in original.backbone.layers:
            block.mixer.dt_proj.bias.fill_(8)
            block.mixer.A_log.add_(8)
            block.mixer.A_log[0].fill_(-14)
    model = Mamba1LM(original)
    expected_ids, expected = model.generate([5, 7], 3)
    poly, _ = calibrate(model, [list(range(1, 97))], 3)
    assert min(model.decay_squarings) > 10
    actual_ids, actual = model.generate([5, 7], 3, poly)
    assert actual_ids == expected_ids
    torch.testing.assert_close(torch.cat(actual), torch.cat(expected), atol=1e-5, rtol=1e-5)
    program = PackedProgram(poly.polynomials, slots=256, bound=1e8, extended=True)
    states = model.initial_states(slots=256, program=program)
    for step, token in enumerate([5, 7, *actual_ids[:-1]]):
        hidden, states = model.step(program.input(model.embedding[token][None]), states, program)
        np.testing.assert_allclose(hidden.value, actual[step].numpy(), atol=1e-8, rtol=1e-8)


@pytest.fixture(scope="module")
def mamba1_request(tmp_path_factory, model_factory):
    root = tmp_path_factory.mktemp("mamba1-packed")
    checkpoint = root / "checkpoint"
    original = model_factory(architecture=1, state_size=12, use_bias=True)
    original.save_pretrained(checkpoint)
    model = load_model(checkpoint)
    options = Mamba1Preparation(
        calibration_input_ids=(tuple(range(1, 97)),), calibration_new_tokens=3
    )
    prepared = model.prepare(
        [5, 7],
        max_new_tokens=3,
        output=root / "request",
        profile="mamba1-experimental",
        options=options,
    )
    return model, prepared, options


def test_prepared_polynomial_evaluates_the_full_model(mamba1_request):
    model, prepared, _ = mamba1_request
    restored = load_prepared(prepared.path)
    result = model.generate([5, 7], max_new_tokens=3, backend="polynomial", prepared=restored)
    exact = model.generate([5, 7], max_new_tokens=3)
    assert result.passed
    assert not result.encrypted
    assert result.generated_ids == exact.generated_ids
    assert result.report["max_abs_error_vs_exact"] <= 0.001
    assert result.report["max_abs_error_vs_polynomial"] <= 1e-12
    manifest = restored.manifest()
    assert manifest["profile"] == "mamba1-experimental"
    assert manifest["security"] == "128-classic"
    assert manifest["client_protocol"] == "packed-greedy-v1"
    assert manifest["program"]["operations"]["feedback"] == 2
    assert manifest["program"]["operations"]["linear_ref"] > 0
    assert "torch" not in json.dumps(manifest)


def test_native_parser_and_depth_planner_accept_export(mamba1_request, tmp_path):
    root = Path(__file__).resolve().parents[1] / "native/fideslib_stage0"
    binary = tmp_path / "inventory"
    subprocess.run(
        [
            "c++",
            "-std=c++20",
            "-O0",
            "-I",
            str(root / "include"),
            str(root / "src/packed_resource_inventory.cpp"),
            str(root / "src/stage1_mamba2_plan.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [str(binary), str(mamba1_request[1].path / "program.txt")],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = json.loads(result.stdout)
    assert inventory["encrypted"] is False
    assert inventory["public_weights"] > 0
    assert inventory["ciphertext_slots"] == 32768


@pytest.mark.parametrize("name", ["program.txt", "client_head.f32", "fixture.npz", "manifest.json"])
def test_request_mutation_is_rejected_before_native_execution(mamba1_request, tmp_path, name):
    path = tmp_path / "changed"
    shutil.copytree(mamba1_request[1].path, path)
    prepared = load_prepared(path)
    changed = path / name
    changed.chmod(0o644)
    with changed.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match=r"changed|digest differs"):
        prepared.generate(binary=tmp_path / "must-not-run", output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_polynomial_coefficients_are_executed_and_domains_enforced(mamba1_request, tmp_path):
    model, prepared, _ = mamba1_request
    path = tmp_path / "changed"
    shutil.copytree(prepared.path, path)
    manifest = prepared.manifest()
    next(iter(manifest["polynomials"].values())).update(lo=1e9, hi=1e9 + 1)
    write_json(path / "manifest.json", manifest)
    (path / "request.json").unlink()
    altered = load_prepared(path)
    with pytest.raises(ValueError, match="outside frozen polynomial domain"):
        model.generate([5, 7], max_new_tokens=3, backend="polynomial", prepared=altered)


@pytest.mark.parametrize(
    "changes",
    [
        {"complete_backbone": False},
        {"client_protocol": "reference-replay"},
        {"tokens": 1},
        {"polynomial_hidden_max_abs_error": 0.002},
        {"polynomial_token_ids": [1, 2, 3]},
        {"calibration_input_ids": [[5, 7]]},
        {"decay_squarings": [-1, 0]},
    ],
)
def test_inconsistent_prepared_metadata_is_rejected(mamba1_request, tmp_path, changes):
    path = tmp_path / "changed"
    shutil.copytree(mamba1_request[1].path, path)
    manifest = mamba1_request[1].manifest()
    write_json(path / "manifest.json", {**manifest, **changes})
    (path / "request.json").unlink()
    with pytest.raises(ValueError, match=r"requires a complete|do not match|error gate|squaring"):
        load_prepared(path)


def test_default_calibration_uses_local_tokenizer(mamba1_request, tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    from fhemamba.workloads.mamba1_export import CALIBRATION_PROMPTS

    seen = []

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            seen.append(text)
            return list(range(1, 97))

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    prepared = mamba1_request[0].prepare(
        [5, 7],
        max_new_tokens=3,
        output=tmp_path / "request",
        tokenizer=tmp_path,
        profile="mamba1-experimental",
    )
    assert seen == list(CALIBRATION_PROMPTS)
    assert prepared.manifest()["calibration_generated_tokens"] == 16


@pytest.mark.parametrize(("ids", "length"), [([7, 5], 3), ([5, 7], 2)])
def test_request_identity_is_checked(mamba1_request, ids, length):
    model, prepared, _ = mamba1_request
    with pytest.raises(ValueError, match="differ from the prepared request"):
        model.generate(ids, max_new_tokens=length, backend="ckks", prepared=prepared)


@pytest.mark.parametrize(
    "options",
    [
        {"calibration_input_ids": []},
        {"calibration_input_ids": [[True]]},
        {"calibration_input_ids": [[97]]},
        {"calibration_input_ids": [[5, 7]]},
        {"calibration_new_tokens": True},
        {"calibration_new_tokens": 65},
        {"calibration_new_tokens": 2},
        {"unknown": True},
    ],
)
def test_invalid_calibration_leaves_no_output(mamba1_request, tmp_path, options):
    model, _, valid = mamba1_request
    value = {
        "calibration_input_ids": valid.calibration_input_ids,
        "calibration_new_tokens": valid.calibration_new_tokens,
        **options,
    }
    with pytest.raises(
        ValueError, match=r"calibration|max_new_tokens|input_ids|prepared requests|invalid Mamba1"
    ):
        model.prepare(
            [5, 7],
            max_new_tokens=3,
            output=tmp_path / "request",
            profile="mamba1-experimental",
            options=value,
        )
    assert not (tmp_path / "request").exists()


@pytest.mark.parametrize(
    "defect",
    [None, "wrong_tokens", "error", "security", "decrypt", "plaintext", "missing", "truncated"],
)
def test_native_adapter_checks_real_returned_tokens_and_protocol(
    mamba1_request, tmp_path, defect, capsys
):
    prepared = mamba1_request[1]
    manifest = prepared.manifest()
    baseline = Path(__file__).resolve().parents[1] / (
        "results/b300/2026-09-28/public-state-reuse/measurements/full-b1/native.json"
    )
    # A protocol fixture, not an encrypted execution or Mamba-1 qualification.
    native = json.loads(baseline.read_text())
    native.update(
        generated_token_ids=manifest["exact_token_ids"],
        client_output_decrypt_count=3,
        per_output_errors=[{"exact": 1e-5, "polynomial": 1e-5}] * manifest["tokens"],
    )
    if defect == "wrong_tokens":
        native["generated_token_ids"] = [1, 2, 3]
    elif defect == "error":
        native["per_output_errors"][-1] = {"exact": 0.01, "polynomial": 1e-5}
    elif defect == "security":
        native["security"] = "not-set"
    elif defect == "decrypt":
        native["evaluation_decryptions"] = 1
    elif defect == "plaintext":
        native["encrypted"] = False
    contents = "{" if defect == "truncated" else json.dumps(native)
    binary = tmp_path / "native"
    binary.write_text(
        "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
        "assert '--client-head' in sys.argv\n"
        "assert sys.argv[sys.argv.index('--security') + 1] == '128-classic'\n"
        "assert '--s2c-first' in sys.argv and '--merge-refresh-correction' in sys.argv\n"
        + ("" if defect == "missing" else f"Path(sys.argv[2]).write_text({contents!r})\n")
    )
    binary.chmod(0o755)
    output = tmp_path / "run"
    code = main(
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
    assert code == (0 if defect is None else 1)
    assert result["passed"] is (defect is None)
    assert result["generated_ids"] == (
        [] if defect in ("missing", "truncated") else native["generated_token_ids"]
    )
    assert result["report"]["profile"] == "mamba1-experimental"
    assert result == json.loads((output / "generation.json").read_text())
    if defect is not None:
        assert result["report"]["error"]


def test_cli_prepares_from_ids_outside_checkout(mamba1_request, tmp_path):
    model, _, options = mamba1_request
    settings = tmp_path / "options.json"
    write_json(
        settings,
        {
            "calibration_input_ids": options.calibration_input_ids,
            "calibration_new_tokens": options.calibration_new_tokens,
        },
    )
    output = tmp_path / "request"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhemamba",
            "prepare",
            "--model",
            str(model.checkpoint),
            "--input-ids",
            "-",
            "--max-new-tokens",
            "3",
            "--profile",
            "mamba1-experimental",
            "--prepare-options",
            str(settings),
            "--output",
            str(output),
        ],
        input="[5, 7]",
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["input_ids"] == [5, 7]
    assert (
        load_prepared(output).manifest()["polynomial_token_ids"]
        == (mamba1_request[1].manifest()["polynomial_token_ids"])
    )
    code = """
import sys
from fhemamba import load_prepared
assert load_prepared(sys.argv[1]).profile == 'mamba1-experimental'
assert not {'torch', 'numpy', 'transformers'} & sys.modules.keys()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(output)], cwd=tmp_path, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
