"""The shared API must keep Mamba-2's frozen surrogate and distinct FHE gates."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from fhemamba import load_model, load_prepared
from fhemamba.benchmarks.io import file_sha256, payload_sha256, write_json
from fhemamba.cli import main
from fhemamba.m1_payload import export_chain_payload
from fhemamba.normalization import plan_invsqrt
from fhemamba.payload_surrogate import public_gate_domains


@pytest.fixture(scope="module")
def mamba2_case(tmp_path_factory, model_factory):
    from types import SimpleNamespace

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("mamba2-api")
    checkpoint = root / "checkpoint"
    original = model_factory()
    original.save_pretrained(checkpoint)
    vocabulary = {f"token{i}": i for i in range(97)}
    for word, i in (("[UNK]", 0), ("The", 5), ("capital", 7)):
        del vocabulary[f"token{i}"]
        vocabulary[word] = i
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(checkpoint)

    class CalibrationTokenizer:
        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=torch.tensor([[ord(c) % 90 + 3 for c in text[:16]]]))

    normalization = root / "norms.json"
    write_json(
        normalization,
        {
            "format": "fhemamba-normalization-schedules-v1",
            "operators": [
                {"layer": layer, "site": site, "recipe": plan_invsqrt(1e-6, hi).recipe()}
                for layer, site, hi in (
                    (0, "rms_invsqrt", 50),
                    (0, "gated_rms_invsqrt", 100),
                    (1, "rms_invsqrt", 50),
                    (1, "gated_rms_invsqrt", 100),
                    (2, "rms_invsqrt", 200),
                )
            ],
        },
    )
    arrays = {}
    for layer, block in enumerate(original.backbone.layers):
        lo, hi = public_gate_domains(block)
        for name, value in {
            "lo": lo,
            "hi": hi,
            "p": np.full((1, len(lo)), 0.5),
            "q": np.full((1, len(lo)), 0.75),
            "rates": block.mixer.A_log.detach().exp().double().numpy(),
        }.items():
            arrays[f"layer_{layer}_{name}"] = value
    arrays["manifest"] = np.array(
        json.dumps(
            {
                "format": "dissipative-mamba2-gates-v1",
                "layers": [0, 1],
                "input_payload_sha256": "0" * 64,
            }
        )
    )
    np.savez(root / "gates.npz", **arrays)
    source = export_chain_payload(
        original,
        CalibrationTokenizer(),
        root / "base",
        n_test_tokens=2,
        cal_tokens=16,
        normalization_bundle=normalization,
        stabilized_gate_bundle=root / "gates.npz",
    )
    model = load_model(checkpoint)
    before = payload_sha256(source)
    prepared = model.prepare(
        [5, 7],
        max_new_tokens=2,
        output=root / "prepared",
        base_chain=source,
        profile="mamba2-experimental",
    )
    assert payload_sha256(source) == before
    return root, model, original, source, prepared


def test_model_detection_and_exact_generation_match_hf(mamba2_case):
    _, model, original, _, _ = mamba2_case
    assert model.architecture == "mamba2"
    ids = torch.tensor([[5, 7]])
    with torch.no_grad():
        expected = []
        for _ in range(2):
            token = int(original(ids, use_cache=False).logits[0, -1].argmax())
            expected.append(token)
            ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
    for _ in range(2):
        result = model.generate([5, 7], max_new_tokens=2)
        assert result.generated_ids == expected
        assert result.passed
        assert not result.encrypted


def test_prepare_preserves_frozen_coefficients_and_runs_polynomials(mamba2_case):
    _, model, _, source, prepared = mamba2_case
    original = json.loads((source / "chain.json").read_text())
    for name in original["layer_dirs"]:
        before = json.loads((source / name / "meta.json").read_text())
        after = json.loads((prepared.path / "payload" / name / "meta.json").read_text())
        for key in ("polys", "joint_gates", "carried_bounds"):
            assert before[key] == after[key]
    result = model.generate(
        [5, 7],
        max_new_tokens=2,
        backend="polynomial",
        prepared=load_prepared(prepared.path),
    )
    assert result.passed
    assert result.generated_ids == prepared.manifest()["polynomial_token_ids"]
    assert result.report["max_abs_error_vs_polynomial"] <= 0.05
    assert "matches_exact_tokens" not in result.report["checks"]
    assert prepared.profile == "mamba2-experimental"


@pytest.mark.parametrize("change", ["input", "length", "weight", "manifest"])
def test_prepared_identity_is_enforced(mamba2_case, tmp_path, change):
    _, model, _, _, source = mamba2_case
    shutil.copytree(source.path, tmp_path / "request")
    prepared = load_prepared(tmp_path / "request")
    ids, length = [5, 7], 2
    if change == "input":
        ids = [7, 5]
    elif change == "length":
        length = 3
    else:
        path = (
            prepared.path / "manifest.json"
            if change == "manifest"
            else prepared.path / "payload/final_norm_w.bin"
        )
        with path.open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(ValueError, match=r"differ|changed"):
        model.generate(ids, max_new_tokens=length, backend="ckks", prepared=prepared)


def test_mamba2_preparation_requires_explicit_frozen_source_and_security(mamba2_case, tmp_path):
    model = mamba2_case[1]
    with pytest.raises(ValueError, match="explicit profile"):
        model.prepare([5, 7], output=tmp_path / "implicit")
    with pytest.raises(ValueError, match="base_chain"):
        model.prepare([5, 7], output=tmp_path / "missing", profile="mamba2-experimental")
    assert not (tmp_path / "implicit").exists()
    assert not (tmp_path / "missing").exists()


def test_cli_detects_mamba2_and_tokenizer_at_checkpoint_root(mamba2_case, capsys):
    _, model, _, _, _ = mamba2_case
    assert (
        main(
            [
                "generate",
                "--model",
                str(model.checkpoint),
                "--prompt",
                "The capital",
                "--max-new-tokens",
                "2",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["input_ids"] == [5, 7]
    assert result["report"]["architecture"] == "mamba2"
    assert result["generated_ids"] == model.generate([5, 7], max_new_tokens=2).generated_ids


def test_cli_prepare_and_polynomial_round_trip(mamba2_case, tmp_path, capsys):
    _, model, _, source, _ = mamba2_case
    assert (
        main(
            [
                "prepare",
                "--model",
                str(model.checkpoint),
                "--prompt",
                "The capital",
                "--base-chain",
                str(source),
                "--profile",
                "mamba2-experimental",
                "--max-new-tokens",
                "2",
                "--output",
                str(tmp_path / "request"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["profile"] == "mamba2-experimental"
    assert (
        main(
            [
                "generate",
                "--model",
                str(model.checkpoint),
                "--prepared",
                str(tmp_path / "request"),
                "--backend",
                "polynomial",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["passed"]


@pytest.mark.parametrize(
    "defect", [None, "tokens", "error", "sync", "security", "missing", "truncated", "parameters"]
)
def test_native_adapter_keeps_mamba2_gates_and_failed_output(mamba2_case, tmp_path, defect):
    prepared = mamba2_case[4]
    manifest = prepared.manifest()
    ids = manifest["polynomial_token_ids"]
    native = {
        "version": "0.5.0",
        "repo_commit": "working-tree",
        "backend": "fideslib",
        "stage": "stage1-mamba2-decode",
        "status": "passed",
        "passed": True,
        "encrypted": True,
        "config": {},
        "parameters": {
            "n_layers_loaded": 2,
            "tokens": 3,
            "autoregressive_client_loop": True,
            "autoregressive_prompt_tokens": 2,
            "autoregressive_generate_tokens": 2,
            "final_norm_applied": True,
            "security": "not-set",
            "fideslib_sync_profile": "full",
        },
        "measurements": {
            "autoregressive_selected_ids": ids,
            "autoregressive_expected_ids": ids,
            "autoregressive_tokens_match": True,
            "max_abs_error": 0.001,
            "per_token_max_abs_error": [0.001] * 3,
            # A different exact-model trajectory is valid for this frozen surrogate.
            "per_token_max_abs_error_vs_exact": [1.0] * 3,
            "per_token_decrypt_ok": [True] * 3,
            "peak_rss_gib": 1,
        },
        "measurement_scope": {
            "fideslib_encrypted_execution": True,
            "full_layer_chain": True,
            "multi_token_ciphertext_state_carry": True,
            "ciphertext_conv_fifo": True,
            "per_token_fresh_embedding_encryption": True,
            "client_output_decrypts_are_protocol_boundary": True,
            "zero_intermediate_decrypts": True,
            "autoregressive_client_output_decrypt_count": 2,
        },
        "operation_counts": {},
        "timing": {"evaluation_seconds": 1},
    }
    if defect == "tokens":
        native["measurements"]["autoregressive_selected_ids"] = [96, 95]
    elif defect == "error":
        native["measurements"]["per_token_max_abs_error"][-1] = 0.051
    elif defect == "sync":
        native["parameters"]["fideslib_sync_profile"] = "unspecified"
    elif defect == "security":
        native["parameters"]["security"] = "128-classic"
    elif defect == "parameters":
        native["parameters"] = None
    binary = tmp_path / "native"
    binary.write_text(
        "#!/usr/bin/env python3\nimport json, sys\nfrom pathlib import Path\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "assert args['--security'] == 'not-set'\n"
        "assert args['--ring-dim'] == '65536' and args['--meta-bts-alpha'] == '12'\n"
        "assert args['--debug-decrypt'] == '0' and args['--autoregressive-client-loop'] == '1'\n"
        f"native = json.loads({json.dumps(native)!r})\n"
        "native['binary_sha256'] = args['--binary-sha256']\n"
        + (
            ""
            if defect == "missing"
            else "Path(args['--output-json']).write_text("
            + ("'{'" if defect == "truncated" else "json.dumps(native)")
            + ")\n"
        )
    )
    binary.chmod(0o755)
    # The wheel-side execution path must not require a source checkout or Torch.
    code = """
import sys
from fhemamba import load_prepared
request = load_prepared(sys.argv[1])
result = request.generate(binary=sys.argv[2], output=sys.argv[3], timeout=10)
assert not {'torch', 'transformers'} & sys.modules.keys()
import json
print(json.dumps(result.to_dict()))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(prepared.path), str(binary), str(tmp_path / "run")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["passed"] is (defect is None)
    assert result["generated_ids"] == (
        []
        if defect in ("missing", "truncated")
        else native["measurements"]["autoregressive_selected_ids"]
    )
    assert result["report"]["security"] == "not-set"
    assert result == json.loads((tmp_path / "run/generation.json").read_text())
    if defect not in ("missing", "truncated"):
        raw = json.loads((tmp_path / "run/native.json").read_text())
        assert "input_payload_sha256" not in raw  # Original evidence is never rewritten.


def test_checkpoint_hash_includes_safetensor_shards(mamba2_case, tmp_path):
    from fhemamba.checkpoints import checkpoint_identity

    original = mamba2_case[2]
    original.save_pretrained(tmp_path / "shards", max_shard_size="10KB")
    identity = checkpoint_identity(tmp_path / "shards", "mamba2")
    index = json.loads((tmp_path / "shards/model.safetensors.index.json").read_text())
    assert set(identity) == {
        "config.json",
        "model.safetensors.index.json",
        *index["weight_map"].values(),
    }
    for name, digest in identity.items():
        assert digest == file_sha256(tmp_path / "shards" / name)


def test_typed_preparation_options_keep_the_same_payload(mamba2_case, tmp_path):
    from fhemamba.models.mamba2 import Mamba2Preparation

    _, model, _, source, previous = mamba2_case
    prepared = model.prepare(
        [5, 7],
        max_new_tokens=2,
        output=tmp_path / "typed",
        profile="mamba2-experimental",
        options=Mamba2Preparation(base_chain=source),
    )
    assert prepared.manifest() == previous.manifest()
    assert payload_sha256(prepared.path / "payload") == payload_sha256(previous.path / "payload")
    with pytest.raises(ValueError, match="choose options or base_chain"):
        model.prepare(
            [5, 7],
            output=tmp_path / "invalid",
            profile="mamba2-experimental",
            options=Mamba2Preparation(source),
            base_chain=source,
        )
    assert not (tmp_path / "invalid").exists()
