"""Scheduled norms must reach both native payloads and their actual references."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from fhemamba.generation import prepare_generation
from fhemamba.m1_payload import _normalization_specs, _poly_ops_from_export, export_chain_payload
from fhemamba.normalization import ScheduledInvSqrt, plan_invsqrt
from fhemamba.payload_surrogate import public_gate_domains, stabilized_specs

transformers = pytest.importorskip("transformers")


@pytest.fixture
def scheduled_model():
    torch.manual_seed(19)
    model = (
        transformers.Mamba2ForCausalLM(
            transformers.Mamba2Config(
                vocab_size=97,
                hidden_size=32,
                expand=2,
                num_heads=4,
                head_dim=16,
                state_size=8,
                n_groups=1,
                num_hidden_layers=2,
                conv_kernel=4,
                chunk_size=8,
            )
        )
        .float()
        .eval()
    )
    bundle = {
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
    }
    return model, bundle


@pytest.mark.parametrize("joint", [False, True])
def test_scheduled_payload_uses_dedicated_final_norm(tmp_path, scheduled_model, joint):
    model, bundle = scheduled_model
    bundle_path = tmp_path / "schedules.json"
    bundle_path.write_text(json.dumps(bundle))
    gate_path = write_gate_bundle(tmp_path, model) if joint else None

    class Tokenizer:
        def __call__(self, text, return_tensors=None):
            class Result:
                input_ids = torch.tensor([[(ord(c) % 90) + 3 for c in text[:16]]])

            return Result()

    out = export_chain_payload(
        model,
        Tokenizer(),
        tmp_path / "payload",
        n_test_tokens=2,
        cal_tokens=16,
        normalization_bundle=bundle_path,
        stabilized_gate_bundle=gate_path,
        autoregressive_prompt_tokens=2,
        autoregressive_generate_tokens=2,
    )
    chain = json.loads((out / "chain.json").read_text())
    ops = _poly_ops_from_export(out, 2)
    assert len(ops.joint_gates) == (2 if joint else 0)
    if joint:
        assert chain["format"] == "fhemamba-m2-chain-joint-v1"
        # The distinct constant joint circuit must survive reload and replace
        # both independently fitted nonlinearities, including reference export.
        z = torch.zeros(1, 1, 4)
        write, decay = ops.mamba2_gates(
            z, -model.backbone.layers[0].mixer.A_log.exp(), 0, (0.0, float("inf"))
        )
        torch.testing.assert_close(write, torch.full_like(z, 0.140625))
        torch.testing.assert_close(decay, torch.full_like(z, 0.75))
    assert chain["final_norm_poly"] == bundle["operators"][-1]["recipe"]
    assert ops.layer_polys[(2, "rms_invsqrt")].hi == 200
    assert ops.layer_polys[(1, "rms_invsqrt")].hi == 50
    for entry in bundle["operators"]:
        operator = ops.layer_polys[(entry["layer"], entry["site"])]
        assert isinstance(operator, ScheduledInvSqrt)
        assert operator.recipe() == entry["recipe"]
    for name, shape in chain["tensors"].items():
        values = np.fromfile(out / f"{name}.bin", dtype="<f4").reshape(shape)
        assert np.isfinite(values).all(), name
    with pytest.raises(ValueError, match="new output directory"):
        export_chain_payload(model, Tokenizer(), out, normalization_bundle=bundle_path)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "uncertified", "epsilon"])
def test_bundle_validation_before_payload_write(tmp_path, scheduled_model, defect):
    model, original = scheduled_model
    bundle = copy.deepcopy(original)
    if defect == "missing":
        bundle["operators"].pop()
    elif defect == "duplicate":
        bundle["operators"].append(bundle["operators"][0])
    elif defect == "uncertified":
        bundle["operators"][0]["recipe"]["coefficients"] = [[1.5, 0.5]]
    else:
        bundle["operators"][0]["recipe"] = plan_invsqrt(0.01, 50).recipe()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="normalization"):
        _normalization_specs(model, path)


def write_gate_bundle(tmp_path, model, defect=None):
    arrays = {}
    for layer, block in enumerate(model.backbone.layers):
        lo, hi = public_gate_domains(block)
        for name, value in {
            "lo": lo,
            "hi": hi,
            "p": np.full((1, len(lo)), 0.5),
            "q": np.full((1, len(lo)), 0.75),
            "rates": block.mixer.A_log.detach().exp().double().numpy(),
        }.items():
            arrays[f"layer_{layer}_{name}"] = value
    if defect == "domain":
        arrays["layer_0_hi"] = arrays["layer_0_hi"] + 1
    elif defect == "certificate":
        arrays["layer_0_p"] = np.full((1, 4), 1.1)
    elif defect == "rates":
        arrays["layer_0_rates"] = arrays["layer_0_rates"] * 2
    arrays["manifest"] = np.array(
        json.dumps(
            {
                "format": "dissipative-mamba2-gates-v1",
                "layers": [0, 1],
                "input_payload_sha256": "0" * 64,
            }
        )
    )
    path = tmp_path / "gates.npz"
    np.savez(path, **arrays)
    return path


@pytest.mark.parametrize("defect", ["domain", "certificate", "rates"])
def test_joint_bundle_must_match_checkpoint_and_certify(tmp_path, scheduled_model, defect):
    model, _ = scheduled_model
    with pytest.raises(ValueError, match="joint gate"):
        stabilized_specs(model, write_gate_bundle(tmp_path, model, defect))


@pytest.fixture
def generation_payload(tmp_path, scheduled_model):
    model, bundle = scheduled_model
    bundle_path = tmp_path / "schedules.json"
    bundle_path.write_text(json.dumps(bundle))

    class Tokenizer:
        def __call__(self, text, return_tensors=None):
            class Result:
                input_ids = torch.tensor([[(ord(c) % 90) + 3 for c in text[:16]]])

            return Result()

    tokenizer = Tokenizer()
    source = export_chain_payload(
        model,
        tokenizer,
        tmp_path / "frozen",
        n_test_tokens=2,
        cal_tokens=16,
        normalization_bundle=bundle_path,
        stabilized_gate_bundle=write_gate_bundle(tmp_path, model),
    )
    return model, tokenizer, source


def test_generation_keeps_full_prompt_and_frozen_source(tmp_path, generation_payload):
    model, tokenizer, source = generation_payload

    def inventory():
        return {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.rglob("*")
            if path.is_file()
        }

    original = inventory()
    target = tmp_path / "generation"
    request = prepare_generation(model, tokenizer, source, target, prompt="abcd", generate_tokens=3)
    assert request["prompt_ids"] == tokenizer("abcd").input_ids[0].tolist()
    assert request["prompt_tokens"] == 4
    assert request["server_evaluations"] == 6
    assert inventory() == original
    chain = json.loads((target / "chain.json").read_text())
    assert chain["tensors"]["autoregressive_poly_embeddings"][0] == 6
    assert len(chain["autoregressive"]["poly_generated_ids"]) == 3
    assert all(
        value[0] == 0 for value in chain["autoregressive"]["operator_domain_violations"].values()
    )
    with pytest.raises(FileExistsError):
        prepare_generation(model, tokenizer, source, target, prompt="abcd", generate_tokens=3)
    with pytest.raises(ValueError, match="outside the source"):
        prepare_generation(
            model, tokenizer, source, source / "nested", prompt="a", generate_tokens=1
        )


@pytest.mark.parametrize("changed", ["weight", "epsilon"])
def test_generation_rejects_different_checkpoint(tmp_path, generation_payload, changed):
    model, tokenizer, source = generation_payload
    if changed == "weight":
        with torch.no_grad():
            model.backbone.layers[0].mixer.D.add_(1)
    else:
        model.backbone.norm_f.variance_epsilon *= 2
    with pytest.raises(ValueError, match="checkpoint differs"):
        prepare_generation(
            model, tokenizer, source, tmp_path / "new", prompt="abcd", generate_tokens=3
        )
    assert not (tmp_path / "new").exists()


def test_existing_payload_generation_checks_final_norm_domain(generation_payload):
    from fhemamba.m1_payload import export_autoregressive_client_payload

    model, tokenizer, source = generation_payload
    path = source / "chain.json"
    chain = json.loads(path.read_text())
    # Narrow the declared domain without changing the finite polynomial. The
    # domain guard must catch this even when the logits remain finite.
    chain["final_norm_poly"]["lo"] = 50.0
    path.write_text(json.dumps(chain))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="normalization reference inputs leave"):
        export_autoregressive_client_payload(model, tokenizer, source, prompt="abcd")
    assert path.read_bytes() == original
