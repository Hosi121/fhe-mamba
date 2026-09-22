"""Generated text must describe measured output, including failed experiments."""

import pytest
from fhemamba.generation import generation_report


class Tokenizer:
    def decode(self, ids, **kwargs):
        return " ".join(str(value) for value in ids)


@pytest.fixture
def evidence():
    request = {
        "prompt_ids": [2, 3, 4],
        "prompt_tokens": 3,
        "generate_tokens": 2,
        "n_layers": 24,
        "server_evaluations": 4,
    }
    chain = {
        "tensors": {"client_embedding_w": [97, 32]},
        "autoregressive": {
            "prompt_ids": request["prompt_ids"],
            "poly_generated_ids": [7, 8],
            "exact_generated_ids": [7, 9],
            "operator_domain_violations": {"rms_invsqrt": [0, 100]},
        },
    }
    native = {
        "input_payload_sha256": "a" * 64,
        "repo_commit": "b" * 40,
        "binary_sha256": "c" * 64,
        "backend": "fideslib",
        "config": {"input_mode": "fhemamba-m1-payload"},
        "encrypted": True,
        "passed": True,
        "parameters": {
            "autoregressive_client_loop": True,
            "autoregressive_prompt_tokens": 3,
            "autoregressive_generate_tokens": 2,
            "n_layers_loaded": 24,
            "tokens": 4,
            "final_norm_applied": True,
            "security": "not-set",
        },
        "measurements": {
            "autoregressive_selected_ids": [7, 8],
            "per_token_max_abs_error": [0.001] * 4,
            "per_token_max_abs_error_vs_exact": [0.1] * 4,
            "per_token_decrypt_ok": [True] * 4,
            "peak_rss_gib": 36,
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
        "operation_counts": {"bootstraps": 5},
        "timing": {"evaluation_seconds": 123},
    }
    return native, chain, request


def report(evidence):
    return generation_report(*evidence, Tokenizer(), payload_sha256="a" * 64)


def test_text_comes_from_actual_selection(evidence):
    result = report(evidence)
    assert result["passed"]
    assert result["text"]["full"] == "2 3 4 7 8"
    assert not result["measurements"]["matches_exact_tokens"]
    evidence[0]["measurements"]["autoregressive_selected_ids"] = [7, 10]
    result = report(evidence)
    assert not result["passed"]
    assert result["text"]["generated"] == "7 10"
    assert result["text"]["polynomial_reference"] == "7 8"


@pytest.mark.parametrize("defect", ["stale", "short_horizon", "prompt", "invalid_id"])
def test_report_rejects_unrelated_or_invalid_evidence(evidence, defect):
    native, _, request = evidence
    if defect == "stale":
        native["input_payload_sha256"] = "d" * 64
    elif defect == "short_horizon":
        native["parameters"]["tokens"] -= 1
    elif defect == "prompt":
        request["prompt_ids"] = [2, 3, 5]
    else:
        native["measurements"]["autoregressive_selected_ids"] = [-1, 8]
    with pytest.raises(ValueError, match=r"payload|requested generation|prompt|token IDs"):
        report(evidence)


@pytest.mark.parametrize("defect", ["missing", "nan", "decrypt", "diagnostic", "clear_state"])
def test_numerical_or_scope_failure_cannot_be_reported_as_passed(evidence, defect):
    native = evidence[0]
    if defect == "missing":
        native["measurements"]["autoregressive_selected_ids"].pop()
    elif defect == "nan":
        native["measurements"]["per_token_max_abs_error"][0] = float("nan")
    elif defect == "decrypt":
        native["measurements"]["per_token_decrypt_ok"][1] = False
    elif defect == "diagnostic":
        native["measurement_scope"]["zero_intermediate_decrypts"] = False
    else:
        native["measurement_scope"]["multi_token_ciphertext_state_carry"] = False
    assert not report(evidence)["passed"]
