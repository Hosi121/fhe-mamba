"""Frozen input immutability and fail-closed output handling for reusable calibration."""

import json

import numpy as np
import pytest

from fhemamba.benchmarks.io import payload_sha256, read_object, write_json
from fhemamba.calibration import gates, schedules, state
from fhemamba.calibration.__main__ import main
from fhemamba.normalization import plan_invsqrt


def test_state_derivative_preserves_observations_and_unrelated_bytes(tmp_path):
    payload, output = tmp_path / "payload", tmp_path / "derived"
    layer = payload / "layer_00"
    layer.mkdir(parents=True)
    write_json(payload / "chain.json", {"n_layers": 1, "layer_dirs": [layer.name]})
    original = [1.0, 0.01, 2.0, 0.02]
    metadata = {
        "layer_index": 0,
        "dims": {"num_heads": 2, "head_dim": 2},
        "time_step_limit": [0, float("inf")],
        "carried_bounds": {
            "source": "calibration_text",
            "state_row_abs_max": original,
            "state_coordinate_calibration": {"description": "independent"},
        },
    }
    (layer / "meta.json").write_text(json.dumps(metadata))
    (layer / "weights.bin").write_bytes(b"public weights")
    (layer / "reference.bin").write_bytes(b"frozen reference")
    identity = payload_sha256(payload)
    report = state.regularize_state(
        payload, output_chain=output, group_heads=2, group_scale_floor=0.25
    )
    changed = json.loads((output / layer.name / "meta.json").read_text())
    assert changed["carried_bounds"]["state_row_abs_max"] == [1, 0.5, 2, 0.5]
    assert changed["carried_bounds"]["state_row_observed_abs_max"] == original
    assert changed["time_step_limit"] == metadata["time_step_limit"]
    for path in layer.glob("*.bin"):
        assert (output / layer.name / path.name).read_bytes() == path.read_bytes()
    assert payload_sha256(payload) == identity
    assert report["output_payload_sha256"] == payload_sha256(output)
    with pytest.raises(ValueError, match="new directory"):
        state.regularize_state(payload, output_chain=output, group_scale_floor=0.5)
    with pytest.raises(ValueError, match="outside"):
        state.regularize_state(payload, output_chain=payload / "nested", group_scale_floor=0.5)


def test_failed_gate_certificate_writes_failure_report_without_bundle(tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    layer = payload / "layer_00"
    layer.mkdir(parents=True)
    write_json(payload / "chain.json", {"layer_dirs": [layer.name]})
    write_json(layer / "meta.json", {"layer_index": 0, "dims": {"d_model": 1, "num_heads": 1}})
    for name, value in (("in_proj_w", 1), ("block_norm_w", 1), ("dt_bias", 0), ("a_log", 0)):
        np.array([value], dtype="<f4").tofile(layer / (name + ".bin"))
    # This actual polynomial has decay 1 - 2^2 = -3, outside the invariant.
    monkeypatch.setattr(
        gates, "fit_gate_roots", lambda *a, **kw: (np.array([[2.0]]), np.array([[1.0]]))
    )
    bundle, output = tmp_path / "gates.npz", tmp_path / "failed.json"
    code = main(
        [
            "gate-fit",
            "--payload",
            str(payload),
            "--bundle",
            str(bundle),
            "--output",
            str(output),
            "--grid-size",
            "3",
        ]
    )
    report = read_object(output)
    assert code == 1
    assert not report["all_certified"]
    assert not report["bundle_written"]
    assert not bundle.exists()
    assert report["measurement_scope"]["encrypted_execution"] is False


def test_normalization_export_refuses_duplicate_or_uncertified_sites(tmp_path):
    entry = {"layer": 0, "site": "rms_invsqrt", "recipe": plan_invsqrt(0.01, 1).recipe()}
    bundle, output = tmp_path / "schedules.json", tmp_path / "native"
    record = {
        "format": "fhemamba-normalization-schedules-v1",
        "input_payload_sha256": "a" * 64,
        "operators": [entry, entry],
    }
    write_json(bundle, record)
    with pytest.raises(ValueError, match="duplicate"):
        schedules.export_normalization(bundle, output_dir=output)
    assert not output.exists()
    record["operators"] = [entry]
    entry["recipe"]["seed"] = 100
    write_json(bundle, record)
    with pytest.raises(ValueError, match="uncertified"):
        schedules.export_normalization(bundle, output_dir=output)
    assert not output.exists()
