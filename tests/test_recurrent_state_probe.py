"""State-only comparisons must share inputs and retain every state coordinate."""

import json
import shutil

import numpy as np
import pytest

from fhemamba.benchmarks.io import file_identity
from fhemamba.recurrent import analysis
from fhemamba.recurrent.probe import INPUTS, export


def _trace(directory):
    generator = np.random.default_rng(90)
    steps, heads, channels, states = 6, 3, 4, 8
    arrays = {
        "b": generator.normal(size=(steps, 1, heads, states)),
        "c": generator.normal(size=(steps, 1, heads, states)),
        "v": generator.normal(size=(steps, 1, heads, channels)),
        "decay": generator.uniform(0.8, 0.99, size=(steps, 1, heads)),
        "lag": generator.uniform(0.001, 0.1, size=(steps, 1, heads)),
        "write": generator.uniform(0.001, 0.1, size=(steps, 1, heads)),
        "skip_weight": generator.normal(size=heads),
    }
    state = np.zeros((1, heads, channels, states))
    key, value = np.zeros((1, heads, states)), np.zeros((1, heads, channels))
    outputs = []
    for step in range(steps):
        b, c, v, decay, lag, write = (arrays[name][step] for name in INPUTS)
        state = state * decay[..., None, None]
        state += value[..., None] * key[..., None, :] * lag[..., None, None]
        state += v[..., None] * b[..., None, :] * write[..., None, None]
        outputs.append((state * c[..., None, :]).sum(-1) + v * arrays["skip_weight"][:, None])
        key, value = b, v
    arrays["reference_y"] = np.stack(outputs)
    directory.mkdir()
    np.savez_compressed(directory / "trace.npz", **arrays)
    manifest = {
        "schema": "fhemamba-recurrence-trace-v1",
        "trace_sha256": file_identity(directory / "trace.npz")["sha256"],
        "steps": steps,
        "heads": heads,
        "channels": channels,
        "state_size": states,
        "layer": 0,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_fixed_state_and_exact_history_export_same_recurrence(tmp_path):
    trace = tmp_path / "trace"
    _trace(trace)
    tiled = export(trace, tmp_path / "tiled", steps=6, representation="tiled", slots=64)
    factored = export(trace, tmp_path / "factored", steps=6, representation="factored", slots=64)
    assert tiled["trace_sha256"] == factored["trace_sha256"]
    assert tiled["max_plaintext_error_vs_dense"] < 1e-12
    assert factored["max_plaintext_error_vs_dense"] < 1e-12
    for report in (tiled, factored):
        assert [item["step"] for item in report["output_names"] if "readout" in item["name"]] == [
            1,
            2,
            3,
            4,
            5,
            6,
        ]
    # Three complete heads occupy two ciphertexts, plus the previous key and value.
    assert {step["carried_ciphertexts"] for step in tiled["steps"]} == {4}
    assert {step["carried_coordinates"] for step in tiled["steps"]} == {3 * 4 * 8 + 3 * (8 + 4)}
    assert len({step["stop"] - step["start"] for step in tiled["steps"]}) == 1
    assert [step["carried_ciphertexts"] for step in factored["steps"]] == [3, 6, 9, 12, 15, 18]
    assert factored["steps"][-1]["operations"]["mul"] > factored["steps"][0]["operations"]["mul"]
    assert [output["name"] for output in tiled["output_names"]][-2:] == [
        "final.state0",
        "final.state1",
    ]
    for representation, report in (("tiled", tiled), ("factored", factored)):
        for name, expected in report["files_sha256"].items():
            assert file_identity(tmp_path / representation / name)["sha256"] == expected


def test_trace_tampering_is_rejected_before_export(tmp_path):
    trace = tmp_path / "trace"
    _trace(trace)
    with (trace / "trace.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest differs"):
        export(trace, tmp_path / "rejected", steps=4, representation="tiled", slots=64)
    assert not (tmp_path / "rejected").exists()


def _measurement_pair(tmp_path, steps=6):
    trace, payload, results = tmp_path / "trace", tmp_path / "payload", tmp_path / "results"
    if not trace.exists():
        _trace(trace)
    for mode in ("factored", "tiled"):
        case = f"{mode}-{steps}"
        manifest = export(trace, payload / case, steps=steps, representation=mode, slots=64)
        measured = results / case
        measured.mkdir(parents=True)
        inputs = {
            name: {"sha256": "a" * 64}
            for name in ("binary", "fideslib", "openfhe_core", "source", "native_source")
        }
        inputs.update(
            trace={"sha256": manifest["trace_sha256"]},
            manifest=file_identity(payload / case / "manifest.json"),
            program={"sha256": manifest["files_sha256"]["program.txt"]},
            fixture={"sha256": manifest["files_sha256"]["fixture.npz"]},
        )
        (measured / "run.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "exit_code": 0,
                    "inputs": inputs,
                    "wall_seconds": 100,
                }
            )
        )
        native = dict.fromkeys(analysis.CONTRACT["equal"], 0)
        native["security_audit"] = {}
        for field, value in analysis.CONTRACT["require"].items():
            target = native
            parts = field.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        arithmetic = 2 if mode == "factored" else 4
        native.update(
            eval_seconds=8 + arithmetic,
            setup_seconds=20,
            bootstrap_seconds=1,
            bootstraps=2,
            ct_ct_mul=100,
            ct_pt_mul=100,
            rotations=100,
            peak_rss_gib=2,
            max_abs_error_vs_exact=1e-5,
            max_abs_error_vs_polynomial=1e-5,
            per_output_errors=[{"exact": 1e-5, "polynomial": 1e-5}] * len(manifest["output_names"]),
            operation_stats={
                "input": {"seconds": 8, "nodes": 30},
                "sum": {"seconds": arithmetic, "nodes": manifest["program"]["operations"]["sum"]},
            },
        )
        (measured / "native.json").write_text(json.dumps(native))
    return payload, results


def test_analysis_reports_slower_arm_and_separates_input_encryption(tmp_path):
    payload, results = _measurement_pair(tmp_path)
    report = analysis.analyze(payload, results, [6])
    assert report["passed"]
    assert report["first_measured_tiled_evaluation_win"] is None
    assert report["first_measured_tiled_arithmetic_win"] is None
    row = report["pairs"][0]
    assert row["arms"]["factored"]["input_materialization_seconds"] == 8
    assert row["arms"]["tiled"]["state_arithmetic_seconds"] == 4
    assert row["state_arithmetic_speedup"] == 0.5


def test_analysis_rejects_pruned_readouts(tmp_path):
    payload, results = _measurement_pair(tmp_path)
    path = results / "tiled-6/native.json"
    native = json.loads(path.read_text())
    native["operation_stats"]["sum"]["nodes"] -= 1
    path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="readout operations were pruned"):
        analysis.analyze(payload, results, [6])


def test_analysis_rejects_mixed_binaries(tmp_path):
    payload, results = _measurement_pair(tmp_path)
    path = results / "tiled-6/run.json"
    record = json.loads(path.read_text())
    record["inputs"]["binary"]["sha256"] = "b" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="do not share a binary"):
        analysis.analyze(payload, results, [6])


def test_analysis_rejects_zero_exit_without_native_result(tmp_path):
    payload, results = _measurement_pair(tmp_path)
    (results / "factored-6/native.json").unlink()
    (results / "factored-6/run.log").write_text("Cuda failure: 'out of memory'\n")
    # With no successful baseline at any length, security matching is impossible.
    with pytest.raises(ValueError, match="must both contain samples"):
        analysis.analyze(payload, results, [6])


def test_failed_partner_has_no_speedup_and_does_not_become_a_pass(tmp_path):
    payload, results = _measurement_pair(tmp_path, steps=4)
    _measurement_pair(tmp_path, steps=6)
    (results / "factored-6/native.json").unlink()
    (results / "factored-6/run.log").write_text("Cuda failure: 'out of memory'\n")
    report = analysis.analyze(payload, results, [4, 6])
    assert report["passed"] is False
    assert report["all_tiled_runs_qualified"] is True
    assert report["failures"][0]["cuda_out_of_memory"] is True
    assert report["pairs"][-1]["complete_pair"] is False
    assert "speedup" not in report["pairs"][-1]
    assert report["pairs"][-1]["arms"]["tiled"]["qualified"] is True


def _schedule_pair(tmp_path):
    payload, results = _measurement_pair(tmp_path)
    cases = []
    for name, limit in (("control", 0), ("candidate", 4)):
        directory = results / name
        shutil.copytree(results / "tiled-6", directory)
        native = json.loads((directory / "native.json").read_text())
        native.update(
            frontier_live_limit=limit,
            frontier_limit_selections=1 if limit else 0,
            maximum_live_dag_values=12,
            sampled_peak_device_bytes=1000,
            device_total_bytes=2000,
            plaintext_cache_entries=8,
        )
        (directory / "native.json").write_text(json.dumps(native))
        record = json.loads((directory / "run.json").read_text())
        record["spec"] = {
            "command": [
                "docker",
                "--name",
                name,
                "--gpus",
                "device=0",
                "--cpuset-cpus",
                "0-4",
                "program.txt",
                f"/work/{name}/native.json",
                "--frontier-live-limit",
                str(limit),
            ]
        }
        (directory / "run.json").write_text(json.dumps(record))
        (directory / "run.log").write_text(
            "node=10/20 op=mul seconds=1.5 eta_seconds=1.5 bootstraps=0 "
            "live_values=11 peak_live_values=12 cache_entries=8 "
            "device_used_bytes=900 sampled_peak_device_bytes=1000\n"
        )
        cases.append({"name": name, "mode": "tiled", "steps": 6, "limit": limit})
    return (
        payload,
        results,
        {
            "schema_version": 1,
            "cases": cases,
            "pairs": [
                {"baseline": "control", "candidate": "candidate", "compare_timing": True},
            ],
        },
    )


def test_schedule_analysis_does_not_turn_oom_into_a_speedup(tmp_path):
    payload, results, study = _schedule_pair(tmp_path)
    (results / "control/native.json").unlink()
    with (results / "control/run.log").open("a") as log:
        log.write("CUDA error: out of memory\n")
    report = analysis.analyze_schedule(payload, results, study)
    assert not report["all_runs_qualified"]
    assert report["cases"]["candidate"]["qualified"]
    assert report["cases"]["control"]["failure"]["cuda_out_of_memory"]
    assert report["cases"]["control"]["progress"]["last"]["live_values"] == 11
    assert not report["pairs"][0]["complete_pair"]
    assert "evaluation" not in report["pairs"][0]
    path = results / "candidate/native.json"
    native = json.loads(path.read_text())
    native["max_abs_error_vs_exact"] = 0.002
    path.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="error/limit exceeded"):
        analysis.analyze_schedule(payload, results, study)


def test_schedule_comparison_rejects_different_device(tmp_path):
    payload, results, study = _schedule_pair(tmp_path)
    assert analysis.analyze_schedule(payload, results, study)["all_runs_qualified"]
    path = results / "candidate/run.json"
    record = json.loads(path.read_text())
    command = record["spec"]["command"]
    command[command.index("--gpus") + 1] = "device=1"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="changed command, device or CPU placement"):
        analysis.analyze_schedule(payload, results, study)
