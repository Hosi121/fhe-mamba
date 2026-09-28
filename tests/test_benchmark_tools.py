from __future__ import annotations

import gzip
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from fhemamba.benchmarks.comparison import compare
from fhemamba.benchmarks.evidence import Redactor, extract_provenance, publish, verify
from fhemamba.benchmarks.jobs import run_job
from fhemamba.benchmarks.process import run_process


def test_common_runner_stops_children_after_leader_exits(tmp_path):
    import time

    marker = tmp_path / "orphan-finished"
    child = f"import time; from pathlib import Path; time.sleep(0.3); Path({str(marker)!r}).touch()"
    command = [
        sys.executable,
        "-c",
        f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}])",
    ]
    assert run_process(command, timeout=5) == (0, False)
    time.sleep(0.5)
    assert not marker.exists()


def test_common_runner_reaps_process_on_cancellation(monkeypatch):
    from fhemamba.benchmarks import process as runner

    popen = subprocess.Popen
    children = []

    def start(*args, **kwargs):
        process = popen(*args, **kwargs)
        children.append(process)
        wait = process.wait

        def interrupted(*args, **kwargs):
            process.wait = wait
            raise KeyboardInterrupt

        process.wait = interrupted
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    with pytest.raises(KeyboardInterrupt):
        runner.run_process([sys.executable, "-c", "import time; time.sleep(10)"])
    assert children[0].poll() is not None


def test_publication_preserves_measurements_and_originals(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    payload = {
        "passed": False,
        "eval_seconds": 1.234567891011,
        "security_audit": {"qp_bits": 3376, "passed": True},
        "generated_token_ids": [315, 279],
        "hostname": "private-machine",
        "path": "/home/researcher/study/native.json",
    }
    raw = json.dumps(payload).encode()
    (source / "native.json").write_bytes(raw)
    (source / "run.log").write_text("ssh private-machine at 10.20.30.40\n")
    (source / "controller.py").write_text("raise RuntimeError('failed control')\n")
    public = tmp_path / "public"
    manifest = publish(source, public, Redactor({"private-machine": "gpu-host"}))
    assert verify(public) == {"passed": True, "files": 2, "archived_files": 1}
    assert manifest["excluded_disposable_scripts"] == 1
    report = json.loads((public / "native.json").read_text())
    assert report == {**payload, "hostname": "<private>", "path": "${HOME}/study/native.json"}
    assert (source / "native.json").read_bytes() == raw
    assert not (public / "controller.py").exists()
    assert next(r for r in manifest["files"] if r["path"] == "native.json")["transformed"]
    extracted = tmp_path / "inspection"
    extract_provenance(public, extracted)
    assert (extracted / "run.log").read_text() == "ssh gpu-host at <private-ip>\n"
    assert not (extracted / "controller.py").exists()
    (public / "native.json").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(public)


def test_publication_is_deterministic_and_checks_nested_archives(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        data = b"compiler: /home/researcher/install/g++\n"
        member = tarfile.TarInfo("src/flags.txt")
        member.uid = 12345
        member.uname = "researcher"
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        for name in ["job.py", "src/fhemamba/reference.py"]:
            member = tarfile.TarInfo(name)
            archive.addfile(member, io.BytesIO())
    (source / "source.tar.gz").write_bytes(gzip.compress(stream.getvalue()))
    publish(source, tmp_path / "first")
    publish(source, tmp_path / "second")
    assert (tmp_path / "first/publication.json").read_bytes() == (
        tmp_path / "second/publication.json"
    ).read_bytes()
    extracted = tmp_path / "inspection"
    extract_provenance(tmp_path / "first", extracted)
    with tarfile.open(extracted / "source.tar.gz") as archive:
        member = archive.getmember("src/flags.txt")
        assert member.uid == 0
        assert member.uname == ""
        assert archive.extractfile(member).read() == b"compiler: ${HOME}/install/g++\n"
        assert "job.py" not in archive.getnames()
        assert "src/fhemamba/reference.py" in archive.getnames()


def test_publication_refuses_overwrite_traversal_and_symlink(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    with pytest.raises(ValueError, match="separate"):
        publish(source, source / "output")
    with pytest.raises(ValueError, match="already exists"):
        publish(source, tmp_path)
    (source / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlink"):
        publish(source, tmp_path / "public")
    (source / "link").unlink()
    with tarfile.open(source / "source.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("../escape")
        archive.addfile(member, io.BytesIO())
    with pytest.raises(ValueError, match="unsafe"):
        publish(source, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_publication_rejects_colliding_redactions(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    (source / "data.json").write_text('{"first": 1, "second": 2}')
    with pytest.raises(ValueError, match="merge"):
        publish(source, tmp_path / "public", Redactor({"first": "same", "second": "same"}))


@pytest.mark.parametrize(("exit_code", "state"), [(0, "completed"), (7, "failed")])
def test_job_records_exit_and_event_without_environment_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int, state: str
) -> None:
    monkeypatch.setenv("UNRECORDED_TOKEN", "not-for-the-report")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": [sys.executable, "-c", f"print('sample');raise SystemExit({exit_code})"],
                "cwd": "${root}",
                "env": {"EXPERIMENT_SETTING": "public-value"},
                "inputs": {"spec": "spec.json"},
                "timeout_seconds": 10,
            }
        )
    )
    result = run_job(
        spec,
        tmp_path / "output",
        settings={"root": str(tmp_path)},
        lock_path=tmp_path / "resource.lock",
        events_path=tmp_path / "events.jsonl",
    )
    assert result["state"] == state
    assert result["exit_code"] == exit_code
    assert len(result["inputs"]["spec"]["sha256"]) == 64
    assert "not-for-the-report" not in json.dumps(result)
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["state"] == state
    assert json.loads((tmp_path / "output/completion.json").read_text()) == event
    with pytest.raises(FileExistsError):
        run_job(spec, tmp_path / "output", settings={"root": str(tmp_path)})


def test_job_timeout_stops_children_before_releasing_lock(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    sentinel = tmp_path / "unexpected"
    child_code = f"import time,pathlib;time.sleep(1);pathlib.Path({str(sentinel)!r}).touch()"
    code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);time.sleep(10)"
    )
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "command": [sys.executable, "-c", code],
                "timeout_seconds": 0.2,
            }
        )
    )
    result = run_job(spec, tmp_path / "output", lock_path=tmp_path / "device.lock")
    assert result["state"] == "timed_out"
    # A subsequent process occupying the same interval cannot observe a late child write.
    subprocess.run([sys.executable, "-c", "import time;time.sleep(1.1)"], check=True)
    assert not sentinel.exists()
    assert json.loads((tmp_path / "output/completion.json").read_text())["state"] == "timed_out"


def test_job_requires_explicit_variables_and_deadline(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "command": ["${missing}"]}))
    with pytest.raises(KeyError):
        run_job(spec, tmp_path / "output")
    spec.write_text(json.dumps({"schema_version": 1, "command": [sys.executable]}))
    with pytest.raises(ValueError, match="timeout"):
        run_job(spec, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_comparison_requires_matched_conditions_before_reporting_gain(tmp_path: Path) -> None:
    contract = {
        "schema_version": 1,
        "equal": ["security", "generated_token_ids"],
        "require": {"passed": True},
        "maximum": {"error": 0.001},
        "minimum_reduction": 0.03,
    }
    payload = {
        "security": "128-classic",
        "generated_token_ids": [1],
        "passed": True,
        "error": 0.0001,
    }
    paths = [tmp_path / f"sample-{i}.json" for i in range(4)]
    for path, seconds in zip(paths, [10, 12, 8, 10], strict=True):
        path.write_text(json.dumps({**payload, "eval_seconds": seconds}))
    result = compare(paths[:2], paths[2:], contract)
    assert result["passed"]
    assert result["baseline"]["mean"] == 11
    assert result["candidate"]["mean"] == 9
    assert result["reduction"] == pytest.approx(2 / 11)
    paths[-1].write_text(json.dumps({**payload, "eval_seconds": 8, "error": 0.002}))
    with pytest.raises(ValueError, match="limit exceeded"):
        compare(paths[:2], paths[2:], contract)
    paths[-1].write_text(json.dumps({**payload, "eval_seconds": 8, "generated_token_ids": [True]}))
    with pytest.raises(ValueError, match="condition changed"):
        compare(paths[:2], paths[2:], contract)
    with pytest.raises(ValueError, match="both arms"):
        compare([paths[0]], [paths[0]], contract)


def test_portable_even_seed_derivation_matches_recorded_exact_bound(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = root / "results/cpu/2026-09-27/refresh-even-polynomial"
    output = tmp_path / "seed.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "experiments/refresh_even_seed/derive.py"),
            str(evidence / "input.json"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
    )
    derived = json.loads(output.read_text())
    analysis = json.loads((evidence / "analysis.json").read_text())
    coefficients = json.loads((evidence / "candidate-coefficients.json").read_text())
    assert derived["coefficients_hex"] == coefficients["binary64_hex"]
    assert (
        derived["difference_bound_exact"]
        == analysis["seed_uniform_absolute_difference_bound"]["exact_fraction"]
    )
