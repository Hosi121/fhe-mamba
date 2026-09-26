"""Serialized small qualification and matched inference, with durable status."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("small", "full"))
    args = parser.parse_args()
    root = args.root.resolve()
    candidate = root / "spark/kernel/packed_fideslib"
    status = {"state": "running", "started_at_unix": time.time(), "stage": args.stage}
    records = []

    def run(name, command, expected_ids=None):
        output = root / name
        output.mkdir()
        status["current"] = name
        save(root / f"{args.stage}-status.json", status)
        before = time.time()
        with (output / "run.log").open("w") as log:
            result = subprocess.run(
                command,
                env=dict(os.environ, OMP_NUM_THREADS="4"),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=2400,
            )
        save(
            output / "run.json",
            {
                "command": command,
                "exit_code": result.returncode,
                "started_at_unix": before,
                "wall_seconds": time.time() - before,
                "binary_sha256": sha(Path(command[3])),
                "environment": {
                    "OMP_NUM_THREADS": "4",
                    "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
                },
            },
        )
        if result.returncode:
            raise RuntimeError(f"{name}: exit {result.returncode}")
        data = json.loads((output / "native.json").read_text())
        assert data["passed"]
        assert data["non_finite"] == data["evaluation_decryptions"] == 0
        assert data["polynomial_tolerance"] == data["exact_tolerance"] == 0.001
        assert data["generated_token_ids"] == expected_ids
        assert data["s2c_first"]
        assert data["bootstrap_passes"] == 2
        assert data["scale_bits"] == 59
        assert data["hoist_rotations"]
        assert data["share_chebyshev"]
        assert data["gpu_plaintext_rns"]
        if "--gpu-dual-ring" in command:
            assert data["gpu_dual_ring"]
            assert (data["ring_dimension"], data["slots"]) == (32768, 16384)
            assert data["ring_switch_up_calls"] > 0
            assert data["ring_switch_down_calls"] == 2 * data["ring_switch_up_calls"]
            assert data["bootstraps"] == 2 * data["ring_switch_up_calls"]
        else:
            assert (data["ring_dimension"], data["slots"]) == (65536, 32768)
        assert data["depth"] == 44
        assert data["refreshed_level"] == 18
        assert data["refresh_ceiling"] == 35
        for key in ("max_abs_error_vs_exact", "max_abs_error_vs_polynomial"):
            assert math.isfinite(data[key])
            assert 0 <= data[key] <= 0.001
        records.append(
            {
                "name": name,
                "eval_seconds": data["eval_seconds"],
                "refresh_seconds": data["bootstrap_seconds"],
                "bootstraps": data["bootstraps"],
            }
        )
        save(root / f"{args.stage}-samples.json", records)
        print(name, data["eval_seconds"], flush=True)

    try:
        source = root / "repo/native/fideslib_stage0"
        save(
            root / f"{args.stage}-source-manifest.json",
            {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()},
        )
        if args.stage == "small":
            for mode in ("qualification", "abba", "baab"):
                status["current"] = f"micro-{mode}"
                save(root / "small-status.json", status)
                command = [
                    "python3",
                    str(root / "repo/experiments/gpu_dual_ring/run_probe.py"),
                    "--binary",
                    str(root / "probe-build/gpu_dual_ring"),
                    "--output",
                    str(root / f"micro-{mode}"),
                    "--mode",
                    mode,
                    "--commit",
                    "e6bde4d498582071a50e134e5046f09ba28e87dd",
                ]
                subprocess.run(command, check=True, timeout=1800)
                data = json.loads((root / f"micro-{mode}/probe.json").read_text())
                assert data["passed"]
                assert data["exact_ntt_maps"] == 6
                assert data["transfer_cases"] == 30
                assert data["max_abs_error"] <= 1e-6
        else:
            assert json.loads((root / "small-status.json").read_text())["state"] == "completed"
            assert json.loads((root / "full-selection.json").read_text())["eligible"]
        study = "prefix" if args.stage == "small" else "full"
        reference = json.loads((root / f"{study}-baseline-command.json").read_text())
        assert sha(Path(reference["command"][3])) == reference["binary_sha256"]
        payload = Path(reference["command"][4]).parent
        manifest = json.loads((payload / "manifest.json").read_text())
        for name, digest in manifest["files_sha256"].items():
            assert sha(payload / name) == digest, name
        save(
            root / f"{study}-payload-verification.json",
            {
                "passed": True,
                "manifest_sha256": sha(payload / "manifest.json"),
                "files_sha256": manifest["files_sha256"],
            },
        )
        for i, mode in enumerate(("baseline", "candidate", "candidate", "baseline")):
            name = f"{study}-{i}-{mode}"
            command = reference["command"].copy()
            command[5] = str(root / name / "native.json")
            if mode == "candidate":
                command[3] = str(candidate)
                command.append("--gpu-dual-ring")
            run(name, command, [6864, 6864] if study == "prefix" else [315, 279, 1614, 315])
        status["state"] = "completed"
    except Exception as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        status["finished_at_unix"] = time.time()
        save(root / f"{args.stage}-status.json", status)


if __name__ == "__main__":
    main()
