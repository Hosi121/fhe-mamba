#!/usr/bin/env python3
"""Run a full prompt -> encrypted recurrence -> client-selected text experiment."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhemamba._env import block_broken_torchvision
from fhemamba.generation import generation_report, prepare_generation
from manage_dgx_build import payload_sha256, sha256

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/mamba2-130m-hf")
    parser.add_argument("--base-chain", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument("--generate-tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--joint-periodic-coefficients",
        action="store_true",
        help="encode public joint-gate coefficients with a compact period and masked basis",
    )
    parser.add_argument("--ssh-host", default="dgx")
    parser.add_argument(
        "--joint-subring-encoding",
        action="store_true",
        help="use a small NTT for periodic joint-gate coefficients (requires periodic mode)",
    )
    parser.add_argument("--ssh-option", action="append", default=[])
    parser.add_argument("--remote-root", required=True, type=Path)
    parser.add_argument("--remote-git-dir", type=Path)
    args = parser.parse_args()
    if args.joint_subring_encoding and not args.joint_periodic_coefficients:
        parser.error("joint-subring-encoding requires joint-periodic-coefficients")
    if args.generate_tokens < 1 or args.threads < 1:
        parser.error("generation length and threads must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", args.output_dir.name):
        parser.error("output directory name must use letters, digits, underscores, dots or hyphens")
    if not args.remote_root.is_absolute():
        parser.error("remote-root must be an absolute path")
    if args.ssh_host.startswith("-") or not args.ssh_host:
        parser.error("invalid SSH host")
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    if not prompt.strip():
        parser.error("prompt must not be empty")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = args.output_dir / "payload"
    request = {"source_payload_sha256": payload_sha256(args.base_chain)}
    block_broken_torchvision()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).float().eval().to(args.device)
    print("Preparing the complete prompt and frozen polynomial references...", flush=True)
    request.update(
        prepare_generation(
            model,
            tokenizer,
            args.base_chain,
            payload,
            prompt=prompt,
            generate_tokens=args.generate_tokens,
        )
    )
    del model
    request["input_payload_sha256"] = payload_sha256(payload)
    request["joint_periodic_coefficients"] = args.joint_periodic_coefficients
    request["joint_subring_encoding"] = args.joint_subring_encoding
    write_json(args.output_dir / "request.json", request)
    manifest = json.loads(
        (ROOT / "fhemamba/experiments/dgx_spark_stabilized_generation.json").read_text()
    )
    evaluations = request["server_evaluations"]
    manifest["defaults"].update(
        {
            "TOKENS": str(evaluations),
            "LAYERS": str(request["n_layers"]),
            "JOINT_PERIODIC_COEFFICIENTS": "1" if args.joint_periodic_coefficients else "0",
            "JOINT_SUBRING_ENCODING": "1" if args.joint_subring_encoding else "0",
        }
    )
    manifest["acceptance"].update({"tokens": evaluations, "layers": request["n_layers"]})
    manifest["timeout_seconds"] = 600 + 1200 * evaluations
    manifest_path = args.output_dir / "generation-manifest.json"
    write_json(manifest_path, manifest)
    sources = [
        *sorted((ROOT / "fhemamba/src/fhemamba").glob("*.py")),
        Path(__file__).resolve(),
        ROOT / "fhemamba/experiments/manage_dgx_build.py",
        ROOT / "fhemamba/experiments/dgx_spark_stabilized_generation.json",
    ]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    write_json(args.output_dir / "source-manifest.json", source_hashes)
    with tarfile.open(args.output_dir / "source.tar.gz", "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname=path.relative_to(ROOT))

    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    for option in args.ssh_option:
        ssh += ["-o", option]
    remote_run = args.remote_root / "generation" / args.output_dir.name

    def remote(command, **kwargs):
        return subprocess.run([*ssh, args.ssh_host, command], **kwargs)

    # Refuse remote reuse before transfer; old numerical results remain immutable.
    remote(
        shlex.join(["mkdir", "-p", "--", str(remote_run.parent)])
        + " && "
        + shlex.join(["mkdir", "--", str(remote_run)]),
        check=True,
    )
    rsync = ["rsync", "-az", "--protect-args", "-e", shlex.join(ssh)]
    subprocess.run(
        [*rsync, str(args.output_dir) + "/", f"{args.ssh_host}:{remote_run}/"], check=True
    )
    run_tag = args.output_dir.name
    repo = args.remote_root / "cipher"
    command = ["env"]
    if args.remote_git_dir:
        command.append(f"GIT_DIR={args.remote_git_dir}")
    command += [
        "python3",
        "fhemamba/experiments/run_dgx_campaign.py",
        "--manifest",
        str(remote_run / manifest_path.name),
        "--runner",
        "scripts/run_dgx_spark.sh",
        "--env",
        f"FHEMAMBA_REMOTE_ROOT={args.remote_root}",
        "--env",
        f"INPUT_CHAIN={remote_run / 'payload'}",
        "--env",
        f"RESULTS_DIR={remote_run / 'results'}",
        "--env",
        f"RUN_TAG={run_tag}",
        "--output-json",
        str(remote_run / "results/campaign.json"),
    ]
    print(
        f"Running {evaluations} encrypted evaluations; log: {args.output_dir / 'native.log'}",
        flush=True,
    )
    with (args.output_dir / "native.log").open("w") as log:
        result = remote(
            "cd " + shlex.quote(str(repo)) + " && " + shlex.join(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    local_results = args.output_dir / "results"
    local_results.mkdir()
    subprocess.run(
        [*rsync, f"{args.ssh_host}:{remote_run}/results/", str(local_results) + "/"], check=True
    )
    artifact = local_results / f"m2_chain_{run_tag}_l{request['n_layers']}_t{evaluations}.json"
    native = json.loads(artifact.read_text())
    chain = json.loads((payload / "chain.json").read_text())
    report = generation_report(
        native, chain, request, tokenizer, payload_sha256=payload_sha256(payload)
    )
    report["checks"]["periodic_coefficient_mode"] = (
        native["parameters"]["joint_gate_schedule"].get("periodic_coefficients", False)
        is args.joint_periodic_coefficients
    )
    report["checks"]["subring_encoding_mode"] = (
        native["parameters"]["joint_gate_schedule"].get("subring_encoding", False)
        is args.joint_subring_encoding
    )
    campaign = json.loads((local_results / "campaign.json").read_text())
    report["checks"]["campaign_passed"] = (
        result.returncode == 0 and campaign.get("acceptance", {}).get("passed") is True
    )
    report["passed"] = all(report["checks"].values())
    report["status"] = "passed" if report["passed"] else "failed"
    report["source_sha256"] = source_hashes
    report["native_artifact"] = {"path": str(artifact), "sha256": sha256(artifact)}
    write_json(args.output_dir / "generation.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "text": report["text"],
                "result": str(args.output_dir / "generation.json"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
