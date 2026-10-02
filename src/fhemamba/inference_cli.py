"""Thin CLI adapters for the public token-based generation API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fhemamba.benchmarks.io import file_sha256
from fhemamba.checkpoints import tokenizer_directory
from fhemamba.inference import load_model, load_prepared
from fhemamba.inputs import generation_length, token_ids


def _parser(command):
    parser = argparse.ArgumentParser(prog=f"fhemamba {command}")
    parser.add_argument(
        "--model",
        type=Path,
        required=command == "prepare",
        help="local Mamba-2 or Mamba-3 SISO checkpoint directory (detected from config)",
    )
    parser.add_argument(
        "--tokenizer", type=Path, help="local tokenizer; defaults to MODEL/tokenizer or MODEL"
    )
    inputs = parser.add_mutually_exclusive_group(required=command == "prepare")
    inputs.add_argument("--prompt", help="text encoded without added special tokens")
    inputs.add_argument("--prompt-file", type=Path, help="UTF-8 text file; '-' reads stdin")
    inputs.add_argument("--input-ids", type=Path, help="JSON array of token IDs; '-' reads stdin")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="fixed greedy length, ignoring EOS (default: prepared length or 16)",
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU model threads (default: 4)")
    if command == "prepare":
        parser.add_argument("--output", type=Path, required=True, help="fresh payload directory")
        parser.add_argument(
            "--profile",
            choices=("classical-128", "mamba2-experimental"),
            help="Mamba-3 defaults to classical-128; Mamba-2 needs explicit experimental selection",
        )
        parser.add_argument(
            "--base-chain", type=Path, help="frozen joint-gate chain required for Mamba-2"
        )
    else:
        parser.add_argument("--backend", choices=("exact", "polynomial", "ckks"), default="exact")
        parser.add_argument("--prepared", type=Path, help="request produced by fhemamba prepare")
        parser.add_argument("--binary", type=Path, help="model-specific native executable for CKKS")
        parser.add_argument("--output", type=Path, help="fresh CKKS run directory")
        parser.add_argument("--timeout", type=float, default=2400, help="CKKS timeout in seconds")
        parser.add_argument("--json", action="store_true", help="emit the structured result")
    return parser


def _read(path):
    return sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8")


def _tokenizer(args, manifest=None):
    path = args.tokenizer
    if path is None and args.model is not None:
        path = tokenizer_directory(args.model)
    if path is None:
        return None
    if manifest is not None:
        for name, expected in manifest["tokenizer_files_sha256"].items():
            if file_sha256(path / name) != expected:
                raise ValueError("tokenizer differs from the prepared request")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        if (
            args.tokenizer is None
            and args.prompt is None
            and args.prompt_file is None
            and hasattr(args, "backend")
        ):
            return None  # Raw token generation does not require a text decoder.
        raise ValueError("text input/output requires pip install 'fhemamba[experiments]'") from exc
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _inputs(args, tokenizer, manifest=None):
    if args.input_ids is not None:
        ids = token_ids(json.loads(_read(args.input_ids)))
    elif args.prompt is not None or args.prompt_file is not None:
        if tokenizer is None:
            raise ValueError(
                "text input requires a local tokenizer in MODEL/tokenizer or MODEL, "
                "or an explicit --tokenizer"
            )
        prompt = args.prompt if args.prompt is not None else _read(args.prompt_file)
        ids = token_ids(tokenizer.encode(prompt, add_special_tokens=False))
    elif manifest is not None:
        ids = list(manifest["prompt_ids"])
    else:
        raise ValueError("provide --prompt, --prompt-file or --input-ids")
    length = args.max_new_tokens
    if length is None:
        length = manifest["generated_tokens"] if manifest is not None else 16
    generation_length(length)
    if manifest is not None and (
        ids != manifest["prompt_ids"] or length != manifest["generated_tokens"]
    ):
        raise ValueError("input IDs or generation length differ from the prepared request")
    return ids, length


def _cpu_threads(count):
    if count < 1:
        raise ValueError("threads must be positive")
    import torch

    torch.set_num_threads(count)


def prepare_main(argv=None):
    parser = _parser("prepare")
    args = parser.parse_args(argv)
    try:
        tokenizer = _tokenizer(args)
        ids, length = _inputs(args, tokenizer)
        _cpu_threads(args.threads)
        prepared = load_model(args.model).prepare(
            ids,
            max_new_tokens=length,
            output=args.output,
            tokenizer=args.tokenizer,
            profile=args.profile,
            base_chain=args.base_chain,
        )
        print(
            json.dumps(
                {
                    "prepared": str(prepared.path),
                    "input_ids": ids,
                    "max_new_tokens": length,
                    "profile": prepared.profile,
                    "manifest_sha256": prepared.manifest_sha256,
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


def generate_main(argv=None):
    parser = _parser("generate")
    args = parser.parse_args(argv)
    try:
        if args.backend == "ckks":
            if args.prepared is None or args.binary is None or args.output is None:
                raise ValueError("CKKS requires --prepared, --binary and --output")
            if args.model is not None:
                raise ValueError("CKKS uses --prepared; --model is only needed for CPU generation")
        elif args.model is None:
            raise ValueError("CPU generation requires --model")
        elif args.binary is not None or args.output is not None:
            raise ValueError("--binary and --output are CKKS execution options")
        if args.backend == "polynomial" and args.prepared is None:
            raise ValueError("polynomial generation requires --prepared")
        prepared = load_prepared(args.prepared) if args.prepared is not None else None
        manifest = prepared.manifest() if prepared is not None else None
        tokenizer = _tokenizer(args, manifest)
        ids, length = _inputs(args, tokenizer, manifest)
        if args.backend == "ckks":
            result = prepared.generate(binary=args.binary, output=args.output, timeout=args.timeout)
        else:
            _cpu_threads(args.threads)
            result = load_model(args.model).generate(
                ids,
                max_new_tokens=length,
                backend=args.backend,
                prepared=prepared,
            )
        if args.json:
            report = result.to_dict()
            if tokenizer is not None and result.passed:
                report["generated_text"] = tokenizer.decode(
                    result.generated_ids, clean_up_tokenization_spaces=False
                )
            print(json.dumps(report, indent=2, allow_nan=False))
        elif result.passed:
            print(
                tokenizer.decode(result.generated_ids, clean_up_tokenization_spaces=False)
                if tokenizer is not None
                else json.dumps(result.generated_ids)
            )
        if not result.passed:
            print(
                f"generation failed: {result.report.get('error') or result.stop_reason}",
                file=sys.stderr,
            )
        return 0 if result.passed else 1
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
