#!/usr/bin/env python3
"""Decode actual FHE client tokens after verifying the run and payload bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_packed_probe import digest


def report_generation(payload, run, tokenizer_path):
    from transformers import AutoTokenizer

    manifest = json.loads((payload / "manifest.json").read_text())
    record = json.loads((run / "run.json").read_text())
    native = json.loads((run / "native.json").read_text())
    if manifest.get("schema") != "fhemamba-mamba3-lm-v1":
        raise ValueError("not a language-model payload")
    if record.get("manifest_sha256") != digest(payload / "manifest.json"):
        raise ValueError("run belongs to a different payload")
    if record.get("native_sha256") != digest(run / "native.json"):
        raise ValueError("native output was changed after the run")
    for name, expected in manifest["tokenizer_files_sha256"].items():
        if digest(tokenizer_path / name) != expected:
            raise ValueError("tokenizer differs from the export")
    tokens = native.get("generated_token_ids", [])
    if not record.get("passed") or tokens != manifest["exact_token_ids"]:
        raise ValueError("encrypted generation did not pass its reference gates")
    if native.get("evaluation_decryptions") != 0:
        raise ValueError("unexpected decryption inside evaluation")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return {
        "passed": True,
        "complete_backbone": manifest["complete_backbone"],
        "layers": manifest["layers"],
        "prompt": manifest["prompt"],
        "generated_token_ids": tokens,
        "generated_text": tokenizer.decode(tokens),
        "text": tokenizer.decode(manifest["prompt_ids"] + tokens),
        "eval_seconds": native["eval_seconds"],
        "max_abs_error_vs_polynomial": native["max_abs_error_vs_polynomial"],
        "max_abs_error_vs_exact": native["max_abs_error_vs_exact"],
        "security": native["security"],
        "client_protocol": manifest["client"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = report_generation(args.payload, args.run, args.tokenizer)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
