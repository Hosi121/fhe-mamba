"""Frozen payload calibration; command help does not initialize PyTorch."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from fhemamba.benchmarks.io import write_json

COMMANDS = {
    "ranges": ("gates", "certify_ranges", "certify Newton basins and public gate envelopes"),
    "gate-screen": ("gates", "screen_gates", "screen SiLU degrees on public-weight intervals"),
    "gate-fit": ("gates", "fit_gates", "fit and certify dissipative selective gates"),
    "normalization": ("schedules", "plan_normalization", "plan certified inverse-root schedules"),
    "normalization-export": (
        "schedules",
        "export_normalization",
        "export certified native probe recipes",
    ),
    "state": ("state", "calibrate_state", "measure state scales on independent token windows"),
    "state-regularize": ("state", "regularize_state", "condition existing public row scales"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, _, description) in COMMANDS.items():
        command = subparsers.add_parser(name, help=description, description=description)
        if name == "normalization-export":
            command.add_argument("--bundle", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
            continue
        command.add_argument("--payload", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name in {"gate-screen", "gate-fit", "normalization"}:
            command.add_argument("--grid-size", type=int, default=8193)
        if name in {"gate-fit", "normalization"}:
            command.add_argument("--bundle", type=Path, required=True)
        if name == "gate-screen":
            command.add_argument("--degrees", type=int, nargs="+", default=[64, 128, 192, 256, 384])
            command.add_argument("--margin", type=float, default=1.1)
        elif name == "gate-fit":
            command.add_argument("--dissipation-degree", type=int, default=1024)
            command.add_argument("--equilibrium-degree", type=int, default=512)
            command.add_argument("--inward-margin", type=float, default=2e-6)
        elif name == "normalization":
            command.add_argument("--relative-error", type=float, default=1e-7)
        if name in {"state", "state-regularize"}:
            command.add_argument("--output-chain", type=Path, required=True)
            command.add_argument("--group-heads", type=int, default=4)
            command.add_argument(
                "--group-scale-floor", type=float, default=0.0, required=name == "state-regularize"
            )
        if name == "state":
            command.add_argument("--checkpoint", type=Path, required=True)
            inputs = command.add_mutually_exclusive_group(required=True)
            inputs.add_argument("--calibration-text", type=Path)
            inputs.add_argument("--calibration-tokens-pt", type=Path)
            command.add_argument("--calibration-description", required=True)
            command.add_argument("--offsets", type=int, nargs="+", default=[0])
            command.add_argument("--tokens", type=int, default=512)
            command.add_argument("--threads", type=int, default=4)
    options = vars(parser.parse_args(argv))
    name, output = options.pop("command"), options.pop("output", None)
    if name in {"gate-fit", "normalization"} and output.exists():
        parser.error("refusing to overwrite an existing frozen bundle/report")
    module, function, _ = COMMANDS[name]
    try:
        report = getattr(import_module(f"fhemamba.calibration.{module}"), function)(**options)
    except ValueError as error:
        parser.error(str(error))
    if output is not None:
        write_json(output, report)
    failed = report.get("status") == "failed" or report.get("bundle_written") is False
    print(f"{'failed' if failed else 'wrote'} {output or options['output_dir']}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
