"""Small lazy dispatcher shared by the public command groups."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Mapping, Sequence

# command -> (module, callable, help); importing help never initializes a model/GPU.
Command = tuple[str, str, str]


def dispatch(prog: str, commands: Mapping[str, Command], argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, _, help_text) in commands.items():
        subparsers.add_parser(name, help=help_text)
    if arguments and arguments[0] in commands:
        module, function, _ = commands[arguments[0]]
        handler = getattr(importlib.import_module(module), function)
        result = handler(arguments[1:])
        return int(result or 0)
    parser.parse_args(arguments)
    return 0
