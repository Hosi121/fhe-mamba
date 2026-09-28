#!/usr/bin/env python3
"""Bind diagnostic node references to the exact frozen packed program."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(program, metadata):
    full, prefix = hashlib.sha256(), hashlib.sha256()
    count = metadata["node_count"]
    with program.open("rb") as stream:
        header = stream.readline()
        full.update(header)
        for _ in range(count):
            line = stream.readline()
            if not line:
                raise ValueError("program is shorter than the diagnostic prefix")
            full.update(line)
            prefix.update(line)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            full.update(chunk)
    if prefix.hexdigest() != metadata["node_prefix_sha256"]:
        raise ValueError("rebuilt diagnostic nodes differ from the original program")
    if full.hexdigest() != metadata["program_sha256"]:
        raise ValueError("program differs from the frozen manifest")
    return {
        "passed": True,
        "program_sha256": full.hexdigest(),
        "node_prefix_sha256": prefix.hexdigest(),
        "nodes_verified": count,
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="fhemamba diagnose verify", description=__doc__)
    p.add_argument("--program", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    a = p.parse_args(argv)
    print(json.dumps(verify(a.program, json.loads(a.metadata.read_text())), indent=2))


if __name__ == "__main__":
    main()
