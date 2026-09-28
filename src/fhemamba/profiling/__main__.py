"""Offline Nsight analysis; no NVIDIA tools needed for help or CSV/SQLite reads."""

from fhemamba._commands import dispatch

COMMANDS = {
    "nsys": ("fhemamba.profiling.nsys", "main", "summarize a Nsight Systems SQLite export"),
    "ncu": ("fhemamba.profiling.ncu", "main", "summarize a Nsight Compute CSV export"),
}


def main(argv=None):
    return dispatch("fhemamba profile", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
