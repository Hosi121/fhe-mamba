"""Capture, export and analyze recurrent-state component workloads."""

from fhemamba._commands import dispatch

COMMANDS = {
    "probe": ("fhemamba.recurrent.probe", "main", "capture/export/audit a recurrence fixture"),
    "analyze": ("fhemamba.recurrent.analysis", "main", "compare storage and schedule results"),
    "plot": ("fhemamba.recurrent.plot", "main", "plot recorded memory; requires matplotlib"),
}


def main(argv=None):
    return dispatch("fhemamba recurrent", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
