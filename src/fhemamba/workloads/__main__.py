"""Model fixture export and upstream reference checks."""

from fhemamba._commands import dispatch

COMMANDS = {
    "export-mamba3": ("fhemamba.workloads.mamba3_export", "main", "export trained generation"),
    "export-mixer": ("fhemamba.workloads.mamba3_probe", "main", "export a small mixer fixture"),
    "check-mixer": ("fhemamba.workloads.mamba3_parity", "main", "check upstream mixer parity"),
    "check-mamba3": ("fhemamba.workloads.mamba3_lm_parity", "main", "check upstream model parity"),
}


def main(argv=None):
    return dispatch("fhemamba workload", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
