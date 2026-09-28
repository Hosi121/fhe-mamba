"""Inspect frozen circuits without changing model qualification."""

from fhemamba._commands import dispatch

COMMANDS = {
    "ckks": ("fhemamba.diagnostics.ckks", "main", "recipe-defined CPU CKKS feasibility probes"),
    "domain": ("fhemamba.diagnostics.domain", "main", "diagnose Mamba-2 polynomial domains"),
    "export": ("fhemamba.diagnostics.references", "main", "export frozen CPU references"),
    "verify": ("fhemamba.diagnostics.prefix", "main", "verify the exact program prefix"),
    "analyze": ("fhemamba.diagnostics.analysis", "main", "attribute observed operation errors"),
    "extract": ("fhemamba.diagnostics.cases", "main", "extract a primitive replay case"),
    "propagate": ("fhemamba.diagnostics.counterfactual", "main", "propagate selected CPU errors"),
    "levels": ("fhemamba.diagnostics.levels", "main", "analyze a level schedule"),
    "polynomials": ("fhemamba.diagnostics.polynomials", "main", "summarize polynomial degrees"),
}


def main(argv=None):
    return dispatch("fhemamba diagnose", COMMANDS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
