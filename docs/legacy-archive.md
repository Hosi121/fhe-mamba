# Legacy implementation archive

The pre-rebuild `fhe_native_mamba3` Python package, its `fhe-mamba3` CLI,
historical root experiment scripts, Slurm launchers, and compatibility tests
were removed from `main` after the active `fhemamba` tree became independent.

They remain available on the remote branch:

```text
archive/pre-compat-retirement-20260811
```

The branch points at commit `be8af52438ba6594abe5fb9b94b20ca7657a500e`,
the fully tested state immediately before retirement. To inspect an old command
without changing the current checkout:

```bash
git worktree add ../fhemamba-legacy archive/pre-compat-retirement-20260811
```

Do not merge that branch back into `main`. Port an individual helper only when
there is a current consumer, rename it for the active architecture, and add a
focused test. Historical artifact and research documentation remains on `main`
because it records provenance, not a supported execution path.
