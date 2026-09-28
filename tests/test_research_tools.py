"""Portable commands and package boundaries survive repository reorganization."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from fhemamba import cli
from fhemamba.benchmarks import __main__ as benchmark
from fhemamba.benchmarks.io import repository_root
from fhemamba.calibration import __main__ as calibration
from fhemamba.diagnostics import __main__ as diagnostics
from fhemamba.profiling import __main__ as profiling
from fhemamba.recurrent import __main__ as recurrent
from fhemamba.workloads import __main__ as workloads

GROUPS = {
    "calibrate": calibration.COMMANDS,
    "benchmark": benchmark.TOOLS,
    "diagnose": diagnostics.COMMANDS,
    "profile": profiling.COMMANDS,
    "recurrent": recurrent.COMMANDS,
    "workload": workloads.COMMANDS,
}


@pytest.mark.parametrize(
    ("group", "command"), [(g, c) for g, names in GROUPS.items() for c in names]
)
def test_each_installed_tool_has_callable_help(group, command, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main([group, command, "--help"])
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_lightweight_tools_work_outside_checkout_without_loading_models(tmp_path):
    code = """
import sys
from fhemamba.cli import main
for args in [['--help'], ['diagnose', 'verify', '--help'],
             ['diagnose', 'ckks', '--help'], ['benchmark', 'packed', '--help'],
             ['calibrate', 'state', '--help'], ['calibrate', 'normalization', '--help'],
             ['benchmark', 'normalization', 'probe', '--help'],
             ['benchmark', 'normalization', 'campaign', '--help']]:
    try:
        main(args)
    except SystemExit as error:
        assert error.code == 0
assert not {'torch', 'numpy', 'openfhe', 'matplotlib', 'einops'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_checkout_lookup_is_explicit_and_rejects_unrelated_directory(tmp_path):
    with pytest.raises(ValueError, match="source checkout"):
        repository_root(tmp_path)
    (tmp_path / "pyproject.toml").write_text("")
    nested = tmp_path / "native/fideslib_stage0/build"
    nested.mkdir(parents=True)
    assert repository_root(nested) == tmp_path


def test_python_imports_have_no_sibling_script_or_sys_path_dependency():
    root = Path(__file__).resolve().parents[1]
    scripts = {p.stem for p in (root / "experiments").rglob("*.py")}
    offenders = []
    for folder in ("src/fhemamba", "experiments"):
        for path in (root / folder).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.split(".")[0] in scripts | {"experiments"}
                ):
                    offenders.append((str(path.relative_to(root)), node.module))
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func.value) == "sys.path"
                ):
                    offenders.append((str(path.relative_to(root)), "sys.path mutation"))
    assert not offenders
