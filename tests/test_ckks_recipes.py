"""Circuit and record compatibility without claiming encrypted qualification."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fhemamba.benchmarks.io import read_object
from fhemamba.diagnostics import ckks, ckks_growth

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "experiments/local_ckks"
KEYS = SimpleNamespace(publicKey="public", secretKey="secret")


class Value:
    def __init__(self, data, index, level=0, failed=False):
        self.data, self.index, self.level, self.failed = data, index, level, failed

    def GetLevel(self):  # noqa: N802 - OpenFHE interface
        return self.level

    def SetLength(self, length):  # noqa: N802
        self.data = self.data[:length]

    def GetRealPackedValue(self):  # noqa: N802
        return self.data


class TraceBackend:
    """Array arithmetic plus deterministic refresh perturbations, not a CKKS model.

    The topology hash pins calls and operand reuse from the deleted scripts.
    Numeric records independently pin replicas, inputs and diagnostic reductions.
    """

    def __init__(self, *, fail_bootstrap=False):
        self.trace, self.scalars = [], []
        self.fail_bootstrap = fail_bootstrap
        self.bootstraps = 0

    def value(self, op, args, data, level=0, failed=False):
        operands = [arg.index if isinstance(arg, Value) else "scalar" for arg in args]
        self.trace.append([op, *operands])
        self.scalars.append([float(arg) for arg in args if not isinstance(arg, Value)])
        return Value(np.asarray(data, dtype=np.float64), len(self.trace), level, failed)

    def MakeCKKSPackedPlaintext(self, values):  # noqa: N802
        return self.value("plain", (), values)

    def Encrypt(self, key, plain):  # noqa: N802
        return self.value("encrypt", (plain,), plain.data.copy())

    def Decrypt(self, ct, key):  # noqa: N802
        result = self.value("decrypt", (ct,), ct.data.copy())
        if ct.failed:
            raise RuntimeError("synthetic decode failure")
        return result

    def binary(self, name, left, right, function):
        args = (left, right)
        values = [arg.data if isinstance(arg, Value) else arg for arg in args]
        level = max(arg.level for arg in args if isinstance(arg, Value)) + (name == "mult")
        return self.value(
            name,
            args,
            function(*values),
            level,
            any(isinstance(arg, Value) and arg.failed for arg in args),
        )

    def EvalMult(self, left, right):  # noqa: N802
        return self.binary("mult", left, right, np.multiply)

    def EvalAdd(self, left, right):  # noqa: N802
        return self.binary("add", left, right, np.add)

    def EvalSub(self, left, right):  # noqa: N802
        return self.binary("sub", left, right, np.subtract)

    def EvalRotate(self, ct, rotation):  # noqa: N802
        return self.value("rotate", (ct, rotation), np.roll(ct.data, -rotation), ct.level)

    def Compress(self, ct, towers):  # noqa: N802
        return self.value("compress", (ct, towers), ct.data.copy(), 41 - towers)

    def EvalBootstrap(self, ct):  # noqa: N802
        self.bootstraps += 1
        noise = np.sin(np.arange(ct.data.size)) * 1e-5 * self.bootstraps
        return self.value("bootstrap", (ct,), ct.data + noise, 21, self.fail_bootstrap)

    def EvalBootstrapSetup(self, budget, dims, slots, correction):  # noqa: N802
        self.value("bootstrap_setup", (*budget, *dims, slots, correction), [])

    def EvalBootstrapKeyGen(self, key, slots):  # noqa: N802
        self.value("bootstrap_keygen", (slots,), [])

    def GetRingDimension(self):  # noqa: N802
        return 16384


def without_timing(value):
    if isinstance(value, dict):
        return {
            key: without_timing(item)
            for key, item in value.items()
            if "seconds" not in key
            and key != "recipe_sha256"
            and not (key == "notes" and item and isinstance(item[0], str))
        }
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return None if isinstance(value, float) and np.isnan(value) else value


def assert_record(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_record(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            assert_record(got, want)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-10, rel=1e-9)
    else:
        assert actual == expected


@pytest.mark.parametrize("suite", ["primitives", "error-growth"])
def test_recipes_preserve_original_circuit_and_records(suite, monkeypatch):
    recipe = read_object(RECIPES / f"{suite}.json")
    contexts = []

    def build(*args, **kwargs):
        backend = TraceBackend(fail_bootstrap=suite == "primitives" and not contexts)
        contexts.append(backend)
        return backend, KEYS

    monkeypatch.setattr(ckks, "make_context", build)
    monkeypatch.setattr(ckks_growth, "make_context", build)
    if suite == "primitives":
        payload = ckks.run_primitives(recipe)
    else:
        payload = ckks_growth.run_growth(recipe, list(recipe["arms"]))
    golden = read_object(ROOT / "tests/fixtures/ckks-recipes.json")[suite]
    digest = hashlib.sha256(json.dumps([c.trace for c in contexts]).encode()).hexdigest()
    assert digest == golden["trace_sha256"]
    assert_record(without_timing(payload), golden["record"])


def test_partial_growth_does_not_claim_missing_attribution():
    recipe = read_object(RECIPES / "error-growth.json")
    arms = {"F12_meta_bts_amp12": {"errors": [0.1, 0.2]}}
    report = ckks_growth.summarize(arms, recipe["arms"], 0.05)
    assert report["dominant_component"] is None
    assert report["verdict"] is None
    assert report["refresh_strategy_verdict"] is None
    assert arms["F12_meta_bts_amp12"]["projected_tokens_to_5e-2"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "suite", "unknown"),
        (None, "typo", 1),
        ("context", "batch", 1024),
        ("context", "depth", True),
        ("bootstrap", "forced_input_towers", 99),
        ("arms", "Z", {"name": "Z", "mode": "unknown"}),
    ],
)
def test_invalid_recipes_fail_before_context_creation(section, key, value):
    recipe = read_object(RECIPES / "error-growth.json")
    (recipe if section is None else recipe[section])[key] = value
    with pytest.raises(
        ValueError, match=r"expected|positive|full packing|forced_input_towers|unknown"
    ):
        ckks.validate_recipe(recipe)


def test_cli_binds_resume_to_recipe_and_keeps_existing_evidence(tmp_path, monkeypatch):
    recipe = RECIPES / "error-growth.json"
    output = tmp_path / "result.json"
    calls = []

    def run(recipe, selected, previous):
        calls.append((selected, previous))
        return {"arms": {"A_refresh_only": {"errors": [0.1, 0.2]}}}

    monkeypatch.setattr(ckks_growth, "run_growth", run)
    args = ["--recipe", str(recipe), "--output", str(output), "--arms", "A"]
    assert ckks.main(args) == 0
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        ckks.main(args)
    assert output.read_bytes() == original
    args[3] = str(tmp_path / "next.json")
    assert ckks.main([*args, "--resume", str(output)]) == 0
    assert calls[-1] == (["A"], read_object(output)["arms"])
    changed = tmp_path / "changed.json"
    payload = read_object(recipe)
    payload["seed"] = 1
    changed.write_text(json.dumps(payload))
    args[1], args[3] = str(changed), str(tmp_path / "rejected.json")
    with pytest.raises(SystemExit, match="2"):
        ckks.main([*args, "--resume", str(output)])
    assert len(calls) == 2
    assert not Path(args[3]).exists()


@pytest.mark.parametrize("arms", ["", "A,A", "Z"])
def test_cli_rejects_invalid_arms_before_loading_openfhe(arms, tmp_path):
    with pytest.raises(SystemExit, match="2"):
        ckks.main(
            [
                "--recipe",
                str(RECIPES / "error-growth.json"),
                "--output",
                str(tmp_path / "out.json"),
                "--arms",
                arms,
            ]
        )
