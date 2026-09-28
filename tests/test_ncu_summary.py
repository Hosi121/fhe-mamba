import csv
import importlib
import io

import pytest


def test_raw_counters_preserve_units_and_reject_invalid_measurements(monkeypatch):
    module = importlib.import_module("fhemamba.profiling.ncu")
    extra = "sm__issue_active.avg.pct_of_peak_sustained_elapsed"
    fields = [
        "ID",
        "Kernel Name",
        "Device",
        "Block Size",
        "Grid Size",
        *module.METRICS.values(),
        extra,
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerow({**dict.fromkeys(module.METRICS.values(), "unit"), extra: "%"})
    row = dict.fromkeys(module.METRICS.values(), "1,024.5")
    row.update({"ID": "0", "Kernel Name": "test", "Device": "0"})
    row[extra] = "58.5"
    writer.writerow(row)
    text = stream.getvalue()
    result = module.summarize(io.StringIO(text))
    assert len(result["kernels"]) == 1
    assert result["kernels"][0]["metrics"]["duration"]["value"] == 1024.5
    assert result["kernels"][0]["metrics"]["duration"]["unit"] == "unit"
    assert extra not in result["kernels"][0]["metrics"]
    selected = module.summarize(io.StringIO(text), [extra])
    assert selected["kernels"][0]["metrics"][extra] == {
        "value": 58.5,
        "unit": "%",
        "counter": extra,
    }
    with pytest.raises(ValueError, match="required metrics"):
        module.summarize(io.StringIO(text), ["missing-counter"])
    with pytest.raises(ValueError, match="invalid hardware counter"):
        module.summarize(io.StringIO(text.replace('"1,024.5"', "nan")))
    writer.writerow(row)
    with pytest.raises(ValueError, match="duplicate"):
        module.summarize(io.StringIO(stream.getvalue()))
