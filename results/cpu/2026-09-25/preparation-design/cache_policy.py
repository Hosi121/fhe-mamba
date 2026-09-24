"""Reproduce capacity/admission counts from the static public-weight trace."""
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "cache-access-trace.json"
data = json.loads(source.read_text())
trace = [(call["weight_node"], diagonal) for call in data["linear_calls"]
         for diagonal in range(call["diagonals"])]
counts = Counter(trace)
identities = list(counts)
size = len(identities)
assert len(trace) == 22440 and size == 4488
assert set(counts.values()) == {5}
cycle = trace[:size]
assert len(set(cycle)) == size and trace == cycle * 5

def lru(capacity):
    resident = OrderedDict()
    hits = 0
    for key in trace:
        if key in resident:
            hits += 1
            resident.move_to_end(key)
        elif capacity:
            if len(resident) == capacity:
                resident.popitem(last=False)
            resident[key] = None
    return hits

rows = []
for capacity in [0, 64, 128, 512, 1024, 2048, 4096, 4488, 8192]:
    capacity = min(capacity, size)
    if rows and rows[-1]["entries"] == capacity:
        continue
    selected = set(identities[:capacity])
    pinned_hits = sum(counts[key] - 1 for key in selected)
    hits = lru(capacity)
    assert hits == (4 * size if capacity >= size else 0)
    assert pinned_hits == 4 * capacity
    rows.append({"entries": capacity, "ifft_bytes": capacity * data["slots"] * 16,
                 "lru_hits": hits, "pinned_subset_hits": pinned_hits,
                 "pinned_subset_fraction_of_encodes": pinned_hits / len(trace)})

result = {"trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "linear_calls": len(data["linear_calls"]), "accesses": len(trace),
          "unique_identities": size, "identical_cycles": 5,
          "reuse_distance_distinct_identities": size - 1, "capacity_curve": rows,
          "scope": "Static program-order identity counts, not runtime hits or speed. The inverse-FFT identity is independent of scale/level; compressed final coefficients additionally need their actual scale/level/degree keys. All resident values remain immutable. Pinned subset is a simple admission policy, not a claim of optimal scheduling."}
(ROOT / "cache-policy.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
