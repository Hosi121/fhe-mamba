# Static waste audit evidence

These files support the [static audit](../../../../docs/research/2026-09-24-static-waste-audit.md).
They contain operation opportunities, not measured speedups. No new encrypted
inference was run for this audit.

- `analyze.cpp`, `inventory.json`: source-level call inventory of the actual
  frozen 12-layer, five-evaluation Mamba-3 program; includes all-integer polynomial
  split scan at the existing coefficient cutoff and no higher symbolic depth.
- `inventory-powers-of-two.json`: initial narrower scan retained for provenance.
- `m3-native.json`, `m3-run.json`: unchanged completed borrowed-upload/routing
  candidate artifacts (1,004.62 s).
- `m2-native.json`, `m2-run.json`: unchanged prior complete direct-upload artifacts
  from `../packed-resources/m2-full-direct/` (2,029.90 s).
- `compiled-sources.json`, `target-provenance.json`: unchanged Mamba-3 campaign
  source manifest and target identity. `provenance.json` binds the analyzer,
  payload and inspected dependency source; 111 local source files were checked.
- `derive.py`, `findings.json`: raw-hash checks, static/runtime count agreement,
  candidate counts and explicitly conditional memory/phase bounds.
- `sha256.json`: checksums of this evidence bundle, excluding itself.

Recompute the derived findings without the model or GPU:

```bash
python3 results/dgx/2026-09-24/static-waste-audit/derive.py
```

To regenerate the inventory, use the source versions in `compiled-sources.json`
and a `program.txt` matching `provenance.json`. From the repository root:

```bash
g++ -O2 -std=c++20 -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -Inative/fideslib_stage0/include -Inative/fideslib_stage0/src \
  results/dgx/2026-09-24/static-waste-audit/analyze.cpp \
  native/fideslib_stage0/src/stage1_mamba2_plan.cpp -o /tmp/fhemamba-static-audit
/tmp/fhemamba-static-audit /path/to/program.txt > /tmp/fhemamba-static-inventory.json
```

The inventory parses public coefficients and executes planning arithmetic only.
It does not encrypt, decrypt, evaluate ciphertexts or benchmark model inference.
The original analyzer binary was removed after its hash and compiler command
were recorded. The large model payload and third-party source copies are not
duplicated in this evidence bundle.
