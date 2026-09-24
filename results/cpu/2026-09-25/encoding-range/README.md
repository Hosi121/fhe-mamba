# Local OpenFHE Encode range reduction

The prototype replaces per-component logarithms in the 64-bit Encode range
scan with one logarithm of the largest absolute component. This is a local
upstream OpenFHE experiment, not an installed FIDESlib/DGX optimization.

The source commit is `aa391988d354d4360f390f223a90e0d1b98839d7`.
Its encoding source is byte-identical to the pinned DGX encoding source; the
other FIDESlib integration patches are absent from this CPU build. Original
and modified encoding sources retain their upstream license headers, and
`OPENFHE-LICENSE` gives the dependency license. Only the PKE shared library
differs between the immutable baseline/candidate library copies.

`encoding_probe.cpp` compares every coefficient and metadata word in 96
successful encodes, plus 24 matching expected small-scale failures. It tests
real/complex contexts, two slot counts, three levels, two scale degrees and
zero/dense/sparse/large/tiny fixtures. 167,772,160 coefficient words agree.
`identity.json` records the 1.25-GiB reference stream hash and length; the
stream is reproducible from fixed fixtures and is omitted from Git. Candidate
validation reads and compares its full contents; it does not rely on a digest
alone. The probe uses identity-root parameters, as the shared GPU preparation
path does before replacing roots and uploading. No GPU or decryption is used.

`measure.py` runs eight independent processes in ABBA/ABBA order, with one
probe executable and alternate library paths. It verifies actual `ldd` bindings.
Both modes pin CPU 0, use the same empty-context setup and warm-up, and suppress
OpenMP during encoding. Each of the four fixture/context cases averages 24
Encode constructions per sample; input creation, context setup and release
are outside the timer. All samples and the noisy initial extracted-loop
experiment (in the preceding ownership evidence) remain recorded.

Run `python3 compare.py` here to reproduce the aggregation. For fresh execution,
`python3 reproduce.py /absolute/path/to/an/unused/output-directory` builds both
CPU library variants, compares the full coefficient stream and repeats the
timing schedule. It requires Git, CMake and a C++20-capable GCC toolchain; it
keeps build intermediates in the selected directory for inspection.

For manual execution,
build the pinned source with the options in `configure-base.log`, compile the
probe using `probe-build-command.json` with paths adjusted to the new directory,
and preserve all baseline shared libraries. Write the golden stream with
`encoding_probe write GOLDEN OUTPUT_JSON`; apply `reduce-encode-logarithms.patch`
and rebuild PKE into a separate immutable library directory. With that directory
in `LD_LIBRARY_PATH`, run `encoding_probe compare GOLDEN OUTPUT_JSON`. The
`bench` mode and `measure.py` reproduce the timing schedule. All paths in raw
commands describe the original local run and are not portable recipes as-is.

The initial probe compile failure from a `Params` name collision is retained
under `failed-probe-build/`; the retry uses `EncodingPolyParams`. A dependent
launch attempted before that retry had no executable and produced no timing.
The actual numerical and benchmark runs all passed after the probe compiled.

See the [study](../../../../docs/research/2026-09-25-encoding-range.md) for
2.3–3.9% local full-encode reductions and limits. Finite scaled inputs are the
prototype's tested contract; invalid/nonfinite/overflow policy, target exact-RNS
checks, target timing and model-wide effects remain unresolved. The production
encoder and DGX dependency are unchanged.
