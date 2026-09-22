# Reproduction configuration

- `reproduction.json` pins the public model revision, its six required file
  hashes, and the two approximation bundles below.
- `mamba2-130m-normalization-20260921.json` contains 49 public normalization
  recipes and their input/error contract.
- `mamba2-130m-gates-20260921.npz` is the byte-identical frozen joint-gate
  coefficient bundle used in the September 22 generation measurements.
  SHA-256: `c15a58e757b928881d11602f162ba095ac597c9de1cc23cb94a09d669cdaf743`.
- `dgx-spark.env` identifies the CUDA/FIDESlib build used by the Spark scripts.
- `b300-platform.env` retains the historical B300 build configuration.

See [the reproduction guide](../docs/reproducing.md). The export path validates
the approximation certificates against the checkpoint and regenerates matching
references. A certificate over a public interval does not prove that every
possible model input remains inside that interval.
