# Delayed modular reduction in the forward NTT

This optional FIDESlib experiment keeps 64-bit Shoup butterfly residues in
`[0, 4p)` for `p < 2^62`. It restores canonical residues at each NTT half's
boundary, before cross-twiddles and fused consumers. Other arithmetic paths
retain the original butterfly. The inverse NTT is unchanged.

The [classical-128 B300 trial](../../docs/research/2026-09-28-long-accuracy.md)
passes 373,293,056 exact output words but is 1.32% slower in an all-layer
two-token screen. This implementation is retained for reproduction, not adopted.

The arithmetic follows Algorithm 4 of [Harvey's paper](https://arxiv.org/abs/1205.2926).
That paper's inverse butterfly corresponds to the CT network used here.
Reduced modular correction work is a hypothesis about performance, not a
guarantee: instruction scheduling and register allocation also matter.

```bash
python experiments/ntt_lazy/check_arithmetic.py
python experiments/ntt_lazy/rebuild.py --help
```

`rebuild.py` copies an existing backend source into a new output directory,
checks and applies the patch, and rebuilds only `NTT.cu.o` and its CUDA device
link. It reuses the original CMake compiler/link flags. Every other archive
member must retain identical contents and order, including duplicate names.
Use `--container NAME` when the original toolchain is in a retained container
with the source/build paths mounted at the same absolute paths. Python runs
on the host; the compiler and archive commands run in the container.

Configure this directory twice, once against each archive, to build
`ntt_lazy_oracle`. It shares the maintained probe with
[`ntt_warp_tail`](../ntt_warp_tail/README.md). For the classical-128 experiment,
record with the baseline and compare with the candidate:

```text
ntt_lazy_oracle baseline.json golden.bin 131072 --digits 4 --special-primes --record
ntt_lazy_oracle candidate.json golden.bin 131072 --digits 4 --special-primes
```

The oracle covers both NTT halves, all six fusion modes, 1/4/44 Q limbs,
zero/near-modulus/alternating/random inputs, the remaining Q source and all P
primes. Reported output-word counts exclude the parameter header. A passing
coefficient comparison does not establish inference speed or qualification;
follow with a frozen all-layer inference comparison and unchanged error gates.
