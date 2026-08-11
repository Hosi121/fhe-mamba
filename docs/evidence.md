# Evidence registry

This registry distinguishes tracked execution artifacts from measurements that
currently exist only in research notes. It is the provenance source for
headline README claims.

## Evidence classes

- **Raw/tracked**: repository contains the backend output JSON.
- **Raw/tracked, legacy schema**: output is present but predates current commit
  or binary provenance requirements.
- **Derived/tracked**: report generated from named raw artifacts.
- **Documented only**: value is recorded in prose, but the raw success artifact
  is not in the repository. It must not be represented as raw evidence.

## Current headline evidence

| Claim | Value | Class | Source / action |
|---|---|---|---|
| Mamba-2 polynomial quality | PPL `22.307 -> 22.333`, 280 windows | Raw/tracked, legacy schema | `fhemamba/results/ppl_ladder_mamba2_frozen_cert.json` |
| Decode lowering parity | max error `3.0518e-5` over five verified tokens | Raw/tracked, legacy schema | `fhemamba/results/decode_budget_mamba2.json` |
| Official Transformers parity | matching next token; logits max difference `4.158e-4` | Raw/tracked, legacy schema | `fhemamba/results/parity_mamba2-130m-hf.json` |
| 24-layer, three-token B300 chain | errors `0.01295 / 0.01173 / 0.03475`; eval `145.75 s`; 469 physical bootstraps; 120.24 GiB peak RSS | Documented only | `docs/research/2026-07-13-fhe-mamba-bottleneck-survey.md`; recover/rerun under PBI-M4-001 |
| 128-bit layer-0, two-token probe | errors `0.01182 / 0.03124`; eval `393.23 s` | Raw/tracked, legacy schema | `fhemamba/results/dgx/m1_decode_128bit_r131072_d43_s59_t2.json` |
| Three-process key separation probe | max error `1.7867e-12` | Raw/tracked | `fhemamba/results/dgx/client_server_probe.json` |
| Existing tracked B300 raw results | three 24-layer/one-token failures at package `0.4.4` | Raw/tracked | `fhemamba/results/b300/`; negative evidence only |

## B300 `0.4.5` recovery contract

The recovered or regenerated success artifact must satisfy all of the
following before the README evidence class changes to **Raw/tracked**:

- `status=passed` and `passed=true`;
- `repo_commit` identifies the evaluated source;
- `binary_sha256` identifies the executable;
- 24 loaded layers, final RMSNorm, and three sequential tokens;
- zero intermediate decrypts and real ciphertext state/FIFO carry;
- `security=not-set`, ring `65536`, scale `59`;
- fully synchronized FIDESlib profile;
- fused replicated transform enabled for `out-proj` only;
- complex state pairing enabled;
- all token errors `<= 0.05`;
- per-token error, timing, bootstrap, rotation, cache, and RSS telemetry.

If the original JSON cannot be recovered, rerun the exact promoted baseline.
Do not infer missing fields from the prose measurement.

## Promotion policy

A new headline claim requires:

1. a tracked raw artifact or a tracked derived report whose raw inputs are
   named and available;
2. validation with `fhemamba validate-artifacts --require-commit`;
3. matching README, evidence registry, backlog, and package version text;
4. explicit security, model, token-horizon, process-separation, and hardware
   scope.

Failed and negative artifacts remain valuable. They must be labeled as such
and must never be used as substitutes for a missing success artifact.
