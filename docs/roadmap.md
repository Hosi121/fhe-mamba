# Roadmap

The active line is `fhemamba` 0.5.x on DGX Spark. The objective and invariants
are in [design.md](design.md); [backlog.md](backlog.md) records executable gates.
B300 recovery is historical provenance work and does not block Spark progress.

The stabilized prompt-to-text baseline now passes: tokenize the whole prompt,
carry encrypted state through every layer, select the next token at the client
and feed back its encrypted embedding. Retain this end-to-end correctness gate
while shortening its runtime. See the
[five-evaluation result](research/2026-09-22-client-generation.md).

## 1. Reproducible Spark baseline

Build pinned CUDA 13 / SM121 dependencies in an isolated prefix. Use calibrated
payloads and a 5 GiB plaintext cache. Establish bootstrap, full-width layer,
and 24-layer gates. Record source/binary/library identity, exact configuration,
per-token errors, operation counts, cold/warm timings and peak memory.

Bind surrogate-quality measurements to those exact payload coefficients and
head masks. The historical PPL certificate does not identify the current
payload, and largest-step head pruning is not a uniform memory-negligibility
proof. Audit per-head domains and long-context behavior alongside circuit work.
The legacy frozen payload fails 1,024- and 4,096-token plaintext screens;
SiLU/normalization extrapolation is now a blocking quality issue. Exact rational
normalization certificates and public-weight activation envelopes are the
next implementation basis, with domain membership and CKKS slack explicit.

Scheduled RMSNorm now passes an opt-in native **24-layer/two-fixed-token**
gate, including all 49 sites, encrypted state carry and internal refresh.
Maximum error against its polynomial reference is 0.007136 at
`security=not-set`; this does not close the full-chain 128-bit gate. The same
payload still fails a 1,024-token plaintext quality screen because it retains
the legacy non-normalization fits. See the
[integration evidence](research/2026-09-21-normalization-integration.md).

The separate [complete-candidate payload](research/2026-09-21-stabilized-native.md)
now includes joint gates and public-envelope activations, with all 121 frozen
recipes preserved. Its matching 1,024/4,096-token plaintext prefixes are finite
with no observed domain escapes. The native **24-layer/two-fixed-token** gate
passes at error **0.003311**, with zero intermediate decryptions, **831**
bootstraps, **1,192.95 s** evaluation and **36.09 GiB** peak RSS. This remains
`security=not-set`. Joint-gate evaluation accounts for **552.24 s** in that
fixed-input baseline and motivates the encoding optimization below. Longer
encrypted horizons and 128-bit full-chain gates remain open.

Compare the binary replication candidate against linear replication on the
same build and payload. Promotion needs passing multi-token accuracy and a
measured end-to-end benefit; fewer logical operations alone are insufficient.

The first [joint-gate encoding optimization](research/2026-09-22-periodic-gate-coefficients.md)
now passes the same 24×5 generation gate: period-32 coefficients and a masked
basis reduce evaluation **3,038.12 -> 2,573.37 s (15.3%)**, with unchanged generated
IDs, depth and 2187 bootstraps. Maximum polynomial-circuit error is **0.011767**,
within the existing 0.05 limit. The interleaved one-layer comparison also passes.
Keep this opt-in result bound to its frozen payload while investigating the
remaining full-ring plaintext conversion and bootstrap costs.

The [subring follow-up](research/2026-09-22-subring-gate-encoding.md) now replaces
the full-ring coefficient NTT with a private 64-point transform and expansion.
It passes exact plaintext parity, repeated smoke and complete generation:
**2,573.37 -> 2,310.80 s (10.2%)**, maximum polynomial error **0.016255**,
with all generated IDs, operation counts and levels unchanged. It stays opt-in.
The [normalization lower bounds](research/2026-09-22-normalization-bounds.md)
identify the next experiment: preserve the public domain and precision while
seeking one fewer internal refresh. The existing stage-11 refreshes cost
227.28 seconds in the periodic baseline; removing them is not yet validated.

## 2. Autoregressive horizon

The stabilized prompt-2/generate-4 run now passes five sequential encrypted
evaluations and reconstructs text from the measured selected IDs: `The capital`
becomes `The capital of the Republic of`. Maximum polynomial-circuit error is
0.009192, evaluation takes 3,038.12 s and peak RSS is 37.05 GiB. This remains a
one-process client loop with `security=not-set`.
`run_dgx_generation.py` prepares the full prompt without truncation, preserves
the frozen operator recipes/bounds and collects the result from Spark.
Next repeat keys/prompts and expand through 16, 64 and 256 steps on held-out
prompts. Require successful decryption
and polynomial-circuit error <= 0.05 at every step, matching generated IDs,
and no intermediate diagnostic decrypts. Record where failure first appears.
Row-normalized state now passes two fresh-key five-step runs with unchanged
operation count; this does not close calibration coverage or new-prompt gates.

Output fusion, paired normalized-state refresh and local bootstrap policy
remain independently controlled candidates on Spark. Do not copy B300 memory
settings or relax the numerical threshold to promote them.

## 3. Separate client and server

Measure the implemented fixed-vector `client-init -> server-eval ->
client-decrypt` roles. The server must have no secret key or debug-decrypt path.
Track ciphertext/key transfer size and setup amortization. Then add a persistent
interactive session or explicit encrypted state serialization for generation.
This protocol work can proceed alongside numerical horizon research.

## 4. Full-chain security and output protocol

Extend the layer-0 128-bit parameter result to all 24 layers, with bounded key
memory and the same numerical gate. Separately specify the output/decryption
security construction and its compatibility with bootstrapping. Parameter
selection alone is not a complete security proof.

## 5. Long-context prefill and FHE-oriented training

Implement an encrypted affine-scan prefill only after its slot layout, serial
versus parallel chunk carry, live ciphertext memory and level schedule are
validated. The existing analytic prefill model is not measured native execution.

Maintain a separate training branch for bounded nonlinearities, smaller state
and cheaper normalization. Compare held-out quality and encrypted cost against
the checkpoint anchor. New architecture work need not wait for B300 or follow
the newest Mamba variant; it must have a measurable accuracy/cost argument.

The [algebra/cryptography survey](research/2026-09-21-ssm-cryptographic-design.md)
adds concrete experiments: deferred rank-one writes with an immutable state
checkpoint, compatible state-basis changes, constrained composite decay, and
direct complex SISO versus a rotating frame. The first has real-checkpoint
factor parity, and channel-row scaling has encrypted five-step precision
evidence. Deferred state and complex dynamics still need native measurements.
Joint decay/write inequalities can certify a length-independent state bound;
bounded EMA or centered updates require a separate trained-model comparison.

## 6. Autonomous encrypted generation

Keep vocabulary projection, selection and embedding lookup encrypted between
tokens, and deliver only final ciphertext output. Start with a small vocabulary
and CKKS/FHEW comparison; measure switching costs and require a clear
logit-error/margin contract. Public weights remain the initial model. Training
for cheaper selection or discrete state is a separate quality/cost experiment.

## Historical and version boundaries

Recover the missing B300 three-token artifact if the original machine/data
become available. Do not reconstruct it from prose or retag a failed artifact.
B300 synchronization experiments are deferred while that platform is unavailable.

`0.5.0` names the compatibility-stack retirement, not completion of the protocol.
A future `1.0.0` requires reproducible interactive encrypted generation, a
secret-key-free server, full-chain validated security parameters, and an explicit
output-security statement. That version is an interactive milestone; the
autonomous encrypted loop remains the stronger research endpoint. Private
model weights are an additional protocol goal.
