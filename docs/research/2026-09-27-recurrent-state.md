# Fixed-size Mamba-3 recurrent state — 2026-09-27

The shared Mamba-3 implementation now supports a state split across ciphertexts
along the head axis. It retains every state coordinate and uses a fixed amount
of recurrent storage. This is an opt-in prerequisite for long generation, not
a replacement for the qualified short-generation exporter.

The first encrypted comparisons preserve the classical-128 context and pass
the existing 0.001 numerical gate at 4, 16 and 40 steps. The fixed state is slower
at 4/16 steps, then reduces state arithmetic by **12.0% at 40 steps**
(126.229 → 111.084 s); total subcircuit evaluation improves **4.75%**
(291.114 → 277.288 s). Both representations fail the 64-step resource gate with
CUDA out-of-memory errors, before producing a native success record. There is
no measured 64-step speedup or encrypted long-text generation claim.

The subsequent [memory-admission study](2026-09-27-recurrent-memory.md) resolves
both 64-step allocation failures with an optional scheduler threshold. The
measurements below remain the original unlimited-lookahead controls.

## Algebra and shared implementation

For one head, with channel vector v and state vectors b and c, the original
exponential-trapezoidal recurrence is

```
S[t] = decay[t] S[t-1]
     + lag[t] v[t-1] b[t-1]^T
     + write[t] v[t] b[t]^T
y[t] = S[t] c[t] + D v[t]
```

The existing factored path retains each past `(b, v, weight)` and evaluates
every contribution to each readout. It is exact, but its stored history grows
linearly and its total recurrence work grows quadratically with step count.
The new path explicitly updates S. Multiplying each small v by its coefficient
before forming the outer product avoids another multiplication over the whole
state. This changes floating-point operation order, not the mathematical state
or its rank. An independent NumPy oracle keeps the original outer-product order.

`HeadTiledState` and `ssm_update_readout` in `tensor_ops.py` implement the shared
storage/update algebra. `mamba3_state_step` serves both the full mixer and the
probe; the factored branch preserves the original history update.
`PackedProgram.recurrent_state` partitions complete heads and supports a final
partial group. One complete head must fit in the logical slot capacity.

The trained layer has 24 heads, 64 channels and state width 128: 196,608 state
coordinates. At 32,768 logical slots this occupies six ciphertexts. The previous
key and value add two more, for **8 carried ciphertexts / 201,216 coordinates**.
The recurrence probe excludes the rotary angle, which belongs to the upstream
mixer. A complete mixer additionally carries that angle.
The factored carry is **3T ciphertexts / 4,632T coordinates** after T steps.

The exported node counts are `143T + 9` for the fixed state and
`3.5T² + 10.5T` for the factored state. These include inputs and initial constants,
not only live encrypted work. They establish the structural complexity of this
lowering, not a measured latency law.

## Workload and measurement boundary

The [reusable probe](../../experiments/recurrent_state/README.md) captures layer
zero during 64 exact plaintext autoregressive evaluations of the pinned SISO
187M model, starting from `The capital`. Every length uses a prefix of the same
trace. Both encrypted arms receive identical b, c, v, decay, lag and write
values at the recurrence boundary. Projections, nonlinearities, accumulated
rotary angles and token selection are outside this encrypted subcircuit.

Every step's readout is retained and checked; the fixed-state arm also checks
all coordinates of its final state. Earlier instrumentation retained only
selected readouts, allowing dead-node elimination to remove benchmarked work.
Those attempts are excluded. The maintained analyzer checks the executed
readout-reduction count against the exported graph to prevent that mistake.

All runs use the same qualified native binary and dependency hashes from the
[GPU plaintext FFT study](2026-09-27-b300-gpu-plaintext-fft.md), the same B300,
CPU placement, preparation workers and plaintext-cache capacity. The context
uses N=131,072, 65,536 physical slots, uniform ternary secrets, HYBRID digits 4
and an audited QP of 3,376 bits against the 3,523-bit classical-128 guideline.
Two-pass S2C-first refresh, frontier scheduling and merged correction remain
enabled. Both absolute-error limits remain 0.001. This is the existing parameter
qualification, not a new cryptographic proof or complete service protocol.

Refresh bounds use this trace's magnitudes with eightfold headroom and a minimum
of one. They are empirical fixture bounds, not certified bounds for arbitrary
prompts. No intermediate or token-feedback decryption occurs in the evaluator;
the harness decrypts retained outputs for validation.

## Measured short comparisons

Each row is one process, with fresh setup. Setup and full wall time are retained
separately in the evidence. Evaluation includes encryption/materialization of
the offline inputs; a complete model would supply those inputs as ciphertexts
from its preceding operators. The arithmetic column sums operation-profile
times excluding `input` and `public`. It includes any refresh inside those
operations. It is not an autoregressive seconds-per-token measurement.

| Steps | Representation | Evaluation (s) | State arithmetic (s) | Maximum exact error |
| ---: | --- | ---: | ---: | ---: |
| 4 | Factored | 14.892 | 1.868 | 4.37e-11 |
| 4 | Fixed state | 30.497 | 11.405 | 8.94e-9 |
| 16 | Factored | 85.966 | 23.093 | 1.23e-10 |
| 16 | Fixed state | 107.477 | 43.755 | 1.82e-8 |
| 40 | Factored | 291.114 | 126.229 | 3.42e-7 |
| 40 | Fixed state | 277.288 | 111.084 | 1.22e-5 |

The 4/16-step runs use no bootstraps. The 40-step factored run completes six
physical bootstraps in 0.839 seconds; the fixed state completes 46 in 4.206
seconds. Its maximum readout/final-state error remains below 0.001. Fixed-state
arithmetic averages 2.85, 2.73 and 2.78 seconds per step at the three lengths.
These points do not establish unlimited-horizon cost or numerical stability.
At 16 steps, `repeat` accounts for 33.978 of 43.755
seconds (77.65%) of fixed-state arithmetic. Broadcast placement, not the
pointwise product kernel, is the largest measured ordinary-operation target.

At 40 steps, the carried ciphertext count falls from 120 to 8. Ciphertext
multiplications fall from 3,240 to 1,040 and reported rotations from 47,068 to
39,614. Input materialization costs 164.849 versus 166.135 seconds, diluting the
arithmetic improvement in the total evaluation metric. Forty is the first
measured win, not an identified universal crossover: there is one sample per arm,
one prompt and no measurements at intervening lengths. Keep the existing
factored default for short generation; retain tiled state as an opt-in prototype.

## Resource failures and executor limits

Both 64-step jobs run out of device memory while copying/adjusting a plaintext
for ciphertext multiplication. Symbolized allocation stacks identify
`Plaintext::adjustPlaintextToCiphertext` and `Ciphertext::multPt`; they do not
identify which retained allocations dominate the total working set.

| Representation | Last completed-node report | Process wall (s) | Native result |
| --- | --- | ---: | --- |
| Factored | 7,700 / 15,005; zero bootstraps | 523.075 | Missing; CUDA OOM |
| Fixed state | 6,660 / 9,160; zero bootstraps | 515.074 | Missing; CUDA OOM |

The dependency exits zero after these errors. The analyzer therefore requires
the native result and its correctness gates, rather than accepting a process
exit code as success. Neither failed timing enters a speedup calculation.

The static source-order inventory finds peak live DAG handles of 260 versus 87
at 64 steps. This is not measured peak GPU memory: it excludes keys, mask cache,
refresh/rotation scratch, ciphertext levels and the runtime frontier order.
Even the fixed-state probe retains T validation readouts and an unrolled DAG.
The frontier scheduler may advance later independent inputs and writes while
an earlier branch waits for refresh. Only client-feedback nodes delimit epochs;
this offline-input probe has no such barriers. A bounded recurrent carry
therefore does not establish a bounded executor working set. Streaming admission
or a memory budget for ready work needs a separate measured qualification.

For scale, eight two-component ciphertexts at all 45 Q primes contain
`8 × 2 × 131072 × 45 × 8 = 754974720` bytes (720 MiB) of raw RNS coefficients.
That calculation excludes duplicate storage, P limbs, keys, caches and scratch;
it is not peak allocation telemetry. Tracking only the eight logical state
handles cannot account for all allocations present at the failure.

## Full-model domain gate

An exact CPU replay audits every nonlinear input against the independently
frozen short-generation domains, without changing any fitted coefficients.
Across 64 steps, **74 sites leave their domains; the first violation is at
step 8**. Layer-zero sin/cos leave their interval at step 9: the frozen interval
is approximately [-22.83, 23.53], while observed accumulated angles reach
[-153.89, 172.77]. There are also 27 inverse-square-root sites outside their
domains, so rotary handling alone is insufficient.

This prevents simply increasing the existing short-export length limit. Next
gates are bounded executor memory, more efficient state broadcasts and a
long-horizon nonlinear-domain/accuracy qualification with real encrypted token
feedback. Fixed-size state gives this lowering linear algebraic work; it does
not establish long-generation accuracy, text quality or an advantage over a
matched Transformer workload.

A possible exact coordinate change would replace accumulated-angle evaluation
with angle increments. Let `R[t]` rotate the unrotated b/c vectors, and define
`U[t] = S[t] R[t]` and `R_delta = R[t-1]^T R[t]`. Then

```
U[t] = (decay[t] U[t-1] + lag[t] v[t-1] b[t-1]^T) R_delta
     + write[t] v[t] b[t]^T
y[t] = U[t] c[t] + D v[t]
```

Here b and c are unrotated vectors. The identity follows by substituting
`R[t]^T R[t] = I`; it does not discard history. Its trigonometric arguments
would be per-step increments, while rotating the state adds layout/arithmetic
work. This is a design candidate, not an implemented or timed FHE optimization.
It would still require long-horizon domain checks for the other nonlinearities.

## Evidence and reproduction

[Curated artifacts](../../results/b300/2026-09-27/recurrent-state/README.md)
retain successful native results, both resource failures, the trace, manifests,
domain audit, static inventory and measured sources. Local machine identities
are publication substitutions; numerical arrays are unchanged. Original and
published hashes are recorded separately.

The maintained probe documents capture, both exports, domain auditing and
comparison. Tests cover complete-mixer lowering with a multi-ciphertext state,
uneven head groups, repeated updates against an independent dense oracle,
retained readouts, source/input matching and failure handling. Full-model long
encrypted generation remains unqualified.
