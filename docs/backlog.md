# Next work

[Current state](status.md) is the handoff entry point. This file contains open
work; completed experiments and rejected implementations are in the
[evidence registry](evidence.md) and [research index](research/README.md).
A listed task is not approval to start an unbounded campaign.

| Priority | Work | Acceptance / next decision |
| --- | --- | --- |
| P0 | First-call refresh accuracy, Mamba-3 64-token request | Compare original pre/post RNS inputs and cache entries, then packing/bootstrap/correction/extraction with the accurate replay. Repair a demonstrated violation and qualify an ordinary frozen 64-token request at both 0.001 gates. [Diagnosis](research/2026-09-28-long-accuracy.md) |
| P1 | Longer sessions and repeated keys/prompts | Preserve classical-128, frozen references and final-output gates. A recurrent component's 64 steps do not qualify full-model generation. Report all failures. |
| P1 | GPU performance after accuracy closure | Begin with a measured bottleneck and a bounded candidate count. Preserve the adopted refresh and public-state reuse; indexed mask lookup remains unqualified. [Last decisions](../results/b300/2026-09-28/long-accuracy/decision.json) |
| P1 | PBI-SEC-001: full Mamba-2 classical-128 chain | Mamba-3 qualification does not qualify Mamba-2. Report actual QP, memory, errors, generated IDs and whole-request runtime. |
| P1 | PBI-M4-002 / PBI-QUALITY-001: Mamba-2 horizon and quality | Extend repeated-key/prompt and native multi-token coverage while retaining frozen coefficient, PPL and domain evidence. |
| P1 | Client/server separation | Independent processes, protocol messages, key ownership and serialization; the inline client loop is not process-separated private chat. |
| P1 | PBI-OPT-006: native scan prefill | Verify chunk summaries, carry, slot layout and depth/live-memory costs before comparing serial and parallel execution. |
| P1 | PBI-OPT-003: graph refresh planning | Preserve fan-out/live-out constraints and measured error; deleting events from a trace is insufficient. |
| P1 | PBI-ARCH-001: FHE-oriented SSM training | Held-out quality, retrieval, generation and encrypted cost for smaller/structured states and bounded nonlinearities. |
| P1 | PBI-LOOP-001: encrypted token selection | Start with a declared small vocabulary; quantify approximation margins and output security before the complete vocabulary. |
| P2 | Native ownership decomposition | Extract touched responsibilities from the large executor behind tests; keep lifetime changes separate from mathematical optimization. [Maintenance](maintenance.md) |
| P2 | PBI-M4-001 / PBI-OPS-102: historical 0.4.5 evidence | Recover the original artifact or rerun its exact scope before creating that historical tag. New B300 results do not substitute for it. |

A task is complete when the stated gate passes or the bounded experiment ends
with an explicit rejection/limitation. Do not turn a passing repeat into a
replacement for a failed control. Update [status](status.md) when the active
question or qualified baseline changes.
