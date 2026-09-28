# Roadmap

The goal is reproducible encrypted inference with explicit accuracy, security
and protocol boundaries. [Current state](status.md) records what is qualified;
[the backlog](backlog.md) records the next executable gates. Historical timing
comparisons belong in [the evidence registry](evidence.md).

## 1. Reliable long generation

Resolve the first-call refresh failure in the frozen Mamba-3 64-token request.
Then extend prompts, keys and horizons without changing the classical-128 or
0.001 final-hidden gates. Distinguish one-layer recurrent-state experiments,
fixed-input evaluation and actual token generation.

## 2. Measured whole-request performance

Profile the current qualified configuration before selecting a bounded set of
candidates. Compare the same payload, device placement, security and precision.
Count preparation, key/cache memory and evaluation separately. Retain rejected
candidates and failed controls; kernel speed alone does not establish a model
speedup. Avoid restarting older searches without new evidence.

## 3. Protocol and output security

Separate client and server processes, establish ownership and message formats,
and document what the server and client observe. Parameter selection alone
does not establish protocol security. The current client decrypts the final
hidden vector to select the next token; encrypted selection is a separate gate.

## 4. Architecture and prefill

Validate chunk-summary carry and native scan prefill before long-context
speed claims. FHE-oriented training may change state structure, gates and
normalization, with explicit held-out quality and encrypted cost comparisons.
Keep pretrained checkpoint experiments reproducible while exploring new models.

## 5. Sustainable implementation

Maintain one common arithmetic backend for Mamba-2 and Mamba-3. Share experiment
I/O, process lifecycle, qualification, diagnostics and profiling in the Python
package. Add workload parameters to manifests instead of copying controllers.
Keep operating settings local and publish reviewed, hash-bound evidence.
