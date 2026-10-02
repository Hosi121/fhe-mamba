# Running and publishing experiments

## Shared local client heads

Mamba-3 exports and prefix payloads store `client_head.f32` as a relative symlink
to `<output-parent>/.client-heads/<sha256>.f32`. Byte-identical heads in sibling
payloads share one read-only file. The native FP32 format and manifest hashes
are unchanged; normal readers and the native runner follow the link directly.
The exporter refuses existing destinations, verifies existing blobs, and never
writes through a payload link. Do not make a shared blob writable to regenerate
a head: use a new output directory.

Keep `.client-heads` when moving or backing up the whole output group. When
copying just one payload elsewhere, dereference links (`cp -rL` or `rsync -aL`)
to make a standalone copy. Removing one payload does not remove its shared blob;
there is no automatic garbage collection of shared heads. Old payloads with
ordinary files remain readable. This changes storage only, not model arithmetic
or GPU qualification.

The maintained interface is the installed `fhemamba.benchmarks` module and the
workload runners in [`experiments/`](../experiments/README.md). A study consists
of a versioned job specification, local machine settings, raw output, and a
reviewed public export. A new study does not need its own `job.py`, watcher,
completion writer or evidence-hashing script.

## Configure once, reuse the job

```bash
uv sync --locked --extra experiments
mkdir -p config/local
cp config/benchmark-settings.example.json config/local/benchmark.json
```

Edit `config/local/benchmark.json` for your checkout, interpreter, native
binary, payload and allocated GPU. This directory is ignored by Git. Only
explicit `${variable}` references are substituted; there is no shell expansion
or automatic dump of environment variables. `${output}` is supplied by the
runner and cannot be overridden by settings.

Start with the CPU example:

```bash
uv run --no-sync python -m fhemamba.benchmarks run \
  experiments/manifests/cpu-smoke-job.json \
  --settings config/local/benchmark.json \
  --output runs/cpu-smoke \
  --events runs/completions.jsonl
```

An existing output directory is an error: use a new run name for each sample.
`run.json` records input file hashes, explicit arguments, exit status and wall
time; `run.log` contains process output. `completion.json` and the optional
JSONL event stream are written on success, nonzero exit, timeout or interruption.
They can be consumed by a local watcher while other work continues.

For a qualified GPU payload and native build, use
[`packed-job.example.json`](../experiments/manifests/packed-job.example.json),
adding the validated optimization options for your study. It is a minimal
classical-128 probe configuration, not the full performance benchmark recipe.
See [Mamba-3](mamba3.md) and [optimization mechanisms](optimizations.md).

Use `--lock runs/locks/gpu-0.lock` for jobs sharing one device. Every cooperating
build, probe and timed sample must use the same lock. The lock does not reserve
hardware against unrelated users; choose an allocated device. The finite
`timeout_seconds` includes lock waiting and execution. On timeout or interruption,
the process group is terminated before the lock is released. This Linux runner
is intended for foreground commands; detached containers need their scheduler's
own lifecycle management.

Process success is separate from numerical acceptance. Keep using the packed
runner's error/token/security checks or the campaign's declared acceptance gates.
`wall_seconds` includes orchestration and is not native `eval_seconds`.
Generic jobs, packed runs and campaigns use the same process-group runner,
including descendant cleanup after the group leader exits. Their existing
timeout return codes and numerical acceptance checks are preserved.

## Compare without a study-specific script

```bash
uv run --no-sync python -m fhemamba.benchmarks compare \
  --baseline results/b300/2026-09-27/security128/full-d4-r1/native.json \
  --candidate results/b300/2026-09-27/security128/full-cache2048-r1/native.json \
  --contract config/packed-comparison.example.json
```

A comparison contract names the timing field, conditions that must match,
required flags, maximum errors and minimum reduction. Extend the example with
the security audit and workload-specific invariants of a new study. Every
sample is checked before computing means; missing fields, changed conditions
and failed gates cause an error. A passing mean reduction is descriptive evidence,
not a statistical significance test or a replacement for encrypted qualification.

## Public evidence

Keep raw runs, machine-specific settings and scratch analysis in `runs/`,
`config/local/` or `.local/`. Commit maintained analysis code under `experiments/`
or shared library code under `src/fhemamba/`; write technical reports under
`docs/research/`. Reports describe the hypothesis, method, acceptance criteria,
result and limitations. Access instructions and session diaries stay local.

To create a reviewed derivative in a new directory:

```bash
cp config/publication-policy.example.json config/local/publication.json
# Edit replacements for your environment before exporting.
uv run --no-sync python -m fhemamba.benchmarks publish \
  runs/my-study results/my-study --policy config/local/publication.json
uv run --no-sync python -m fhemamba.benchmarks verify results/my-study
```

The exporter leaves the source untouched. JSON measurements, CSV summaries and
numerical arrays remain directly readable. Relevant logs, measured source snapshots
and build records become `provenance.tar.gz`. Disposable launchers, watchers,
cleanup scripts and one-off analyses are excluded, including loose copies inside
source archives. Shared execution/comparison/verification lives in the installed
package. Useful research algorithms belong in maintained experiment tools.
Delete superseded working scripts after checking their replacement; do not keep
a second archive of them. Failure records remain included. Measured package,
build and test sources remain version-identifiable; compiled Python caches are omitted.
Opaque profiler recordings remain private, with their original hash recorded.
Export reviewed text summaries from Nsight instead of publishing raw recordings.

The `publication.json` manifest records **original** and **published** SHA-256
values separately. Original hashes embedded in historical records still refer
to the original bytes. The verifier checks the public derivative and its archive
members; it does not rerun numerical validation or authenticate historical claims.

Home-directory names, private IPv4 addresses, instance IDs, GPU UUIDs and selected
identity fields are normalized. Add literal hostname/path replacements in the
local policy. This is a deterministic publication transform, not a general secret
scanner: review files before export and keep credentials and keys out of the
input directory. Numerical arrays are copied unchanged; other binary formats
require an explicit reviewed `binary_suffixes` allowlist. Archive ownership and
timestamps are normalized, and path traversal or symlinks are rejected.

Inspect the repository's archived provenance without a GPU:

```bash
uv run --no-sync python -m fhemamba.benchmarks verify results
uv run --no-sync python -m fhemamba.benchmarks extract \
  results runs/evidence-inspection
```

The initial repository cleanup has one root publication manifest. New studies
can have independent manifests in their own directories. Publication changes
the current checkout only; it does not rewrite Git history or earlier releases.
