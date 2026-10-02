# Documentation

Start with [getting started](reproducing.md). Commands run from the repository
root unless a guide names another working directory.

## Use the project

| Goal | Guide |
| --- | --- |
| Install Python dependencies and run a CPU example | [Getting started](reproducing.md) |
| Generate from text or token IDs with Python or CLI | [Generation](generation.md) |
| Load Mamba-3, export a workload and run encrypted generation | [Mamba-3](mamba3.md) |
| Build the native backend on DGX Spark | [Spark build and execution](dgx-spark.md) |
| Find importable APIs and CLI commands | [Python package](package.md) |
| Run, compare and publish an experiment | [Experiment workflow](experiments.md) |
| Choose a specialized research tool | [Experiment index](../experiments/README.md) |

## Develop and validate

| Goal | Guide |
| --- | --- |
| Understand the current baseline and next investigation | [Current state](status.md) |
| Set up development and submit a change | [Contributing](../CONTRIBUTING.md) · [Testing](testing.md) |
| Find implementation ownership and conventions | [Repository map](repository.md) · [Maintenance](maintenance.md) |
| Understand the model and protocol | [Design](design.md) |
| Qualify numerical or GPU changes | [Research validation](validation.md) · [Optimizations](optimizations.md) |
| Choose future work | [Roadmap](roadmap.md) · [Backlog](backlog.md) |

## Read the evidence

The [evidence registry](evidence.md) connects claims to measurements and failed
controls. Use the [research index](research/README.md) for detailed studies,
the [result index](../results/README.md) for artifacts, and the
[archive](archive/README.md) for historical designs.
