# Documentation

Start with [reproducing the research snapshot](reproducing.md) to run the CPU
example, obtain the pinned checkpoint, export a payload or run encrypted
generation. Commands use the repository root as their working directory.

## Guides

| Document | Use it for |
| --- | --- |
| [Reproduction](reproducing.md) | A complete path from checkout to the measured workload |
| [DGX Spark](dgx-spark.md) | Native dependencies, builds, machine requirements and campaigns |
| [Python package](package.md) | Model, operator, layout and CLI modules |
| [Mamba-3](mamba3.md) | Trained SISO model, common arithmetic backend and complete encrypted generation |
| [Repository layout](repository.md) | File ownership, generated outputs and previous paths |
| [Testing](testing.md) | Formatting, Python tests, coverage, C++ contracts and artifact checks |
| [Research validation](validation.md) | Approximation quality, encrypted probes and GPU promotion gates |
| [Contributing](../CONTRIBUTING.md) | Development setup, review and evidence requirements |

## Design and evidence

| Document | Use it for |
| --- | --- |
| [Design](design.md) | Protocol invariants, state recurrence and algorithm choices |
| [Evidence registry](evidence.md) | Trace headline claims to raw/derived artifacts and failed controls |
| [Research index](research/README.md) | Read the studies by topic and date |
| [Result index](../results/README.md) | Find recorded measurements without running a GPU |
| [Roadmap](roadmap.md) | Research direction and capability milestones |
| [Backlog](backlog.md) | Concrete next experiments and current status |
| [Maintenance](maintenance.md) | Implementation boundaries and technical debt |
| [Archive](archive/README.md) | Historical designs, ledgers and the retired implementation |

Published artifacts retain the paths, commits and hashes recorded when they
were produced. See [previous layout](repository.md#previous-layout) when an
older study uses the original directory structure.
