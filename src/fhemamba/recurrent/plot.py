#!/usr/bin/env python3
"""Plot a validated scheduler study; requires the optional matplotlib package."""

from __future__ import annotations

import argparse
from pathlib import Path

from fhemamba.benchmarks.io import read_object
from fhemamba.recurrent.analysis import progress_samples


def plot(report, results, output, steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [
        pair for pair in report["pairs"] if report["cases"][pair["baseline"]]["steps"] == steps
    ]
    if not pairs:
        raise ValueError("no comparison pairs at the requested step count")
    figure, axes = plt.subplots(2, len(pairs), squeeze=False, figsize=(6 * len(pairs), 7))
    for column, pair in enumerate(pairs):
        for arm, color in (("baseline", "#64748b"), ("candidate", "#0d9488")):
            name = pair[arm]
            case = report["cases"][name]
            samples = progress_samples(results / name / "run.log")
            if not samples:
                raise ValueError(f"missing progress samples: {name}")
            label = "Unlimited lookahead" if not case["limit"] else f"Admission at {case['limit']}"
            x = [sample["completed_nodes"] for sample in samples]
            for row, key, divisor in ((0, "device_used_bytes", 2**30), (1, "live_values", 1)):
                y = [sample[key] / divisor for sample in samples]
                axes[row, column].plot(x, y, color=color, label=label, linewidth=1.8)
                if case.get("failure", {}).get("cuda_out_of_memory"):
                    axes[row, column].plot(x[-1], y[-1], "x", color=color, markersize=9)
                    axes[row, column].annotate(
                        "OOM", (x[-1], y[-1]), xytext=(5, -12), textcoords="offset points"
                    )
            if case["limit"]:
                axes[1, column].axhline(case["limit"], color=color, linestyle=":", alpha=0.65)
        mode = report["cases"][pair["baseline"]]["mode"]
        axes[0, column].set_title(f"{mode.capitalize()} state, {steps} steps")
        axes[0, column].set_ylabel("Sampled device usage (GiB)")
        axes[1, column].set_ylabel("Retained logical DAG values")
        axes[1, column].set_xlabel("Completed DAG nodes")
        axes[0, column].legend(frameon=False, fontsize=9)
        for axis in axes[:, column]:
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Recurrent subcircuit: memory admission", fontsize=15)
    figure.text(
        0.5,
        0.02,
        "Samples every 10 completed nodes; intra-operation peaks are not captured.\n"
        "Dashed line: soft admission threshold. Required work may exceed it.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fhemamba recurrent plot", description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = read_object(args.report)
    if report.get("schema") != "fhemamba-recurrence-schedule-v1":
        parser.error("expected a validated scheduler report")
    plot(report, args.results, args.output, args.steps)


if __name__ == "__main__":
    main()
