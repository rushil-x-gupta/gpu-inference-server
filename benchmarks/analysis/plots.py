"""
Headline plots for each sweep axis.

Error bars are std across the 3 repeat runs per config -- this is the
direct visual counterpart to the variance_check.py numbers, so a config
with a wide error bar is visibly flagged on the plot itself, not just
in a separate table.

p90 is used as the tail-latency line instead of p99. At num-prompts=300,
p99 rests on ~3 samples in the tail (see project journal) -- not stable
enough to plot with a straight face. p90 has ~30 samples backing it at
this sample size, which is defensible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTDIR = Path("benchmarks/analysis/plots")

AXIS_LABELS = {
    "concurrency": "Concurrency (--max-concurrency)",
    "input_length": "Input length (tokens)",
    "output_length": "Output length (tokens)",
}


def _agg(df: pd.DataFrame, sweep: str, metric: str):
    sub = df[df["sweep"] == sweep]
    g = sub.groupby("x_value")[metric].agg(["mean", "std"]).sort_index()
    return g.index.values, g["mean"].values, g["std"].fillna(0).values


def plot_sweep(df: pd.DataFrame, sweep: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{sweep} sweep")

    x, y, err = _agg(df, sweep, "output_throughput")
    axes[0].errorbar(x, y, yerr=err, marker="o", capsize=3)
    axes[0].set_title("Output throughput")
    axes[0].set_ylabel("tokens/sec")

    x, y, err = _agg(df, sweep, "mean_ttft_ms")
    axes[1].errorbar(x, y, yerr=err, marker="o", capsize=3, color="tab:orange")
    axes[1].set_title("TTFT (mean)")
    axes[1].set_ylabel("ms")

    x, y, err = _agg(df, sweep, "mean_tpot_ms")
    axes[2].errorbar(x, y, yerr=err, marker="o", capsize=3, color="tab:green")
    axes[2].set_title("TPOT (mean)")
    axes[2].set_ylabel("ms")

    for ax in axes:
        ax.set_xlabel(AXIS_LABELS[sweep])
        ax.grid(alpha=0.3)
        if sweep in ("concurrency",):
            ax.set_xscale("log", base=2)

    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTDIR / f"{sweep}_sweep.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    df = pd.read_csv("benchmarks/analysis/combined_summary.csv")
    for sweep in ["concurrency", "input_length", "output_length"]:
        plot_sweep(df, sweep)