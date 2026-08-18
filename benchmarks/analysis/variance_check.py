"""
Repeat-run variance check.

We ran 3 repeats per config specifically to catch thermal drift or other
run-to-run instability (see project journal). This computes the coefficient
of variation (std/mean) across those 3 repeats for the metrics that matter
most, and flags any config where variance is high enough that "the average"
might be hiding something worth a closer look rather than genuine signal
from the swept parameter.

A CV threshold of 5% is a reasonable default for a controlled local
benchmark -- higher than that on a metric like output_throughput across
3 identical-config runs suggests something (thermal, background load,
scheduler nondeterminism) is adding real noise.
"""

import pandas as pd

CV_THRESHOLD = 0.05
METRICS = ["output_throughput", "mean_ttft_ms", "mean_tpot_ms"]


def variance_report(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["sweep", "x_value"]
    rows = []
    for (sweep, x_value), group in df.groupby(group_cols):
        row = {"sweep": sweep, "x_value": x_value, "n_runs": len(group)}
        for metric in METRICS:
            mean = group[metric].mean()
            std = group[metric].std()
            cv = std / mean if mean else float("nan")
            row[f"{metric}_cv"] = cv
            row[f"{metric}_mean"] = mean
        rows.append(row)

    report = pd.DataFrame(rows).sort_values(["sweep", "x_value"])
    return report


def flag_high_variance(report: pd.DataFrame) -> pd.DataFrame:
    cv_cols = [c for c in report.columns if c.endswith("_cv")]
    mask = (report[cv_cols] > CV_THRESHOLD).any(axis=1)
    return report[mask]


if __name__ == "__main__":
    df = pd.read_csv("benchmarks/analysis/combined_summary.csv")
    report = variance_report(df)
    pd.set_option("display.width", 140)
    print(report.to_string(index=False))

    flagged = flag_high_variance(report)
    if len(flagged):
        print(f"\n{len(flagged)} config(s) exceed {CV_THRESHOLD:.0%} CV on at least one metric:")
        print(flagged.to_string(index=False))
        print(
            "\nWorth checking whether these correspond to later runs in the "
            "sweep (thermal drift) or are scattered randomly (more likely "
            "scheduler/OS noise) before trusting the averaged curve at "
            "these points."
        )
    else:
        print(f"\nAll configs within {CV_THRESHOLD:.0%} CV across repeats.")