"""
Run the full analysis pipeline in order:

1. Load and combine the three sweep summary CSVs, sanity-check row counts
   and that "held fixed" params actually stayed fixed.
2. Check repeat-run variance -- do this BEFORE trusting any averaged curve.
3. Marginal scaling efficiency for the concurrency sweep -- where does it
   fall off linear.
4. Generate the headline plots.

Each step prints its own findings; nothing here silently swallows a
warning. If load_results.py prints a WARNING, read it before trusting
anything downstream.
"""

import argparse

from load_results import load_all, scan_for_detailed_results
from variance_check import variance_report, flag_high_variance
from efficiency import marginal_efficiency
from plots import plot_sweep

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", required=True)
    parser.add_argument("--input-length", required=True)
    parser.add_argument("--output-length", required=True)
    parser.add_argument("--results-dir", default="benchmarks/results")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Load and combine")
    print("=" * 60)
    df = load_all(args.concurrency, args.input_length, args.output_length)
    print(f"Loaded {len(df)} rows.")

    out_csv = "benchmarks/analysis/combined_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    detailed = scan_for_detailed_results(args.results_dir)
    if not detailed:
        print(
            "\nNote: no --save-detailed files found. Queueing-vs-TTFT and "
            "per-request ITL-drift questions stay open until a rerun with "
            "that flag added."
        )

    print("\n" + "=" * 60)
    print("STEP 2: Repeat-run variance check")
    print("=" * 60)
    report = variance_report(df)
    pd.set_option("display.width", 140)
    print(report.to_string(index=False))
    flagged = flag_high_variance(report)
    if len(flagged):
        print(f"\n{len(flagged)} config(s) exceed the CV threshold -- see above.")

    print("\n" + "=" * 60)
    print("STEP 3: Concurrency scaling efficiency")
    print("=" * 60)
    eff = marginal_efficiency(df)
    print(eff.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 4: Plots")
    print("=" * 60)
    for sweep in ["concurrency", "input_length", "output_length"]:
        plot_sweep(df, sweep)

    print("\nDone. Combined data: benchmarks/analysis/combined_summary.csv")
    print("Plots: benchmarks/analysis/plots/")


if __name__ == "__main__":
    main()