"""
Test the thermal-drift hypothesis directly instead of assuming it.

Loads the thermal_check sweep results (summary.csv) and the nvidia-smi
temperature log captured alongside it, aligns them by wall-clock time,
and reports:
  1. Does GPU temperature actually rise over the course of the check?
  2. Does output_throughput correlate with temperature, or with run order?

If neither correlation shows up, the earlier variance flag on the
2048/4096 output-length configs is NOT explained by thermal drift, and
something else (background load, scheduler nondeterminism, something
else entirely) needs to be considered instead -- this script's job is
to tell you which case you're in, not to confirm the hypothesis by
assumption.
"""

import sys
from pathlib import Path

import pandas as pd


def load_temp_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # nvidia-smi returns e.g. "45 C" / "1830 MHz" / "35.20 W" as strings
    for col in df.columns:
        if col != "timestamp":
            df[col] = df[col].astype(str).str.extract(r"([\d.]+)").astype(float)
    return df


def load_results(results_dir: str) -> pd.DataFrame:
    summary_path = Path(results_dir) / "thermal_check" / "summary.csv"
    if not summary_path.exists():
        candidates = list(Path(results_dir).rglob("summary.csv"))
        raise FileNotFoundError(
            f"No summary.csv at {summary_path}. "
            f"Found these instead, check --results-dir: {candidates}"
        )
    df = pd.read_csv(summary_path)
    df = df.sort_values("run_number") if "run_number" in df.columns else df.reset_index()
    return df


def main(results_dir: str, temp_log_path: str):
    results = load_results(results_dir)
    temps = load_temp_log(temp_log_path)

    print(f"Loaded {len(results)} runs, {len(temps)} temperature samples.")
    print(f"Temp log spans {temps['timestamp'].min()} to {temps['timestamp'].max()}")

    if "date" not in results.columns:
        print(
            "\nWARNING: no 'date' column in summary.csv -- can't align runs to "
            "the temperature log by wall-clock time. Falling back to run order "
            "as a rough proxy. Treat the correlation below as suggestive only."
        )
        results["order"] = range(len(results))
        x_col = "order"
    else:
        results["end_time"] = pd.to_datetime(results["date"], format="%Y%m%d-%H%M%S")
        results["start_time"] = results["end_time"] - pd.to_timedelta(
            results["duration"], unit="s"
        )

        def mean_temp_during(row):
            window = temps[
                (temps["timestamp"] >= row["start_time"])
                & (temps["timestamp"] <= row["end_time"])
            ]
            return window["temperature.gpu"].mean() if len(window) else float("nan")

        results["mean_temp_c"] = results.apply(mean_temp_during, axis=1)
        x_col = "mean_temp_c"

    print()
    cols = [c for c in ["random-output-len", "run_number", x_col, "output_throughput", "mean_tpot_ms"] if c in results.columns]
    print(results[cols].to_string(index=False))

    if x_col == "mean_temp_c" and results["mean_temp_c"].notna().sum() > 2:
        corr = results["mean_temp_c"].corr(results["output_throughput"])
        print(f"\nCorrelation (temp vs throughput): {corr:.2f}")
        temp_rise = results["mean_temp_c"].max() - results["mean_temp_c"].min()
        print(f"Temperature range during check: {temp_rise:.1f} C")
        if abs(corr) > 0.5 and temp_rise > 5:
            print(
                "-> Meaningful negative correlation with a real temperature "
                "rise: thermal drift is a plausible explanation for the "
                "earlier variance flag."
            )
        else:
            print(
                "-> Weak correlation and/or minimal temperature rise: thermal "
                "drift does NOT look like the explanation. The variance seen "
                "earlier likely has a different cause -- worth checking "
                "background processes or scheduler behavior instead."
            )


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results"
    temp_log = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/analysis/temp_log.csv"
    main(results_dir, temp_log)