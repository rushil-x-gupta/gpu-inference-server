"""
Roofline attribution for the concurrency sweep: at each concurrency level,
was the GPU compute-bound (SM utilization saturated) or memory-bandwidth-
bound (memory controller utilization saturated) -- or neither?

Same alignment pattern as analyze_thermal.py: match each run's wall-clock
window against the telemetry log using the 'date'/'duration' fields from
the summary.csv, then average utilization over that window.
"""

import sys
from pathlib import Path

import pandas as pd

SATURATION_THRESHOLD = 85  # % utilization treated as "saturated" for this analysis


def load_gpu_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ["utilization.gpu [%]", "utilization.memory [%]", "memory.used [MiB]",
                "clocks.sm [MHz]", "power.draw [W]"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r"([\d.]+)").astype(float)
    return df


def load_results(results_dir: str) -> pd.DataFrame:
    summary_path = Path(results_dir) / "gpu_util_sweep" / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No summary.csv at {summary_path}. Run gpu_util_sweep.sh first."
        )
    return pd.read_csv(summary_path)


def classify(row) -> str:
    gpu_sat = row["mean_util_gpu"] >= SATURATION_THRESHOLD
    mem_sat = row["mean_util_mem"] >= SATURATION_THRESHOLD
    if gpu_sat and not mem_sat:
        return "compute-bound"
    if mem_sat and not gpu_sat:
        return "memory-bound"
    if gpu_sat and mem_sat:
        return "both saturated"
    return "neither saturated"


def main(results_dir: str, gpu_log_path: str):
    results = load_results(results_dir)
    gpu_log = load_gpu_log(gpu_log_path)

    if "date" not in results.columns:
        print(
            "ERROR: no 'date' column in summary.csv -- can't align runs to the "
            "telemetry log by wall-clock time. Check that this vLLM version "
            "writes a 'date' field into the result JSON (it did for the "
            "thermal check on this same machine, so this would be unexpected)."
        )
        sys.exit(1)

    results["end_time"] = pd.to_datetime(results["date"], format="%Y%m%d-%H%M%S")
    results["start_time"] = results["end_time"] - pd.to_timedelta(results["duration"], unit="s")

    def window_stats(row):
        window = gpu_log[
            (gpu_log["timestamp"] >= row["start_time"])
            & (gpu_log["timestamp"] <= row["end_time"])
        ]
        if not len(window):
            return pd.Series({"mean_util_gpu": float("nan"), "mean_util_mem": float("nan")})
        return pd.Series({
            "mean_util_gpu": window["utilization.gpu [%]"].mean(),
            "mean_util_mem": window["utilization.memory [%]"].mean(),
        })

    results = pd.concat([results, results.apply(window_stats, axis=1)], axis=1)

    per_config = (
        results.groupby("max-concurrency")[
            ["mean_util_gpu", "mean_util_mem", "output_throughput", "mean_tpot_ms"]
        ]
        .mean()
        .sort_index()
    )
    per_config["classification"] = per_config.apply(classify, axis=1)

    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.1f}")
    print(per_config.to_string())

    print(f"\n(Saturation threshold: {SATURATION_THRESHOLD}% utilization)")

    transitions = per_config["classification"].ne(per_config["classification"].shift())
    if transitions.sum() > 1:
        print("\nBottleneck classification changes across the sweep:")
        for conc, row in per_config[transitions].iterrows():
            print(f"  concurrency={conc}: {row['classification']}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    gpu_log = sys.argv[2] if len(sys.argv) > 2 else "analysis/gpu_util_log.csv"
    main(results_dir, gpu_log)