"""
Load the three sweep summary CSVs into one combined, tagged dataframe.

Each summary.csv already has run_number and the swept params as columns
(from vllm bench sweep serve), so this step is mostly concatenation plus
tagging each row with which axis it belongs to and what its x-value is,
so downstream plotting/analysis code doesn't need to know the per-sweep
column names.
"""

import json
from pathlib import Path

import pandas as pd

# Which column is the "x axis" for each sweep, and the fixed values held
# constant while that axis varies (for labeling plots / sanity checks).
SWEEP_CONFIG = {
    "concurrency": {
        "x_col": "max-concurrency",
        "held_fixed": {"random-input-len": 1024, "random-output-len": 512},
    },
    "input_length": {
        "x_col": "random-input-len",
        "held_fixed": {"max-concurrency": 16, "random-output-len": 256},
    },
    "output_length": {
        "x_col": "random-output-len",
        "held_fixed": {"max-concurrency": 16, "random-input-len": 256},
    },
}


def load_sweep(csv_path: str, sweep_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["sweep"] = sweep_name
    df["x_value"] = df[SWEEP_CONFIG[sweep_name]["x_col"]]

    # Sanity check: confirm the columns that should be fixed for this sweep
    # actually are fixed. If not, something in the run didn't match the
    # intended design and downstream isolation claims (e.g. "TTFT sweep
    # isolates prefill") don't hold.
    for col, expected in SWEEP_CONFIG[sweep_name]["held_fixed"].items():
        actual_values = df[col].unique()
        if len(actual_values) > 1 or actual_values[0] != expected:
            print(
                f"WARNING [{sweep_name}]: expected '{col}' fixed at "
                f"{expected}, found {list(actual_values)}"
            )
    return df


def load_all(concurrency_csv: str, input_len_csv: str, output_len_csv: str) -> pd.DataFrame:
    frames = [
        load_sweep(concurrency_csv, "concurrency"),
        load_sweep(input_len_csv, "input_length"),
        load_sweep(output_len_csv, "output_length"),
    ]
    combined = pd.concat(frames, ignore_index=True)

    expected_rows = 8 * 3 + 5 * 3 + 6 * 3  # configs * 3 repeats each
    if len(combined) != expected_rows:
        print(
            f"WARNING: expected {expected_rows} total rows "
            f"(19 configs x 3 repeats), got {len(combined)}. "
            "Check for missing or extra runs before trusting aggregates."
        )
    return combined


def scan_for_detailed_results(results_dir: str) -> list[str]:
    """
    Check whether any result JSON in results_dir actually has --save-detailed
    fields (ttfts, start_times, itls). The sweep's summary.csv never has
    these — they only live in the per-run JSON files the sweep tool writes
    alongside it. Returns the list of files that DO have detail, so you know
    whether the queueing-vs-TTFT question is answerable without a rerun.
    """
    hits = []
    for path in Path(results_dir).rglob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "ttfts" in data and "start_times" in data:
            hits.append(str(path))
    return hits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", required=True)
    parser.add_argument("--input-length", required=True)
    parser.add_argument("--output-length", required=True)
    parser.add_argument("--results-dir", default="benchmarks/results")
    args = parser.parse_args()

    df = load_all(args.concurrency, args.input_length, args.output_length)
    print(f"Loaded {len(df)} rows across {df['sweep'].nunique()} sweeps.")

    detailed = scan_for_detailed_results(args.results_dir)
    if detailed:
        print(f"\nFound {len(detailed)} result file(s) with --save-detailed data:")
        for f in detailed[:5]:
            print(f"  {f}")
        print("Per-request queueing/ITL-drift analysis is possible.")
    else:
        print(
            "\nNo --save-detailed result files found under "
            f"{args.results_dir}. Per-request analysis (queueing-vs-TTFT, "
            "ITL drift within a request) is NOT possible from this run — "
            "would need a rerun with --save-detailed added to BENCH_CMD."
        )

    df.to_csv("benchmarks/analysis/combined_summary.csv", index=False)
    print("\nSaved combined_summary.csv")