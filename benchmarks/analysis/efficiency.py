"""
Marginal scaling efficiency for the concurrency sweep.

Raw throughput numbers don't directly answer "where does this stop
scaling well" -- you have to compare against what linear scaling from
the concurrency=1 baseline would have predicted. This is the same
calculation used on the smoke test data earlier in the project; this
script applies it to the full sweep (averaged across the 3 repeats
per config) instead of a 20-prompt smoke test.
"""

import pandas as pd


def marginal_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    conc = df[df["sweep"] == "concurrency"]
    means = conc.groupby("x_value")["output_throughput"].mean().sort_index()

    base_concurrency = means.index.min()
    base_throughput = means.loc[base_concurrency]

    rows = []
    for concurrency, throughput in means.items():
        linear_expected = base_throughput * (concurrency / base_concurrency)
        efficiency = throughput / linear_expected
        rows.append(
            {
                "concurrency": concurrency,
                "actual_tok_s": throughput,
                "linear_expected_tok_s": linear_expected,
                "efficiency": efficiency,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("benchmarks/analysis/combined_summary.csv")
    result = marginal_efficiency(df)
    pd.set_option("display.float_format", lambda x: f"{x:.1f}")
    print(result.to_string(index=False))

    # Flag the point where efficiency first drops below 80% -- a reasonable,
    # stated-in-advance threshold for "scaling has meaningfully broken down"
    # rather than picking the cutoff after seeing the numbers.
    below_80 = result[result["efficiency"] < 0.80]
    if len(below_80):
        first = below_80.iloc[0]
        print(
            f"\nScaling efficiency first drops below 80% at "
            f"concurrency={int(first['concurrency'])} "
            f"({first['efficiency']:.0%} of linear)."
        )
    else:
        print("\nScaling stayed above 80% efficiency across the whole sweep -- "
              "the KV-cache wall we calculated may sit beyond concurrency=128, "
              "or the ceiling calculation needs revisiting.")