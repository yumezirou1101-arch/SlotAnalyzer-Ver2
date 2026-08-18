from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Ana-Slo Ver.4.2
# Adaptive vs TOP10
# Practical bankroll / efficiency comparison
#
# Uses the existing long-term OOS daily output (39-series).
# Important:
# - avg_diff is treated as per-selected-machine average difference.
# - total_diff is treated as the actual total difference for that
#   day's selected rule.
# - "10-machine normalized total" = avg_diff * 10, allowing an
#   equal-number-of-machines comparison.
# - This does NOT estimate yen EV because games, investment, payout,
#   and play duration are not present in this input.
# ============================================================

BASE_DIR = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

SEARCH_ROOT = BASE_DIR
INPUT_NAME = "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv"

OUTPUT_DIR = (
    BASE_DIR / "data" / "maruhan_maebashi" / "machine_number"
    / "analysis" / "31days_deep"
    / "41_Ver4_2_adaptive_vs_top10_bankroll_efficiency"
)

OUT_DAILY = OUTPUT_DIR / "41_Ver4_2_bankroll_efficiency_daily.csv"
OUT_SUMMARY = OUTPUT_DIR / "41_Ver4_2_bankroll_efficiency_summary.csv"
OUT_BLOCK = OUTPUT_DIR / "41_Ver4_2_bankroll_efficiency_block.csv"
OUT_DRAWDOWN = OUTPUT_DIR / "41_Ver4_2_bankroll_efficiency_drawdown.csv"


def find_input():
    candidates = sorted(
        p for p in SEARCH_ROOT.rglob(INPUT_NAME)
        if p.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"{INPUT_NAME} が見つかりません。\n検索先: {SEARCH_ROOT}"
        )
    return candidates[-1]


def load_data():
    input_path = find_input()
    print(f"Loading: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required = [
        "date",
        "block",
        "rule",
        "selected_rule",
        "is_adaptive_selection",
        "machines",
        "avg_diff",
        "total_diff",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "必要列がありません: " + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["machines"] = pd.to_numeric(df["machines"], errors="coerce")
    df["avg_diff"] = pd.to_numeric(df["avg_diff"], errors="coerce")
    df["total_diff"] = pd.to_numeric(df["total_diff"], errors="coerce")

    df["is_adaptive_selection"] = (
        df["is_adaptive_selection"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    df = df.dropna(
        subset=["date", "block", "machines", "avg_diff", "total_diff"]
    ).copy()

    return df, input_path


def build_daily(df):
    top10 = df[
        df["rule"].astype(str).str.upper().eq("TOP10")
    ].copy()

    adaptive = df[
        df["is_adaptive_selection"]
    ].copy()

    # One adaptive-selected rule per block/date is expected.
    adaptive = (
        adaptive.sort_values(["date", "block"])
        .drop_duplicates(["date", "block"], keep="last")
    )

    top10 = (
        top10.sort_values(["date", "block"])
        .drop_duplicates(["date", "block"], keep="last")
    )

    a = adaptive[
        [
            "date", "block", "selected_rule", "machines",
            "avg_diff", "total_diff"
        ]
    ].rename(
        columns={
            "selected_rule": "adaptive_rule",
            "machines": "adaptive_machines",
            "avg_diff": "adaptive_avg_diff",
            "total_diff": "adaptive_total_diff",
        }
    )

    t = top10[
        [
            "date", "block", "machines",
            "avg_diff", "total_diff"
        ]
    ].rename(
        columns={
            "machines": "top10_machines",
            "avg_diff": "top10_avg_diff",
            "total_diff": "top10_total_diff",
        }
    )

    daily = a.merge(t, on=["date", "block"], how="inner")

    if daily.empty:
        raise RuntimeError("AdaptiveとTOP10の共通日がありません。")

    # Equal-capital / equal-machine-count comparison.
    # We use 10 machines as the common reference because TOP10
    # explicitly selects 10 machines.
    daily["adaptive_10m_normalized_total"] = (
        daily["adaptive_avg_diff"] * 10.0
    )
    daily["top10_10m_total"] = (
        daily["top10_avg_diff"] * 10.0
    )
    daily["normalized_improvement"] = (
        daily["adaptive_10m_normalized_total"]
        - daily["top10_10m_total"]
    )

    daily["per_machine_improvement"] = (
        daily["adaptive_avg_diff"]
        - daily["top10_avg_diff"]
    )

    daily["machine_reduction"] = (
        daily["top10_machines"] - daily["adaptive_machines"]
    )

    daily["adaptive_better"] = daily["per_machine_improvement"] > 0
    daily["top10_better"] = daily["per_machine_improvement"] < 0
    daily["tie"] = daily["per_machine_improvement"] == 0

    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def max_drawdown(series):
    s = pd.Series(series, dtype=float).fillna(0.0)
    cumulative = s.cumsum()
    peak = cumulative.cummax()
    dd = cumulative - peak
    return float(dd.min()), dd


def longest_streak(values, target=True):
    best = 0
    current = 0
    for v in values:
        if bool(v) == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_strategy(daily, prefix, total_col, avg_col, machines_col):
    values = daily[total_col].to_numpy(dtype=float)
    avg_values = daily[avg_col].to_numpy(dtype=float)
    machine_values = daily[machines_col].to_numpy(dtype=float)

    dd, dd_series = max_drawdown(values)

    positive = values > 0
    negative = values < 0
    ties = values == 0

    return {
        "strategy": prefix,
        "days": len(daily),
        "avg_daily_total_diff": float(values.mean()),
        "median_daily_total_diff": float(np.median(values)),
        "total_diff": float(values.sum()),
        "avg_per_machine_diff": float(avg_values.mean()),
        "median_per_machine_diff": float(np.median(avg_values)),
        "best_day_total_diff": float(values.max()),
        "worst_day_total_diff": float(values.min()),
        "positive_days": int(positive.sum()),
        "negative_days": int(negative.sum()),
        "tie_days": int(ties.sum()),
        "positive_day_rate": float(positive.mean() * 100),
        "avg_machines_per_day": float(machine_values.mean()),
        "total_machine_selections": float(machine_values.sum()),
        "max_losing_streak": int(longest_streak(negative, True)),
        "max_drawdown": float(dd),
        "final_cumulative_diff": float(values.sum()),
    }


def build_block_summary(daily):
    rows = []

    for block, g in daily.groupby("block", sort=True):
        adaptive_values = g["adaptive_total_diff"].to_numpy(float)
        top10_values = g["top10_total_diff"].to_numpy(float)

        rows.append({
            "block": int(block),
            "start": g["date"].min().date(),
            "end": g["date"].max().date(),
            "days": len(g),
            "adaptive_rule": ",".join(
                sorted(g["adaptive_rule"].astype(str).unique())
            ),
            "adaptive_total_diff": float(adaptive_values.sum()),
            "top10_total_diff": float(top10_values.sum()),
            "adaptive_vs_top10_total": float(
                adaptive_values.sum() - top10_values.sum()
            ),
            "adaptive_avg_per_machine": float(
                g["adaptive_avg_diff"].mean()
            ),
            "top10_avg_per_machine": float(
                g["top10_avg_diff"].mean()
            ),
            "adaptive_avg_machines": float(
                g["adaptive_machines"].mean()
            ),
            "top10_avg_machines": float(
                g["top10_machines"].mean()
            ),
            "adaptive_10m_normalized": float(
                g["adaptive_10m_normalized_total"].sum()
            ),
            "top10_10m_normalized": float(
                g["top10_10m_total"].sum()
            ),
            "normalized_improvement": float(
                g["normalized_improvement"].sum()
            ),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("Ana-Slo Ver.4.2 Adaptive vs TOP10")
    print("Practical Bankroll / Efficiency Comparison")
    print("=" * 72)

    df, input_path = load_data()
    print(f"records = {len(df):,}")

    daily = build_daily(df)

    print()
    print("=" * 72)
    print("DAILY PAIRED COMPARISON")
    print("=" * 72)

    print(f"paired days = {len(daily)}")
    print(
        f"date range = {daily['date'].min().date()} "
        f"to {daily['date'].max().date()}"
    )

    print()
    print(
        daily[
            [
                "date", "block", "adaptive_rule",
                "adaptive_machines", "top10_machines",
                "adaptive_avg_diff", "top10_avg_diff",
                "per_machine_improvement",
                "adaptive_total_diff", "top10_total_diff",
                "normalized_improvement",
            ]
        ].to_string(index=False)
    )

    adaptive_total = daily["adaptive_total_diff"].sum()
    top10_total = daily["top10_total_diff"].sum()

    adaptive_avg_machine = daily["adaptive_avg_diff"].mean()
    top10_avg_machine = daily["top10_avg_diff"].mean()

    normalized_a = daily["adaptive_10m_normalized_total"].sum()
    normalized_t = daily["top10_10m_total"].sum()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"Adaptive actual total diff     : {adaptive_total:+.0f}")
    print(f"TOP10 actual total diff        : {top10_total:+.0f}")
    print(
        f"Actual total difference       : "
        f"{adaptive_total - top10_total:+.0f}"
    )

    print()
    print(
        f"Adaptive per-machine avg diff : "
        f"{adaptive_avg_machine:+.2f}"
    )
    print(
        f"TOP10 per-machine avg diff    : "
        f"{top10_avg_machine:+.2f}"
    )
    print(
        f"Per-machine improvement       : "
        f"{adaptive_avg_machine - top10_avg_machine:+.2f}"
    )

    print()
    print("EQUAL 10-MACHINE NORMALIZATION")
    print("-" * 72)
    print(
        f"Adaptive normalized total     : "
        f"{normalized_a:+.0f}"
    )
    print(
        f"TOP10 normalized total        : "
        f"{normalized_t:+.0f}"
    )
    print(
        f"Normalized improvement        : "
        f"{normalized_a - normalized_t:+.0f}"
    )

    print()
    print("SELECTION EFFICIENCY")
    print("-" * 72)
    print(
        f"Adaptive avg machines/day     : "
        f"{daily['adaptive_machines'].mean():.2f}"
    )
    print(
        f"TOP10 avg machines/day        : "
        f"{daily['top10_machines'].mean():.2f}"
    )
    print(
        f"Machine reduction             : "
        f"{daily['top10_machines'].mean() - daily['adaptive_machines'].mean():.2f}"
    )

    print()
    print("DAILY COMPARISON")
    print("-" * 72)
    print(
        f"Adaptive better days          : "
        f"{int(daily['adaptive_better'].sum())}"
    )
    print(
        f"TOP10 better days             : "
        f"{int(daily['top10_better'].sum())}"
    )
    print(
        f"Tie days                      : "
        f"{int(daily['tie'].sum())}"
    )

    # Strategy summaries based on actual selected totals.
    adaptive_summary = summarize_strategy(
        daily,
        "ADAPTIVE",
        "adaptive_total_diff",
        "adaptive_avg_diff",
        "adaptive_machines",
    )
    top10_summary = summarize_strategy(
        daily,
        "TOP10",
        "top10_total_diff",
        "top10_avg_diff",
        "top10_machines",
    )

    summary_df = pd.DataFrame([adaptive_summary, top10_summary])

    # Normalized 10-machine risk comparison.
    norm_adaptive = daily["adaptive_10m_normalized_total"]
    norm_top10 = daily["top10_10m_total"]

    norm_adaptive_dd, _ = max_drawdown(norm_adaptive)
    norm_top10_dd, _ = max_drawdown(norm_top10)

    normalized_summary = pd.DataFrame([
        {
            "strategy": "ADAPTIVE_10M_NORMALIZED",
            "days": len(daily),
            "total_diff": float(norm_adaptive.sum()),
            "avg_daily_diff": float(norm_adaptive.mean()),
            "median_daily_diff": float(norm_adaptive.median()),
            "positive_day_rate": float((norm_adaptive > 0).mean() * 100),
            "max_drawdown": norm_adaptive_dd,
            "avg_machines_actual": float(
                daily["adaptive_machines"].mean()
            ),
        },
        {
            "strategy": "TOP10_10M",
            "days": len(daily),
            "total_diff": float(norm_top10.sum()),
            "avg_daily_diff": float(norm_top10.mean()),
            "median_daily_diff": float(norm_top10.median()),
            "positive_day_rate": float((norm_top10 > 0).mean() * 100),
            "max_drawdown": norm_top10_dd,
            "avg_machines_actual": float(
                daily["top10_machines"].mean()
            ),
        },
    ])

    block_df = build_block_summary(daily)

    draw_rows = []
    for name, series in [
        ("ADAPTIVE_ACTUAL", daily["adaptive_total_diff"]),
        ("TOP10_ACTUAL", daily["top10_total_diff"]),
        ("ADAPTIVE_10M_NORMALIZED", norm_adaptive),
        ("TOP10_10M", norm_top10),
    ]:
        cumulative = series.cumsum()
        peak = cumulative.cummax()
        dd = cumulative - peak
        draw_rows.append(
            pd.DataFrame({
                "date": daily["date"],
                "strategy": name,
                "daily_diff": series.to_numpy(),
                "cumulative_diff": cumulative.to_numpy(),
                "drawdown": dd.to_numpy(),
            })
        )

    drawdown_df = pd.concat(draw_rows, ignore_index=True)

    print()
    print("=" * 72)
    print("RISK / RETURN COMPARISON")
    print("=" * 72)
    print(summary_df.to_string(index=False))

    print()
    print("10-machine normalized:")
    print(normalized_summary.to_string(index=False))

    print()
    print("BLOCK COMPARISON:")
    print(block_df.to_string(index=False))

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    if normalized_summary.loc[
        normalized_summary["strategy"] == "ADAPTIVE_10M_NORMALIZED",
        "total_diff"
    ].iloc[0] > normalized_summary.loc[
        normalized_summary["strategy"] == "TOP10_10M",
        "total_diff"
    ].iloc[0]:
        print("Equal 10-machine basis : ADAPTIVE better")
    else:
        print("Equal 10-machine basis : TOP10 better")

    if daily["adaptive_avg_diff"].mean() > daily["top10_avg_diff"].mean():
        print("Per-machine efficiency : ADAPTIVE better")
    else:
        print("Per-machine efficiency : TOP10 better")

    if daily["adaptive_machines"].mean() < daily["top10_machines"].mean():
        print("Capital concentration   : ADAPTIVE uses fewer machines")
    else:
        print("Capital concentration   : no reduction")

    print()
    print(
        "NOTE: This analysis does not convert diff into yen EV because "
        "the input does not contain game counts, investment, payout, "
        "or play-duration data."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily.to_csv(OUT_DAILY, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    block_df.to_csv(OUT_BLOCK, index=False, encoding="utf-8-sig")
    drawdown_df.to_csv(OUT_DRAWDOWN, index=False, encoding="utf-8-sig")

    print()
    print("=" * 72)
    print("FILES SAVED")
    print("=" * 72)
    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_BLOCK)
    print(OUT_DRAWDOWN)
    print()
    print("Bankroll / efficiency comparison complete.")


if __name__ == "__main__":
    main()
