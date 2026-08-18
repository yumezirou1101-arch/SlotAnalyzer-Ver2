from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Ana-Slo Ver.4.2
# Adaptive Risk / Selection-Intensity Filter Test
#
# IMPORTANT:
# The current 39-series daily CSV does NOT contain a confidence
# score. Therefore this version does NOT invent a confidence
# threshold and does NOT use future OOS results to filter days.
#
# We compare:
#   1) TOP10: every day, 10 machines
#   2) ADAPTIVE: the rule selected by prior OOS blocks
#   3) ADAPTIVE_SWITCH_ONLY:
#        play only when Adaptive selects a rule different from TOP10
#        (currently this means the concentrated TOP4_5 regime)
#
# This is a valid pre-decision regime filter because selected_rule
# is determined before the corresponding OOS block.
#
# No look-ahead filtering is used.
# ============================================================

BASE_DIR = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
INPUT_NAME = "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv"

OUTPUT_DIR = (
    BASE_DIR / "data" / "maruhan_maebashi" / "machine_number"
    / "analysis" / "31days_deep"
    / "42_Ver4_2_adaptive_risk_filter"
)

OUT_DAILY = OUTPUT_DIR / "42_Ver4_2_adaptive_risk_filter_daily.csv"
OUT_SUMMARY = OUTPUT_DIR / "42_Ver4_2_adaptive_risk_filter_summary.csv"
OUT_BLOCK = OUTPUT_DIR / "42_Ver4_2_adaptive_risk_filter_block.csv"
OUT_MONTHLY = OUTPUT_DIR / "42_Ver4_2_adaptive_risk_filter_monthly.csv"
OUT_DRAWDOWN = OUTPUT_DIR / "42_Ver4_2_adaptive_risk_filter_drawdown.csv"


def find_input():
    candidates = sorted(
        p for p in BASE_DIR.rglob(INPUT_NAME)
        if p.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"{INPUT_NAME} が見つかりません。\n検索先: {BASE_DIR}"
        )
    return candidates[-1]


def load_daily():
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

    df["rule"] = df["rule"].astype(str).str.strip()
    df["selected_rule"] = df["selected_rule"].astype(str).str.strip()

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

    return df


def max_drawdown(values):
    s = pd.Series(values, dtype=float).fillna(0.0)
    cumulative = s.cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    return float(drawdown.min()), cumulative, drawdown


def max_losing_streak(values):
    best = 0
    cur = 0
    for v in values:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def make_paired_daily(df):
    # TOP10 baseline
    top10 = df[
        df["rule"].str.upper().eq("TOP10")
    ].copy()

    top10 = (
        top10.sort_values(["date", "block"])
        .drop_duplicates(["date", "block"], keep="last")
    )

    top10 = top10[
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

    # Adaptive-selected result
    adaptive = df[df["is_adaptive_selection"]].copy()

    adaptive = (
        adaptive.sort_values(["date", "block"])
        .drop_duplicates(["date", "block"], keep="last")
    )

    adaptive = adaptive[
        [
            "date", "block", "selected_rule",
            "machines", "avg_diff", "total_diff"
        ]
    ].rename(
        columns={
            "selected_rule": "adaptive_rule",
            "machines": "adaptive_machines",
            "avg_diff": "adaptive_avg_diff",
            "total_diff": "adaptive_total_diff",
        }
    )

    paired = adaptive.merge(
        top10,
        on=["date", "block"],
        how="inner",
    )

    if paired.empty:
        raise RuntimeError("AdaptiveとTOP10の共通日がありません。")

    paired = paired.sort_values("date").reset_index(drop=True)

    # No-lookahead regime indicator.
    paired["adaptive_switch"] = (
        paired["adaptive_rule"].str.upper() != "TOP10"
    )

    paired["adaptive_vs_top10_per_machine"] = (
        paired["adaptive_avg_diff"]
        - paired["top10_avg_diff"]
    )

    paired["adaptive_vs_top10_total"] = (
        paired["adaptive_total_diff"]
        - paired["top10_total_diff"]
    )

    return paired


def strategy_summary(
    daily,
    strategy,
    active_mask,
    diff_col,
    machine_col,
):
    active = daily.loc[active_mask].copy()

    if active.empty:
        return {
            "strategy": strategy,
            "total_days": len(daily),
            "played_days": 0,
            "skipped_days": len(daily),
            "play_rate": 0.0,
            "total_diff": 0.0,
            "avg_diff_per_played_day": np.nan,
            "median_diff_per_played_day": np.nan,
            "avg_diff_per_machine": np.nan,
            "total_machine_selections": 0.0,
            "avg_machines_per_played_day": np.nan,
            "positive_days": 0,
            "negative_days": 0,
            "positive_day_rate": np.nan,
            "max_losing_streak": 0,
            "max_drawdown": 0.0,
            "profit_factor_simple": np.nan,
        }

    values = active[diff_col].astype(float).to_numpy()
    machines = active[machine_col].astype(float).to_numpy()

    total_diff = float(values.sum())
    positive = values[values > 0].sum()
    negative_abs = abs(values[values < 0].sum())

    dd, _, _ = max_drawdown(values)

    return {
        "strategy": strategy,
        "total_days": len(daily),
        "played_days": len(active),
        "skipped_days": len(daily) - len(active),
        "play_rate": len(active) / len(daily) * 100,
        "total_diff": total_diff,
        "avg_diff_per_played_day": float(values.mean()),
        "median_diff_per_played_day": float(np.median(values)),
        "avg_diff_per_machine": float(
            active["adaptive_avg_diff"].mean()
            if strategy != "TOP10"
            else active["top10_avg_diff"].mean()
        ),
        "total_machine_selections": float(machines.sum()),
        "avg_machines_per_played_day": float(machines.mean()),
        "positive_days": int((values > 0).sum()),
        "negative_days": int((values < 0).sum()),
        "positive_day_rate": float((values > 0).mean() * 100),
        "max_losing_streak": int(max_losing_streak(values)),
        "max_drawdown": float(dd),
        "profit_factor_simple": (
            float(positive / negative_abs)
            if negative_abs > 0 else np.inf
        ),
    }


def build_daily_output(daily):
    rows = []

    for _, r in daily.iterrows():
        # TOP10
        rows.append({
            "date": r["date"],
            "block": int(r["block"]),
            "strategy": "TOP10",
            "active": True,
            "selected_rule": "TOP10",
            "machines": r["top10_machines"],
            "avg_diff": r["top10_avg_diff"],
            "total_diff": r["top10_total_diff"],
        })

        # Adaptive actual
        rows.append({
            "date": r["date"],
            "block": int(r["block"]),
            "strategy": "ADAPTIVE",
            "active": True,
            "selected_rule": r["adaptive_rule"],
            "machines": r["adaptive_machines"],
            "avg_diff": r["adaptive_avg_diff"],
            "total_diff": r["adaptive_total_diff"],
        })

        # Adaptive switch-only
        active = bool(r["adaptive_switch"])
        rows.append({
            "date": r["date"],
            "block": int(r["block"]),
            "strategy": "ADAPTIVE_SWITCH_ONLY",
            "active": active,
            "selected_rule": r["adaptive_rule"],
            "machines": r["adaptive_machines"] if active else 0,
            "avg_diff": r["adaptive_avg_diff"] if active else 0.0,
            "total_diff": r["adaptive_total_diff"] if active else 0.0,
        })

    return pd.DataFrame(rows)


def build_block_summary(daily):
    rows = []

    for block, g in daily.groupby("block", sort=True):
        active_switch = g[g["adaptive_switch"]]

        rows.append({
            "block": int(block),
            "start": g["date"].min().date(),
            "end": g["date"].max().date(),
            "days": len(g),
            "adaptive_rule": ",".join(
                sorted(g["adaptive_rule"].astype(str).unique())
            ),
            "switch_days": int(g["adaptive_switch"].sum()),
            "adaptive_total": float(g["adaptive_total_diff"].sum()),
            "top10_total": float(g["top10_total_diff"].sum()),
            "adaptive_vs_top10": float(
                g["adaptive_total_diff"].sum()
                - g["top10_total_diff"].sum()
            ),
            "switch_only_total": float(
                active_switch["adaptive_total_diff"].sum()
            ),
            "switch_only_machines": float(
                active_switch["adaptive_machines"].sum()
            ),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("Ana-Slo Ver.4.2 Adaptive Risk / Selection-Intensity Filter Test")
    print("=" * 72)

    df = load_daily()
    print(f"records = {len(df):,}")

    paired = make_paired_daily(df)

    print()
    print("=" * 72)
    print("DATA CHECK")
    print("=" * 72)
    print(f"paired days = {len(paired)}")
    print(
        f"date range = {paired['date'].min().date()} "
        f"to {paired['date'].max().date()}"
    )
    print(
        f"Adaptive switch days = "
        f"{int(paired['adaptive_switch'].sum())}/{len(paired)}"
    )
    print(
        "switch rules = "
        + ", ".join(
            sorted(
                paired.loc[
                    paired["adaptive_switch"],
                    "adaptive_rule"
                ].unique()
            )
        )
        if paired["adaptive_switch"].any()
        else "switch rules = NONE"
    )

    # Strategy masks
    mask_all = pd.Series(True, index=paired.index)
    mask_switch = paired["adaptive_switch"]

    summaries = [
        strategy_summary(
            paired,
            "TOP10",
            mask_all,
            "top10_total_diff",
            "top10_machines",
        ),
        strategy_summary(
            paired,
            "ADAPTIVE",
            mask_all,
            "adaptive_total_diff",
            "adaptive_machines",
        ),
        strategy_summary(
            paired,
            "ADAPTIVE_SWITCH_ONLY",
            mask_switch,
            "adaptive_total_diff",
            "adaptive_machines",
        ),
    ]

    summary = pd.DataFrame(summaries)

    print()
    print("=" * 72)
    print("STRATEGY SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))

    # Daily paired comparison for switch days only.
    switch = paired[paired["adaptive_switch"]].copy()

    if not switch.empty:
        print()
        print("=" * 72)
        print("SWITCH-DAY ANALYSIS")
        print("=" * 72)

        print(
            switch[
                [
                    "date",
                    "block",
                    "adaptive_rule",
                    "adaptive_machines",
                    "top10_machines",
                    "adaptive_avg_diff",
                    "top10_avg_diff",
                    "adaptive_total_diff",
                    "top10_total_diff",
                    "adaptive_vs_top10_per_machine",
                    "adaptive_vs_top10_total",
                ]
            ].to_string(index=False)
        )

        print()
        print(
            f"Switch-day Adaptive total : "
            f"{switch['adaptive_total_diff'].sum():+.0f}"
        )
        print(
            f"Switch-day TOP10 total    : "
            f"{switch['top10_total_diff'].sum():+.0f}"
        )
        print(
            f"Switch-day difference     : "
            f"{(
                switch['adaptive_total_diff'].sum()
                - switch['top10_total_diff'].sum()
            ):+.0f}"
        )

        print(
            f"Switch-day per-machine improvement : "
            f"{switch['adaptive_vs_top10_per_machine'].mean():+.2f}"
        )

    # Monthly
    paired["month"] = paired["date"].dt.to_period("M").astype(str)

    monthly_rows = []
    for month, g in paired.groupby("month", sort=True):
        sw = g[g["adaptive_switch"]]

        monthly_rows.append({
            "month": month,
            "days": len(g),
            "adaptive_rule_switch_days": int(g["adaptive_switch"].sum()),
            "adaptive_total": float(g["adaptive_total_diff"].sum()),
            "top10_total": float(g["top10_total_diff"].sum()),
            "adaptive_vs_top10": float(
                g["adaptive_total_diff"].sum()
                - g["top10_total_diff"].sum()
            ),
            "switch_only_total": float(
                sw["adaptive_total_diff"].sum()
            ),
            "switch_only_days": len(sw),
        })

    monthly = pd.DataFrame(monthly_rows)

    # Block
    block = build_block_summary(paired)

    # Drawdown / cumulative
    draw_rows = []

    for strategy, active, diff_col in [
        ("TOP10", mask_all, "top10_total_diff"),
        ("ADAPTIVE", mask_all, "adaptive_total_diff"),
    ]:
        values = paired[diff_col].where(active, 0.0).astype(float)
        cumulative = values.cumsum()
        peak = cumulative.cummax()
        drawdown = cumulative - peak

        draw_rows.append(
            pd.DataFrame({
                "date": paired["date"],
                "strategy": strategy,
                "active": active.astype(bool),
                "daily_diff": values,
                "cumulative_diff": cumulative,
                "drawdown": drawdown,
            })
        )

    switch_values = paired["adaptive_total_diff"].where(
        paired["adaptive_switch"], 0.0
    ).astype(float)

    cumulative = switch_values.cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak

    draw_rows.append(
        pd.DataFrame({
            "date": paired["date"],
            "strategy": "ADAPTIVE_SWITCH_ONLY",
            "active": paired["adaptive_switch"],
            "daily_diff": switch_values,
            "cumulative_diff": cumulative,
            "drawdown": drawdown,
        })
    )

    drawdown_df = pd.concat(draw_rows, ignore_index=True)

    # Detailed daily output
    daily_output = build_daily_output(paired)

    print()
    print("=" * 72)
    print("BLOCK SUMMARY")
    print("=" * 72)
    print(block.to_string(index=False))

    print()
    print("=" * 72)
    print("IMPORTANT INTERPRETATION")
    print("=" * 72)
    print(
        "1. ADAPTIVE_SWITCH_ONLY is a pre-decision regime filter, "
        "not a confidence score filter."
    )
    print(
        "2. It only plays when prior-OOS Adaptive selects a rule other "
        "than TOP10."
    )
    print(
        "3. No future test-day result is used to decide whether to play."
    )
    print(
        "4. This does not prove profitability or statistical significance."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_output.to_csv(
        OUT_DAILY, index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        OUT_SUMMARY, index=False, encoding="utf-8-sig"
    )
    block.to_csv(
        OUT_BLOCK, index=False, encoding="utf-8-sig"
    )
    monthly.to_csv(
        OUT_MONTHLY, index=False, encoding="utf-8-sig"
    )
    drawdown_df.to_csv(
        OUT_DRAWDOWN, index=False, encoding="utf-8-sig"
    )

    print()
    print("=" * 72)
    print("FILES SAVED")
    print("=" * 72)
    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_BLOCK)
    print(OUT_MONTHLY)
    print(OUT_DRAWDOWN)
    print()
    print("Adaptive risk / selection-intensity filter test complete.")


if __name__ == "__main__":
    main()
