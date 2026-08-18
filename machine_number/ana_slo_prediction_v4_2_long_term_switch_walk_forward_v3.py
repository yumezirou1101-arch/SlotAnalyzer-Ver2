# -*- coding: utf-8 -*-
"""
Ana-Slo Ver.4.2 Long-Term Adaptive Switch Walk-Forward Validation

目的:
    既存の「long-term adaptive rank band walk-forward」の完全OOS結果から、
    AdaptiveがTOP10以外へ切り替えた日だけを「SWITCH_ONLY」として抽出し、
    TOP10 / ADAPTIVE / SWITCH_ONLY を同一期間で比較する。

重要:
    - selected_rule は各ブロックで PRIOR_OOS のみから決定された既存結果を使用。
    - 当日の差枚をルール選択には使用しない。
    - 本スクリプトは既存OOS結果の再集計であり、新しい予測モデルを学習しない。
    - SWITCH_ONLY の判定は selected_rule != TOP10。
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# PATH
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = ROOT / "data"

OUTPUT_DIR = (
    SEARCH_ROOT
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "43_Ver4_2_long_term_switch_walk_forward"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DAILY = OUTPUT_DIR / "43_Ver4_2_long_term_switch_walk_forward_daily.csv"
OUT_SUMMARY = OUTPUT_DIR / "43_Ver4_2_long_term_switch_walk_forward_summary.csv"
OUT_BLOCK = OUTPUT_DIR / "43_Ver4_2_long_term_switch_walk_forward_block.csv"
OUT_MONTHLY = OUTPUT_DIR / "43_Ver4_2_long_term_switch_walk_forward_monthly.csv"
OUT_STABILITY = OUTPUT_DIR / "43_Ver4_2_long_term_switch_walk_forward_stability.csv"


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def find_input() -> Path:
    exact_names = [
        "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv",
    ]

    candidates = []

    for name in exact_names:
        candidates.extend(SEARCH_ROOT.rglob(name))

    if not candidates:
        # fallback: any matching long-term adaptive daily CSV
        candidates = list(
            SEARCH_ROOT.rglob(
                "*long*term*adaptive*rank*band*fixed*daily.csv"
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "long-term adaptive daily CSV が見つかりません。\n"
            f"検索先: {SEARCH_ROOT}\n"
            "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv "
            "を確認してください。"
        )

    # Prefer the path containing the expected 39 directory.
    candidates = sorted(
        candidates,
        key=lambda p: (
            "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv" not in p.name,
            str(p),
        ),
    )

    return candidates[0]


def clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def profit_factor_simple(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    pos = x[x > 0].sum()
    neg = -x[x < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else float("nan")
    return float(pos / neg)


def max_losing_streak(values: pd.Series) -> int:
    streak = 0
    best = 0
    for v in pd.to_numeric(values, errors="coerce").fillna(0):
        if v < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def max_drawdown(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy()
    if len(x) == 0:
        return 0.0
    equity = np.cumsum(x)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    dd = equity - peak
    return float(dd.min())


def summarize(
    df: pd.DataFrame,
    strategy_name: str,
) -> dict:
    if df.empty:
        return {
            "strategy": strategy_name,
            "total_days": 0,
            "played_days": 0,
            "skipped_days": 0,
            "play_rate": 0.0,
            "total_diff": 0.0,
            "avg_diff_per_played_day": np.nan,
            "median_diff_per_played_day": np.nan,
            "avg_diff_per_machine": np.nan,
            "total_machine_selections": 0.0,
            "avg_machines_per_played_day": np.nan,
            "positive_days": 0,
            "negative_days": 0,
            "tie_days": 0,
            "positive_day_rate": np.nan,
            "max_losing_streak": 0,
            "max_drawdown": 0.0,
            "profit_factor_simple": np.nan,
        }

    days = len(df)
    played = int((df["played"] == True).sum())
    skipped = days - played

    played_df = df[df["played"] == True].copy()

    if played_df.empty:
        return {
            "strategy": strategy_name,
            "total_days": days,
            "played_days": 0,
            "skipped_days": skipped,
            "play_rate": 0.0,
            "total_diff": 0.0,
            "avg_diff_per_played_day": np.nan,
            "median_diff_per_played_day": np.nan,
            "avg_diff_per_machine": np.nan,
            "total_machine_selections": 0.0,
            "avg_machines_per_played_day": np.nan,
            "positive_days": 0,
            "negative_days": 0,
            "tie_days": 0,
            "positive_day_rate": np.nan,
            "max_losing_streak": 0,
            "max_drawdown": 0.0,
            "profit_factor_simple": np.nan,
        }

    diffs = played_df["strategy_total_diff"]
    machines = played_df["strategy_machines"]

    pos = int((diffs > 0).sum())
    neg = int((diffs < 0).sum())
    tie = int((diffs == 0).sum())

    total_diff = float(diffs.sum())
    total_machines = float(machines.sum())

    return {
        "strategy": strategy_name,
        "total_days": days,
        "played_days": played,
        "skipped_days": skipped,
        "play_rate": played / days * 100.0 if days else np.nan,
        "total_diff": total_diff,
        "avg_diff_per_played_day": float(diffs.mean()),
        "median_diff_per_played_day": float(diffs.median()),
        "avg_diff_per_machine": (
            total_diff / total_machines
            if total_machines > 0
            else np.nan
        ),
        "total_machine_selections": total_machines,
        "avg_machines_per_played_day": float(machines.mean()),
        "positive_days": pos,
        "negative_days": neg,
        "tie_days": tie,
        "positive_day_rate": (
            pos / played * 100.0 if played else np.nan
        ),
        "max_losing_streak": max_losing_streak(diffs),
        "max_drawdown": max_drawdown(diffs),
        "profit_factor_simple": profit_factor_simple(diffs),
    }


def paired_stats(day_df: pd.DataFrame) -> dict:
    x = day_df["switch_total_diff"].to_numpy(dtype=float)
    y = day_df["top10_total_diff"].to_numpy(dtype=float)

    diff = x - y

    n = len(diff)
    if n == 0:
        return {
            "paired_days": 0,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "switch_better_days": 0,
            "top10_better_days": 0,
            "tie_days": 0,
            "switch_better_rate": np.nan,
        }

    switch_better = int((diff > 0).sum())
    top10_better = int((diff < 0).sum())
    ties = int((diff == 0).sum())

    return {
        "paired_days": n,
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "switch_better_days": switch_better,
        "top10_better_days": top10_better,
        "tie_days": ties,
        "switch_better_rate": switch_better / n * 100.0,
    }


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Ana-Slo Ver.4.2 Long-Term Adaptive Switch Walk-Forward")
    print("=" * 72)

    input_path = find_input()

    print(f"Loading: {input_path}")

    df = pd.read_csv(input_path)
    print(f"records = {len(df):,}")
    print(f"columns = {list(df.columns)}")

    required = {
        "date",
        "block",
        "block_start",
        "block_end",
        "rule",
        "selected_rule",
        "is_adaptive_selection",
        "machines",
        "avg_diff",
        "median_diff",
        "win_rate",
        "plus1000_rate",
        "plus2000_rate",
        "positive",
        "total_diff",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "必要列がありません: " + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["block_start"] = pd.to_datetime(
        df["block_start"], errors="coerce"
    )
    df["block_end"] = pd.to_datetime(
        df["block_end"], errors="coerce"
    )

    df = clean_numeric(
        df,
        [
            "block",
            "machines",
            "avg_diff",
            "median_diff",
            "win_rate",
            "plus1000_rate",
            "plus2000_rate",
            "positive",
            "total_diff",
        ],
    )

    df = df.dropna(subset=["date", "total_diff"]).copy()
    df = df.sort_values(["date", "rule"]).reset_index(drop=True)

    # We need exactly one TOP10 row and one Adaptive-selected row per date.
    # Existing long-term adaptive output has multiple rules per date.
    top10 = (
        df[df["rule"].astype(str).str.upper() == "TOP10"]
        .sort_values(["date", "block"])
        .drop_duplicates("date", keep="last")
        .copy()
    )

    adaptive = (
        df[df["is_adaptive_selection"].astype(bool)]
        .sort_values(["date", "block"])
        .drop_duplicates("date", keep="last")
        .copy()
    )

    # If is_adaptive_selection is unreliable, use selected_rule as fallback.
    if adaptive.empty:
        adaptive = (
            df[
                df["selected_rule"].astype(str).str.upper()
                != "TOP10"
            ]
            .sort_values(["date", "block"])
            .drop_duplicates("date", keep="last")
            .copy()
        )

    top10 = top10[
        [
            "date",
            "block",
            "block_start",
            "block_end",
            "total_diff",
            "machines",
        ]
    ].rename(
        columns={
            "block": "top10_block",
            "block_start": "top10_block_start",
            "block_end": "top10_block_end",
            "total_diff": "top10_total_diff",
            "machines": "top10_machines",
        }
    )

    adaptive = adaptive[
        [
            "date",
            "block",
            "block_start",
            "block_end",
            "rule",
            "selected_rule",
            "total_diff",
            "machines",
        ]
    ].rename(
        columns={
            "block": "adaptive_block",
            "block_start": "adaptive_block_start",
            "block_end": "adaptive_block_end",
            "rule": "adaptive_rule_source",
            "total_diff": "adaptive_total_diff",
            "machines": "adaptive_machines",
        }
    )

    paired = pd.merge(
        top10,
        adaptive,
        on="date",
        how="inner",
    )

    paired = paired.sort_values("date").reset_index(drop=True)

    if paired.empty:
        raise ValueError(
            "TOP10とAdaptiveの対応日が作れませんでした。"
        )

    # Adaptive switch = selected rule is not TOP10.
    paired["is_switch_day"] = (
        paired["selected_rule"]
        .astype(str)
        .str.strip()
        .str.upper()
        .ne("TOP10")
    )

    # Switch-only strategy: play only on switch days.
    paired["switch_total_diff"] = np.where(
        paired["is_switch_day"],
        paired["adaptive_total_diff"],
        0.0,
    )

    paired["switch_machines"] = np.where(
        paired["is_switch_day"],
        paired["adaptive_machines"],
        0.0,
    )

    paired["switch_played"] = paired["is_switch_day"]

    paired["switch_vs_top10_total"] = (
        paired["switch_total_diff"]
        - np.where(
            paired["is_switch_day"],
            paired["top10_total_diff"],
            0.0,
        )
    )

    paired["switch_vs_top10_per_machine"] = np.where(
        paired["is_switch_day"],
        (
            paired["adaptive_total_diff"]
            / paired["adaptive_machines"].replace(0, np.nan)
            - paired["top10_total_diff"]
            / paired["top10_machines"].replace(0, np.nan)
        ),
        np.nan,
    )

    paired["month"] = paired["date"].dt.to_period("M").astype(str)

    # --------------------------------------------------------------
    # Daily output
    # --------------------------------------------------------------

    # merge後は block_start / block_end が
    # top10_block_start / adaptive_block_start 等に分かれるため、
    # Adaptive側のブロック期間を正式な表示列として使用する。
    daily_out = paired[
        [
            "date",
            "top10_block",
            "adaptive_block",
            "adaptive_block_start",
            "adaptive_block_end",
            "selected_rule",
            "is_switch_day",
            "top10_machines",
            "adaptive_machines",
            "top10_total_diff",
            "adaptive_total_diff",
            "switch_total_diff",
            "switch_machines",
            "switch_vs_top10_total",
            "switch_vs_top10_per_machine",
        ]
    ].copy()

    daily_out = daily_out.rename(
        columns={
            "adaptive_block_start": "block_start",
            "adaptive_block_end": "block_end",
        }
    )

    daily_out.to_csv(
        OUT_DAILY,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Strategy summary
    # --------------------------------------------------------------

    total_days = len(paired)

    top10_strategy = paired.copy()
    top10_strategy["played"] = True
    top10_strategy["strategy_total_diff"] = (
        top10_strategy["top10_total_diff"]
    )
    top10_strategy["strategy_machines"] = (
        top10_strategy["top10_machines"]
    )

    adaptive_strategy = paired.copy()
    adaptive_strategy["played"] = True
    adaptive_strategy["strategy_total_diff"] = (
        adaptive_strategy["adaptive_total_diff"]
    )
    adaptive_strategy["strategy_machines"] = (
        adaptive_strategy["adaptive_machines"]
    )

    switch_strategy = paired.copy()
    switch_strategy["played"] = switch_strategy["is_switch_day"]
    switch_strategy["strategy_total_diff"] = np.where(
        switch_strategy["is_switch_day"],
        switch_strategy["adaptive_total_diff"],
        0.0,
    )
    switch_strategy["strategy_machines"] = np.where(
        switch_strategy["is_switch_day"],
        switch_strategy["adaptive_machines"],
        0.0,
    )

    summaries = pd.DataFrame(
        [
            summarize(top10_strategy, "TOP10"),
            summarize(adaptive_strategy, "ADAPTIVE"),
            summarize(switch_strategy, "SWITCH_ONLY"),
        ]
    )

    summaries.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Block analysis
    # --------------------------------------------------------------

    block_rows = []

    for block, g in paired.groupby("adaptive_block", sort=True):
        switch_g = g[g["is_switch_day"]]

        block_rows.append(
            {
                "block": block,
                "block_start": g["adaptive_block_start"].iloc[0],
                "block_end": g["adaptive_block_end"].iloc[0],
                "days": len(g),
                "selected_rule_mode": (
                    g["selected_rule"].mode().iloc[0]
                    if not g["selected_rule"].mode().empty
                    else ""
                ),
                "switch_days": int(g["is_switch_day"].sum()),
                "switch_total": float(
                    switch_g["adaptive_total_diff"].sum()
                ),
                "top10_total_all_days": float(
                    g["top10_total_diff"].sum()
                ),
                "top10_total_switch_days": float(
                    switch_g["top10_total_diff"].sum()
                ),
                "switch_vs_top10_on_switch_days": float(
                    (
                        switch_g["adaptive_total_diff"]
                        - switch_g["top10_total_diff"]
                    ).sum()
                ),
                "switch_machines": float(
                    switch_g["adaptive_machines"].sum()
                ),
            }
        )

    block_out = pd.DataFrame(block_rows)

    block_out.to_csv(
        OUT_BLOCK,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Monthly
    # --------------------------------------------------------------

    monthly_rows = []

    for month, g in paired.groupby("month", sort=True):
        switch_g = g[g["is_switch_day"]]

        switch_diffs = switch_g["adaptive_total_diff"]

        monthly_rows.append(
            {
                "month": month,
                "days": len(g),
                "switch_days": int(len(switch_g)),
                "switch_day_rate": (
                    len(switch_g) / len(g) * 100.0
                ),
                "top10_total": float(g["top10_total_diff"].sum()),
                "adaptive_total": float(
                    g["adaptive_total_diff"].sum()
                ),
                "switch_only_total": float(
                    switch_g["adaptive_total_diff"].sum()
                ),
                "switch_only_avg_per_played_day": (
                    float(switch_diffs.mean())
                    if not switch_g.empty
                    else np.nan
                ),
                "switch_positive_days": int(
                    (switch_diffs > 0).sum()
                ),
                "switch_negative_days": int(
                    (switch_diffs < 0).sum()
                ),
                "switch_positive_rate": (
                    float((switch_diffs > 0).mean() * 100.0)
                    if not switch_g.empty
                    else np.nan
                ),
                "switch_machines": float(
                    switch_g["adaptive_machines"].sum()
                ),
            }
        )

    monthly_out = pd.DataFrame(monthly_rows)

    monthly_out.to_csv(
        OUT_MONTHLY,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Sequential stability / cumulative performance
    # --------------------------------------------------------------

    stab = paired.copy()

    stab["switch_diff"] = np.where(
        stab["is_switch_day"],
        stab["adaptive_total_diff"],
        0.0,
    )

    stab["switch_machines_used"] = np.where(
        stab["is_switch_day"],
        stab["adaptive_machines"],
        0.0,
    )

    stab["switch_cum_diff"] = stab["switch_diff"].cumsum()
    stab["top10_cum_diff"] = stab["top10_total_diff"].cumsum()
    stab["adaptive_cum_diff"] = stab["adaptive_total_diff"].cumsum()

    stab["switch_cum_machines"] = (
        stab["switch_machines_used"].cumsum()
    )

    stab["switch_cum_avg_per_machine"] = np.where(
        stab["switch_cum_machines"] > 0,
        stab["switch_cum_diff"]
        / stab["switch_cum_machines"],
        np.nan,
    )

    stab["switch_cum_played_days"] = (
        stab["is_switch_day"].cumsum()
    )

    stab["switch_cum_positive_days"] = (
        (
            stab["is_switch_day"]
            & (stab["adaptive_total_diff"] > 0)
        ).cumsum()
    )

    stab["switch_cum_negative_days"] = (
        (
            stab["is_switch_day"]
            & (stab["adaptive_total_diff"] < 0)
        ).cumsum()
    )

    stab["switch_cum_positive_rate"] = np.where(
        stab["switch_cum_played_days"] > 0,
        stab["switch_cum_positive_days"]
        / stab["switch_cum_played_days"]
        * 100.0,
        np.nan,
    )

    stability_out = stab[
        [
            "date",
            "adaptive_block",
            "selected_rule",
            "is_switch_day",
            "adaptive_total_diff",
            "top10_total_diff",
            "switch_diff",
            "switch_cum_diff",
            "top10_cum_diff",
            "adaptive_cum_diff",
            "switch_cum_machines",
            "switch_cum_avg_per_machine",
            "switch_cum_played_days",
            "switch_cum_positive_days",
            "switch_cum_negative_days",
            "switch_cum_positive_rate",
        ]
    ].copy()

    stability_out.to_csv(
        OUT_STABILITY,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Console report
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("LONG-TERM SWITCH WALK-FORWARD RESULT")
    print("=" * 72)

    print(f"paired days          : {total_days}")
    print(
        f"switch days          : "
        f"{int(paired['is_switch_day'].sum())}/{total_days}"
    )
    print(
        f"switch day rate      : "
        f"{paired['is_switch_day'].mean() * 100:.2f}%"
    )

    print()
    print(summaries.to_string(index=False))

    stats = paired_stats(
        paired[
            [
                "switch_total_diff",
                "top10_total_diff",
            ]
        ].rename(
            columns={
                "switch_total_diff": "switch_total_diff",
                "top10_total_diff": "top10_total_diff",
            }
        )
    )

    print()
    print("=" * 72)
    print("SWITCH DAYS vs TOP10")
    print("=" * 72)

    switch_days_df = paired[paired["is_switch_day"]].copy()

    print(
        f"switch-day Adaptive total : "
        f"{switch_days_df['adaptive_total_diff'].sum():+.0f}"
    )
    print(
        f"switch-day TOP10 total    : "
        f"{switch_days_df['top10_total_diff'].sum():+.0f}"
    )
    print(
        f"switch-day difference     : "
        f"{(
            switch_days_df['adaptive_total_diff']
            - switch_days_df['top10_total_diff']
        ).sum():+.0f}"
    )

    if not switch_days_df.empty:
        print(
            f"switch-day Adaptive avg   : "
            f"{switch_days_df['adaptive_total_diff'].mean():+.2f}"
        )
        print(
            f"switch-day Adaptive / machine : "
            f"{(
                switch_days_df['adaptive_total_diff'].sum()
                / switch_days_df['adaptive_machines'].sum()
            ):+.2f}"
        )

    print()
    print("=" * 72)
    print("PAIRED COMPARISON ON SWITCH DAYS")
    print("=" * 72)

    if not switch_days_df.empty:
        switch_delta = (
            switch_days_df["adaptive_total_diff"]
            - switch_days_df["top10_total_diff"]
        )

        print(f"paired switch days     : {len(switch_days_df)}")
        print(f"Adaptive better days   : {(switch_delta > 0).sum()}")
        print(f"TOP10 better days      : {(switch_delta < 0).sum()}")
        print(f"tie days               : {(switch_delta == 0).sum()}")
        print(
            f"mean difference        : "
            f"{switch_delta.mean():+.2f}"
        )
        print(
            f"median difference      : "
            f"{switch_delta.median():+.2f}"
        )

    print()
    print("=" * 72)
    print("FILES SAVED")
    print("=" * 72)

    print(OUT_DAILY)
    print(OUT_SUMMARY)
    print(OUT_BLOCK)
    print(OUT_MONTHLY)
    print(OUT_STABILITY)

    print()
    print("IMPORTANT:")
    print("This is an OOS switch extraction / stability analysis.")
    print("The selected rule comes from prior-OOS Adaptive results.")
    print("It does not establish future profitability or statistical significance.")
    print("A longer historical sample is required before production adoption.")


if __name__ == "__main__":
    main()
