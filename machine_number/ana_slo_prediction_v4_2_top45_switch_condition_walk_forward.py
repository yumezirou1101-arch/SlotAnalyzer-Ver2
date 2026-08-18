# -*- coding: utf-8 -*-
"""
Ana-Slo Ver.4.2 TOP4_5 Switch Condition Walk-Forward

目的:
    V4.2_C Adaptive が TOP4_5 へ切り替える条件を、
    「当時までの prior-OOS 成績」だけで判定し、完全 OOS で検証する。

基本ルール（PRIMARY）:
    - 初回ブロックは TOP10
    - prior block が2個未満なら TOP10
    - prior-OOS の累積平均差枚で
          TOP4_5平均 - TOP10平均 >= +300枚/日
      なら次ブロックを TOP4_5
    - それ以外は TOP10

重要:
    - 当該 test block の結果は、その block の選択には絶対に使用しない。
    - PRIMARY ルールを固定して評価する。
    - 複数閾値の診断は探索的分析であり、将来成績を保証しない。
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = ROOT / "data"

INPUT_NAME = "39_Ver4_2_long_term_adaptive_rank_band_fixed_daily.csv"

OUTPUT_DIR = (
    SEARCH_ROOT
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "44_Ver4_2_TOP4_5_switch_condition_walk_forward"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DAILY = OUTPUT_DIR / "44_Ver4_2_TOP4_5_switch_condition_daily.csv"
OUT_BLOCK = OUTPUT_DIR / "44_Ver4_2_TOP4_5_switch_condition_blocks.csv"
OUT_SUMMARY = OUTPUT_DIR / "44_Ver4_2_TOP4_5_switch_condition_summary.csv"
OUT_DIAG = OUTPUT_DIR / "44_Ver4_2_TOP4_5_switch_condition_diagnostic.csv"
OUT_MONTHLY = OUTPUT_DIR / "44_Ver4_2_TOP4_5_switch_condition_monthly.csv"


PRIMARY_THRESHOLD = 300.0
PRIMARY_MIN_PRIOR_BLOCKS = 2

# Exploratory thresholds only. Do not select the production rule from these
# without a subsequent untouched validation period.
DIAGNOSTIC_THRESHOLDS = [0, 100, 200, 300, 400, 500, 750, 1000]


def find_input() -> Path:
    candidates = list(SEARCH_ROOT.rglob(INPUT_NAME))
    if not candidates:
        candidates = list(
            SEARCH_ROOT.rglob(
                "*long*term*adaptive*rank*band*fixed*daily.csv"
            )
        )
    if not candidates:
        raise FileNotFoundError(
            "入力CSVが見つかりません。\n"
            f"検索先: {SEARCH_ROOT}\n"
            f"期待ファイル: {INPUT_NAME}"
        )
    candidates.sort(key=lambda p: (INPUT_NAME not in p.name, str(p)))
    return candidates[0]


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
    return float((equity - peak).min())


def summarize_strategy(
    daily: pd.DataFrame,
    name: str,
    diff_col: str,
    machine_col: str,
    played_col: str,
) -> dict:
    total_days = len(daily)
    played = daily[daily[played_col]].copy()

    if played.empty:
        return {
            "strategy": name,
            "days": total_days,
            "played_days": 0,
            "skipped_days": total_days,
            "play_rate": 0.0,
            "total_diff": 0.0,
            "avg_diff_per_played_day": np.nan,
            "median_diff_per_played_day": np.nan,
            "avg_diff_per_machine": np.nan,
            "total_machine_selections": 0.0,
            "positive_days": 0,
            "negative_days": 0,
            "positive_day_rate": np.nan,
            "max_losing_streak": 0,
            "max_drawdown": 0.0,
        }

    diffs = played[diff_col].astype(float)
    machines = played[machine_col].astype(float)

    return {
        "strategy": name,
        "days": total_days,
        "played_days": len(played),
        "skipped_days": total_days - len(played),
        "play_rate": len(played) / total_days * 100,
        "total_diff": float(diffs.sum()),
        "avg_diff_per_played_day": float(diffs.mean()),
        "median_diff_per_played_day": float(diffs.median()),
        "avg_diff_per_machine": (
            float(diffs.sum() / machines.sum())
            if machines.sum() > 0 else np.nan
        ),
        "total_machine_selections": float(machines.sum()),
        "positive_days": int((diffs > 0).sum()),
        "negative_days": int((diffs < 0).sum()),
        "positive_day_rate": float((diffs > 0).mean() * 100),
        "max_losing_streak": max_losing_streak(diffs),
        "max_drawdown": max_drawdown(diffs),
    }


def build_block_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for block, g in df.groupby("block", sort=True):
        start = g["block_start"].iloc[0]
        end = g["block_end"].iloc[0]

        top10 = g[g["rule"].str.upper() == "TOP10"]
        top45 = g[g["rule"].str.upper() == "TOP4_5"]

        if top10.empty or top45.empty:
            continue

        rows.append({
            "block": int(block),
            "block_start": start,
            "block_end": end,
            "top10_total_diff": float(top10["total_diff"].sum()),
            "top45_total_diff": float(top45["total_diff"].sum()),
            "top10_days": int(len(top10)),
            "top45_days": int(len(top45)),
            "top10_avg_diff": float(top10["total_diff"].mean()),
            "top45_avg_diff": float(top45["total_diff"].mean()),
            "top45_minus_top10": float(
                top45["total_diff"].mean()
                - top10["total_diff"].mean()
            ),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("TOP10 / TOP4_5 のブロック比較を作れませんでした。")
    return out.sort_values("block").reset_index(drop=True)


def evaluate_threshold(
    blocks: pd.DataFrame,
    threshold: float,
    min_prior_blocks: int,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    for _, row in blocks.iterrows():
        block = int(row["block"])
        prior = blocks[blocks["block"] < block]

        prior_n = len(prior)

        if prior_n < min_prior_blocks:
            selected = "TOP10"
            reason = "INSUFFICIENT_PRIOR_BLOCKS"
        else:
            prior_top45 = prior["top45_avg_diff"].mean()
            prior_top10 = prior["top10_avg_diff"].mean()
            prior_edge = prior_top45 - prior_top10

            if prior_edge >= threshold:
                selected = "TOP4_5"
                reason = "PRIOR_TOP45_EDGE_ABOVE_THRESHOLD"
            else:
                selected = "TOP10"
                reason = "PRIOR_TOP45_EDGE_BELOW_THRESHOLD"

        if selected == "TOP4_5":
            oos_diff = float(row["top45_total_diff"])
        else:
            oos_diff = float(row["top10_total_diff"])

        top10_diff = float(row["top10_total_diff"])

        rows.append({
            "block": block,
            "block_start": row["block_start"],
            "block_end": row["block_end"],
            "selected_rule": selected,
            "selection_reason": reason,
            "prior_blocks_used": prior_n,
            "prior_top45_minus_top10": (
                float(
                    prior["top45_avg_diff"].mean()
                    - prior["top10_avg_diff"].mean()
                )
                if prior_n else np.nan
            ),
            "threshold": threshold,
            "oos_selected_total_diff": oos_diff,
            "oos_top10_total_diff": top10_diff,
            "oos_vs_top10": oos_diff - top10_diff,
        })

    result = pd.DataFrame(rows)

    summary = {
        "threshold": threshold,
        "min_prior_blocks": min_prior_blocks,
        "blocks": len(result),
        "TOP4_5_selected_blocks": int(
            (result["selected_rule"] == "TOP4_5").sum()
        ),
        "TOP10_selected_blocks": int(
            (result["selected_rule"] == "TOP10").sum()
        ),
        "adaptive_total_diff": float(
            result["oos_selected_total_diff"].sum()
        ),
        "top10_total_diff": float(
            result["oos_top10_total_diff"].sum()
        ),
        "adaptive_vs_top10": float(
            result["oos_vs_top10"].sum()
        ),
        "adaptive_avg_block_diff": float(
            result["oos_selected_total_diff"].mean()
        ),
        "top10_avg_block_diff": float(
            result["oos_top10_total_diff"].mean()
        ),
        "positive_blocks": int(
            (result["oos_selected_total_diff"] > 0).sum()
        ),
        "negative_blocks": int(
            (result["oos_selected_total_diff"] < 0).sum()
        ),
    }

    return result, summary


def main() -> None:
    print("=" * 72)
    print("Ana-Slo Ver.4.2 TOP4_5 Switch Condition Walk-Forward")
    print("=" * 72)

    input_path = find_input()
    print(f"Loading: {input_path}")

    df = pd.read_csv(input_path)
    print(f"records = {len(df):,}")

    required = {
        "date", "block", "block_start", "block_end",
        "rule", "total_diff"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("必要列がありません: " + ", ".join(missing))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["block_start"] = pd.to_datetime(
        df["block_start"], errors="coerce"
    )
    df["block_end"] = pd.to_datetime(
        df["block_end"], errors="coerce"
    )
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["total_diff"] = pd.to_numeric(
        df["total_diff"], errors="coerce"
    )
    df["rule"] = df["rule"].astype(str).str.strip()

    df = df.dropna(
        subset=["date", "block", "total_diff"]
    ).copy()

    blocks = build_block_table(df)

    print()
    print(f"walk-forward blocks = {len(blocks)}")
    for _, r in blocks.iterrows():
        print(
            f"BLOCK {int(r['block'])}: "
            f"{r['block_start'].date()} to {r['block_end'].date()}"
        )

    # --------------------------------------------------------------
    # PRIMARY rule
    # --------------------------------------------------------------

    primary_daily, primary_summary = evaluate_threshold(
        blocks,
        PRIMARY_THRESHOLD,
        PRIMARY_MIN_PRIOR_BLOCKS,
    )

    print()
    print("=" * 72)
    print("PRIMARY SWITCH CONDITION")
    print("=" * 72)
    print(
        f"Condition: prior TOP4_5 edge >= "
        f"+{PRIMARY_THRESHOLD:.0f}枚/day"
    )
    print(
        f"Minimum prior blocks: {PRIMARY_MIN_PRIOR_BLOCKS}"
    )

    print()
    print(
        primary_daily[
            [
                "block",
                "block_start",
                "block_end",
                "selected_rule",
                "selection_reason",
                "prior_blocks_used",
                "prior_top45_minus_top10",
                "oos_selected_total_diff",
                "oos_top10_total_diff",
                "oos_vs_top10",
            ]
        ].to_string(index=False)
    )

    # Daily approximation:
    # A selected block's total is distributed according to its actual
    # daily rule rows. For the final primary comparison we use block-level
    # totals because the switching decision is made at block boundaries.
    top10_daily = (
        df[df["rule"].str.upper() == "TOP10"]
        .groupby("date", as_index=False)["total_diff"]
        .sum()
        .rename(columns={"total_diff": "top10_daily_diff"})
    )

    top45_daily = (
        df[df["rule"].str.upper() == "TOP4_5"]
        .groupby("date", as_index=False)["total_diff"]
        .sum()
        .rename(columns={"total_diff": "top45_daily_diff"})
    )

    all_days = (
        df[["date", "block", "block_start", "block_end"]]
        .drop_duplicates("date")
        .merge(top10_daily, on="date", how="left")
        .merge(top45_daily, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    selection_map = primary_daily[
        ["block", "selected_rule"]
    ].drop_duplicates()

    all_days = all_days.merge(
        selection_map,
        on="block",
        how="left",
    )

    all_days["selected_daily_diff"] = np.where(
        all_days["selected_rule"].eq("TOP4_5"),
        all_days["top45_daily_diff"],
        all_days["top10_daily_diff"],
    )

    all_days["switch_day"] = all_days["selected_rule"].eq("TOP4_5")

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    primary_day_summary = pd.DataFrame([
        summarize_strategy(
            all_days.assign(
                played=True,
                diff=all_days["top10_daily_diff"],
                machines=10,
            ),
            "TOP10",
            "diff",
            "machines",
            "played",
        ),
        summarize_strategy(
            all_days.assign(
                played=True,
                diff=all_days["selected_daily_diff"],
                machines=np.where(
                    all_days["selected_rule"].eq("TOP4_5"),
                    2,
                    10,
                ),
            ),
            "CONDITION_ADAPTIVE",
            "diff",
            "machines",
            "played",
        ),
        summarize_strategy(
            all_days.assign(
                played=all_days["switch_day"],
                diff=np.where(
                    all_days["switch_day"],
                    all_days["top45_daily_diff"],
                    0,
                ),
                machines=np.where(
                    all_days["switch_day"], 2, 0
                ),
            ),
            "SWITCH_ONLY_TOP4_5",
            "diff",
            "machines",
            "played",
        ),
    ])

    primary_day_summary.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    primary_daily.to_csv(
        OUT_BLOCK,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Exploratory threshold diagnostic
    # --------------------------------------------------------------

    diag_rows = []

    for threshold in DIAGNOSTIC_THRESHOLDS:
        _, s = evaluate_threshold(
            blocks,
            threshold,
            PRIMARY_MIN_PRIOR_BLOCKS,
        )
        diag_rows.append(s)

    diagnostic = pd.DataFrame(diag_rows)

    diagnostic["rank_by_total"] = (
        diagnostic["adaptive_total_diff"]
        .rank(method="min", ascending=False)
    )

    diagnostic.to_csv(
        OUT_DIAG,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Monthly primary
    # --------------------------------------------------------------

    all_days["month"] = (
        all_days["date"].dt.to_period("M").astype(str)
    )

    monthly_rows = []

    for month, g in all_days.groupby("month", sort=True):
        sw = g[g["switch_day"]]

        monthly_rows.append({
            "month": month,
            "days": len(g),
            "switch_days": int(len(sw)),
            "switch_rate": len(sw) / len(g) * 100,
            "top10_total": float(g["top10_daily_diff"].sum()),
            "condition_adaptive_total": float(
                g["selected_daily_diff"].sum()
            ),
            "switch_only_total": float(
                sw["top45_daily_diff"].sum()
            ),
            "switch_only_avg_per_day": (
                float(sw["top45_daily_diff"].mean())
                if not sw.empty else np.nan
            ),
        })

    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(
        OUT_MONTHLY,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Console result
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("PRIMARY RESULT")
    print("=" * 72)
    print(primary_day_summary.to_string(index=False))

    print()
    print("=" * 72)
    print("EXPLORATORY THRESHOLD DIAGNOSTIC")
    print("=" * 72)
    print(diagnostic.to_string(index=False))

    print()
    print("=" * 72)
    print("MONTHLY")
    print("=" * 72)
    print(monthly.to_string(index=False))

    print()
    print("=" * 72)
    print("FILES SAVED")
    print("=" * 72)
    print(OUT_SUMMARY)
    print(OUT_BLOCK)
    print(OUT_DIAG)
    print(OUT_MONTHLY)
    print(OUT_DAILY)

    print()
    print("IMPORTANT:")
    print(
        "PRIMARY threshold is fixed at "
        f"+{PRIMARY_THRESHOLD:.0f}枚/day."
    )
    print(
        "Test-block results are never used to choose that block."
    )
    print(
        "Threshold diagnostic is exploratory and may not be used "
        "as final production tuning."
    )
    print(
        "The current sample is short; future profitability is not proven."
    )


if __name__ == "__main__":
    main()
