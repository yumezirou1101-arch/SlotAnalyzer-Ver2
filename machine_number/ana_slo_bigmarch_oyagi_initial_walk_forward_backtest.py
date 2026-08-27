from __future__ import annotations

from pathlib import Path
import argparse
import math
import pandas as pd
import numpy as np


# ============================================================
# Big March Takasaki Oyagi - Initial Walk-Forward Backtest
# ============================================================
#
# Purpose
# -------
# Build a simple, leakage-safe baseline backtest for the 31-day
# integrated Big March Takasaki Oyagi dataset.
#
# This is NOT a production model.
#
# Candidate baseline signals:
#   1) AVG_HISTORY      : historical average diff per machine
#   2) RECENT7_AVG      : average diff over previous 7 observed days
#   3) RECENT7_WIN      : win rate over previous 7 observed days
#   4) WEEKDAY_AVG      : historical average diff on same weekday
#   5) PLUS1000_RATE    : historical rate of diff >= +1000
#   6) SIMPLE_BLEND     : simple standardized blend of the above
#
# Leakage control
# ---------------
# For each target date, features are built ONLY from prior dates.
# The target day's diff is used only for evaluation.
#
# Output
# ------
# - daily TOP10 results per baseline
# - overall model comparison
# - individual picks
# ============================================================


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

QUALITY_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
    / "analysis_31days_deep"
    / "01_data_quality"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
    / "analysis_31days_deep"
    / "02_initial_walk_forward_backtest"
)


def header(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def parse_args():
    p = argparse.ArgumentParser(
        description="Initial leakage-safe walk-forward baseline backtest."
    )
    p.add_argument(
        "--min-history-days",
        type=int,
        default=7,
        help="Minimum prior calendar days before evaluation starts. Default: 7.",
    )
    p.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Number of machines selected each day. Default: 10.",
    )
    return p.parse_args()


def find_integrated_file() -> Path:
    files = sorted(
        QUALITY_DIR.glob("01_bigmarch_oyagi_integrated_*.csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No integrated quality dataset found in:\n{QUALITY_DIR}"
        )
    return files[-1]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    required = {
        "date",
        "machine_name",
        "machine_no",
        "G",
        "diff",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="raise").dt.normalize()
    x["machine_no"] = pd.to_numeric(x["machine_no"], errors="raise").astype(int)
    x["G"] = pd.to_numeric(x["G"], errors="coerce")
    x["diff"] = pd.to_numeric(x["diff"], errors="coerce")
    x["machine_name"] = x["machine_name"].astype(str).str.strip()
    x["win"] = (x["diff"] > 0).astype(int)
    x["plus1000"] = (x["diff"] >= 1000).astype(int)
    x["plus2000"] = (x["diff"] >= 2000).astype(int)
    x["weekday"] = x["date"].dt.weekday
    return x.sort_values(["date", "machine_no"]).reset_index(drop=True)


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def build_features(history: pd.DataFrame, target_panel: pd.DataFrame) -> pd.DataFrame:
    # Historical per-machine aggregates.
    hist_machine = (
        history.groupby("machine_no")
        .agg(
            avg_history=("diff", "mean"),
            plus1000_rate=("plus1000", "mean"),
            history_n=("date", "nunique"),
        )
        .reset_index()
    )

    # Recent 7 observed rows for each machine.
    recent_parts = []
    for machine_no, grp in history.groupby("machine_no"):
        g = grp.sort_values("date").tail(7)
        recent_parts.append(
            {
                "machine_no": machine_no,
                "recent7_avg": g["diff"].mean(),
                "recent7_win": g["win"].mean(),
                "recent7_n": g["date"].nunique(),
            }
        )
    recent = pd.DataFrame(recent_parts)

    # Same-weekday historical average.
    target_weekday = int(target_panel["weekday"].iloc[0])
    weekday_hist = history[history["weekday"] == target_weekday]

    weekday_avg = (
        weekday_hist.groupby("machine_no")["diff"]
        .mean()
        .rename("weekday_avg")
        .reset_index()
    )

    features = (
        target_panel[
            ["date", "machine_no", "machine_name", "diff", "win", "plus1000", "plus2000"]
        ]
        .merge(hist_machine, on="machine_no", how="left")
        .merge(recent, on="machine_no", how="left")
        .merge(weekday_avg, on="machine_no", how="left")
    )

    # Fill missing signals conservatively with same-day panel medians/zeros.
    for col in (
        "avg_history",
        "recent7_avg",
        "recent7_win",
        "weekday_avg",
        "plus1000_rate",
    ):
        med = features[col].median()
        if pd.isna(med):
            med = 0.0
        features[col] = features[col].fillna(med)

    features["history_n"] = features["history_n"].fillna(0)
    features["recent7_n"] = features["recent7_n"].fillna(0)

    # Simple blend: equal-weight standardized baseline.
    features["simple_blend"] = (
        zscore(features["avg_history"])
        + zscore(features["recent7_avg"])
        + zscore(features["recent7_win"])
        + zscore(features["weekday_avg"])
        + zscore(features["plus1000_rate"])
    ) / 5.0

    return features


def evaluate_selection(
    selected: pd.DataFrame,
    target_date: pd.Timestamp,
    model_name: str,
) -> dict:
    n = len(selected)
    if n == 0:
        return {
            "target_date": target_date.date(),
            "model": model_name,
            "selected_n": 0,
            "avg_diff": np.nan,
            "median_diff": np.nan,
            "total_diff": 0.0,
            "win_rate": np.nan,
            "plus1000_rate": np.nan,
            "plus2000_rate": np.nan,
            "positive_day": False,
        }

    return {
        "target_date": target_date.date(),
        "model": model_name,
        "selected_n": n,
        "avg_diff": selected["diff"].mean(),
        "median_diff": selected["diff"].median(),
        "total_diff": selected["diff"].sum(),
        "win_rate": selected["win"].mean() * 100,
        "plus1000_rate": selected["plus1000"].mean() * 100,
        "plus2000_rate": selected["plus2000"].mean() * 100,
        "positive_day": selected["diff"].mean() > 0,
    }


def main():
    args = parse_args()

    if args.min_history_days < 1:
        raise ValueError("--min-history-days must be >= 1")
    if args.topn < 1:
        raise ValueError("--topn must be >= 1")

    source = find_integrated_file()
    data = load_data(source)

    dates = sorted(data["date"].drop_duplicates())
    if len(dates) <= args.min_history_days:
        raise RuntimeError("Not enough dates for walk-forward backtest.")

    header("Big March Takasaki Oyagi - Initial Walk-Forward Backtest")
    print(f"source                : {source}")
    print(f"records               : {len(data):,}")
    print(f"days                  : {len(dates)}")
    print(f"date range            : {dates[0].date()} to {dates[-1].date()}")
    print(f"min history days      : {args.min_history_days}")
    print(f"top N                 : {args.topn}")

    model_cols = {
        "AVG_HISTORY": "avg_history",
        "RECENT7_AVG": "recent7_avg",
        "RECENT7_WIN": "recent7_win",
        "WEEKDAY_AVG": "weekday_avg",
        "PLUS1000_RATE": "plus1000_rate",
        "SIMPLE_BLEND": "simple_blend",
    }

    daily_rows = []
    pick_rows = []

    eval_dates = dates[args.min_history_days:]

    for target_date in eval_dates:
        history = data[data["date"] < target_date].copy()
        target_panel = data[data["date"] == target_date].copy()

        if target_panel.empty:
            continue

        features = build_features(history, target_panel)

        header(f"TARGET {target_date.date()}")

        for model_name, score_col in model_cols.items():
            ranked = (
                features.sort_values(
                    [score_col, "machine_no"],
                    ascending=[False, True],
                )
                .head(args.topn)
                .copy()
            )

            ranked["prediction_rank"] = range(1, len(ranked) + 1)
            ranked["model"] = model_name
            ranked["score"] = ranked[score_col]

            result = evaluate_selection(
                ranked,
                target_date,
                model_name,
            )
            daily_rows.append(result)

            for _, row in ranked.iterrows():
                pick_rows.append(
                    {
                        "target_date": target_date.date(),
                        "model": model_name,
                        "prediction_rank": int(row["prediction_rank"]),
                        "machine_no": int(row["machine_no"]),
                        "machine_name": row["machine_name"],
                        "score": float(row["score"]),
                        "actual_diff": float(row["diff"]),
                        "actual_win": int(row["win"]),
                        "actual_plus1000": int(row["plus1000"]),
                        "actual_plus2000": int(row["plus2000"]),
                        "avg_history": float(row["avg_history"]),
                        "recent7_avg": float(row["recent7_avg"]),
                        "recent7_win": float(row["recent7_win"]),
                        "weekday_avg": float(row["weekday_avg"]),
                        "plus1000_rate": float(row["plus1000_rate"]),
                        "history_n": int(row["history_n"]),
                        "recent7_n": int(row["recent7_n"]),
                    }
                )

            print(
                f"{model_name:<16} "
                f"avg={result['avg_diff']:>8.1f} "
                f"win={result['win_rate']:>5.1f}% "
                f"+1000={result['plus1000_rate']:>5.1f}% "
                f"+2000={result['plus2000_rate']:>5.1f}% "
                f"total={result['total_diff']:>9.0f}"
            )

    daily = pd.DataFrame(daily_rows)
    picks = pd.DataFrame(pick_rows)

    overall = (
        daily.groupby("model")
        .agg(
            evaluated_days=("target_date", "nunique"),
            mean_daily_avg_diff=("avg_diff", "mean"),
            median_daily_avg_diff=("avg_diff", "median"),
            mean_win_rate=("win_rate", "mean"),
            mean_plus1000_rate=("plus1000_rate", "mean"),
            mean_plus2000_rate=("plus2000_rate", "mean"),
            positive_day_rate=("positive_day", "mean"),
            total_diff=("total_diff", "sum"),
        )
        .reset_index()
    )

    overall["positive_day_rate"] = overall["positive_day_rate"] * 100
    overall = overall.sort_values(
        ["mean_daily_avg_diff", "total_diff"],
        ascending=[False, False],
    ).reset_index(drop=True)

    header("OVERALL MODEL COMPARISON")
    print(overall.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_path = OUTPUT_DIR / "02_initial_walk_forward_daily.csv"
    picks_path = OUTPUT_DIR / "02_initial_walk_forward_picks.csv"
    overall_path = OUTPUT_DIR / "02_initial_walk_forward_overall.csv"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    picks.to_csv(picks_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")

    header("FILES SAVED")
    print(daily_path)
    print(picks_path)
    print(overall_path)

    print()
    print("Initial baseline backtest complete.")
    print("This is for model selection research only; no production model was changed.")
    print("No Maruhan Maebashi files were modified.")


if __name__ == "__main__":
    main()
