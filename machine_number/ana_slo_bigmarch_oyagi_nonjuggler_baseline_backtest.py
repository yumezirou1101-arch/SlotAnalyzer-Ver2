from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# Big March Takasaki Oyagi
# NON_JUGGLER Baseline Walk-Forward Backtest
# ============================================================
#
# Purpose
# -------
# Compare baseline prediction signals using NON_JUGGLER machines only.
#
# Candidate signals:
#   1) AVG_HISTORY
#   2) RECENT7_AVG
#   3) RECENT7_WIN
#   4) WEEKDAY_AVG
#   5) PLUS1000_RATE
#   6) SIMPLE_BLEND
#
# Leakage control:
#   For each target date, all features are calculated only from
#   dates strictly before the target date.
#
# This script is for research only.
# It does NOT modify any production model.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

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
    / "10_nonjuggler_baseline_backtest"
)

JUGGLER_KEYWORD = "ジャグラー"


def header(title: str) -> None:
    print()
    print("=" * 124)
    print(title)
    print("=" * 124)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe NON_JUGGLER baseline walk-forward backtest."
        )
    )

    parser.add_argument(
        "--min-history-days",
        type=int,
        default=7,
        help=(
            "Minimum prior calendar days before evaluation starts. "
            "Default: 7."
        ),
    )

    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Number of NON_JUGGLER machines selected per day. Default: 10.",
    )

    return parser.parse_args()


def find_integrated_file() -> Path:
    files = sorted(
        QUALITY_DIR.glob(
            "01_bigmarch_oyagi_integrated_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No integrated quality dataset found in:\n{QUALITY_DIR}"
        )

    return files[-1]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required = {
        "date",
        "machine_name",
        "machine_no",
        "G",
        "diff",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    x = df.copy()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="raise",
    ).dt.normalize()

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="raise",
    ).astype(int)

    x["G"] = pd.to_numeric(
        x["G"],
        errors="coerce",
    )

    x["diff"] = pd.to_numeric(
        x["diff"],
        errors="coerce",
    )

    if x["diff"].isna().any():
        raise RuntimeError(
            "Missing or invalid diff found."
        )

    duplicate_mask = x.duplicated(
        subset=[
            "date",
            "machine_no",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = x.loc[
            duplicate_mask,
            [
                "date",
                "machine_no",
                "machine_name",
            ],
        ]

        raise RuntimeError(
            "Duplicate date-machine rows found:\n"
            + duplicate_rows.head(20).to_string(index=False)
        )

    x["win"] = (
        x["diff"] > 0
    ).astype(int)

    x["plus1000"] = (
        x["diff"] >= 1000
    ).astype(int)

    x["plus2000"] = (
        x["diff"] >= 2000
    ).astype(int)

    x["weekday"] = (
        x["date"].dt.weekday
    )

    x["is_juggler"] = (
        x["machine_name"]
        .str.contains(
            JUGGLER_KEYWORD,
            na=False,
        )
    )

    x = x[
        ~x["is_juggler"]
    ].copy()

    if x.empty:
        raise RuntimeError(
            "No NON_JUGGLER rows found. "
            "Check JUGGLER_KEYWORD and source data encoding."
        )

    return (
        x.sort_values(
            [
                "date",
                "machine_no",
            ]
        )
        .reset_index(drop=True)
    )


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(
        series,
        errors="coerce",
    )

    mean = s.mean()
    std = s.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(
            0.0,
            index=s.index,
        )

    return (
        s - mean
    ) / std


def build_features(
    history: pd.DataFrame,
    target_panel: pd.DataFrame,
) -> pd.DataFrame:

    hist_machine = (
        history
        .groupby("machine_no")
        .agg(
            avg_history=(
                "diff",
                "mean",
            ),
            plus1000_rate=(
                "plus1000",
                "mean",
            ),
            history_n=(
                "date",
                "nunique",
            ),
        )
        .reset_index()
    )

    recent_rows = []

    for machine_no, group in history.groupby(
        "machine_no"
    ):
        recent = (
            group
            .sort_values("date")
            .tail(7)
        )

        recent_rows.append(
            {
                "machine_no": int(
                    machine_no
                ),
                "recent7_avg": (
                    recent["diff"].mean()
                ),
                "recent7_win": (
                    recent["win"].mean()
                ),
                "recent7_n": (
                    recent["date"].nunique()
                ),
            }
        )

    recent = pd.DataFrame(
        recent_rows
    )

    target_weekday = int(
        target_panel["weekday"].iloc[0]
    )

    weekday_history = history[
        history["weekday"]
        == target_weekday
    ]

    weekday_avg = (
        weekday_history
        .groupby("machine_no")["diff"]
        .mean()
        .rename("weekday_avg")
        .reset_index()
    )

    features = (
        target_panel[
            [
                "date",
                "machine_no",
                "machine_name",
                "diff",
                "win",
                "plus1000",
                "plus2000",
            ]
        ]
        .merge(
            hist_machine,
            on="machine_no",
            how="left",
        )
        .merge(
            recent,
            on="machine_no",
            how="left",
        )
        .merge(
            weekday_avg,
            on="machine_no",
            how="left",
        )
    )

    feature_columns = (
        "avg_history",
        "recent7_avg",
        "recent7_win",
        "weekday_avg",
        "plus1000_rate",
    )

    for column in feature_columns:
        median = features[
            column
        ].median()

        if pd.isna(median):
            median = 0.0

        features[column] = (
            features[column]
            .fillna(median)
        )

    features["history_n"] = (
        features["history_n"]
        .fillna(0)
        .astype(int)
    )

    features["recent7_n"] = (
        features["recent7_n"]
        .fillna(0)
        .astype(int)
    )

    features["simple_blend"] = (
        zscore(
            features["avg_history"]
        )
        + zscore(
            features["recent7_avg"]
        )
        + zscore(
            features["recent7_win"]
        )
        + zscore(
            features["weekday_avg"]
        )
        + zscore(
            features["plus1000_rate"]
        )
    ) / 5.0

    return features


def bootstrap_daily_mean_ci(
    values,
    n_boot: int = 10000,
    seed: int = 20260829,
):
    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    samples = rng.choice(
        arr,
        size=(
            n_boot,
            len(arr),
        ),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    return (
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def evaluate_selection(
    selected: pd.DataFrame,
    target_date: pd.Timestamp,
    model_name: str,
) -> dict:

    if selected.empty:
        return {
            "target_date": (
                target_date.date()
            ),
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
        "target_date": (
            target_date.date()
        ),
        "model": model_name,
        "selected_n": (
            len(selected)
        ),
        "avg_diff": (
            selected["diff"].mean()
        ),
        "median_diff": (
            selected["diff"].median()
        ),
        "total_diff": (
            selected["diff"].sum()
        ),
        "win_rate": (
            selected["win"].mean()
            * 100
        ),
        "plus1000_rate": (
            selected["plus1000"].mean()
            * 100
        ),
        "plus2000_rate": (
            selected["plus2000"].mean()
            * 100
        ),
        "positive_day": (
            selected["diff"].mean()
            > 0
        ),
    }


def main():
    args = parse_args()

    if args.min_history_days < 1:
        raise ValueError(
            "--min-history-days must be >= 1"
        )

    if args.topn < 1:
        raise ValueError(
            "--topn must be >= 1"
        )

    source = find_integrated_file()
    data = load_data(source)

    dates = sorted(
        data["date"].drop_duplicates()
    )

    if len(dates) <= args.min_history_days:
        raise RuntimeError(
            "Not enough dates for walk-forward backtest."
        )

    eval_dates = dates[
        args.min_history_days:
    ]

    header(
        "Big March Takasaki Oyagi - "
        "NON_JUGGLER Baseline Walk-Forward Backtest"
    )

    print(
        f"source                : {source}"
    )
    print(
        f"records               : {len(data):,}"
    )
    print(
        f"days                  : {len(dates)}"
    )
    print(
        f"date range            : "
        f"{dates[0].date()} to "
        f"{dates[-1].date()}"
    )
    print(
        f"juggler keyword       : {JUGGLER_KEYWORD}"
    )
    print(
        "segment               : NON_JUGGLER only"
    )
    print(
        f"min history days      : {args.min_history_days}"
    )
    print(
        f"evaluation days       : {len(eval_dates)}"
    )
    print(
        f"evaluation range      : "
        f"{eval_dates[0].date()} to "
        f"{eval_dates[-1].date()}"
    )
    print(
        f"top N                 : {args.topn}"
    )

    daily_counts = (
        data
        .groupby("date")
        .size()
        .rename("nonjuggler_machine_count")
    )

    header(
        "DAILY NON_JUGGLER MACHINE COUNTS"
    )

    print(
        daily_counts.to_string()
    )

    model_columns = {
        "AVG_HISTORY": (
            "avg_history"
        ),
        "RECENT7_AVG": (
            "recent7_avg"
        ),
        "RECENT7_WIN": (
            "recent7_win"
        ),
        "WEEKDAY_AVG": (
            "weekday_avg"
        ),
        "PLUS1000_RATE": (
            "plus1000_rate"
        ),
        "SIMPLE_BLEND": (
            "simple_blend"
        ),
    }

    daily_rows = []
    pick_rows = []

    for target_date in eval_dates:
        history = data[
            data["date"]
            < target_date
        ].copy()

        target_panel = data[
            data["date"]
            == target_date
        ].copy()

        if target_panel.empty:
            continue

        features = build_features(
            history,
            target_panel,
        )

        header(
            f"TARGET {target_date.date()}"
        )

        print(
            f"history rows          : {len(history):,}"
        )
        print(
            f"target machines       : {len(target_panel):,}"
        )

        for (
            model_name,
            score_column,
        ) in model_columns.items():

            ranked = (
                features
                .sort_values(
                    [
                        score_column,
                        "machine_no",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .head(
                    args.topn
                )
                .copy()
            )

            ranked[
                "prediction_rank"
            ] = range(
                1,
                len(ranked) + 1,
            )

            ranked[
                "model"
            ] = model_name

            ranked[
                "score"
            ] = ranked[
                score_column
            ]

            result = evaluate_selection(
                ranked,
                target_date,
                model_name,
            )

            daily_rows.append(
                result
            )

            for _, row in ranked.iterrows():
                pick_rows.append(
                    {
                        "target_date": (
                            target_date.date()
                        ),
                        "model": (
                            model_name
                        ),
                        "prediction_rank": int(
                            row[
                                "prediction_rank"
                            ]
                        ),
                        "machine_no": int(
                            row[
                                "machine_no"
                            ]
                        ),
                        "machine_name": (
                            row[
                                "machine_name"
                            ]
                        ),
                        "score": float(
                            row[
                                "score"
                            ]
                        ),
                        "actual_diff": float(
                            row[
                                "diff"
                            ]
                        ),
                        "actual_win": int(
                            row[
                                "win"
                            ]
                        ),
                        "actual_plus1000": int(
                            row[
                                "plus1000"
                            ]
                        ),
                        "actual_plus2000": int(
                            row[
                                "plus2000"
                            ]
                        ),
                        "avg_history": float(
                            row[
                                "avg_history"
                            ]
                        ),
                        "recent7_avg": float(
                            row[
                                "recent7_avg"
                            ]
                        ),
                        "recent7_win": float(
                            row[
                                "recent7_win"
                            ]
                        ),
                        "weekday_avg": float(
                            row[
                                "weekday_avg"
                            ]
                        ),
                        "plus1000_rate": float(
                            row[
                                "plus1000_rate"
                            ]
                        ),
                        "history_n": int(
                            row[
                                "history_n"
                            ]
                        ),
                        "recent7_n": int(
                            row[
                                "recent7_n"
                            ]
                        ),
                    }
                )

            print(
                f"{model_name:<16} "
                f"avg={result['avg_diff']:>9.1f} "
                f"win={result['win_rate']:>5.1f}% "
                f"+1000={result['plus1000_rate']:>5.1f}% "
                f"+2000={result['plus2000_rate']:>5.1f}% "
                f"total={result['total_diff']:>10.0f}"
            )

    daily = pd.DataFrame(
        daily_rows
    )

    picks = pd.DataFrame(
        pick_rows
    )

    if daily.empty:
        raise RuntimeError(
            "No backtest results were generated."
        )

    overall_rows = []

    for model_name, group in daily.groupby(
        "model"
    ):
        ci_low, ci_high = (
            bootstrap_daily_mean_ci(
                group["avg_diff"]
            )
        )

        overall_rows.append(
            {
                "model": (
                    model_name
                ),
                "evaluated_days": (
                    group[
                        "target_date"
                    ].nunique()
                ),
                "mean_daily_avg_diff": (
                    group[
                        "avg_diff"
                    ].mean()
                ),
                "median_daily_avg_diff": (
                    group[
                        "avg_diff"
                    ].median()
                ),
                "mean_win_rate": (
                    group[
                        "win_rate"
                    ].mean()
                ),
                "mean_plus1000_rate": (
                    group[
                        "plus1000_rate"
                    ].mean()
                ),
                "mean_plus2000_rate": (
                    group[
                        "plus2000_rate"
                    ].mean()
                ),
                "positive_day_rate": (
                    group[
                        "positive_day"
                    ].mean()
                    * 100
                ),
                "total_diff": (
                    group[
                        "total_diff"
                    ].sum()
                ),
                "daily_mean_ci95_low": (
                    ci_low
                ),
                "daily_mean_ci95_high": (
                    ci_high
                ),
            }
        )

    overall = (
        pd.DataFrame(
            overall_rows
        )
        .sort_values(
            [
                "mean_daily_avg_diff",
                "total_diff",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    header(
        "OVERALL NON_JUGGLER MODEL COMPARISON"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    best_model = (
        overall.iloc[0]["model"]
    )

    header(
        "BEST MODEL"
    )

    print(
        overall.iloc[
            [0]
        ].to_string(
            index=False
        )
    )

    best_picks = picks[
        picks["model"]
        == best_model
    ].copy()

    rank_diagnostics = (
        best_picks
        .groupby(
            "prediction_rank"
        )
        .agg(
            n=(
                "actual_diff",
                "size",
            ),
            avg_actual_diff=(
                "actual_diff",
                "mean",
            ),
            median_actual_diff=(
                "actual_diff",
                "median",
            ),
            total_actual_diff=(
                "actual_diff",
                "sum",
            ),
            win_rate=(
                "actual_win",
                "mean",
            ),
            plus1000_rate=(
                "actual_plus1000",
                "mean",
            ),
            plus2000_rate=(
                "actual_plus2000",
                "mean",
            ),
        )
        .reset_index()
    )

    for column in (
        "win_rate",
        "plus1000_rate",
        "plus2000_rate",
    ):
        rank_diagnostics[
            column
        ] *= 100

    rank_diagnostics[
        "model"
    ] = best_model

    header(
        "RANK DIAGNOSTICS FOR BEST MODEL"
    )

    print(
        rank_diagnostics.to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_path = (
        OUTPUT_DIR
        / "10_nonjuggler_baseline_daily.csv"
    )

    picks_path = (
        OUTPUT_DIR
        / "10_nonjuggler_baseline_picks.csv"
    )

    overall_path = (
        OUTPUT_DIR
        / "10_nonjuggler_baseline_overall.csv"
    )

    rank_path = (
        OUTPUT_DIR
        / "10_nonjuggler_baseline_best_rank.csv"
    )

    counts_path = (
        OUTPUT_DIR
        / "10_nonjuggler_daily_machine_counts.csv"
    )

    daily.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    picks.to_csv(
        picks_path,
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    rank_diagnostics.to_csv(
        rank_path,
        index=False,
        encoding="utf-8-sig",
    )

    daily_counts.to_csv(
        counts_path,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    for path in (
        daily_path,
        picks_path,
        overall_path,
        rank_path,
        counts_path,
    ):
        print(
            path
        )

    print()
    print(
        "NON_JUGGLER baseline backtest complete."
    )
    print(
        "Research only. No production model was changed."
    )
    print(
        "No Maruhan Maebashi files were modified."
    )


if __name__ == "__main__":
    main()