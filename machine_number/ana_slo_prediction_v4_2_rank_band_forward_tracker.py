from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 71 - V4.2 Live Prediction Rank-Band Forward Tracker
# ============================================================
#
# Purpose
# -------
# Track live, pre-saved prediction performance by rank band.
#
# Input:
#   69_live_prediction_detail.csv
#
# Rank bands:
#   TOP1
#   TOP3
#   TOP5
#   TOP10
#   PRIMARY_1_5
#   NEXT_6_10
#   RANK_7_9
#
# Safety:
# - does NOT recalculate predictions
# - does NOT change V4.2_C weights
# - does NOT modify 63 / 64 / 69 / 70 outputs
# - evaluates only rows already matched to actual results by 69
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
)

INPUT_FILE = (
    ANALYSIS_DIR
    / "69_Ver4_2_live_prediction_backtest"
    / "69_live_prediction_detail.csv"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "71_Ver4_2_rank_band_forward_tracker"
)

EXPECTED_ROWS_PER_DAY = 10
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260825

RANK_BANDS = {
    "TOP1": (1, 1),
    "TOP3": (1, 3),
    "TOP5": (1, 5),
    "TOP10": (1, 10),
    "PRIMARY_1_5": (1, 5),
    "NEXT_6_10": (6, 10),
    "RANK_7_9": (7, 9),
}


def header(title: str) -> None:
    print()
    print("=" * 116)
    print(title)
    print("=" * 116)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None

    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CSV read failed: {path}\nlast_error={last_error}"
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    seed_offset: int = 0,
) -> tuple[float, float]:

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(
        BOOTSTRAP_SEED + seed_offset
    )

    samples = rng.choice(
        x,
        size=(BOOTSTRAP_REPS, len(x)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> float:

    tmp = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(tmp) < 5:
        return np.nan

    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan

    try:
        return float(
            tmp[["x", "y"]]
            .corr(method="spearman")
            .iloc[0, 1]
        )
    except Exception:
        return np.nan


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:

    required = [
        "machine_no",
        "machine_name",
        "score",
        "prediction_rank",
        "tier",
        "target_date",
        "latest_data_date",
        "actual_diff",
        "actual_win",
        "actual_plus1000",
        "actual_plus2000",
        "machine_name_match",
        "prediction_file",
        "prediction_sha256",
    ]

    missing = [c for c in required if c not in raw.columns]

    if missing:
        raise ValueError(
            f"Required columns missing from 69 detail file: {missing}"
        )

    df = raw.copy()

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="coerce",
    )

    df["latest_data_date"] = pd.to_datetime(
        df["latest_data_date"],
        errors="coerce",
    )

    for col in (
        "machine_no",
        "score",
        "prediction_rank",
        "actual_diff",
        "actual_win",
        "actual_plus1000",
        "actual_plus2000",
    ):
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "target_date",
            "latest_data_date",
            "machine_no",
            "prediction_rank",
            "actual_diff",
        ]
    ).copy()

    df["machine_no"] = df["machine_no"].astype(int)
    df["prediction_rank"] = df["prediction_rank"].astype(int)

    df["machine_name"] = (
        df["machine_name"]
        .astype(str)
        .str.strip()
    )

    df["tier"] = (
        df["tier"]
        .astype(str)
        .str.strip()
    )

    df["actual_win_calc"] = (
        df["actual_diff"] > 0
    ).astype(int)

    df["actual_plus1000_calc"] = (
        df["actual_diff"] >= 1000
    ).astype(int)

    df["actual_plus2000_calc"] = (
        df["actual_diff"] >= 2000
    ).astype(int)

    return (
        df.sort_values(
            ["target_date", "prediction_rank"]
        )
        .reset_index(drop=True)
    )


def build_quality_table(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for target_date, group in df.groupby(
        "target_date",
        sort=True,
    ):

        ranks = sorted(
            group["prediction_rank"].astype(int).tolist()
        )

        rows.append(
            {
                "target_date": target_date,
                "rows": int(len(group)),
                "unique_machines": int(
                    group["machine_no"].nunique()
                ),
                "unique_ranks": int(
                    group["prediction_rank"].nunique()
                ),
                "duplicate_machines": int(
                    group.duplicated(
                        subset=["machine_no"]
                    ).sum()
                ),
                "duplicate_ranks": int(
                    group.duplicated(
                        subset=["prediction_rank"]
                    ).sum()
                ),
                "rank_1_to_10_complete":
                    ranks == list(range(1, 11)),
                "machine_name_match_all": bool(
                    group["machine_name_match"]
                    .astype(bool)
                    .all()
                ),
                "latest_before_target_all": bool(
                    (
                        group["latest_data_date"]
                        < group["target_date"]
                    ).all()
                ),
                "actual_diff_missing": int(
                    group["actual_diff"].isna().sum()
                ),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["eligible"] = (
        (out["rows"] == EXPECTED_ROWS_PER_DAY)
        & (out["unique_machines"] == EXPECTED_ROWS_PER_DAY)
        & (out["unique_ranks"] == EXPECTED_ROWS_PER_DAY)
        & (out["duplicate_machines"] == 0)
        & (out["duplicate_ranks"] == 0)
        & out["rank_1_to_10_complete"]
        & out["machine_name_match_all"]
        & out["latest_before_target_all"]
        & (out["actual_diff_missing"] == 0)
    )

    return out


def summarize_band(
    group: pd.DataFrame,
    band_name: str,
    rank_start: int,
    rank_end: int,
) -> dict:

    selected = group[
        group["prediction_rank"].between(
            rank_start,
            rank_end,
        )
    ].copy()

    diffs = selected["actual_diff"].astype(float)

    return {
        "band": band_name,
        "rank_start": rank_start,
        "rank_end": rank_end,
        "selected_n": int(len(selected)),
        "avg_diff": float(diffs.mean()),
        "median_diff": float(diffs.median()),
        "total_diff": float(diffs.sum()),
        "win_rate": float(
            (diffs > 0).mean() * 100.0
        ),
        "plus1000_rate": float(
            (diffs >= 1000).mean() * 100.0
        ),
        "plus2000_rate": float(
            (diffs >= 2000).mean() * 100.0
        ),
        "min_diff": float(diffs.min()),
        "max_diff": float(diffs.max()),
    }


def build_daily_band_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for target_date, group in df.groupby(
        "target_date",
        sort=True,
    ):

        score_spearman = safe_spearman(
            group["score"],
            group["actual_diff"],
        )

        rank_spearman = safe_spearman(
            group["prediction_rank"],
            group["actual_diff"],
        )

        for band_name, (
            rank_start,
            rank_end,
        ) in RANK_BANDS.items():

            row = summarize_band(
                group,
                band_name,
                rank_start,
                rank_end,
            )

            row["target_date"] = target_date
            row["score_spearman"] = score_spearman
            row["rank_spearman"] = rank_spearman

            rows.append(row)

    out = pd.DataFrame(rows)

    return out[
        [
            "target_date",
            "band",
            "rank_start",
            "rank_end",
            "selected_n",
            "avg_diff",
            "median_diff",
            "total_diff",
            "win_rate",
            "plus1000_rate",
            "plus2000_rate",
            "min_diff",
            "max_diff",
            "score_spearman",
            "rank_spearman",
        ]
    ]


def build_overall_band_table(
    daily: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for band_name, (
        rank_start,
        rank_end,
    ) in RANK_BANDS.items():

        selected = detail[
            detail["prediction_rank"].between(
                rank_start,
                rank_end,
            )
        ].copy()

        band_daily = daily[
            daily["band"] == band_name
        ].copy()

        diffs = selected["actual_diff"].astype(float)

        row_ci_low, row_ci_high = bootstrap_mean_ci(
            diffs.to_numpy(),
            seed_offset=rank_start * 100 + rank_end,
        )

        daily_avg_ci_low, daily_avg_ci_high = bootstrap_mean_ci(
            band_daily["avg_diff"].to_numpy(),
            seed_offset=rank_start * 1000 + rank_end,
        )

        rows.append(
            {
                "band": band_name,
                "rank_start": rank_start,
                "rank_end": rank_end,
                "evaluated_days": int(
                    band_daily["target_date"].nunique()
                ),
                "selected_rows": int(len(selected)),
                "avg_diff_per_machine": float(
                    diffs.mean()
                ),
                "median_diff_per_machine": float(
                    diffs.median()
                ),
                "total_diff": float(
                    diffs.sum()
                ),
                "win_rate": float(
                    (diffs > 0).mean() * 100.0
                ),
                "plus1000_rate": float(
                    (diffs >= 1000).mean() * 100.0
                ),
                "plus2000_rate": float(
                    (diffs >= 2000).mean() * 100.0
                ),
                "positive_day_rate": float(
                    (
                        band_daily["total_diff"] > 0
                    ).mean()
                    * 100.0
                ),
                "mean_daily_avg_diff": float(
                    band_daily["avg_diff"].mean()
                ),
                "median_daily_avg_diff": float(
                    band_daily["avg_diff"].median()
                ),
                "machine_avg_diff_ci95_low": row_ci_low,
                "machine_avg_diff_ci95_high": row_ci_high,
                "daily_avg_diff_ci95_low": daily_avg_ci_low,
                "daily_avg_diff_ci95_high": daily_avg_ci_high,
            }
        )

    return pd.DataFrame(rows)


def build_rank_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for rank, group in df.groupby(
        "prediction_rank",
        sort=True,
    ):

        diffs = group["actual_diff"].astype(float)

        rows.append(
            {
                "prediction_rank": int(rank),
                "n": int(len(group)),
                "avg_actual_diff": float(
                    diffs.mean()
                ),
                "median_actual_diff": float(
                    diffs.median()
                ),
                "total_actual_diff": float(
                    diffs.sum()
                ),
                "win_rate": float(
                    (diffs > 0).mean() * 100.0
                ),
                "plus1000_rate": float(
                    (diffs >= 1000).mean() * 100.0
                ),
                "plus2000_rate": float(
                    (diffs >= 2000).mean() * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_daily_rank_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:

    matrix = df.pivot(
        index="prediction_rank",
        columns="target_date",
        values="actual_diff",
    )

    matrix = matrix.sort_index()

    matrix.columns = [
        pd.Timestamp(c).strftime("%Y-%m-%d")
        for c in matrix.columns
    ]

    matrix = matrix.reset_index()

    date_cols = [
        c
        for c in matrix.columns
        if c != "prediction_rank"
    ]

    matrix["total_diff"] = matrix[
        date_cols
    ].sum(axis=1)

    matrix["avg_diff"] = matrix[
        date_cols
    ].mean(axis=1)

    matrix["win_days"] = (
        matrix[date_cols] > 0
    ).sum(axis=1)

    matrix["loss_days"] = (
        matrix[date_cols] <= 0
    ).sum(axis=1)

    return matrix


def build_daily_order_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for target_date, group in df.groupby(
        "target_date",
        sort=True,
    ):

        rows.append(
            {
                "target_date": target_date,
                "n": int(len(group)),
                "score_spearman_vs_actual_diff":
                    safe_spearman(
                        group["score"],
                        group["actual_diff"],
                    ),
                "rank_spearman_vs_actual_diff":
                    safe_spearman(
                        group["prediction_rank"],
                        group["actual_diff"],
                    ),
                "top10_avg_diff": float(
                    group["actual_diff"].mean()
                ),
                "top10_total_diff": float(
                    group["actual_diff"].sum()
                ),
                "top10_win_rate": float(
                    (
                        group["actual_diff"] > 0
                    ).mean()
                    * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_checkpoint_table(
    overall: pd.DataFrame,
    total_days: int,
) -> pd.DataFrame:

    rows = []

    for row in overall.itertuples(
        index=False
    ):

        if total_days < 21:
            status = "ACCUMULATING"
            decision = (
                "No model/rank-band change. "
                "Continue live forward tracking."
            )
        else:
            status = "FIRST_REVIEW_READY"
            decision = (
                "Review persistence, confidence intervals, "
                "store-relative performance, and rank ordering "
                "before any challenger change."
            )

        rows.append(
            {
                "band": row.band,
                "evaluated_days": int(
                    row.evaluated_days
                ),
                "selected_rows": int(
                    row.selected_rows
                ),
                "avg_diff_per_machine": float(
                    row.avg_diff_per_machine
                ),
                "total_diff": float(
                    row.total_diff
                ),
                "win_rate": float(
                    row.win_rate
                ),
                "plus2000_rate": float(
                    row.plus2000_rate
                ),
                "positive_day_rate": float(
                    row.positive_day_rate
                ),
                "first_review_target_days": 21,
                "days_remaining_to_first_review":
                    max(0, 21 - total_days),
                "status": status,
                "decision_rule": decision,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:

    header(
        "71 - V4.2 Live Prediction Rank-Band Forward Tracker"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"69 detail file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = read_csv_flexible(
        INPUT_FILE
    )

    df = prepare_data(
        raw
    )

    print(f"input file            : {INPUT_FILE}")
    print(f"rows                  : {len(df)}")
    print(f"evaluated days        : {df['target_date'].nunique()}")
    print(
        "date range            : "
        f"{df['target_date'].min().date()} "
        "to "
        f"{df['target_date'].max().date()}"
    )

    quality = build_quality_table(
        df
    )

    header("DATA QUALITY")
    print(
        quality.to_string(
            index=False
        )
    )

    if quality.empty or not quality["eligible"].all():
        raise RuntimeError(
            "71 tracker aborted because "
            "69 detail quality check failed."
        )

    daily = build_daily_band_table(
        df
    )

    overall = build_overall_band_table(
        daily,
        df,
    )

    rank = build_rank_table(
        df
    )

    rank_matrix = build_daily_rank_matrix(
        df
    )

    daily_order = build_daily_order_table(
        df
    )

    total_days = int(
        df["target_date"].nunique()
    )

    checkpoint = build_checkpoint_table(
        overall,
        total_days,
    )

    header("OVERALL RANK-BAND RESULTS")
    print(
        overall[
            [
                "band",
                "evaluated_days",
                "selected_rows",
                "avg_diff_per_machine",
                "total_diff",
                "win_rate",
                "plus1000_rate",
                "plus2000_rate",
                "positive_day_rate",
                "machine_avg_diff_ci95_low",
                "machine_avg_diff_ci95_high",
            ]
        ].to_string(
            index=False
        )
    )

    header("DAILY RANK-BAND RESULTS")
    print(
        daily[
            [
                "target_date",
                "band",
                "selected_n",
                "avg_diff",
                "total_diff",
                "win_rate",
                "plus2000_rate",
            ]
        ].to_string(
            index=False
        )
    )

    header("DAILY SCORE / RANK ORDER")
    print(
        daily_order.to_string(
            index=False
        )
    )

    header("RANK-BY-RANK MATRIX")
    print(
        rank_matrix.to_string(
            index=False
        )
    )

    header("21-DAY CHECKPOINT")
    print(
        checkpoint[
            [
                "band",
                "evaluated_days",
                "days_remaining_to_first_review",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    paths = {
        "quality":
            OUTPUT_DIR / "71_data_quality.csv",
        "daily_bands":
            OUTPUT_DIR / "71_rank_band_daily.csv",
        "overall_bands":
            OUTPUT_DIR / "71_rank_band_overall.csv",
        "rank_summary":
            OUTPUT_DIR / "71_rank_summary.csv",
        "rank_matrix":
            OUTPUT_DIR / "71_daily_rank_matrix.csv",
        "daily_order":
            OUTPUT_DIR / "71_daily_score_rank_order.csv",
        "checkpoint":
            OUTPUT_DIR / "71_21day_checkpoint.csv",
    }

    quality.to_csv(
        paths["quality"],
        index=False,
        encoding="utf-8-sig",
    )

    daily.to_csv(
        paths["daily_bands"],
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        paths["overall_bands"],
        index=False,
        encoding="utf-8-sig",
    )

    rank.to_csv(
        paths["rank_summary"],
        index=False,
        encoding="utf-8-sig",
    )

    rank_matrix.to_csv(
        paths["rank_matrix"],
        index=False,
        encoding="utf-8-sig",
    )

    daily_order.to_csv(
        paths["daily_order"],
        index=False,
        encoding="utf-8-sig",
    )

    checkpoint.to_csv(
        paths["checkpoint"],
        index=False,
        encoding="utf-8-sig",
    )

    header("FILES SAVED")

    for path in paths.values():
        print(path)

    print()
    print(
        "71 rank-band forward tracker complete."
    )
    print(
        "No predictions or model weights were changed."
    )
    print(
        f"First formal review target: 21 evaluated days "
        f"({max(0, 21 - total_days)} days remaining)."
    )


if __name__ == "__main__":
    main()
