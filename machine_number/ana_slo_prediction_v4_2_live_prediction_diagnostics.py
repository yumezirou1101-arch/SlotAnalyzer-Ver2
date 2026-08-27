from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 70 - V4.2 Live Prediction Diagnostics
# ============================================================
#
# Purpose
# -------
# Diagnose why the already-saved live predictions performed
# well or poorly.
#
# Input:
#   69_live_prediction_detail.csv
#
# Safety:
# - no ranking recalculation
# - no weight changes
# - no modification of 63 / 64 / 69 outputs
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

ANALYSIS_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)

INPUT_FILE = (
    ANALYSIS_DIR
    / "69_Ver4_2_live_prediction_backtest"
    / "69_live_prediction_detail.csv"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "70_Ver4_2_live_prediction_diagnostics"
)

EXPECTED_ROWS_PER_DAY = 10
MIN_ROWS_FOR_CORRELATION = 5

FEATURE_COLUMNS = [
    "avg31",
    "recent7_avg",
    "recent7_win",
    "last_diff",
    "prev_change",
    "weekday_avg",
    "type_avg",
    "plus1000_rate",
    "plus2000_rate",
    "neighbor_avg",
    "bounce_signal",
    "machine_history",
    "number_band_10",
    "number_band_50",
    "number_run_edge",
    "score",
    "prediction_rank",
]

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260825


def header(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


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


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
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


def safe_corr(
    x: pd.Series,
    y: pd.Series,
    method: str,
) -> tuple[float, int]:

    tmp = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    n = int(len(tmp))

    if n < MIN_ROWS_FOR_CORRELATION:
        return (np.nan, n)

    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return (np.nan, n)

    try:
        corr = (
            tmp[["x", "y"]]
            .corr(method=method)
            .iloc[0, 1]
        )
        return (float(corr), n)
    except Exception:
        return (np.nan, n)


def safe_float(value) -> float:
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return np.nan


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:

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

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from 69 detail file: {missing}"
        )

    x = df.copy()

    x["target_date"] = pd.to_datetime(
        x["target_date"],
        errors="coerce",
    )
    x["latest_data_date"] = pd.to_datetime(
        x["latest_data_date"],
        errors="coerce",
    )

    numeric_cols = [
        "machine_no",
        "score",
        "prediction_rank",
        "actual_diff",
        "actual_win",
        "actual_plus1000",
        "actual_plus2000",
        *[
            c
            for c in FEATURE_COLUMNS
            if c not in ("score", "prediction_rank")
        ],
    ]

    for col in numeric_cols:
        if col in x.columns:
            x[col] = pd.to_numeric(
                x[col],
                errors="coerce",
            )

    x = x.dropna(
        subset=[
            "target_date",
            "machine_no",
            "prediction_rank",
            "actual_diff",
        ]
    ).copy()

    x["machine_no"] = x["machine_no"].astype(int)
    x["prediction_rank"] = x["prediction_rank"].astype(int)
    x["tier"] = x["tier"].astype(str).str.strip()
    x["machine_name"] = x["machine_name"].astype(str).str.strip()

    x["actual_positive"] = (x["actual_diff"] > 0).astype(int)
    x["actual_plus1000_flag"] = (x["actual_diff"] >= 1000).astype(int)
    x["actual_plus2000_flag"] = (x["actual_diff"] >= 2000).astype(int)
    x["actual_big_loss_flag"] = (x["actual_diff"] <= -3000).astype(int)
    x["actual_big_win_flag"] = (x["actual_diff"] >= 3000).astype(int)

    return (
        x.sort_values(
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
            int(v)
            for v in group["prediction_rank"].dropna().tolist()
        )

        rows.append(
            {
                "target_date": target_date,
                "rows": int(len(group)),
                "unique_machines": int(group["machine_no"].nunique()),
                "unique_ranks": int(group["prediction_rank"].nunique()),
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
                "name_match_all": bool(
                    group["machine_name_match"].astype(bool).all()
                ),
                "actual_diff_missing": int(
                    group["actual_diff"].isna().sum()
                ),
                "latest_before_target_all": bool(
                    (
                        group["latest_data_date"]
                        < group["target_date"]
                    ).all()
                ),
            }
        )

    q = pd.DataFrame(rows)

    if q.empty:
        return q

    q["eligible"] = (
        (q["rows"] == EXPECTED_ROWS_PER_DAY)
        & (q["unique_machines"] == EXPECTED_ROWS_PER_DAY)
        & (q["unique_ranks"] == EXPECTED_ROWS_PER_DAY)
        & (q["duplicate_machines"] == 0)
        & (q["duplicate_ranks"] == 0)
        & q["rank_1_to_10_complete"]
        & q["name_match_all"]
        & (q["actual_diff_missing"] == 0)
        & q["latest_before_target_all"]
    )

    return q


def build_feature_correlation_table(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for feature in FEATURE_COLUMNS:

        if feature not in df.columns:
            continue

        pearson, n_p = safe_corr(
            df[feature],
            df["actual_diff"],
            "pearson",
        )
        spearman, n_s = safe_corr(
            df[feature],
            df["actual_diff"],
            "spearman",
        )

        rows.append(
            {
                "feature": feature,
                "n": min(n_p, n_s),
                "pearson_vs_actual_diff": pearson,
                "spearman_vs_actual_diff": spearman,
                "abs_spearman":
                    abs(spearman)
                    if np.isfinite(spearman)
                    else np.nan,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = (
            out.sort_values(
                ["abs_spearman", "feature"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

    return out


def build_daily_correlation_table(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for target_date, group in df.groupby(
        "target_date",
        sort=True,
    ):

        score_pearson, n1 = safe_corr(
            group["score"],
            group["actual_diff"],
            "pearson",
        )
        score_spearman, n2 = safe_corr(
            group["score"],
            group["actual_diff"],
            "spearman",
        )
        rank_pearson, n3 = safe_corr(
            group["prediction_rank"],
            group["actual_diff"],
            "pearson",
        )
        rank_spearman, n4 = safe_corr(
            group["prediction_rank"],
            group["actual_diff"],
            "spearman",
        )

        rows.append(
            {
                "target_date": target_date,
                "n": min(n1, n2, n3, n4),
                "score_pearson": score_pearson,
                "score_spearman": score_spearman,
                "rank_pearson": rank_pearson,
                "rank_spearman": rank_spearman,
                "top10_avg_diff": float(group["actual_diff"].mean()),
                "top10_total_diff": float(group["actual_diff"].sum()),
                "top10_win_rate": float(
                    (group["actual_diff"] > 0).mean() * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


def build_binary_group_comparison(
    df: pd.DataFrame,
    flag_col: str,
    label: str,
) -> pd.DataFrame:

    rows = []

    for feature in FEATURE_COLUMNS:

        if feature not in df.columns:
            continue

        temp = df[[feature, flag_col]].copy()
        temp[feature] = pd.to_numeric(
            temp[feature],
            errors="coerce",
        )
        temp[flag_col] = pd.to_numeric(
            temp[flag_col],
            errors="coerce",
        )
        temp = temp.dropna()

        positive = temp[temp[flag_col] == 1][feature]
        negative = temp[temp[flag_col] == 0][feature]

        if positive.empty or negative.empty:
            mean_positive = np.nan
            mean_negative = np.nan
            median_positive = np.nan
            median_negative = np.nan
            mean_diff = np.nan
        else:
            mean_positive = float(positive.mean())
            mean_negative = float(negative.mean())
            median_positive = float(positive.median())
            median_negative = float(negative.median())
            mean_diff = float(
                mean_positive - mean_negative
            )

        rows.append(
            {
                "comparison": label,
                "feature": feature,
                "positive_n": int(len(positive)),
                "negative_n": int(len(negative)),
                "positive_mean": mean_positive,
                "negative_mean": mean_negative,
                "positive_minus_negative_mean": mean_diff,
                "positive_median": median_positive,
                "negative_median": median_negative,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out["abs_mean_difference"] = (
            out["positive_minus_negative_mean"].abs()
        )
        out = (
            out.sort_values(
                ["abs_mean_difference", "feature"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

    return out


def build_high_low_split_table(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for feature in FEATURE_COLUMNS:

        if feature not in df.columns:
            continue

        temp = df[[feature, "actual_diff"]].copy()
        temp[feature] = pd.to_numeric(
            temp[feature],
            errors="coerce",
        )
        temp["actual_diff"] = pd.to_numeric(
            temp["actual_diff"],
            errors="coerce",
        )
        temp = temp.dropna()

        if len(temp) < 6 or temp[feature].nunique() < 2:
            continue

        median_value = float(temp[feature].median())

        low = temp[
            temp[feature] <= median_value
        ]["actual_diff"]

        high = temp[
            temp[feature] > median_value
        ]["actual_diff"]

        if low.empty or high.empty:
            continue

        low_mean = float(low.mean())
        high_mean = float(high.mean())

        rows.append(
            {
                "feature": feature,
                "split_median": median_value,
                "low_n": int(len(low)),
                "high_n": int(len(high)),
                "low_avg_actual_diff": low_mean,
                "high_avg_actual_diff": high_mean,
                "high_minus_low_avg_diff":
                    float(high_mean - low_mean),
                "low_win_rate":
                    float((low > 0).mean() * 100.0),
                "high_win_rate":
                    float((high > 0).mean() * 100.0),
                "low_plus2000_rate":
                    float((low >= 2000).mean() * 100.0),
                "high_plus2000_rate":
                    float((high >= 2000).mean() * 100.0),
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out["abs_high_low_diff"] = (
            out["high_minus_low_avg_diff"].abs()
        )
        out = (
            out.sort_values(
                ["abs_high_low_diff", "feature"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

    return out


def build_tier_summary(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for tier, group in df.groupby(
        "tier",
        sort=True,
    ):

        diffs = pd.to_numeric(
            group["actual_diff"],
            errors="coerce",
        ).dropna()

        ci_low, ci_high = bootstrap_mean_ci(
            diffs.to_numpy()
        )

        rows.append(
            {
                "tier": tier,
                "rows": int(len(diffs)),
                "days": int(group["target_date"].nunique()),
                "avg_diff": float(diffs.mean()),
                "median_diff": float(diffs.median()),
                "total_diff": float(diffs.sum()),
                "win_rate":
                    float((diffs > 0).mean() * 100.0),
                "plus1000_rate":
                    float((diffs >= 1000).mean() * 100.0),
                "plus2000_rate":
                    float((diffs >= 2000).mean() * 100.0),
                "avg_diff_ci95_low": ci_low,
                "avg_diff_ci95_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


def build_rank_summary(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for rank, group in df.groupby(
        "prediction_rank",
        sort=True,
    ):

        diffs = pd.to_numeric(
            group["actual_diff"],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "prediction_rank": int(rank),
                "n": int(len(diffs)),
                "avg_actual_diff": float(diffs.mean()),
                "median_actual_diff": float(diffs.median()),
                "total_actual_diff": float(diffs.sum()),
                "win_rate":
                    float((diffs > 0).mean() * 100.0),
                "plus2000_rate":
                    float((diffs >= 2000).mean() * 100.0),
            }
        )

    return pd.DataFrame(rows)


def build_extreme_table(df: pd.DataFrame) -> pd.DataFrame:

    keep_cols = [
        "target_date",
        "prediction_rank",
        "tier",
        "machine_no",
        "machine_name",
        "score",
        "actual_diff",
        "avg31",
        "recent7_avg",
        "recent7_win",
        "last_diff",
        "prev_change",
        "weekday_avg",
        "type_avg",
        "plus1000_rate",
        "plus2000_rate",
        "neighbor_avg",
        "bounce_signal",
        "machine_history",
        "number_band_10",
        "number_band_50",
    ]

    keep_cols = [
        c
        for c in keep_cols
        if c in df.columns
    ]

    return (
        df[keep_cols]
        .sort_values(
            ["actual_diff", "target_date", "prediction_rank"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )


def build_diagnostic_flags(
    corr_df: pd.DataFrame,
    high_low_df: pd.DataFrame,
    tier_df: pd.DataFrame,
    daily_corr_df: pd.DataFrame,
    total_rows: int,
    total_days: int,
) -> pd.DataFrame:

    rows = [
        {
            "category": "SAMPLE_SIZE",
            "finding":
                f"{total_rows} rows / {total_days} days only",
            "severity": "WARNING",
            "interpretation":
                "Too early for model-weight changes. "
                "Use findings only as hypotheses.",
        }
    ]

    if not corr_df.empty:

        score_row = corr_df[
            corr_df["feature"] == "score"
        ]

        if not score_row.empty:

            score_spear = safe_float(
                score_row.iloc[0][
                    "spearman_vs_actual_diff"
                ]
            )

            if np.isfinite(score_spear):

                if score_spear <= -0.20:
                    rows.append(
                        {
                            "category": "SCORE_ORDER",
                            "finding":
                                "score has negative Spearman "
                                f"correlation ({score_spear:.3f})",
                            "severity": "WATCH",
                            "interpretation":
                                "Higher score may not currently "
                                "correspond to better next-day diff.",
                        }
                    )

                elif score_spear >= 0.20:
                    rows.append(
                        {
                            "category": "SCORE_ORDER",
                            "finding":
                                "score has positive Spearman "
                                f"correlation ({score_spear:.3f})",
                            "severity": "POSITIVE_SIGNAL",
                            "interpretation":
                                "Score ordering shows some directional "
                                "consistency, but sample is small.",
                        }
                    )

    if not tier_df.empty:

        primary = tier_df[
            tier_df["tier"] == "PRIMARY"
        ]
        next_ = tier_df[
            tier_df["tier"] == "NEXT"
        ]

        if not primary.empty and not next_.empty:

            p_avg = float(
                primary.iloc[0]["avg_diff"]
            )
            n_avg = float(
                next_.iloc[0]["avg_diff"]
            )

            if p_avg < n_avg:
                rows.append(
                    {
                        "category": "PRIMARY_VS_NEXT",
                        "finding":
                            f"PRIMARY avg {p_avg:.1f} "
                            f"< NEXT avg {n_avg:.1f}",
                        "severity": "WATCH",
                        "interpretation":
                            "Ranks 1-5 are not outperforming "
                            "ranks 6-10 in the current sample.",
                    }
                )

    if not high_low_df.empty:

        suspicious = (
            high_low_df[
                high_low_df[
                    "high_minus_low_avg_diff"
                ] < -1000
            ]
            .head(5)
        )

        for row in suspicious.itertuples(
            index=False
        ):
            rows.append(
                {
                    "category": "FEATURE_HIGH_LOW",
                    "finding":
                        f"{row.feature}: high group "
                        f"underperforms low group by "
                        f"{abs(row.high_minus_low_avg_diff):.1f} coins",
                    "severity": "HYPOTHESIS",
                    "interpretation":
                        "Possible over-weighting or mean-reversion "
                        "candidate. Do not change weights yet.",
                }
            )

    if not daily_corr_df.empty:

        neg_days = int(
            (
                daily_corr_df["score_spearman"] < 0
            ).sum()
        )

        valid_days = int(
            daily_corr_df[
                "score_spearman"
            ].notna().sum()
        )

        if valid_days > 0 and neg_days > valid_days / 2:
            rows.append(
                {
                    "category": "DAILY_SCORE_ORDER",
                    "finding":
                        "negative score-order correlation "
                        f"on {neg_days}/{valid_days} days",
                    "severity": "WATCH",
                    "interpretation":
                        "Score ordering may be unstable day to day.",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:

    header(
        "70 - V4.2 Live Prediction Diagnostics"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"69 detail file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = read_csv_flexible(INPUT_FILE)
    df = prepare_data(raw)

    print(f"input file            : {INPUT_FILE}")
    print(f"rows                  : {len(df)}")
    print(f"days                  : {df['target_date'].nunique()}")
    print(
        "date range            : "
        f"{df['target_date'].min().date()} "
        "to "
        f"{df['target_date'].max().date()}"
    )

    quality_df = build_quality_table(df)

    header("DATA QUALITY")
    print(quality_df.to_string(index=False))

    if quality_df.empty or not quality_df["eligible"].all():
        raise RuntimeError(
            "70 diagnostics aborted because "
            "69 detail quality check failed."
        )

    corr_df = build_feature_correlation_table(df)
    daily_corr_df = build_daily_correlation_table(df)

    positive_compare_df = build_binary_group_comparison(
        df,
        "actual_positive",
        "WIN_VS_LOSS",
    )

    plus1000_compare_df = build_binary_group_comparison(
        df,
        "actual_plus1000_flag",
        "PLUS1000_VS_OTHER",
    )

    plus2000_compare_df = build_binary_group_comparison(
        df,
        "actual_plus2000_flag",
        "PLUS2000_VS_OTHER",
    )

    high_low_df = build_high_low_split_table(df)
    tier_df = build_tier_summary(df)
    rank_df = build_rank_summary(df)
    extremes_df = build_extreme_table(df)

    flags_df = build_diagnostic_flags(
        corr_df,
        high_low_df,
        tier_df,
        daily_corr_df,
        total_rows=int(len(df)),
        total_days=int(
            df["target_date"].nunique()
        ),
    )

    header("SCORE / RANK CORRELATION")
    print(
        corr_df[
            corr_df["feature"].isin(
                ["score", "prediction_rank"]
            )
        ].to_string(index=False)
    )

    header("TOP FEATURE CORRELATIONS")
    print(
        corr_df.head(12).to_string(
            index=False
        )
    )

    header("DAILY SCORE / RANK CORRELATION")
    print(
        daily_corr_df.to_string(
            index=False
        )
    )

    header("PRIMARY VS NEXT")
    print(
        tier_df.to_string(
            index=False
        )
    )

    header("RANK-BY-RANK RESULTS")
    print(
        rank_df.to_string(
            index=False
        )
    )

    header("HIGH / LOW FEATURE SPLITS")
    print(
        high_low_df.head(12).to_string(
            index=False
        )
    )

    header("DIAGNOSTIC FLAGS")
    print(
        flags_df.to_string(
            index=False
        )
    )

    paths = {
        "quality":
            OUTPUT_DIR / "70_data_quality.csv",
        "correlations":
            OUTPUT_DIR / "70_feature_correlations.csv",
        "daily_correlations":
            OUTPUT_DIR / "70_daily_correlations.csv",
        "win_vs_loss":
            OUTPUT_DIR / "70_feature_win_vs_loss.csv",
        "plus1000":
            OUTPUT_DIR / "70_feature_plus1000_vs_other.csv",
        "plus2000":
            OUTPUT_DIR / "70_feature_plus2000_vs_other.csv",
        "high_low":
            OUTPUT_DIR / "70_feature_high_low_splits.csv",
        "tier":
            OUTPUT_DIR / "70_primary_vs_next.csv",
        "rank":
            OUTPUT_DIR / "70_rank_summary.csv",
        "extremes":
            OUTPUT_DIR / "70_extreme_results_review.csv",
        "flags":
            OUTPUT_DIR / "70_diagnostic_flags.csv",
    }

    quality_df.to_csv(
        paths["quality"],
        index=False,
        encoding="utf-8-sig",
    )
    corr_df.to_csv(
        paths["correlations"],
        index=False,
        encoding="utf-8-sig",
    )
    daily_corr_df.to_csv(
        paths["daily_correlations"],
        index=False,
        encoding="utf-8-sig",
    )
    positive_compare_df.to_csv(
        paths["win_vs_loss"],
        index=False,
        encoding="utf-8-sig",
    )
    plus1000_compare_df.to_csv(
        paths["plus1000"],
        index=False,
        encoding="utf-8-sig",
    )
    plus2000_compare_df.to_csv(
        paths["plus2000"],
        index=False,
        encoding="utf-8-sig",
    )
    high_low_df.to_csv(
        paths["high_low"],
        index=False,
        encoding="utf-8-sig",
    )
    tier_df.to_csv(
        paths["tier"],
        index=False,
        encoding="utf-8-sig",
    )
    rank_df.to_csv(
        paths["rank"],
        index=False,
        encoding="utf-8-sig",
    )
    extremes_df.to_csv(
        paths["extremes"],
        index=False,
        encoding="utf-8-sig",
    )
    flags_df.to_csv(
        paths["flags"],
        index=False,
        encoding="utf-8-sig",
    )

    header("FILES SAVED")

    for path in paths.values():
        print(path)

    print()
    print(
        "70 live prediction diagnostics complete."
    )
    print(
        "No model weights were changed."
    )
    print(
        "Treat all findings as hypotheses until "
        "more live prediction days accumulate."
    )


if __name__ == "__main__":
    main()
