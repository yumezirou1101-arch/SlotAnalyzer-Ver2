from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# Ana-Slo Ver.4.2 Candidate Stability Test
#
# Compare:
#   V4_BASE
#   V4.2_A : recent7_win excluded
#   V4.2_B : bounce_signal excluded
#   V4.2_C : recent7_win + bounce_signal excluded
#
# Split1:
#   TRAIN 2026-07-11 to 2026-07-25
#   TEST  2026-07-26 to 2026-08-02
#
# Split2:
#   TRAIN 2026-07-11 to 2026-08-02
#   TEST  2026-08-03 to 2026-08-10
#
# IMPORTANT:
# We do NOT optimize weights inside each split.
# We only remove candidate factors from the fixed Ver.4 weights
# and renormalize the remaining weights.
# ============================================================

BASE = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_DIR = (
    BASE
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)

FACTORS = [
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
]

# ------------------------------------------------------------
# Ver.4 fixed weights
# ------------------------------------------------------------

V4_WEIGHTS = {
    "avg31": 0.0670952025611345,
    "recent7_avg": 0.05164896703284082,
    "recent7_win": 0.06602967770818714,
    "last_diff": 0.12382294629381808,
    "prev_change": 0.10484738021281044,
    "weekday_avg": 0.05672674990073483,
    "type_avg": 0.05843723530102936,
    "plus1000_rate": 0.17725354845070532,
    "plus2000_rate": 0.13298938481323394,
    "neighbor_avg": 0.06161296683628432,
    "bounce_signal": 0.09953594088922124,
}

# ------------------------------------------------------------
# Split definitions
# ------------------------------------------------------------

SPLITS = [
    {
        "name": "SPLIT1",
        "train_start": pd.Timestamp("2026-07-11"),
        "train_end": pd.Timestamp("2026-07-25"),
        "test_start": pd.Timestamp("2026-07-26"),
        "test_end": pd.Timestamp("2026-08-02"),
    },
    {
        "name": "SPLIT2",
        "train_start": pd.Timestamp("2026-07-11"),
        "train_end": pd.Timestamp("2026-08-02"),
        "test_start": pd.Timestamp("2026-08-03"),
        "test_end": pd.Timestamp("2026-08-10"),
    },
]

# ------------------------------------------------------------
# Import existing feature builder
# ------------------------------------------------------------

from ana_slo_prediction_v4_oos import load_data, build_features


# ============================================================
# Model weights
# ============================================================

def make_weights(exclude_factors=None):

    weights = V4_WEIGHTS.copy()

    if exclude_factors is None:
        exclude_factors = []

    for factor in exclude_factors:
        if factor in weights:
            weights[factor] = 0.0

    total = sum(weights.values())

    if total <= 0:
        raise ValueError(
            "Weight sum became zero."
        )

    for factor in weights:
        weights[factor] /= total

    return weights


MODELS = {
    "V4_BASE": {
        "exclude": [],
    },

    "V4.2_A": {
        "exclude": [
            "recent7_win"
        ],
    },

    "V4.2_B": {
        "exclude": [
            "bounce_signal"
        ],
    },

    "V4.2_C": {
        "exclude": [
            "recent7_win",
            "bounce_signal"
        ],
    },
}


# ============================================================
# Z-score
# ============================================================

def zscore(series):

    s = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0.0)

    std = float(
        s.std(ddof=0)
    )

    if std == 0 or np.isnan(std):

        return pd.Series(
            0.0,
            index=s.index
        )

    return (
        s - s.mean()
    ) / std


# ============================================================
# Ranking
# ============================================================

def rank_score(
    df,
    weights
):

    x = df.copy()

    score = pd.Series(
        0.0,
        index=x.index
    )

    for factor in FACTORS:

        if factor not in x.columns:
            continue

        z = zscore(
            x[factor]
        )

        transformed = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100
        )

        score += (
            transformed
            * weights.get(
                factor,
                0.0
            )
        )

    x["score"] = score

    return x.sort_values(
        "score",
        ascending=False
    )


# ============================================================
# Evaluate one day
# ============================================================

def evaluate_day(
    panel,
    weights,
    top_n
):

    if panel.empty:
        return None

    ranked = rank_score(
        panel,
        weights
    )

    top = ranked.head(
        min(
            top_n,
            len(ranked)
        )
    )

    d = pd.to_numeric(
        top["diff"],
        errors="coerce"
    ).dropna()

    if d.empty:
        return None

    return {
        "avg_diff": float(
            d.mean()
        ),

        "median_diff": float(
            d.median()
        ),

        "win_rate": float(
            (d > 0).mean()
            * 100
        ),

        "plus1000_rate": float(
            (d >= 1000).mean()
            * 100
        ),

        "plus2000_rate": float(
            (d >= 2000).mean()
            * 100
        ),

        "positive": int(
            d.sum() > 0
        ),

        "total_diff": float(
            d.sum()
        ),
    }


# ============================================================
# Aggregate daily results
# ============================================================

def summarize(
    rows,
    model,
    split,
    period
):

    result_rows = []

    for top_n in (
        5,
        10,
        20,
        30
    ):

        subset = [
            r for r in rows
            if r["model"] == model
            and r["split"] == split
            and r["period"] == period
            and r["top_n"] == top_n
        ]

        if not subset:
            continue

        result_rows.append({

            "split":
                split,

            "period":
                period,

            "model":
                model,

            "top_n":
                top_n,

            "days":
                len(subset),

            "avg_diff":
                float(
                    np.mean([
                        r["avg_diff"]
                        for r in subset
                    ])
                ),

            "median_daily_avg":
                float(
                    np.median([
                        r["avg_diff"]
                        for r in subset
                    ])
                ),

            "win_rate":
                float(
                    np.mean([
                        r["win_rate"]
                        for r in subset
                    ])
                ),

            "plus1000_rate":
                float(
                    np.mean([
                        r["plus1000_rate"]
                        for r in subset
                    ])
                ),

            "plus2000_rate":
                float(
                    np.mean([
                        r["plus2000_rate"]
                        for r in subset
                    ])
                ),

            "positive_days":
                float(
                    np.mean([
                        r["positive"]
                        for r in subset
                    ])
                    * 100
                ),

            "total_diff":
                float(
                    np.sum([
                        r["total_diff"]
                        for r in subset
                    ])
                ),
        })

    return result_rows


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Ana-Slo Ver.4.2 Candidate Stability Test"
    )
    print("=" * 70)

    print()
    print("MODELS")
    print("-" * 70)

    weight_map = {}

    for model_name, config in MODELS.items():

        weights = make_weights(
            config["exclude"]
        )

        weight_map[model_name] = weights

        print(
            f"{model_name:<12} "
            f"exclude="
            f"{','.join(config['exclude']) if config['exclude'] else 'NONE'}"
        )

    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    print(
        f"records = {len(df):,}"
    )

    print()

    # --------------------------------------------------------
    # Build panels for all required dates
    # --------------------------------------------------------

    all_start = min(
        split["test_start"]
        for split in SPLITS
    )

    all_end = max(
        split["test_end"]
        for split in SPLITS
    )

    panels = {}

    print(
        "Building daily feature panels..."
    )

    for target_date in pd.date_range(
        all_start,
        all_end
    ):

        panel = build_features(
            df,
            target_date
        )

        if panel.empty:

            print(
                f"{target_date.date()} "
                f"EMPTY"
            )

            continue

        panels[target_date] = panel

        print(
            f"{target_date.date()} "
            f"machines={len(panel)}"
        )

    print()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    daily_rows = []
    summary_rows = []

    for split_info in SPLITS:

        split_name = split_info["name"]

        train_start = split_info["train_start"]
        train_end = split_info["train_end"]

        test_start = split_info["test_start"]
        test_end = split_info["test_end"]

        print("=" * 70)
        print(split_name)
        print("=" * 70)

        print(
            f"TRAIN: "
            f"{train_start.date()} "
            f"to "
            f"{train_end.date()}"
        )

        print(
            f"TEST : "
            f"{test_start.date()} "
            f"to "
            f"{test_end.date()}"
        )

        print()

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        for model_name in MODELS:

            weights = weight_map[
                model_name
            ]

            print(
                f"Evaluating "
                f"{model_name}..."
            )

            # TEST only
            for target_date in pd.date_range(
                test_start,
                test_end
            ):

                if target_date not in panels:
                    continue

                panel = panels[
                    target_date
                ]

                for top_n in (
                    5,
                    10,
                    20,
                    30
                ):

                    result = evaluate_day(
                        panel,
                        weights,
                        top_n
                    )

                    if result is None:
                        continue

                    daily_rows.append({

                        "split":
                            split_name,

                        "period":
                            "TEST",

                        "date":
                            target_date.date(),

                        "model":
                            model_name,

                        "top_n":
                            top_n,

                        **result,
                    })

            # TRAIN only for diagnostic
            for target_date in pd.date_range(
                train_start,
                train_end
            ):

                panel = build_features(
                    df,
                    target_date
                )

                if panel.empty:
                    continue

                for top_n in (
                    5,
                    10,
                    20,
                    30
                ):

                    result = evaluate_day(
                        panel,
                        weights,
                        top_n
                    )

                    if result is None:
                        continue

                    daily_rows.append({

                        "split":
                            split_name,

                        "period":
                            "TRAIN",

                        "date":
                            target_date.date(),

                        "model":
                            model_name,

                        "top_n":
                            top_n,

                        **result,
                    })

        # ----------------------------------------------------
        # Summaries
        # ----------------------------------------------------

        for model_name in MODELS:

            summary_rows.extend(
                summarize(
                    daily_rows,
                    model_name,
                    split_name,
                    "TRAIN"
                )
            )

            summary_rows.extend(
                summarize(
                    daily_rows,
                    model_name,
                    split_name,
                    "TEST"
                )
            )

    # --------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------

    daily_df = pd.DataFrame(
        daily_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Compare TEST against V4 BASE
    # --------------------------------------------------------

    base_test = summary_df[
        (
            summary_df["model"]
            == "V4_BASE"
        )
        &
        (
            summary_df["period"]
            == "TEST"
        )
    ][
        [
            "split",
            "top_n",
            "avg_diff",
            "total_diff",
            "win_rate",
            "positive_days"
        ]
    ].copy()

    base_test = base_test.rename(
        columns={
            "avg_diff":
                "base_avg_diff",

            "total_diff":
                "base_total_diff",

            "win_rate":
                "base_win_rate",

            "positive_days":
                "base_positive_days",
        }
    )

    compare_df = summary_df.merge(
        base_test,
        on=[
            "split",
            "top_n"
        ],
        how="left"
    )

    compare_df[
        "avg_diff_change_vs_v4"
    ] = (
        compare_df["avg_diff"]
        - compare_df["base_avg_diff"]
    )

    compare_df[
        "total_diff_change_vs_v4"
    ] = (
        compare_df["total_diff"]
        - compare_df["base_total_diff"]
    )

    compare_df[
        "win_rate_change_vs_v4"
    ] = (
        compare_df["win_rate"]
        - compare_df["base_win_rate"]
    )

    compare_df[
        "positive_days_change_vs_v4"
    ] = (
        compare_df["positive_days"]
        - compare_df["base_positive_days"]
    )

    # --------------------------------------------------------
    # TEST TOP10 only
    # --------------------------------------------------------

    test_top10 = compare_df[
        (
            compare_df["period"]
            == "TEST"
        )
        &
        (
            compare_df["top_n"]
            == 10
        )
    ].copy()

    test_top10 = test_top10.sort_values(
        [
            "split",
            "total_diff"
        ],
        ascending=[
            True,
            False
        ]
    )

    # --------------------------------------------------------
    # Stability score
    #
    # Count how many TEST splits beat V4.
    # Also calculate mean change.
    # --------------------------------------------------------

    candidate_rows = []

    for model_name in MODELS:

        subset = test_top10[
            test_top10["model"]
            == model_name
        ]

        if subset.empty:
            continue

        candidate_rows.append({

            "model":
                model_name,

            "test_splits":
                len(subset),

            "splits_better_than_v4":
                int(
                    (
                        subset[
                            "avg_diff_change_vs_v4"
                        ] > 0
                    ).sum()
                ),

            "mean_avg_diff":
                float(
                    subset[
                        "avg_diff"
                    ].mean()
                ),

            "mean_change_vs_v4":
                float(
                    subset[
                        "avg_diff_change_vs_v4"
                    ].mean()
                ),

            "total_diff_all_tests":
                float(
                    subset[
                        "total_diff"
                    ].sum()
                ),

            "total_change_vs_v4":
                float(
                    subset[
                        "total_diff_change_vs_v4"
                    ].sum()
                ),

            "mean_win_rate":
                float(
                    subset[
                        "win_rate"
                    ].mean()
                ),

            "mean_positive_days":
                float(
                    subset[
                        "positive_days"
                    ].mean()
                ),
        })

    stability_df = pd.DataFrame(
        candidate_rows
    )

    stability_df = stability_df.sort_values(
        [
            "splits_better_than_v4",
            "mean_change_vs_v4"
        ],
        ascending=[
            False,
            False
        ]
    )

    # --------------------------------------------------------
    # Weights output
    # --------------------------------------------------------

    weight_rows = []

    for model_name, weights in weight_map.items():

        for factor in FACTORS:

            weight_rows.append({

                "model":
                    model_name,

                "factor":
                    factor,

                "weight":
                    weights[factor],
            })

    weights_df = pd.DataFrame(
        weight_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out_daily = (
        OUT_DIR
        / "23_Ver4_2_stability_daily.csv"
    )

    out_summary = (
        OUT_DIR
        / "23_Ver4_2_stability_summary.csv"
    )

    out_compare = (
        OUT_DIR
        / "23_Ver4_2_stability_compare.csv"
    )

    out_stability = (
        OUT_DIR
        / "23_Ver4_2_stability_ranking.csv"
    )

    out_weights = (
        OUT_DIR
        / "23_Ver4_2_stability_weights.csv"
    )

    daily_df.to_csv(
        out_daily,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        out_summary,
        index=False,
        encoding="utf-8-sig"
    )

    compare_df.to_csv(
        out_compare,
        index=False,
        encoding="utf-8-sig"
    )

    stability_df.to_csv(
        out_stability,
        index=False,
        encoding="utf-8-sig"
    )

    weights_df.to_csv(
        out_weights,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VER.4.2 STABILITY TEST RESULT"
    )
    print("=" * 70)

    print()

    print(
        "TEST TOP10 BY SPLIT"
    )

    print(
        test_top10[
            [
                "split",
                "model",
                "avg_diff",
                "total_diff",
                "win_rate",
                "positive_days",
                "avg_diff_change_vs_v4",
                "total_diff_change_vs_v4",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "STABILITY RANKING"
    )
    print("=" * 70)

    print(
        stability_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "TRAIN vs TEST"
    )
    print("=" * 70)

    train_test = summary_df[
        summary_df["top_n"] == 10
    ][
        [
            "split",
            "period",
            "model",
            "avg_diff",
            "total_diff",
            "win_rate",
            "positive_days",
        ]
    ]

    print(
        train_test.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print(
        "DIAGNOSTIC"
    )
    print("=" * 70)

    for split_name in [
        "SPLIT1",
        "SPLIT2"
    ]:

        subset = test_top10[
            test_top10["split"]
            == split_name
        ]

        print()
        print(
            f"{split_name} TOP10:"
        )

        for _, row in subset.iterrows():

            print(
                f"  {row['model']:<12} "
                f"{row['avg_diff']:+8.2f} "
                f"vs V4 "
                f"{row['avg_diff_change_vs_v4']:+8.2f}"
            )

    print()
    print(
        "Saved:"
    )

    print(
        out_daily
    )

    print(
        out_summary
    )

    print(
        out_compare
    )

    print(
        out_stability
    )

    print(
        out_weights
    )

    print()
    print(
        "Ver.4.2 stability test complete."
    )


if __name__ == "__main__":
    main()