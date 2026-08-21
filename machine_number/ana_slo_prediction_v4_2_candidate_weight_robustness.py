from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

SOURCE_56 = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "62_Ver4_2_candidate_weight_robustness"
)

TOP_N = 10

EXPECTED_TOTALS = {
    "BASE": 130400.0,
    "AVG31_X0.50": 140700.0,
    "PLUS1000_X1.50": 135400.0,
}

SPECIAL_DATES = [
    pd.Timestamp("2026-08-03"),
    pd.Timestamp("2026-08-17"),
]


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


def load_source_56():
    if not SOURCE_56.exists():
        raise FileNotFoundError(
            f"56 source script not found: {SOURCE_56}"
        )

    spec = importlib.util.spec_from_file_location(
        "slotanalyzer_56",
        SOURCE_56,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not import 56 source."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def normalize_weights(
    weights: dict[str, float],
) -> dict[str, float]:

    total = sum(weights.values())

    if total <= 0:
        raise ValueError(
            "Weight sum must be positive."
        )

    return {
        k: v / total
        for k, v in weights.items()
    }


def make_candidate_weights(
    base_weights: dict[str, float],
) -> dict[str, dict[str, float]]:

    # BASE
    base = base_weights.copy()

    # avg31 x 0.50
    avg31 = base_weights.copy()
    avg31["avg31"] *= 0.50
    avg31 = normalize_weights(avg31)

    # plus1000_rate x 1.50
    plus1000 = base_weights.copy()
    plus1000["plus1000_rate"] *= 1.50
    plus1000 = normalize_weights(plus1000)

    # Combined exploratory candidate
    combined = base_weights.copy()
    combined["avg31"] *= 0.50
    combined["plus1000_rate"] *= 1.50
    combined = normalize_weights(combined)

    return {
        "BASE": base,
        "AVG31_X0.50": avg31,
        "PLUS1000_X1.50": plus1000,
        "COMBINED": combined,
    }


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    std = float(
        s.std(ddof=0)
    )

    if (
        std == 0
        or np.isnan(std)
    ):
        return pd.Series(
            0.0,
            index=s.index,
        )

    return (
        s - s.mean()
    ) / std


def rank_panel(
    panel: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:

    x = panel.copy()

    score = pd.Series(
        0.0,
        index=x.index,
    )

    for factor, weight in weights.items():

        if factor not in x.columns:
            raise RuntimeError(
                f"Feature missing from panel: {factor}"
            )

        z = zscore(
            x[factor]
        )

        component = (
            50.0
            + z * 12.5
        ).clip(
            0,
            100,
        )

        score += (
            component
            * weight
        )

    x["score"] = score

    return x.sort_values(
        [
            "score",
            "machine_no",
        ],
        ascending=[
            False,
            True,
        ],
    )


def evaluate_ranked(
    ranked: pd.DataFrame,
) -> dict:

    selected = ranked.head(TOP_N).copy()

    diffs = pd.to_numeric(
        selected["diff"],
        errors="coerce",
    ).dropna()

    selected_nos = tuple(
        int(x)
        for x in selected[
            "machine_no"
        ].tolist()
    )

    return {
        "avg_diff":
            float(diffs.mean()),

        "median_diff":
            float(diffs.median()),

        "win_rate":
            float(
                (diffs > 0).mean()
                * 100.0
            ),

        "positive":
            int(
                diffs.sum() > 0
            ),

        "total_diff":
            float(
                diffs.sum()
            ),

        "selected_nos":
            selected_nos,
    }


def safe_pct(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return np.nan

    return float(
        numerator
        / denominator
        * 100.0
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "62 - V4.2_C TOP10 Candidate-Weight Robustness Diagnostic"
    )

    m56 = load_source_56()

    df = m56.load_data()
    rolling_splits = m56.ROLLING_SPLITS

    base_weights = (
        m56.V42_C_WEIGHTS.copy()
    )

    candidate_weights = (
        make_candidate_weights(
            base_weights
        )
    )

    print(
        f"records              : {len(df):,}"
    )
    print(
        f"days                 : {df['date'].nunique()}"
    )
    print(
        f"machines             : {df['machine_no'].nunique()}"
    )
    print(
        f"top_n                : {TOP_N}"
    )

    print()
    print(
        "Candidates:"
    )

    for name, weights in candidate_weights.items():
        print(
            f"  {name:<18} "
            f"sum={sum(weights.values()):.12f}"
        )

    print()
    print(
        "Purpose:"
    )
    print(
        "- Recheck the two positive 61 candidates."
    )
    print(
        "- Test leave-one-day and leave-one-split robustness."
    )
    print(
        "- Check 2026-08-03 and 2026-08-17 exclusions."
    )
    print(
        "- Measure how often TOP10 selections actually change."
    )
    print(
        "- Evaluate the combined adjustment as exploratory only."
    )

    # --------------------------------------------------------
    # Build panels
    # --------------------------------------------------------

    edge_distance_map = (
        m56.build_number_edge_distance(
            df["machine_no"].tolist()
        )
    )

    all_test_dates = sorted(
        {
            d
            for (
                _,
                _,
                _,
                test_start,
                test_end,
            ) in rolling_splits
            for d in pd.date_range(
                test_start,
                test_end,
            )
        }
    )

    panels = {}

    header(
        "BUILDING LEAKAGE-SAFE PANELS"
    )

    for target_date in all_test_dates:

        panel = m56.build_features(
            df,
            target_date,
            edge_distance_map,
        )

        panels[target_date] = panel

        print(
            f"{target_date.date()} "
            f"machines={len(panel)}"
        )

    # --------------------------------------------------------
    # Rolling OOS evaluation
    # --------------------------------------------------------

    daily_rows = []

    header(
        "ROLLING OOS"
    )

    for (
        split_name,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in rolling_splits:

        print(
            f"{split_name}: "
            f"{test_start.date()} "
            f"to {test_end.date()}"
        )

        for target_date in pd.date_range(
            test_start,
            test_end,
        ):

            panel = panels.get(
                target_date
            )

            if (
                panel is None
                or panel.empty
            ):
                continue

            for (
                mode,
                weights,
            ) in candidate_weights.items():

                ranked = rank_panel(
                    panel,
                    weights,
                )

                result = evaluate_ranked(
                    ranked
                )

                result.update(
                    {
                        "mode":
                            mode,

                        "split":
                            split_name,

                        "date":
                            target_date,

                        "train_start":
                            train_start,

                        "train_end":
                            train_end,

                        "test_start":
                            test_start,

                        "test_end":
                            test_end,
                    }
                )

                daily_rows.append(
                    result
                )

    daily_df = pd.DataFrame(
        daily_rows
    )

    if daily_df.empty:
        raise RuntimeError(
            "No OOS results."
        )

    # --------------------------------------------------------
    # Overall + safety checks
    # --------------------------------------------------------

    overall_df = (
        daily_df.groupby(
            "mode",
            as_index=False,
        )
        .agg(
            days=(
                "date",
                "nunique",
            ),

            avg_diff=(
                "avg_diff",
                "mean",
            ),

            win_rate=(
                "win_rate",
                "mean",
            ),

            positive_days=(
                "positive",
                "mean",
            ),

            total_diff=(
                "total_diff",
                "sum",
            ),
        )
    )

    overall_df[
        "positive_days"
    ] *= 100.0

    header(
        "SAFETY CHECK"
    )

    safety_ok = True

    for mode, expected in (
        EXPECTED_TOTALS.items()
    ):

        row = overall_df[
            overall_df["mode"]
            == mode
        ]

        if row.empty:
            raise RuntimeError(
                f"Safety row missing: {mode}"
            )

        actual = float(
            row.iloc[0][
                "total_diff"
            ]
        )

        ok = bool(
            np.isclose(
                actual,
                expected,
                atol=0.01,
            )
        )

        print(
            f"{mode:<18} "
            f"actual={actual:+.1f} "
            f"expected={expected:+.1f} "
            f"=> {ok}"
        )

        safety_ok = (
            safety_ok and ok
        )

    if not safety_ok:
        raise RuntimeError(
            "SAFETY CHECK FAILED. "
            "Do not interpret robustness results."
        )

    base_total = float(
        overall_df.loc[
            overall_df["mode"]
            == "BASE",
            "total_diff",
        ].iloc[0]
    )

    overall_df[
        "total_change_vs_base"
    ] = (
        overall_df["total_diff"]
        - base_total
    )

    # --------------------------------------------------------
    # Paired daily + selection overlap
    # --------------------------------------------------------

    base_daily = daily_df[
        daily_df["mode"]
        == "BASE"
    ][
        [
            "date",
            "split",
            "total_diff",
            "selected_nos",
        ]
    ].rename(
        columns={
            "total_diff":
                "base_total_diff",

            "selected_nos":
                "base_selected_nos",
        }
    )

    pair_frames = []

    for mode in (
        candidate_weights.keys()
    ):

        if mode == "BASE":
            continue

        candidate = daily_df[
            daily_df["mode"]
            == mode
        ][
            [
                "date",
                "split",
                "total_diff",
                "selected_nos",
            ]
        ].rename(
            columns={
                "total_diff":
                    "candidate_total_diff",

                "selected_nos":
                    "candidate_selected_nos",
            }
        )

        pair = base_daily.merge(
            candidate,
            on=[
                "date",
                "split",
            ],
            how="inner",
            validate="one_to_one",
        )

        pair["mode"] = mode

        pair[
            "change_vs_base"
        ] = (
            pair[
                "candidate_total_diff"
            ]
            - pair[
                "base_total_diff"
            ]
        )

        overlap_counts = []
        jaccards = []
        changed_slots = []

        for _, row in pair.iterrows():

            base_set = set(
                row[
                    "base_selected_nos"
                ]
            )

            candidate_set = set(
                row[
                    "candidate_selected_nos"
                ]
            )

            overlap = len(
                base_set
                & candidate_set
            )

            union = len(
                base_set
                | candidate_set
            )

            overlap_counts.append(
                overlap
            )

            jaccards.append(
                (
                    overlap / union
                    if union > 0
                    else 1.0
                )
            )

            changed_slots.append(
                TOP_N - overlap
            )

        pair[
            "top10_overlap"
        ] = overlap_counts

        pair[
            "top10_jaccard"
        ] = jaccards

        pair[
            "changed_slots"
        ] = changed_slots

        pair_frames.append(
            pair
        )

    paired_df = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    paired_summary_df = (
        paired_df.groupby(
            "mode",
            as_index=False,
        )
        .agg(
            paired_days=(
                "date",
                "size",
            ),

            better_days=(
                "change_vs_base",
                lambda s:
                    int(
                        (s > 0).sum()
                    ),
            ),

            same_days=(
                "change_vs_base",
                lambda s:
                    int(
                        (s == 0).sum()
                    ),
            ),

            worse_days=(
                "change_vs_base",
                lambda s:
                    int(
                        (s < 0).sum()
                    ),
            ),

            mean_daily_change=(
                "change_vs_base",
                "mean",
            ),

            median_daily_change=(
                "change_vs_base",
                "median",
            ),

            total_change=(
                "change_vs_base",
                "sum",
            ),

            min_daily_change=(
                "change_vs_base",
                "min",
            ),

            max_daily_change=(
                "change_vs_base",
                "max",
            ),

            mean_top10_overlap=(
                "top10_overlap",
                "mean",
            ),

            mean_top10_jaccard=(
                "top10_jaccard",
                "mean",
            ),

            mean_changed_slots=(
                "changed_slots",
                "mean",
            ),

            days_with_any_change=(
                "changed_slots",
                lambda s:
                    int(
                        (s > 0).sum()
                    ),
            ),
        )
    )

    paired_summary_df[
        "better_rate_ex_ties"
    ] = (
        paired_summary_df[
            "better_days"
        ]
        / (
            paired_summary_df[
                "better_days"
            ]
            + paired_summary_df[
                "worse_days"
            ]
        ).replace(
            0,
            np.nan,
        )
        * 100.0
    )

    # --------------------------------------------------------
    # By split
    # --------------------------------------------------------

    split_rows = []

    for mode in (
        candidate_weights.keys()
    ):

        mode_daily = daily_df[
            daily_df["mode"]
            == mode
        ]

        for split, g in (
            mode_daily.groupby(
                "split",
                sort=False,
            )
        ):

            split_rows.append(
                {
                    "mode":
                        mode,

                    "split":
                        split,

                    "days":
                        int(
                            g[
                                "date"
                            ].nunique()
                        ),

                    "avg_diff":
                        float(
                            g[
                                "avg_diff"
                            ].mean()
                        ),

                    "total_diff":
                        float(
                            g[
                                "total_diff"
                            ].sum()
                        ),

                    "positive_days":
                        float(
                            g[
                                "positive"
                            ].mean()
                            * 100.0
                        ),
                }
            )

    split_df = pd.DataFrame(
        split_rows
    )

    base_split = split_df[
        split_df["mode"]
        == "BASE"
    ][
        [
            "split",
            "total_diff",
        ]
    ].rename(
        columns={
            "total_diff":
                "base_split_total",
        }
    )

    candidate_split_compare = split_df[
        split_df["mode"]
        != "BASE"
    ].merge(
        base_split,
        on="split",
        how="left",
    )

    candidate_split_compare[
        "split_change_vs_base"
    ] = (
        candidate_split_compare[
            "total_diff"
        ]
        - candidate_split_compare[
            "base_split_total"
        ]
    )

    # --------------------------------------------------------
    # Leave-one-day-out
    # --------------------------------------------------------

    loo_rows = []

    for mode in (
        paired_df["mode"]
        .drop_duplicates()
    ):

        p = paired_df[
            paired_df["mode"]
            == mode
        ].copy()

        for idx, row in (
            p.iterrows()
        ):

            remaining = p.drop(
                index=idx
            )

            loo_rows.append(
                {
                    "mode":
                        mode,

                    "excluded_date":
                        row[
                            "date"
                        ],

                    "excluded_split":
                        row[
                            "split"
                        ],

                    "excluded_change":
                        float(
                            row[
                                "change_vs_base"
                            ]
                        ),

                    "remaining_total_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].sum()
                        ),

                    "remaining_mean_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].mean()
                        ),
                }
            )

    loo_df = pd.DataFrame(
        loo_rows
    )

    # --------------------------------------------------------
    # Leave-one-split-out
    # --------------------------------------------------------

    loso_rows = []

    for mode in (
        paired_df["mode"]
        .drop_duplicates()
    ):

        p = paired_df[
            paired_df["mode"]
            == mode
        ].copy()

        for split in (
            p[
                "split"
            ].drop_duplicates()
        ):

            remaining = p[
                p["split"]
                != split
            ]

            loso_rows.append(
                {
                    "mode":
                        mode,

                    "excluded_split":
                        split,

                    "remaining_total_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].sum()
                        ),

                    "remaining_mean_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].mean()
                        ),

                    "remaining_median_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].median()
                        ),
                }
            )

    loso_df = pd.DataFrame(
        loso_rows
    )

    # --------------------------------------------------------
    # Special-date exclusion
    # --------------------------------------------------------

    special_rows = []

    exclusion_sets = {
        "EXCLUDE_2026-08-03":
            {
                pd.Timestamp(
                    "2026-08-03"
                )
            },

        "EXCLUDE_2026-08-17":
            {
                pd.Timestamp(
                    "2026-08-17"
                )
            },

        "EXCLUDE_BOTH":
            set(
                SPECIAL_DATES
            ),
    }

    for mode in (
        paired_df["mode"]
        .drop_duplicates()
    ):

        p = paired_df[
            paired_df["mode"]
            == mode
        ].copy()

        for label, excluded_dates in (
            exclusion_sets.items()
        ):

            remaining = p[
                ~p["date"].isin(
                    excluded_dates
                )
            ]

            special_rows.append(
                {
                    "mode":
                        mode,

                    "exclusion":
                        label,

                    "remaining_days":
                        int(
                            len(
                                remaining
                            )
                        ),

                    "remaining_total_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].sum()
                        ),

                    "remaining_mean_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].mean()
                        ),

                    "remaining_median_change":
                        float(
                            remaining[
                                "change_vs_base"
                            ].median()
                        ),
                }
            )

    special_df = pd.DataFrame(
        special_rows
    )

    # --------------------------------------------------------
    # Advantage concentration
    # --------------------------------------------------------

    concentration_rows = []

    for mode in (
        paired_df["mode"]
        .drop_duplicates()
    ):

        p = paired_df[
            paired_df["mode"]
            == mode
        ].copy()

        positive = (
            p.loc[
                p[
                    "change_vs_base"
                ] > 0,
                "change_vs_base",
            ]
            .sort_values(
                ascending=False
            )
        )

        positive_total = float(
            positive.sum()
        )

        concentration_rows.append(
            {
                "mode":
                    mode,

                "positive_advantage_total":
                    positive_total,

                "top1_positive_share_pct":
                    safe_pct(
                        float(
                            positive.head(
                                1
                            ).sum()
                        ),
                        positive_total,
                    ),

                "top3_positive_share_pct":
                    safe_pct(
                        float(
                            positive.head(
                                3
                            ).sum()
                        ),
                        positive_total,
                    ),

                "top5_positive_share_pct":
                    safe_pct(
                        float(
                            positive.head(
                                5
                            ).sum()
                        ),
                        positive_total,
                    ),
            }
        )

    concentration_df = pd.DataFrame(
        concentration_rows
    )

    # --------------------------------------------------------
    # Robustness assessment
    # --------------------------------------------------------

    assessment_rows = []

    for mode in (
        paired_df["mode"]
        .drop_duplicates()
    ):

        pair_row = paired_summary_df[
            paired_summary_df["mode"]
            == mode
        ].iloc[0]

        split_part = (
            candidate_split_compare[
                candidate_split_compare[
                    "mode"
                ]
                == mode
            ]
        )

        improved_splits = int(
            (
                split_part[
                    "split_change_vs_base"
                ]
                > 0
            ).sum()
        )

        loo_part = loo_df[
            loo_df["mode"]
            == mode
        ]

        loso_part = loso_df[
            loso_df["mode"]
            == mode
        ]

        special_part = (
            special_df[
                special_df["mode"]
                == mode
            ]
        )

        loo_all_positive = bool(
            (
                loo_part[
                    "remaining_total_change"
                ]
                > 0
            ).all()
        )

        loso_all_positive = bool(
            (
                loso_part[
                    "remaining_total_change"
                ]
                > 0
            ).all()
        )

        special_all_positive = bool(
            (
                special_part[
                    "remaining_total_change"
                ]
                > 0
            ).all()
        )

        if (
            pair_row[
                "total_change"
            ] > 0
            and pair_row[
                "median_daily_change"
            ] >= 0
            and pair_row[
                "better_days"
            ] >= pair_row[
                "worse_days"
            ]
            and improved_splits >= 4
            and loo_all_positive
            and loso_all_positive
            and special_all_positive
        ):
            status = (
                "ROBUST_V4_3_CANDIDATE"
            )

        elif (
            pair_row[
                "total_change"
            ] > 0
        ):
            status = (
                "POSITIVE_BUT_NOT_ROBUST"
            )

        else:
            status = (
                "NO_IMPROVEMENT"
            )

        assessment_rows.append(
            {
                "mode":
                    mode,

                "status":
                    status,

                "total_change_vs_base":
                    float(
                        pair_row[
                            "total_change"
                        ]
                    ),

                "better_days":
                    int(
                        pair_row[
                            "better_days"
                        ]
                    ),

                "same_days":
                    int(
                        pair_row[
                            "same_days"
                        ]
                    ),

                "worse_days":
                    int(
                        pair_row[
                            "worse_days"
                        ]
                    ),

                "median_daily_change":
                    float(
                        pair_row[
                            "median_daily_change"
                        ]
                    ),

                "improved_splits":
                    improved_splits,

                "loo_all_positive":
                    loo_all_positive,

                "loso_all_positive":
                    loso_all_positive,

                "special_exclusions_all_positive":
                    special_all_positive,

                "mean_changed_slots":
                    float(
                        pair_row[
                            "mean_changed_slots"
                        ]
                    ),

                "days_with_any_top10_change":
                    int(
                        pair_row[
                            "days_with_any_change"
                        ]
                    ),

                "production_adopted":
                    False,
            }
        )

    assessment_df = pd.DataFrame(
        assessment_rows
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    header(
        "OVERALL RESULTS"
    )

    print(
        overall_df.sort_values(
            "total_diff",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    header(
        "PAIRED DAILY + TOP10 SELECTION CHANGE"
    )

    print(
        paired_summary_df.sort_values(
            "total_change",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    header(
        "BY SPLIT VS BASE"
    )

    print(
        candidate_split_compare.sort_values(
            [
                "mode",
                "split",
            ]
        ).to_string(
            index=False
        )
    )

    header(
        "SPECIAL-DATE EXCLUSIONS"
    )

    print(
        special_df.sort_values(
            [
                "mode",
                "exclusion",
            ]
        ).to_string(
            index=False
        )
    )

    header(
        "ADVANTAGE CONCENTRATION"
    )

    print(
        concentration_df.to_string(
            index=False
        )
    )

    header(
        "ROBUSTNESS ASSESSMENT"
    )

    print(
        assessment_df.sort_values(
            "total_change_vs_base",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "Important: COMBINED is exploratory and was not selected independently "
        "before this test."
    )
    print(
        "No production weight is changed by this script."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "62_candidate_weight_daily.csv":
            daily_df,

        "62_candidate_weight_overall.csv":
            overall_df,

        "62_candidate_weight_paired_daily.csv":
            paired_df,

        "62_candidate_weight_paired_summary.csv":
            paired_summary_df,

        "62_candidate_weight_by_split.csv":
            split_df,

        "62_candidate_weight_split_compare.csv":
            candidate_split_compare,

        "62_candidate_weight_leave_one_day_out.csv":
            loo_df,

        "62_candidate_weight_leave_one_split_out.csv":
            loso_df,

        "62_candidate_weight_special_exclusions.csv":
            special_df,

        "62_candidate_weight_concentration.csv":
            concentration_df,

        "62_candidate_weight_assessment.csv":
            assessment_df,
    }

    header(
        "FILES SAVED"
    )

    for filename, frame in (
        outputs.items()
    ):

        path = (
            OUTPUT_DIR
            / filename
        )

        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(path)

    print()
    print(
        "62 candidate-weight robustness diagnostic complete."
    )
    print(
        "No production model change has been made."
    )


if __name__ == "__main__":
    main()
