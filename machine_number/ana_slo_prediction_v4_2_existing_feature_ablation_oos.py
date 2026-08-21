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
    / "60_Ver4_2_existing_feature_ablation_oos"
)

EXPECTED_BASE_TOTAL_DIFF = 130400.0
TOP_N = 10


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 102)
    print(title)
    print("=" * 102)


def load_source_56():
    if not SOURCE_56.exists():
        raise FileNotFoundError(
            f"56 source script not found: {SOURCE_56}"
        )

    spec = importlib.util.spec_from_file_location(
        "slotanalyzer_56",
        SOURCE_56,
    )

    if (
        spec is None
        or spec.loader is None
    ):
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

    total = sum(
        weights.values()
    )

    if total <= 0:
        raise ValueError(
            "Weight sum must be positive."
        )

    return {
        k: v / total
        for k, v
        in weights.items()
    }


def ablated_weights(
    base_weights: dict[str, float],
    removed_factor: str | None,
) -> dict[str, float]:

    if removed_factor is None:
        return base_weights.copy()

    if removed_factor not in base_weights:
        raise KeyError(
            f"Factor not in base weights: {removed_factor}"
        )

    weights = base_weights.copy()
    weights.pop(
        removed_factor
    )

    return normalize_weights(
        weights
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "60 - V4.2_C TOP10 Existing Feature Ablation Rolling OOS"
    )

    m56 = load_source_56()

    df = m56.load_data()

    base_weights = (
        m56.V42_C_WEIGHTS.copy()
    )

    rolling_splits = (
        m56.ROLLING_SPLITS
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
        "Exact V4.2_C factors:"
    )

    for factor, weight in (
        base_weights.items()
    ):
        print(
            f"  {factor:<20} "
            f"{weight:.12f}"
        )

    print(
        f"  {'TOTAL':<20} "
        f"{sum(base_weights.values()):.12f}"
    )

    print()
    print(
        "Method:"
    )
    print(
        "- BASE keeps the exact V4.2_C weights."
    )
    print(
        "- Each ablation removes exactly ONE existing factor."
    )
    print(
        "- Remaining weights are renormalized to sum to 1.0."
    )
    print(
        "- Feature construction is imported from validated 56 logic."
    )
    print(
        "- No new feature is added in this test."
    )

    # --------------------------------------------------------
    # Build panels once
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

        panels[
            target_date
        ] = panel

        print(
            f"{target_date.date()} "
            f"machines={len(panel)}"
        )

    # --------------------------------------------------------
    # Define modes
    # --------------------------------------------------------

    modes: dict[str, str | None] = {
        "BASE":
            None,
    }

    for factor in (
        base_weights.keys()
    ):
        modes[
            f"DROP_{factor.upper()}"
        ] = factor

    # --------------------------------------------------------
    # Rolling OOS
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

        for (
            mode,
            removed_factor,
        ) in modes.items():

            weights = ablated_weights(
                base_weights,
                removed_factor,
            )

            for target_date in (
                pd.date_range(
                    test_start,
                    test_end,
                )
            ):

                panel = panels.get(
                    target_date
                )

                if (
                    panel is None
                    or panel.empty
                ):
                    continue

                result = m56.evaluate_day(
                    panel,
                    weights,
                )

                if result is None:
                    continue

                result.update(
                    {
                        "mode":
                            mode,

                        "removed_factor":
                            (
                                "NONE"
                                if removed_factor
                                is None
                                else removed_factor
                            ),

                        "split":
                            split_name,

                        "model":
                            "V4.2_C",

                        "top_n":
                            TOP_N,

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
    # Summaries
    # --------------------------------------------------------

    split_df = (
        daily_df.groupby(
            [
                "mode",
                "removed_factor",
                "split",
            ],
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

    split_df[
        "positive_days"
    ] *= 100.0

    overall_df = (
        daily_df.groupby(
            [
                "mode",
                "removed_factor",
            ],
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

    split_stability = (
        split_df.groupby(
            [
                "mode",
                "removed_factor",
            ],
            as_index=False,
        )
        .agg(
            min_split_avg=(
                "avg_diff",
                "min",
            ),

            max_split_avg=(
                "avg_diff",
                "max",
            ),

            positive_split_rate=(
                "total_diff",
                lambda s:
                    float(
                        (
                            s > 0
                        ).mean()
                        * 100.0
                    ),
            ),
        )
    )

    overall_df = overall_df.merge(
        split_stability,
        on=[
            "mode",
            "removed_factor",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    base = overall_df[
        overall_df["mode"]
        == "BASE"
    ]

    if base.empty:
        raise RuntimeError(
            "BASE result missing."
        )

    base_total = float(
        base.iloc[0][
            "total_diff"
        ]
    )

    base_avg = float(
        base.iloc[0][
            "avg_diff"
        ]
    )

    base_ok = bool(
        np.isclose(
            base_total,
            EXPECTED_BASE_TOTAL_DIFF,
            atol=0.01,
        )
    )

    header(
        "BASELINE SAFETY CHECK"
    )

    print(
        f"actual   : {base_total:+.1f}"
    )
    print(
        f"expected : {EXPECTED_BASE_TOTAL_DIFF:+.1f}"
    )
    print(
        f"match    : {base_ok}"
    )

    if not base_ok:
        raise RuntimeError(
            "BASELINE SAFETY CHECK FAILED. "
            "Do not interpret ablation results."
        )

    overall_df[
        "total_change_vs_base"
    ] = (
        overall_df[
            "total_diff"
        ]
        - base_total
    )

    overall_df[
        "avg_change_vs_base"
    ] = (
        overall_df[
            "avg_diff"
        ]
        - base_avg
    )

    # --------------------------------------------------------
    # Paired daily comparison vs BASE
    # --------------------------------------------------------

    base_daily = daily_df[
        daily_df["mode"]
        == "BASE"
    ][
        [
            "date",
            "split",
            "total_diff",
        ]
    ].rename(
        columns={
            "total_diff":
                "base_total_diff",
        }
    )

    paired_frames = []

    for (
        mode,
        removed_factor,
    ) in modes.items():

        if mode == "BASE":
            continue

        candidate = daily_df[
            daily_df["mode"]
            == mode
        ][
            [
                "date",
                "split",
                "removed_factor",
                "total_diff",
            ]
        ].rename(
            columns={
                "total_diff":
                    "candidate_total_diff",
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

        pair[
            "mode"
        ] = mode

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

        paired_frames.append(
            pair
        )

    paired_df = pd.concat(
        paired_frames,
        ignore_index=True,
    )

    paired_summary_df = (
        paired_df.groupby(
            [
                "mode",
                "removed_factor",
            ],
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
                        (
                            s > 0
                        ).sum()
                    ),
            ),

            same_days=(
                "change_vs_base",
                lambda s:
                    int(
                        (
                            s == 0
                        ).sum()
                    ),
            ),

            worse_days=(
                "change_vs_base",
                lambda s:
                    int(
                        (
                            s < 0
                        ).sum()
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
    # Leave-one-day / leave-one-split robustness
    # --------------------------------------------------------

    loo_rows = []
    loso_rows = []

    for mode in modes:

        if mode == "BASE":
            continue

        p = paired_df[
            paired_df["mode"]
            == mode
        ].copy()

        removed_factor = str(
            p.iloc[0][
                "removed_factor"
            ]
        )

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

                    "removed_factor":
                        removed_factor,

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
                }
            )

        for split in (
            p["split"]
            .drop_duplicates()
        ):

            remaining = p[
                p["split"]
                != split
            ]

            loso_rows.append(
                {
                    "mode":
                        mode,

                    "removed_factor":
                        removed_factor,

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

    loo_df = pd.DataFrame(
        loo_rows
    )

    loso_df = pd.DataFrame(
        loso_rows
    )

    # --------------------------------------------------------
    # Robustness assessment
    # --------------------------------------------------------

    base_split = split_df[
        split_df["mode"]
        == "BASE"
    ][
        [
            "split",
            "avg_diff",
        ]
    ].rename(
        columns={
            "avg_diff":
                "base_avg_diff",
        }
    )

    assessment_rows = []

    for mode in modes:

        if mode == "BASE":
            continue

        o = overall_df[
            overall_df["mode"]
            == mode
        ].iloc[0]

        p = paired_summary_df[
            paired_summary_df["mode"]
            == mode
        ].iloc[0]

        candidate_split = split_df[
            split_df["mode"]
            == mode
        ][
            [
                "split",
                "avg_diff",
            ]
        ]

        sc = candidate_split.merge(
            base_split,
            on="split",
            how="inner",
        )

        improved_splits = int(
            (
                sc[
                    "avg_diff"
                ]
                > sc[
                    "base_avg_diff"
                ]
            ).sum()
        )

        lp = loo_df[
            loo_df["mode"]
            == mode
        ]

        sp = loso_df[
            loso_df["mode"]
            == mode
        ]

        loo_all_positive = bool(
            (
                lp[
                    "remaining_total_change"
                ]
                > 0
            ).all()
        )

        loso_all_positive = bool(
            (
                sp[
                    "remaining_total_change"
                ]
                > 0
            ).all()
        )

        if (
            o[
                "total_change_vs_base"
            ] > 0
            and p[
                "median_daily_change"
            ] >= 0
            and p[
                "better_days"
            ] >= p[
                "worse_days"
            ]
            and improved_splits >= 4
            and loo_all_positive
            and loso_all_positive
        ):
            status = (
                "ROBUST_DROP_CANDIDATE"
            )

        elif (
            o[
                "total_change_vs_base"
            ] > 0
        ):
            status = (
                "POSITIVE_BUT_NOT_ROBUST"
            )

        else:
            status = (
                "KEEP_FACTOR"
            )

        assessment_rows.append(
            {
                "mode":
                    mode,

                "removed_factor":
                    o[
                        "removed_factor"
                    ],

                "status":
                    status,

                "total_diff":
                    float(
                        o[
                            "total_diff"
                        ]
                    ),

                "total_change_vs_base":
                    float(
                        o[
                            "total_change_vs_base"
                        ]
                    ),

                "avg_diff":
                    float(
                        o[
                            "avg_diff"
                        ]
                    ),

                "win_rate":
                    float(
                        o[
                            "win_rate"
                        ]
                    ),

                "positive_days":
                    float(
                        o[
                            "positive_days"
                        ]
                    ),

                "better_days":
                    int(
                        p[
                            "better_days"
                        ]
                    ),

                "same_days":
                    int(
                        p[
                            "same_days"
                        ]
                    ),

                "worse_days":
                    int(
                        p[
                            "worse_days"
                        ]
                    ),

                "median_daily_change":
                    float(
                        p[
                            "median_daily_change"
                        ]
                    ),

                "improved_splits":
                    improved_splits,

                "loo_all_positive":
                    loo_all_positive,

                "loso_all_positive":
                    loso_all_positive,

                "production_change":
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
        "PAIRED DAILY COMPARISON VS BASE"
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

    header(
        "BY SPLIT"
    )

    print(
        split_df.sort_values(
            [
                "mode",
                "split",
            ]
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "60_existing_feature_ablation_daily.csv":
            daily_df,

        "60_existing_feature_ablation_by_split.csv":
            split_df,

        "60_existing_feature_ablation_overall.csv":
            overall_df,

        "60_existing_feature_ablation_paired_daily.csv":
            paired_df,

        "60_existing_feature_ablation_paired_summary.csv":
            paired_summary_df,

        "60_existing_feature_ablation_leave_one_day_out.csv":
            loo_df,

        "60_existing_feature_ablation_leave_one_split_out.csv":
            loso_df,

        "60_existing_feature_ablation_assessment.csv":
            assessment_df,
    }

    header(
        "FILES SAVED"
    )

    for (
        filename,
        frame,
    ) in outputs.items():

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
        "60 existing-feature ablation OOS complete."
    )
    print(
        "No production model change has been made."
    )


if __name__ == "__main__":
    main()
