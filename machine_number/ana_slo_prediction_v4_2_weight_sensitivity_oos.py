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
    / "61_Ver4_2_weight_sensitivity_oos"
)

EXPECTED_BASE_TOTAL_DIFF = 130400.0
TOP_N = 10

MULTIPLIERS = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
]


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


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
        key: value / total
        for key, value
        in weights.items()
    }


def adjusted_weights(
    base_weights: dict[str, float],
    factor: str | None,
    multiplier: float,
) -> dict[str, float]:

    if factor is None:
        return base_weights.copy()

    if factor not in base_weights:
        raise KeyError(
            f"Factor not found: {factor}"
        )

    weights = base_weights.copy()

    weights[factor] = (
        weights[factor]
        * multiplier
    )

    return normalize_weights(
        weights
    )


def mode_name(
    factor: str | None,
    multiplier: float,
) -> str:

    if factor is None:
        return "BASE"

    return (
        f"{factor}"
        f"_x{multiplier:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "61 - V4.2_C TOP10 Existing-Weight Sensitivity Rolling OOS"
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
        "Base V4.2_C weights:"
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
        "Multipliers:"
    )
    print(
        "  "
        + ", ".join(
            f"{x:.2f}x"
            for x in MULTIPLIERS
        )
    )

    print()
    print(
        "Method:"
    )
    print(
        "- Change only ONE existing factor weight at a time."
    )
    print(
        "- Multiply it by 0.50 / 0.75 / 1.00 / 1.25 / 1.50."
    )
    print(
        "- Renormalize all weights back to 1.0."
    )
    print(
        "- 1.00x is the exact baseline and is evaluated only once."
    )
    print(
        "- No production model change is made by this script."
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
    # Experiment definitions
    # --------------------------------------------------------

    experiments = [
        {
            "mode":
                "BASE",

            "factor":
                "NONE",

            "multiplier":
                1.00,
        }
    ]

    for factor in (
        base_weights.keys()
    ):

        for multiplier in MULTIPLIERS:

            if np.isclose(
                multiplier,
                1.00,
            ):
                continue

            experiments.append(
                {
                    "mode":
                        mode_name(
                            factor,
                            multiplier,
                        ),

                    "factor":
                        factor,

                    "multiplier":
                        float(
                            multiplier
                        ),
                }
            )

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

        for experiment in experiments:

            factor = (
                None
                if experiment[
                    "factor"
                ] == "NONE"
                else experiment[
                    "factor"
                ]
            )

            multiplier = float(
                experiment[
                    "multiplier"
                ]
            )

            weights = adjusted_weights(
                base_weights,
                factor,
                multiplier,
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
                            experiment[
                                "mode"
                            ],

                        "factor":
                            experiment[
                                "factor"
                            ],

                        "multiplier":
                            multiplier,

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
                "factor",
                "multiplier",
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
                "factor",
                "multiplier",
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

    stability_df = (
        split_df.groupby(
            [
                "mode",
                "factor",
                "multiplier",
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
        stability_df,
        on=[
            "mode",
            "factor",
            "multiplier",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # Baseline safety check
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
            "Do not interpret sensitivity results."
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
    # Paired daily comparison
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

    pair_frames = []

    for experiment in experiments:

        if experiment[
            "mode"
        ] == "BASE":
            continue

        candidate = daily_df[
            daily_df["mode"]
            == experiment[
                "mode"
            ]
        ][
            [
                "date",
                "split",
                "factor",
                "multiplier",
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
        ] = experiment[
            "mode"
        ]

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

        pair_frames.append(
            pair
        )

    paired_df = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    paired_summary_df = (
        paired_df.groupby(
            [
                "mode",
                "factor",
                "multiplier",
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
    # Best multiplier per factor - exploratory only
    # --------------------------------------------------------

    non_base_overall = (
        overall_df[
            overall_df["mode"]
            != "BASE"
        ]
        .copy()
    )

    best_rows = []

    for factor, group in (
        non_base_overall.groupby(
            "factor"
        )
    ):

        best = (
            group.sort_values(
                [
                    "total_diff",
                    "min_split_avg",
                    "positive_split_rate",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

        pair = paired_summary_df[
            paired_summary_df[
                "mode"
            ]
            == best[
                "mode"
            ]
        ].iloc[0]

        best_rows.append(
            {
                "factor":
                    factor,

                "best_multiplier":
                    float(
                        best[
                            "multiplier"
                        ]
                    ),

                "mode":
                    best[
                        "mode"
                    ],

                "total_diff":
                    float(
                        best[
                            "total_diff"
                        ]
                    ),

                "total_change_vs_base":
                    float(
                        best[
                            "total_change_vs_base"
                        ]
                    ),

                "avg_diff":
                    float(
                        best[
                            "avg_diff"
                        ]
                    ),

                "win_rate":
                    float(
                        best[
                            "win_rate"
                        ]
                    ),

                "positive_days":
                    float(
                        best[
                            "positive_days"
                        ]
                    ),

                "min_split_avg":
                    float(
                        best[
                            "min_split_avg"
                        ]
                    ),

                "positive_split_rate":
                    float(
                        best[
                            "positive_split_rate"
                        ]
                    ),

                "better_days":
                    int(
                        pair[
                            "better_days"
                        ]
                    ),

                "same_days":
                    int(
                        pair[
                            "same_days"
                        ]
                    ),

                "worse_days":
                    int(
                        pair[
                            "worse_days"
                        ]
                    ),

                "median_daily_change":
                    float(
                        pair[
                            "median_daily_change"
                        ]
                    ),
            }
        )

    best_df = pd.DataFrame(
        best_rows
    )

    # --------------------------------------------------------
    # Candidate screening
    # --------------------------------------------------------

    screening_rows = []

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

    for _, row in best_df.iterrows():

        mode = row["mode"]

        split_candidate = split_df[
            split_df["mode"]
            == mode
        ][
            [
                "split",
                "avg_diff",
            ]
        ]

        sc = split_candidate.merge(
            base_split,
            on="split",
            how="inner",
        )

        improved_splits = int(
            (
                sc["avg_diff"]
                > sc[
                    "base_avg_diff"
                ]
            ).sum()
        )

        if (
            row[
                "total_change_vs_base"
            ] > 0
            and row[
                "median_daily_change"
            ] >= 0
            and row[
                "better_days"
            ] >= row[
                "worse_days"
            ]
            and improved_splits >= 4
        ):
            status = (
                "PROMISING_REQUIRE_ROBUSTNESS_TEST"
            )

        elif (
            row[
                "total_change_vs_base"
            ] > 0
        ):
            status = (
                "POSITIVE_BUT_UNSTABLE"
            )

        else:
            status = (
                "NO_WEIGHT_IMPROVEMENT"
            )

        screening_rows.append(
            {
                "factor":
                    row[
                        "factor"
                    ],

                "best_multiplier":
                    row[
                        "best_multiplier"
                    ],

                "status":
                    status,

                "total_change_vs_base":
                    row[
                        "total_change_vs_base"
                    ],

                "better_days":
                    row[
                        "better_days"
                    ],

                "same_days":
                    row[
                        "same_days"
                    ],

                "worse_days":
                    row[
                        "worse_days"
                    ],

                "median_daily_change":
                    row[
                        "median_daily_change"
                    ],

                "improved_splits":
                    improved_splits,

                "production_adopted":
                    False,
            }
        )

    screening_df = pd.DataFrame(
        screening_rows
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    header(
        "TOP 20 OVERALL SETTINGS"
    )

    print(
        overall_df.sort_values(
            [
                "total_diff",
                "min_split_avg",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(20).to_string(
            index=False
        )
    )

    header(
        "BEST MULTIPLIER PER FACTOR - EXPLORATORY"
    )

    print(
        best_df.sort_values(
            "total_change_vs_base",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    header(
        "CANDIDATE SCREENING"
    )

    print(
        screening_df.sort_values(
            "total_change_vs_base",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "The best multiplier is selected after scanning multiple settings."
    )
    print(
        "Therefore any apparent gain contains selection bias."
    )
    print(
        "Do not adopt a new production weight from this script alone."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "61_weight_sensitivity_daily.csv":
            daily_df,

        "61_weight_sensitivity_by_split.csv":
            split_df,

        "61_weight_sensitivity_overall.csv":
            overall_df,

        "61_weight_sensitivity_paired_daily.csv":
            paired_df,

        "61_weight_sensitivity_paired_summary.csv":
            paired_summary_df,

        "61_weight_sensitivity_best_per_factor.csv":
            best_df,

        "61_weight_sensitivity_screening.csv":
            screening_df,
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
        "61 V4.2_C weight sensitivity OOS complete."
    )
    print(
        "No production model change has been made."
    )


if __name__ == "__main__":
    main()
