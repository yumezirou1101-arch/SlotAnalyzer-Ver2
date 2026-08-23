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
    / "57_Ver4_2_number_band10_weight_robustness"
)

# Test only the 10-number band signal that survived 56 as a weak candidate.
BAND10_WEIGHTS = [
    0.000,
    0.025,
    0.050,
    0.075,
    0.100,
    0.125,
    0.150,
]

EXPECTED_BASE_TOTAL_DIFF = 130400.0
EXPECTED_W010_TOTAL_DIFF = 134900.0


# ============================================================
# HELPERS
# ============================================================

def print_header(
    title: str,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


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
            "Could not create import specification for 56."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def mode_name(
    weight: float,
) -> str:
    return (
        "BAND10_W"
        + f"{weight:.3f}"
    )


def make_weights(
    base_weights: dict[str, float],
    band10_weight: float,
) -> dict[str, float]:
    if not (
        0.0
        <= band10_weight
        < 1.0
    ):
        raise ValueError(
            "band10_weight must be >= 0 and < 1."
        )

    if band10_weight == 0.0:
        return base_weights.copy()

    scale = (
        1.0
        - band10_weight
    )

    weights = {
        factor:
            value * scale
        for factor, value
        in base_weights.items()
    }

    weights[
        "number_band_10"
    ] = band10_weight

    total = sum(
        weights.values()
    )

    if not np.isclose(
        total,
        1.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Weight sum error: {total}"
        )

    return weights


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
    print_header(
        "57 - V4.2_C TOP10 Number-Band-10 Weight Robustness"
    )

    m56 = load_source_56()

    base_weights = (
        m56.V42_C_WEIGHTS.copy()
    )

    rolling_splits = (
        m56.ROLLING_SPLITS
    )

    top_n = int(
        m56.TOP_N
    )

    print(
        f"56 source            : {SOURCE_56}"
    )
    print(
        f"top_n                : {top_n}"
    )
    print(
        "band10 weights       : "
        + ", ".join(
            f"{w:.3f}"
            for w in BAND10_WEIGHTS
        )
    )

    print()
    print(
        "Important:"
    )
    print(
        "- The 56 feature-construction code is imported directly."
    )
    print(
        "- Every historical feature therefore remains leakage-safe "
        "(date < target_date)."
    )
    print(
        "- Only the weight assigned to number_band_10 changes."
    )
    print(
        "- Weight 0.000 must reproduce the 48/56 baseline."
    )
    print(
        "- Weight 0.100 must reproduce the 56 ADD_NUMBER_BAND_10 result."
    )

    df = m56.load_data()

    print()
    print(
        f"records              : {len(df):,}"
    )
    print(
        f"days                 : {df['date'].nunique()}"
    )
    print(
        f"machines             : {df['machine_no'].nunique()}"
    )

    edge_distance_map = (
        m56.build_number_edge_distance(
            df[
                "machine_no"
            ].tolist()
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

    # --------------------------------------------------------
    # Build panels once using the exact 56 feature code.
    # --------------------------------------------------------

    panels: dict[
        pd.Timestamp,
        pd.DataFrame,
    ] = {}

    print_header(
        "BUILDING PANELS FROM 56 FEATURE LOGIC"
    )

    for target_date in (
        all_test_dates
    ):
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
    # Rolling OOS
    # --------------------------------------------------------

    daily_rows = []

    print_header(
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
            f"TEST {test_start.date()} "
            f"to {test_end.date()}"
        )

        for weight in (
            BAND10_WEIGHTS
        ):
            weights = make_weights(
                base_weights,
                weight,
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
                        "split":
                            split_name,

                        "mode":
                            mode_name(
                                weight
                            ),

                        "band10_weight":
                            float(
                                weight
                            ),

                        "model":
                            "V4.2_C",

                        "top_n":
                            top_n,

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
    # Summary
    # --------------------------------------------------------

    split_df = (
        daily_df.groupby(
            [
                "band10_weight",
                "mode",
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
                "band10_weight",
                "mode",
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
                "band10_weight",
                "mode",
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

    overall_df = (
        overall_df.merge(
            split_stability,
            on=[
                "band10_weight",
                "mode",
            ],
            how="left",
        )
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    base_row = overall_df[
        np.isclose(
            overall_df[
                "band10_weight"
            ],
            0.0,
        )
    ]

    w010_row = overall_df[
        np.isclose(
            overall_df[
                "band10_weight"
            ],
            0.100,
        )
    ]

    if (
        base_row.empty
        or w010_row.empty
    ):
        raise RuntimeError(
            "Required safety-check rows missing."
        )

    base_total = float(
        base_row.iloc[0][
            "total_diff"
        ]
    )

    w010_total = float(
        w010_row.iloc[0][
            "total_diff"
        ]
    )

    base_ok = bool(
        np.isclose(
            base_total,
            EXPECTED_BASE_TOTAL_DIFF,
            atol=0.01,
        )
    )

    w010_ok = bool(
        np.isclose(
            w010_total,
            EXPECTED_W010_TOTAL_DIFF,
            atol=0.01,
        )
    )

    print_header(
        "SAFETY CHECK"
    )

    print(
        f"W=0.000 actual/expected : "
        f"{base_total:+.1f} / "
        f"{EXPECTED_BASE_TOTAL_DIFF:+.1f} "
        f"=> {base_ok}"
    )

    print(
        f"W=0.100 actual/expected : "
        f"{w010_total:+.1f} / "
        f"{EXPECTED_W010_TOTAL_DIFF:+.1f} "
        f"=> {w010_ok}"
    )

    if not (
        base_ok
        and w010_ok
    ):
        raise RuntimeError(
            "SAFETY CHECK FAILED. "
            "Do not interpret weight-grid results."
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
        - float(
            base_row.iloc[0][
                "avg_diff"
            ]
        )
    )

    # --------------------------------------------------------
    # Paired daily comparison versus W=0.
    # --------------------------------------------------------

    base_daily = daily_df[
        np.isclose(
            daily_df[
                "band10_weight"
            ],
            0.0,
        )
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

    for weight in (
        BAND10_WEIGHTS
    ):
        if np.isclose(
            weight,
            0.0,
        ):
            continue

        wd = daily_df[
            np.isclose(
                daily_df[
                    "band10_weight"
                ],
                weight,
            )
        ][
            [
                "date",
                "split",
                "total_diff",
            ]
        ].rename(
            columns={
                "total_diff":
                    "weight_total_diff",
            }
        )

        pair = base_daily.merge(
            wd,
            on=[
                "date",
                "split",
            ],
            how="inner",
            validate="one_to_one",
        )

        pair[
            "band10_weight"
        ] = float(
            weight
        )

        pair[
            "mode"
        ] = mode_name(
            weight
        )

        pair[
            "change_vs_base"
        ] = (
            pair[
                "weight_total_diff"
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
                "band10_weight",
                "mode",
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
    # Leave-one-day-out and leave-one-split-out robustness.
    # --------------------------------------------------------

    loo_rows = []
    loso_rows = []

    for weight in (
        BAND10_WEIGHTS
    ):
        if np.isclose(
            weight,
            0.0,
        ):
            continue

        p = paired_df[
            np.isclose(
                paired_df[
                    "band10_weight"
                ],
                weight,
            )
        ].copy()

        for idx, row in (
            p.iterrows()
        ):
            remaining = p.drop(
                index=idx
            )

            loo_rows.append(
                {
                    "band10_weight":
                        float(
                            weight
                        ),

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
                    "band10_weight":
                        float(
                            weight
                        ),

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

    robustness_rows = []

    for weight in (
        BAND10_WEIGHTS
    ):
        if np.isclose(
            weight,
            0.0,
        ):
            continue

        pair_row = (
            paired_summary_df[
                np.isclose(
                    paired_summary_df[
                        "band10_weight"
                    ],
                    weight,
                )
            ].iloc[0]
        )

        split_weight = split_df[
            np.isclose(
                split_df[
                    "band10_weight"
                ],
                weight,
            )
        ][
            [
                "split",
                "avg_diff",
            ]
        ]

        split_base = split_df[
            np.isclose(
                split_df[
                    "band10_weight"
                ],
                0.0,
            )
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

        split_compare = (
            split_weight.merge(
                split_base,
                on="split",
                how="inner",
            )
        )

        improved_splits = int(
            (
                split_compare[
                    "avg_diff"
                ]
                > split_compare[
                    "base_avg_diff"
                ]
            ).sum()
        )

        loo_part = loo_df[
            np.isclose(
                loo_df[
                    "band10_weight"
                ],
                weight,
            )
        ]

        loso_part = loso_df[
            np.isclose(
                loso_df[
                    "band10_weight"
                ],
                weight,
            )
        ]

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

        overall_row = overall_df[
            np.isclose(
                overall_df[
                    "band10_weight"
                ],
                weight,
            )
        ].iloc[0]

        if (
            overall_row[
                "total_change_vs_base"
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
        ):
            status = (
                "ROBUST_PROMISING"
            )

        elif (
            overall_row[
                "total_change_vs_base"
            ] > 0
        ):
            status = (
                "POSITIVE_BUT_NOT_ROBUST"
            )

        else:
            status = (
                "NO_OOS_IMPROVEMENT"
            )

        robustness_rows.append(
            {
                "band10_weight":
                    float(
                        weight
                    ),

                "mode":
                    mode_name(
                        weight
                    ),

                "status":
                    status,

                "total_change_vs_base":
                    float(
                        overall_row[
                            "total_change_vs_base"
                        ]
                    ),

                "avg_change_vs_base":
                    float(
                        overall_row[
                            "avg_change_vs_base"
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

                "production_adopted":
                    False,
            }
        )

    robustness_df = pd.DataFrame(
        robustness_rows
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_header(
        "OVERALL WEIGHT GRID"
    )

    print(
        overall_df.sort_values(
            "band10_weight"
        ).to_string(
            index=False
        )
    )

    print_header(
        "PAIRED DAILY VS W=0"
    )

    print(
        paired_summary_df.sort_values(
            "band10_weight"
        ).to_string(
            index=False
        )
    )

    print_header(
        "ROBUSTNESS ASSESSMENT"
    )

    print(
        robustness_df.sort_values(
            "band10_weight"
        ).to_string(
            index=False
        )
    )

    print_header(
        "BY SPLIT"
    )

    print(
        split_df.sort_values(
            [
                "band10_weight",
                "split",
            ]
        ).to_string(
            index=False
        )
    )

    # Exploratory top performer is shown but is NOT adopted.
    best_row = overall_df.sort_values(
        [
            "total_diff",
            "min_split_avg",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[0]

    print_header(
        "EXPLORATORY BEST TOTAL DIFF - NOT AN ADOPTION"
    )

    print(
        f"weight               : "
        f"{best_row['band10_weight']:.3f}"
    )
    print(
        f"total_diff           : "
        f"{best_row['total_diff']:+.1f}"
    )
    print(
        f"change_vs_base       : "
        f"{best_row['total_change_vs_base']:+.1f}"
    )
    print()
    print(
        "The highest in-sample/OOS-grid result is NOT automatically selected."
    )
    print(
        "Production adoption requires a robust pattern and more independent history."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "57_band10_weight_daily.csv":
            daily_df,

        "57_band10_weight_by_split.csv":
            split_df,

        "57_band10_weight_overall.csv":
            overall_df,

        "57_band10_weight_paired_daily.csv":
            paired_df,

        "57_band10_weight_paired_summary.csv":
            paired_summary_df,

        "57_band10_weight_leave_one_day_out.csv":
            loo_df,

        "57_band10_weight_leave_one_split_out.csv":
            loso_df,

        "57_band10_weight_robustness.csv":
            robustness_df,
    }

    print_header(
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
        "57 number-band-10 weight robustness test complete."
    )
    print(
        "No production model change has been made."
    )


if __name__ == "__main__":
    main()
