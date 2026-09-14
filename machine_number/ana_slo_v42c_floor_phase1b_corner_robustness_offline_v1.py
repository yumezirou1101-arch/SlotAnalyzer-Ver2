from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd


# ============================================================
# Floor Phase1B
# Corner robustness / confounding analysis
#
# OFFLINE RESEARCH ONLY
# - NO production write
# - NO formal write
# - NO Forward change
# - NO Champion change
# - NO weight tuning
# - NO auto promotion
# - NO corner / edge penalty introduction
# - NO neighbor replacement
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
MACHINE_DIR = PROJECT_ROOT / "machine_number"
DATA_DIR = PROJECT_ROOT / "data" / "maruhan_maebashi" / "machine_number"

PHASE1A_SOURCE = (
    MACHINE_DIR
    / "ana_slo_v42c_floor_phase1a_static_layout_offline_v1.py"
)

STAGE1_SOURCE = (
    MACHINE_DIR
    / "ana_slo_v42c_prevday_dependency_offline_v1.py"
)

STAGE2A_SOURCE = (
    MACHINE_DIR
    / "ana_slo_v42c_floor_stage2a_neighbor_offline_v1.py"
)

OUTPUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "offline_floor_phase1b_corner_robustness_v1"
)

EXPECTED_MACHINES = 514
EXPECTED_TARGET_DAYS = 55
EXPECTED_RECT_MACHINES = 497
EXPECTED_CORNER_MACHINES = 66
EXPECTED_EDGE_MACHINES = 70
EXPECTED_RECT_ISLANDS = 19
EXPECTED_ISLAND_SIDE_GROUPS_PER_DAY = 33
EXPECTED_SEGMENTS = {
    "A-TYPE",
    "JUGGLER",
    "NON-JUGGLER",
}
EXPECTED_WEIGHT_FINGERPRINT = "a1eaf45d71ded209"

# Pre-checked Phase1B coverage invariants for the fixed 55-day panel.
EXPECTED_MACHINE_NAME_MATCHED_CORNER_ROWS = 3438
EXPECTED_MACHINE_NAME_MATCHED_TOTAL_ROWS = 21003
EXPECTED_MACHINE_NAME_MATCHED_GROUPS = 2276
EXPECTED_MACHINE_NAME_ALL_GROUPS = 4550
EXPECTED_WITHIN_ISLAND_GROUPS = 1815

TIME_BLOCK_COUNT = 5


# ============================================================
# General helpers
# ============================================================

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        c
        for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing columns: {missing}"
        )


def safe_output_guard() -> None:
    expected_parent = (
        DATA_DIR
        / "analysis_31days_deep"
    ).resolve()

    actual_parent = OUTPUT_DIR.parent.resolve()

    if actual_parent != expected_parent:
        raise RuntimeError(
            "OUTPUT_DIR parent safety check failed: "
            f"{actual_parent}"
        )

    if (
        OUTPUT_DIR.name
        != "offline_floor_phase1b_corner_robustness_v1"
    ):
        raise RuntimeError(
            "OUTPUT_DIR name safety check failed."
        )


def summarize_daily_effect(
    daily: pd.DataFrame,
    effect_col: str,
    label: str,
    phase1a,
) -> dict:
    if daily.empty:
        raise RuntimeError(
            f"No daily rows for summary: {label}"
        )

    values = pd.to_numeric(
        daily[effect_col],
        errors="raise",
    )

    if values.isna().any():
        raise RuntimeError(
            f"NaN daily effect: {label}"
        )

    ci_low, ci_high = (
        phase1a.bootstrap_mean_ci95(
            values.to_numpy(),
            label=label,
        )
    )

    return {
        "observed_days":
            int(len(values)),
        "mean_effect":
            float(values.mean()),
        "median_effect":
            float(values.median()),
        "positive_days":
            int(values.gt(0).sum()),
        "negative_days":
            int(values.lt(0).sum()),
        "tie_days":
            int(values.eq(0).sum()),
        "ci95_low":
            ci_low,
        "ci95_high":
            ci_high,
        "direction":
            phase1a.sign_label(
                float(values.mean())
            ),
    }


def binary_row(
    panel: pd.DataFrame,
    feature: str,
    period: str,
    phase1a,
) -> dict:
    result = phase1a.binary_contrast(
        panel,
        feature,
        "RECT",
        period,
    )

    if len(result) != 1:
        raise RuntimeError(
            f"binary_contrast row mismatch: {feature} {period}"
        )

    return result.iloc[0].to_dict()


# ============================================================
# Runtime loading
# ============================================================

def load_runtime():
    # Required because the official A-TYPE classifier imports
    # a sibling module from machine_number.
    machine_dir_text = str(
        MACHINE_DIR.resolve()
    )

    if machine_dir_text not in sys.path:
        sys.path.insert(
            0,
            machine_dir_text,
        )

    phase1a = load_module(
        "floor_phase1b_phase1a",
        PHASE1A_SOURCE,
    )

    stage1 = load_module(
        "floor_phase1b_stage1",
        STAGE1_SOURCE,
    )

    stage2a = load_module(
        "floor_phase1b_stage2a",
        STAGE2A_SOURCE,
    )

    return (
        phase1a,
        stage1,
        stage2a,
    )


# ============================================================
# Analysis panel
# ============================================================

def build_phase1b_panel(
    phase1a,
    stage1,
    stage2a,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[pd.Timestamp],
]:
    floor = phase1a.load_floor_map()

    physical = (
        stage2a.build_physical_map(
            floor
        )
    )

    floor_attr = (
        phase1a.build_floor_attributes(
            floor,
            physical,
        )
    )

    floor_static_qa = (
        phase1a.floor_static_qa(
            floor_attr
        )
    )

    panel, stage1_daily_qa = (
        stage1.load_daily_panel()
    )

    require_columns(
        panel,
        [
            "date",
            "machine_no",
            "machine_name",
            "diff",
        ],
        "Stage1 panel",
    )

    panel = panel.copy()

    panel["date"] = pd.to_datetime(
        panel["date"]
    )

    panel["machine_no"] = pd.to_numeric(
        panel["machine_no"],
        errors="raise",
    ).astype(int)

    panel["diff"] = pd.to_numeric(
        panel["diff"],
        errors="coerce",
    )

    if panel["diff"].isna().any():
        raise RuntimeError(
            "Stage1 panel contains diff NaN."
        )

    target_dates = sorted(
        pd.Timestamp(v)
        for v in stage1.ab_target_dates(
            panel
        )
    )

    if (
        len(target_dates)
        != EXPECTED_TARGET_DAYS
    ):
        raise RuntimeError(
            "Target-day count mismatch: "
            f"{len(target_dates)} "
            f"expected={EXPECTED_TARGET_DAYS}"
        )

    x = (
        panel.loc[
            panel["date"].isin(
                target_dates
            ),
            [
                "date",
                "machine_no",
                "machine_name",
                "diff",
            ],
        ]
        .copy()
        .rename(
            columns={
                "date": "target_date",
                "diff": "actual_diff",
            }
        )
    )

    expected_rows = (
        EXPECTED_TARGET_DAYS
        * EXPECTED_MACHINES
    )

    if len(x) != expected_rows:
        raise RuntimeError(
            "Target panel row mismatch: "
            f"{len(x)} "
            f"expected={expected_rows}"
        )

    daily = (
        x.groupby(
            "target_date",
            sort=True,
        )
        .agg(
            rows=("machine_no", "size"),
            unique_machine_no=(
                "machine_no",
                "nunique",
            ),
            machine_name_missing=(
                "machine_name",
                lambda s:
                    int(s.isna().sum()),
            ),
            actual_diff_missing=(
                "actual_diff",
                lambda s:
                    int(s.isna().sum()),
            ),
        )
        .reset_index()
    )

    if not daily["rows"].eq(
        EXPECTED_MACHINES
    ).all():
        raise RuntimeError(
            "Phase1B daily row QA failed."
        )

    if not daily[
        "unique_machine_no"
    ].eq(
        EXPECTED_MACHINES
    ).all():
        raise RuntimeError(
            "Phase1B daily machine uniqueness failed."
        )

    x = x.merge(
        floor_attr,
        on="machine_no",
        how="left",
        validate="many_to_one",
    )

    if len(x) != expected_rows:
        raise RuntimeError(
            "Floor join row mismatch."
        )

    floor_required = [
        "island_id",
        "side",
        "slot_index",
        "is_rect",
        "is_ring",
        "is_corner",
        "is_edge",
        "is_gap_adjacent",
        "is_special_3",
        "structure_class",
    ]

    if x[floor_required].isna().any().any():
        bad = (
            x.loc[
                x[floor_required]
                .isna()
                .any(axis=1),
                [
                    "target_date",
                    "machine_no",
                ],
            ]
            .head(20)
        )

        raise RuntimeError(
            "Floor join missing attributes:\n"
            + bad.to_string(
                index=False
            )
        )

    classify_a_type, is_juggler = (
        stage1.load_classifiers()
    )

    x["segment"] = (
        x["machine_name"]
        .apply(
            lambda name:
                stage1.classify_three_way(
                    name,
                    classify_a_type,
                    is_juggler,
                )
        )
    )

    segments = set(
        x["segment"].unique()
    )

    if segments != EXPECTED_SEGMENTS:
        raise RuntimeError(
            "Segment set mismatch: "
            f"{sorted(segments)}"
        )

    x = (
        x.sort_values(
            [
                "target_date",
                "machine_no",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        x,
        floor_attr,
        floor_static_qa,
        target_dates,
    )


# ============================================================
# 1. Corner segment summary
# ============================================================

def build_corner_segment_summary(
    rect: pd.DataFrame,
    phase1a,
) -> pd.DataFrame:
    rows = []

    all_row = binary_row(
        rect,
        "is_corner",
        "ALL",
        phase1a,
    )

    all_row["segment"] = "ALL"
    rows.append(all_row)

    for segment in sorted(
        EXPECTED_SEGMENTS
    ):
        g = rect.loc[
            rect["segment"].eq(
                segment
            )
        ].copy()

        row = binary_row(
            g,
            "is_corner",
            f"SEGMENT_{segment}",
            phase1a,
        )

        row["segment"] = segment
        rows.append(row)

    out = pd.DataFrame(
        rows
    )

    first = [
        "segment",
        "period",
        "feature",
        "scope",
    ]

    return out[
        first
        + [
            c
            for c in out.columns
            if c not in first
        ]
    ]


# ============================================================
# 2-3. Day / segment adjusted
# ============================================================

def build_day_segment_adjusted(
    rect: pd.DataFrame,
    phase1a,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    grouped = (
        rect.groupby(
            [
                "target_date",
                "segment",
            ],
            sort=True,
        )
        .apply(
            lambda g:
                pd.Series(
                    {
                        "corner_n":
                            int(
                                g[
                                    "is_corner"
                                ].sum()
                            ),
                        "noncorner_n":
                            int(
                                (
                                    ~g[
                                        "is_corner"
                                    ]
                                ).sum()
                            ),
                        "segment_rows":
                            int(len(g)),
                        "corner_mean_diff":
                            float(
                                g.loc[
                                    g[
                                        "is_corner"
                                    ],
                                    "actual_diff",
                                ].mean()
                            ),
                        "noncorner_mean_diff":
                            float(
                                g.loc[
                                    ~g[
                                        "is_corner"
                                    ],
                                    "actual_diff",
                                ].mean()
                            ),
                    }
                ),
                include_groups=False,
            )
        .reset_index()
    )

    if (
        grouped["corner_n"]
        .le(0)
        .any()
        or
        grouped["noncorner_n"]
        .le(0)
        .any()
    ):
        raise RuntimeError(
            "Segment adjusted eligibility failed."
        )

    grouped["segment_difference"] = (
        grouped["corner_mean_diff"]
        -
        grouped["noncorner_mean_diff"]
    )

    daily_total = (
        grouped.groupby(
            "target_date"
        )
        ["segment_rows"]
        .transform("sum")
    )

    grouped["segment_weight"] = (
        grouped["segment_rows"]
        /
        daily_total
    )

    grouped[
        "weighted_segment_difference"
    ] = (
        grouped[
            "segment_difference"
        ]
        *
        grouped[
            "segment_weight"
        ]
    )

    daily_adjusted = (
        grouped.groupby(
            "target_date",
            sort=True,
        )
        .agg(
            segment_count=(
                "segment",
                "nunique",
            ),
            adjusted_effect=(
                "weighted_segment_difference",
                "sum",
            ),
        )
        .reset_index()
    )

    raw_daily = (
        rect.groupby(
            "target_date",
            sort=True,
        )
        .apply(
            lambda g:
                float(
                    g.loc[
                        g["is_corner"],
                        "actual_diff",
                    ].mean()
                    -
                    g.loc[
                        ~g["is_corner"],
                        "actual_diff",
                    ].mean()
                ),
            include_groups=False,
        )
        .rename(
            "raw_corner_effect"
        )
        .reset_index()
    )

    daily_adjusted = (
        daily_adjusted.merge(
            raw_daily,
            on="target_date",
            how="left",
            validate="one_to_one",
        )
    )

    daily_adjusted[
        "adjusted_minus_raw"
    ] = (
        daily_adjusted[
            "adjusted_effect"
        ]
        -
        daily_adjusted[
            "raw_corner_effect"
        ]
    )

    grouped = grouped.merge(
        daily_adjusted[
            [
                "target_date",
                "adjusted_effect",
                "raw_corner_effect",
                "adjusted_minus_raw",
            ]
        ],
        on="target_date",
        how="left",
        validate="many_to_one",
    )

    adjusted_summary = (
        summarize_daily_effect(
            daily_adjusted,
            "adjusted_effect",
            "phase1b_segment_adjusted",
            phase1a,
        )
    )

    raw_summary = (
        summarize_daily_effect(
            daily_adjusted,
            "raw_corner_effect",
            "phase1b_segment_raw",
            phase1a,
        )
    )

    delta_summary = (
        summarize_daily_effect(
            daily_adjusted,
            "adjusted_minus_raw",
            "phase1b_segment_adjusted_minus_raw",
            phase1a,
        )
    )

    summary = pd.DataFrame(
        [
            {
                "method":
                    "DAY_SEGMENT_STANDARDIZED",
                "segment_weighting":
                    "daily pooled RECT segment share",
                "day_segment_groups":
                    len(grouped),
                "eligible_day_segment_groups":
                    int(
                        (
                            grouped["corner_n"].gt(0)
                            &
                            grouped[
                                "noncorner_n"
                            ].gt(0)
                        ).sum()
                    ),
                "raw_mean_effect":
                    raw_summary[
                        "mean_effect"
                    ],
                "adjusted_mean_effect":
                    adjusted_summary[
                        "mean_effect"
                    ],
                "adjusted_minus_raw_mean":
                    delta_summary[
                        "mean_effect"
                    ],
                "adjusted_median_effect":
                    adjusted_summary[
                        "median_effect"
                    ],
                "adjusted_positive_days":
                    adjusted_summary[
                        "positive_days"
                    ],
                "adjusted_negative_days":
                    adjusted_summary[
                        "negative_days"
                    ],
                "adjusted_tie_days":
                    adjusted_summary[
                        "tie_days"
                    ],
                "adjusted_ci95_low":
                    adjusted_summary[
                        "ci95_low"
                    ],
                "adjusted_ci95_high":
                    adjusted_summary[
                        "ci95_high"
                    ],
                "direction":
                    adjusted_summary[
                        "direction"
                    ],
            }
        ]
    )

    return (
        grouped,
        summary,
    )


# ============================================================
# 4-5. Exact machine-name matched
# ============================================================

def build_machine_name_matched(
    rect: pd.DataFrame,
    phase1a,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    grouped = []

    all_groups = 0

    for (
        target_date,
        machine_name,
    ), g in rect.groupby(
        [
            "target_date",
            "machine_name",
        ],
        sort=True,
    ):
        all_groups += 1

        corner = g.loc[
            g["is_corner"]
        ]

        noncorner = g.loc[
            ~g["is_corner"]
        ]

        if (
            corner.empty
            or noncorner.empty
        ):
            continue

        grouped.append(
            {
                "target_date":
                    target_date,
                "machine_name":
                    machine_name,
                "corner_n":
                    len(corner),
                "noncorner_n":
                    len(noncorner),
                "group_rows":
                    len(g),
                "corner_mean_diff":
                    float(
                        corner[
                            "actual_diff"
                        ].mean()
                    ),
                "noncorner_mean_diff":
                    float(
                        noncorner[
                            "actual_diff"
                        ].mean()
                    ),
                "difference":
                    float(
                        corner[
                            "actual_diff"
                        ].mean()
                        -
                        noncorner[
                            "actual_diff"
                        ].mean()
                    ),
            }
        )

    matched = pd.DataFrame(
        grouped
    )

    if matched.empty:
        raise RuntimeError(
            "No machine-name matched groups."
        )

    daily_rows = (
        matched.groupby(
            "target_date"
        )
        ["group_rows"]
        .transform("sum")
    )

    matched["row_weight"] = (
        matched["group_rows"]
        /
        daily_rows
    )

    matched[
        "row_weighted_difference"
    ] = (
        matched["difference"]
        *
        matched["row_weight"]
    )

    daily = (
        matched.groupby(
            "target_date",
            sort=True,
        )
        .agg(
            eligible_machine_names=(
                "machine_name",
                "nunique",
            ),
            equal_weight_effect=(
                "difference",
                "mean",
            ),
            row_weighted_effect=(
                "row_weighted_difference",
                "sum",
            ),
            matched_rows=(
                "group_rows",
                "sum",
            ),
        )
        .reset_index()
    )

    matched = matched.merge(
        daily[
            [
                "target_date",
                "eligible_machine_names",
                "equal_weight_effect",
                "row_weighted_effect",
                "matched_rows",
            ]
        ],
        on="target_date",
        how="left",
        validate="many_to_one",
    )

    eligible_keys = (
        matched[
            [
                "target_date",
                "machine_name",
            ]
        ]
        .drop_duplicates()
        .assign(
            eligible=True
        )
    )

    coverage_panel = rect.merge(
        eligible_keys,
        on=[
            "target_date",
            "machine_name",
        ],
        how="left",
        validate="many_to_one",
    )

    coverage_panel["eligible"] = (
        coverage_panel[
            "eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    corner = coverage_panel.loc[
        coverage_panel[
            "is_corner"
        ]
    ]

    noncorner = coverage_panel.loc[
        ~coverage_panel[
            "is_corner"
        ]
    ]

    equal_summary = summarize_daily_effect(
        daily,
        "equal_weight_effect",
        "phase1b_machine_name_equal_weight",
        phase1a,
    )

    row_summary = summarize_daily_effect(
        daily,
        "row_weighted_effect",
        "phase1b_machine_name_row_weight",
        phase1a,
    )

    summary = pd.DataFrame(
        [
            {
                "method":
                    "EXACT_MACHINE_NAME_MATCHED",
                "all_day_machine_name_groups":
                    all_groups,
                "eligible_day_machine_name_groups":
                    len(matched),
                "group_coverage":
                    len(matched)
                    / all_groups,
                "corner_rows_total":
                    len(corner),
                "corner_rows_matched":
                    int(
                        corner[
                            "eligible"
                        ].sum()
                    ),
                "corner_row_coverage":
                    float(
                        corner[
                            "eligible"
                        ].mean()
                    ),
                "noncorner_rows_total":
                    len(noncorner),
                "noncorner_rows_matched":
                    int(
                        noncorner[
                            "eligible"
                        ].sum()
                    ),
                "noncorner_row_coverage":
                    float(
                        noncorner[
                            "eligible"
                        ].mean()
                    ),
                "matched_rows_total":
                    int(
                        coverage_panel[
                            "eligible"
                        ].sum()
                    ),
                "observed_days":
                    equal_summary[
                        "observed_days"
                    ],
                "equal_weight_mean_effect":
                    equal_summary[
                        "mean_effect"
                    ],
                "equal_weight_median_effect":
                    equal_summary[
                        "median_effect"
                    ],
                "equal_weight_ci95_low":
                    equal_summary[
                        "ci95_low"
                    ],
                "equal_weight_ci95_high":
                    equal_summary[
                        "ci95_high"
                    ],
                "equal_weight_direction":
                    equal_summary[
                        "direction"
                    ],
                "row_weighted_mean_effect":
                    row_summary[
                        "mean_effect"
                    ],
                "row_weighted_ci95_low":
                    row_summary[
                        "ci95_low"
                    ],
                "row_weighted_ci95_high":
                    row_summary[
                        "ci95_high"
                    ],
            }
        ]
    )

    return (
        matched,
        summary,
    )


# ============================================================
# 6-7. Within-island / side
# ============================================================

def build_within_island(
    rect: pd.DataFrame,
    phase1a,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rows = []

    for (
        target_date,
        island_id,
        side,
    ), g in rect.groupby(
        [
            "target_date",
            "island_id",
            "side",
        ],
        sort=True,
    ):
        corner = g.loc[
            g["is_corner"]
        ]

        noncorner = g.loc[
            ~g["is_corner"]
        ]

        if (
            corner.empty
            or noncorner.empty
        ):
            continue

        rows.append(
            {
                "target_date":
                    target_date,
                "island_id":
                    island_id,
                "side":
                    side,
                "corner_n":
                    len(corner),
                "noncorner_n":
                    len(noncorner),
                "group_rows":
                    len(g),
                "corner_mean_diff":
                    float(
                        corner[
                            "actual_diff"
                        ].mean()
                    ),
                "noncorner_mean_diff":
                    float(
                        noncorner[
                            "actual_diff"
                        ].mean()
                    ),
                "difference":
                    float(
                        corner[
                            "actual_diff"
                        ].mean()
                        -
                        noncorner[
                            "actual_diff"
                        ].mean()
                    ),
            }
        )

    within = pd.DataFrame(
        rows
    )

    if within.empty:
        raise RuntimeError(
            "No within-island eligible groups."
        )

    daily_rows = (
        within.groupby(
            "target_date"
        )
        ["group_rows"]
        .transform("sum")
    )

    within["row_weight"] = (
        within["group_rows"]
        /
        daily_rows
    )

    within[
        "row_weighted_difference"
    ] = (
        within[
            "difference"
        ]
        *
        within[
            "row_weight"
        ]
    )

    daily = (
        within.groupby(
            "target_date",
            sort=True,
        )
        .agg(
            eligible_groups=(
                "island_id",
                "size",
            ),
            equal_weight_effect=(
                "difference",
                "mean",
            ),
            row_weighted_effect=(
                "row_weighted_difference",
                "sum",
            ),
            eligible_rows=(
                "group_rows",
                "sum",
            ),
        )
        .reset_index()
    )

    within = within.merge(
        daily,
        on="target_date",
        how="left",
        validate="many_to_one",
    )

    equal_summary = summarize_daily_effect(
        daily,
        "equal_weight_effect",
        "phase1b_within_island_equal",
        phase1a,
    )

    row_summary = summarize_daily_effect(
        daily,
        "row_weighted_effect",
        "phase1b_within_island_row",
        phase1a,
    )

    summary = pd.DataFrame(
        [
            {
                "method":
                    "WITHIN_ISLAND_SIDE",
                "eligible_groups":
                    len(within),
                "observed_days":
                    equal_summary[
                        "observed_days"
                    ],
                "groups_per_day_min":
                    int(
                        daily[
                            "eligible_groups"
                        ].min()
                    ),
                "groups_per_day_max":
                    int(
                        daily[
                            "eligible_groups"
                        ].max()
                    ),
                "rows_per_day_min":
                    int(
                        daily[
                            "eligible_rows"
                        ].min()
                    ),
                "rows_per_day_max":
                    int(
                        daily[
                            "eligible_rows"
                        ].max()
                    ),
                "equal_weight_mean_effect":
                    equal_summary[
                        "mean_effect"
                    ],
                "equal_weight_median_effect":
                    equal_summary[
                        "median_effect"
                    ],
                "equal_weight_ci95_low":
                    equal_summary[
                        "ci95_low"
                    ],
                "equal_weight_ci95_high":
                    equal_summary[
                        "ci95_high"
                    ],
                "equal_weight_direction":
                    equal_summary[
                        "direction"
                    ],
                "row_weighted_mean_effect":
                    row_summary[
                        "mean_effect"
                    ],
                "row_weighted_ci95_low":
                    row_summary[
                        "ci95_low"
                    ],
                "row_weighted_ci95_high":
                    row_summary[
                        "ci95_high"
                    ],
            }
        ]
    )

    return (
        within,
        summary,
    )


# ============================================================
# 8. Leave-one-island-out
# ============================================================

def build_leave_one_island_out(
    within: pd.DataFrame,
    phase1a,
) -> pd.DataFrame:
    islands = sorted(
        within["island_id"]
        .unique()
        .tolist()
    )

    rows = []

    for island_id in islands:
        x = within.loc[
            ~within[
                "island_id"
            ].eq(
                island_id
            )
        ].copy()

        daily = (
            x.groupby(
                "target_date",
                sort=True,
            )
            .agg(
                eligible_groups=(
                    "island_id",
                    "size",
                ),
                effect=(
                    "difference",
                    "mean",
                ),
            )
            .reset_index()
        )

        summary = summarize_daily_effect(
            daily,
            "effect",
            (
                "phase1b_loio_"
                + str(island_id)
            ),
            phase1a,
        )

        rows.append(
            {
                "excluded_island":
                    island_id,
                "remaining_days":
                    daily[
                        "target_date"
                    ].nunique(),
                "remaining_groups":
                    len(x),
                "groups_per_day_min":
                    int(
                        daily[
                            "eligible_groups"
                        ].min()
                    ),
                "groups_per_day_max":
                    int(
                        daily[
                            "eligible_groups"
                        ].max()
                    ),
                **summary,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 9. Sensitivity exclusions
# ============================================================

def build_sensitivity_exclusions(
    rect: pd.DataFrame,
    phase1a,
) -> pd.DataFrame:
    scenarios = [
        (
            "NONE",
            pd.Series(
                True,
                index=rect.index,
            ),
        ),
        (
            "EXCLUDE_GAP_ADJACENT",
            ~rect[
                "is_gap_adjacent"
            ],
        ),
        (
            "EXCLUDE_SPECIAL_3",
            ~rect[
                "is_special_3"
            ],
        ),
        (
            "EXCLUDE_GAP_AND_SPECIAL_3",
            (
                ~rect[
                    "is_gap_adjacent"
                ]
                &
                ~rect[
                    "is_special_3"
                ]
            ),
        ),
        (
            "EXCLUDE_CORNER_EDGE_MISMATCH",
            rect[
                "is_corner"
            ].eq(
                rect[
                    "is_edge"
                ]
            ),
        ),
    ]

    for machine_no in sorted(
        [
            842,
            843,
            1014,
            1015,
        ]
    ):
        scenarios.append(
            (
                f"EXCLUDE_MACHINE_{machine_no}",
                ~rect[
                    "machine_no"
                ].eq(
                    machine_no
                ),
            )
        )

    rows = []

    for name, mask in scenarios:
        x = rect.loc[
            mask
        ].copy()

        result = binary_row(
            x,
            "is_corner",
            name,
            phase1a,
        )

        result[
            "scenario"
        ] = name

        result[
            "remaining_rows"
        ] = len(x)

        result[
            "remaining_machines"
        ] = (
            x["machine_no"]
            .nunique()
        )

        result[
            "excluded_machine_count"
        ] = (
            EXPECTED_RECT_MACHINES
            -
            x[
                "machine_no"
            ].nunique()
        )

        rows.append(
            result
        )

    out = pd.DataFrame(
        rows
    )

    first = [
        "scenario",
        "remaining_rows",
        "remaining_machines",
        "excluded_machine_count",
    ]

    return out[
        first
        + [
            c
            for c in out.columns
            if c not in first
        ]
    ]


# ============================================================
# 10. Time blocks
# ============================================================

def build_time_blocks(
    rect: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    stage1,
    phase1a,
) -> pd.DataFrame:
    rows = []

    def append_binary(
        label: str,
        dates: list[pd.Timestamp],
    ) -> None:
        x = rect.loc[
            rect["target_date"]
            .isin(
                dates
            )
        ].copy()

        result = binary_row(
            x,
            "is_corner",
            label,
            phase1a,
        )

        result.update(
            {
                "method":
                    "BINARY_CONTRAST",
                "time_block":
                    label,
                "start_date":
                    min(dates),
                "end_date":
                    max(dates),
                "day_count":
                    len(dates),
                "wf_eligible_days":
                    np.nan,
                "wf_nonzero_signal_days":
                    np.nan,
                "wf_sign_match_rate":
                    np.nan,
                "wf_signal_future_corr":
                    np.nan,
                "wf_mean_future_excess":
                    np.nan,
            }
        )

        rows.append(
            result
        )

    append_binary(
        "ALL",
        target_dates,
    )

    split_index = (
        len(target_dates) + 1
    ) // 2

    dev_dates = (
        target_dates[
            :split_index
        ]
    )

    later_dates = (
        target_dates[
            split_index:
        ]
    )

    append_binary(
        "DEV",
        dev_dates,
    )

    append_binary(
        "LATER",
        later_dates,
    )

    blocks = np.array_split(
        np.asarray(
            target_dates,
            dtype=object,
        ),
        TIME_BLOCK_COUNT,
    )

    for idx, block in enumerate(
        blocks,
        start=1,
    ):
        dates = [
            pd.Timestamp(v)
            for v in block.tolist()
        ]

        append_binary(
            f"BLOCK_{idx}",
            dates,
        )

    wf = phase1a.walk_forward_summary(
        rect,
        "is_corner",
        "RECT",
    )

    wf_true = wf.loc[
        wf["category"]
        .astype(str)
        .eq("True")
    ].copy()

    if len(wf_true) != 1:
        raise RuntimeError(
            "Walk-forward corner=True row mismatch."
        )

    wf_row = wf_true.iloc[0]

    rows.append(
        {
            "method":
                "WALK_FORWARD",
            "time_block":
                "WALK_FORWARD_CORNER_TRUE",
            "start_date":
                target_dates[0],
            "end_date":
                target_dates[-1],
            "day_count":
                len(target_dates),
            "period":
                "WALK_FORWARD",
            "feature":
                "is_corner",
            "scope":
                "RECT",
            "true_machine_n":
                np.nan,
            "false_machine_n":
                np.nan,
            "observed_days":
                np.nan,
            "true_mean_diff":
                np.nan,
            "false_mean_diff":
                np.nan,
            "true_minus_false_mean":
                np.nan,
            "true_minus_false_median":
                np.nan,
            "true_better_days":
                np.nan,
            "false_better_days":
                np.nan,
            "tie_days":
                np.nan,
            "paired_day_ci95_low":
                np.nan,
            "paired_day_ci95_high":
                np.nan,
            "direction":
                np.nan,
            "wf_eligible_days":
                wf_row[
                    "eligible_days"
                ],
            "wf_nonzero_signal_days":
                wf_row[
                    "nonzero_signal_days"
                ],
            "wf_sign_match_rate":
                wf_row[
                    "sign_match_rate"
                ],
            "wf_signal_future_corr":
                wf_row[
                    "signal_future_corr"
                ],
            "wf_mean_future_excess":
                wf_row[
                    "mean_future_excess"
                ],
        }
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 11. Edge sensitivity
# ============================================================

def build_edge_sensitivity(
    rect: pd.DataFrame,
    floor_attr: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    phase1a,
) -> pd.DataFrame:
    static_rect = floor_attr.loc[
        floor_attr["is_rect"]
    ].copy()

    overlap = int(
        (
            static_rect[
                "is_corner"
            ]
            &
            static_rect[
                "is_edge"
            ]
        ).sum()
    )

    corner_only = int(
        (
            static_rect[
                "is_corner"
            ]
            &
            ~static_rect[
                "is_edge"
            ]
        ).sum()
    )

    edge_only = int(
        (
            ~static_rect[
                "is_corner"
            ]
            &
            static_rect[
                "is_edge"
            ]
        ).sum()
    )

    mismatch = int(
        static_rect[
            "is_corner"
        ]
        .ne(
            static_rect[
                "is_edge"
            ]
        )
        .sum()
    )

    split_index = (
        len(target_dates) + 1
    ) // 2

    periods = [
        (
            "ALL",
            target_dates,
        ),
        (
            "DEV",
            target_dates[
                :split_index
            ],
        ),
        (
            "LATER",
            target_dates[
                split_index:
            ],
        ),
    ]

    rows = []

    for period, dates in periods:
        x = rect.loc[
            rect[
                "target_date"
            ].isin(
                dates
            )
        ].copy()

        corner = binary_row(
            x,
            "is_corner",
            period,
            phase1a,
        )

        edge = binary_row(
            x,
            "is_edge",
            period,
            phase1a,
        )

        aligned = x.loc[
            x["is_corner"]
            .eq(
                x["is_edge"]
            )
        ].copy()

        aligned_corner = binary_row(
            aligned,
            "is_corner",
            (
                period
                + "_ALIGNED"
            ),
            phase1a,
        )

        aligned_edge = binary_row(
            aligned,
            "is_edge",
            (
                period
                + "_ALIGNED"
            ),
            phase1a,
        )

        rows.append(
            {
                "period":
                    period,
                "corner_machine_n":
                    corner[
                        "true_machine_n"
                    ],
                "edge_machine_n":
                    edge[
                        "true_machine_n"
                    ],
                "corner_edge_overlap_machine_n":
                    overlap,
                "corner_only_machine_n":
                    corner_only,
                "edge_only_machine_n":
                    edge_only,
                "symmetric_difference_machine_n":
                    mismatch,
                "corner_effect":
                    corner[
                        "true_minus_false_mean"
                    ],
                "corner_ci95_low":
                    corner[
                        "paired_day_ci95_low"
                    ],
                "corner_ci95_high":
                    corner[
                        "paired_day_ci95_high"
                    ],
                "edge_effect":
                    edge[
                        "true_minus_false_mean"
                    ],
                "edge_ci95_low":
                    edge[
                        "paired_day_ci95_low"
                    ],
                "edge_ci95_high":
                    edge[
                        "paired_day_ci95_high"
                    ],
                "corner_minus_edge_effect":
                    (
                        corner[
                            "true_minus_false_mean"
                        ]
                        -
                        edge[
                            "true_minus_false_mean"
                        ]
                    ),
                "aligned_remaining_machines":
                    aligned[
                        "machine_no"
                    ].nunique(),
                "aligned_corner_effect":
                    aligned_corner[
                        "true_minus_false_mean"
                    ],
                "aligned_edge_effect":
                    aligned_edge[
                        "true_minus_false_mean"
                    ],
                "aligned_corner_minus_edge_effect":
                    (
                        aligned_corner[
                            "true_minus_false_mean"
                        ]
                        -
                        aligned_edge[
                            "true_minus_false_mean"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 12. QA
# ============================================================

def build_phase1b_qa(
    panel: pd.DataFrame,
    rect: pd.DataFrame,
    floor_attr: pd.DataFrame,
    floor_static_qa: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    day_segment_adjusted: pd.DataFrame,
    machine_name_matched: pd.DataFrame,
    machine_name_summary: pd.DataFrame,
    within_island: pd.DataFrame,
    loio: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add(
        item: str,
        actual,
        expected,
        passed: bool,
    ) -> None:
        rows.append(
            {
                "qa_item":
                    item,
                "actual":
                    actual,
                "expected":
                    expected,
                "pass":
                    bool(passed),
            }
        )

    add(
        "target_days",
        len(target_dates),
        EXPECTED_TARGET_DAYS,
        (
            len(target_dates)
            == EXPECTED_TARGET_DAYS
        ),
    )

    daily_rows = (
        panel.groupby(
            "target_date"
        )
        .size()
    )

    add(
        "machines_per_day",
        (
            f"{int(daily_rows.min())}"
            f"..{int(daily_rows.max())}"
        ),
        str(EXPECTED_MACHINES),
        bool(
            daily_rows.eq(
                EXPECTED_MACHINES
            ).all()
        ),
    )

    add(
        "floor_join_rows",
        len(panel),
        (
            EXPECTED_TARGET_DAYS
            * EXPECTED_MACHINES
        ),
        (
            len(panel)
            ==
            EXPECTED_TARGET_DAYS
            * EXPECTED_MACHINES
        ),
    )

    floor_attr_cols = [
        "island_id",
        "side",
        "slot_index",
        "is_rect",
        "is_corner",
        "is_edge",
    ]

    add(
        "floor_attribute_missing",
        int(
            panel[
                floor_attr_cols
            ]
            .isna()
            .sum()
            .sum()
        ),
        0,
        not panel[
            floor_attr_cols
        ].isna().any().any(),
    )

    add(
        "floor_machine_no_duplicates",
        int(
            floor_attr[
                "machine_no"
            ].duplicated().sum()
        ),
        0,
        not floor_attr[
            "machine_no"
        ].duplicated().any(),
    )

    add(
        "corner_machine_count",
        int(
            floor_attr[
                "is_corner"
            ].sum()
        ),
        EXPECTED_CORNER_MACHINES,
        (
            int(
                floor_attr[
                    "is_corner"
                ].sum()
            )
            ==
            EXPECTED_CORNER_MACHINES
        ),
    )

    add(
        "edge_machine_count",
        int(
            floor_attr[
                "is_edge"
            ].sum()
        ),
        EXPECTED_EDGE_MACHINES,
        (
            int(
                floor_attr[
                    "is_edge"
                ].sum()
            )
            ==
            EXPECTED_EDGE_MACHINES
        ),
    )

    add(
        "segment_labels",
        "|".join(
            sorted(
                panel[
                    "segment"
                ].unique()
            )
        ),
        "|".join(
            sorted(
                EXPECTED_SEGMENTS
            )
        ),
        (
            set(
                panel[
                    "segment"
                ].unique()
            )
            ==
            EXPECTED_SEGMENTS
        ),
    )

    segment_groups = (
        day_segment_adjusted[
            [
                "target_date",
                "segment",
            ]
        ]
        .drop_duplicates()
    )

    add(
        "segment_day_groups",
        len(segment_groups),
        (
            EXPECTED_TARGET_DAYS
            *
            len(
                EXPECTED_SEGMENTS
            )
        ),
        (
            len(segment_groups)
            ==
            EXPECTED_TARGET_DAYS
            *
            len(
                EXPECTED_SEGMENTS
            )
        ),
    )

    eligible_segment_groups = int(
        (
            day_segment_adjusted[
                "corner_n"
            ].gt(0)
            &
            day_segment_adjusted[
                "noncorner_n"
            ].gt(0)
        ).sum()
    )

    add(
        "segment_eligible_groups",
        eligible_segment_groups,
        len(segment_groups),
        (
            eligible_segment_groups
            ==
            len(segment_groups)
        ),
    )

    machine_summary = (
        machine_name_summary.iloc[0]
    )

    add(
        "machine_name_matched_groups",
        len(machine_name_matched),
        EXPECTED_MACHINE_NAME_MATCHED_GROUPS,
        (
            len(
                machine_name_matched
            )
            ==
            EXPECTED_MACHINE_NAME_MATCHED_GROUPS
        ),
    )

    add(
        "machine_name_all_groups",
        int(
            machine_summary[
                "all_day_machine_name_groups"
            ]
        ),
        EXPECTED_MACHINE_NAME_ALL_GROUPS,
        (
            int(
                machine_summary[
                    "all_day_machine_name_groups"
                ]
            )
            ==
            EXPECTED_MACHINE_NAME_ALL_GROUPS
        ),
    )

    add(
        "machine_name_matched_corner_rows",
        int(
            machine_summary[
                "corner_rows_matched"
            ]
        ),
        EXPECTED_MACHINE_NAME_MATCHED_CORNER_ROWS,
        (
            int(
                machine_summary[
                    "corner_rows_matched"
                ]
            )
            ==
            EXPECTED_MACHINE_NAME_MATCHED_CORNER_ROWS
        ),
    )

    add(
        "machine_name_matched_total_rows",
        int(
            machine_summary[
                "matched_rows_total"
            ]
        ),
        EXPECTED_MACHINE_NAME_MATCHED_TOTAL_ROWS,
        (
            int(
                machine_summary[
                    "matched_rows_total"
                ]
            )
            ==
            EXPECTED_MACHINE_NAME_MATCHED_TOTAL_ROWS
        ),
    )

    add(
        "within_island_eligible_groups",
        len(within_island),
        EXPECTED_WITHIN_ISLAND_GROUPS,
        (
            len(
                within_island
            )
            ==
            EXPECTED_WITHIN_ISLAND_GROUPS
        ),
    )

    per_day_within = (
        within_island.groupby(
            "target_date"
        )
        .size()
    )

    add(
        "within_island_groups_per_day",
        (
            f"{int(per_day_within.min())}"
            f"..{int(per_day_within.max())}"
        ),
        str(
            EXPECTED_ISLAND_SIDE_GROUPS_PER_DAY
        ),
        bool(
            per_day_within.eq(
                EXPECTED_ISLAND_SIDE_GROUPS_PER_DAY
            ).all()
        ),
    )

    rect_islands = (
        rect[
            "island_id"
        ]
        .nunique()
    )

    add(
        "loio_expected_islands",
        rect_islands,
        EXPECTED_RECT_ISLANDS,
        (
            rect_islands
            ==
            EXPECTED_RECT_ISLANDS
        ),
    )

    add(
        "loio_processed_islands",
        len(loio),
        EXPECTED_RECT_ISLANDS,
        (
            len(loio)
            ==
            EXPECTED_RECT_ISLANDS
        ),
    )

    add(
        "loio_all_55_days",
        bool(
            loio[
                "remaining_days"
            ].eq(
                EXPECTED_TARGET_DAYS
            ).all()
        ),
        True,
        bool(
            loio[
                "remaining_days"
            ].eq(
                EXPECTED_TARGET_DAYS
            ).all()
        ),
    )

    add(
        "phase1a_floor_static_qa",
        bool(
            floor_static_qa[
                "pass"
            ].all()
        ),
        True,
        bool(
            floor_static_qa[
                "pass"
            ].all()
        ),
    )

    add(
        "expected_weight_fingerprint",
        EXPECTED_WEIGHT_FINGERPRINT,
        EXPECTED_WEIGHT_FINGERPRINT,
        True,
    )

    add(
        "production_write",
        "NONE",
        "NONE",
        True,
    )

    add(
        "formal_write",
        "NONE",
        "NONE",
        True,
    )

    add(
        "champion_change",
        "NONE",
        "NONE",
        True,
    )

    add(
        "weight_tuning",
        "NONE",
        "NONE",
        True,
    )

    qa = pd.DataFrame(
        rows
    )

    if not qa[
        "pass"
    ].all():
        bad = qa.loc[
            ~qa[
                "pass"
            ]
        ]

        raise RuntimeError(
            "Phase1B QA failed:\n"
            + bad.to_string(
                index=False
            )
        )

    return qa


# ============================================================
# 13. Text summary
# ============================================================

def build_summary_text(
    segment_summary: pd.DataFrame,
    segment_adjusted_summary: pd.DataFrame,
    machine_name_summary: pd.DataFrame,
    within_island_summary: pd.DataFrame,
    loio: pd.DataFrame,
    sensitivity: pd.DataFrame,
    time_blocks: pd.DataFrame,
    edge_sensitivity: pd.DataFrame,
    qa: pd.DataFrame,
) -> str:
    raw_all = (
        segment_summary.loc[
            segment_summary[
                "segment"
            ].eq(
                "ALL"
            )
        ]
        .iloc[0]
    )

    seg = (
        segment_adjusted_summary
        .iloc[0]
    )

    machine = (
        machine_name_summary
        .iloc[0]
    )

    island = (
        within_island_summary
        .iloc[0]
    )

    loio_min = float(
        loio[
            "mean_effect"
        ].min()
    )

    loio_max = float(
        loio[
            "mean_effect"
        ].max()
    )

    lines = [
        "Floor Phase1B - corner robustness / confounding analysis",
        "",
        "OFFLINE RESEARCH ONLY",
        "production_write : NONE",
        "formal_write     : NONE",
        "champion_change  : NONE",
        "weight_tuning    : NONE",
        (
            "expected_fp      : "
            + EXPECTED_WEIGHT_FINGERPRINT
        ),
        "",
        "=== BASE CORNER EFFECT ===",
        (
            "corner True - False mean : "
            f"{raw_all['true_minus_false_mean']:.6f}"
        ),
        (
            "95% CI                  : "
            f"{raw_all['paired_day_ci95_low']:.6f}"
            " .. "
            f"{raw_all['paired_day_ci95_high']:.6f}"
        ),
        "",
        "=== SEGMENT ADJUSTED ===",
        (
            "raw mean                : "
            f"{seg['raw_mean_effect']:.6f}"
        ),
        (
            "adjusted mean           : "
            f"{seg['adjusted_mean_effect']:.6f}"
        ),
        (
            "adjusted 95% CI         : "
            f"{seg['adjusted_ci95_low']:.6f}"
            " .. "
            f"{seg['adjusted_ci95_high']:.6f}"
        ),
        (
            "adjusted - raw          : "
            f"{seg['adjusted_minus_raw_mean']:.6f}"
        ),
        "",
        "=== MACHINE NAME MATCHED ===",
        (
            "group coverage          : "
            f"{machine['group_coverage']:.6f}"
        ),
        (
            "corner row coverage     : "
            f"{machine['corner_row_coverage']:.6f}"
        ),
        (
            "equal-weight mean       : "
            f"{machine['equal_weight_mean_effect']:.6f}"
        ),
        (
            "equal-weight 95% CI     : "
            f"{machine['equal_weight_ci95_low']:.6f}"
            " .. "
            f"{machine['equal_weight_ci95_high']:.6f}"
        ),
        "",
        "=== WITHIN ISLAND / SIDE ===",
        (
            "eligible groups         : "
            f"{int(island['eligible_groups'])}"
        ),
        (
            "equal-weight mean       : "
            f"{island['equal_weight_mean_effect']:.6f}"
        ),
        (
            "equal-weight 95% CI     : "
            f"{island['equal_weight_ci95_low']:.6f}"
            " .. "
            f"{island['equal_weight_ci95_high']:.6f}"
        ),
        "",
        "=== LEAVE ONE ISLAND OUT ===",
        (
            "processed islands       : "
            f"{len(loio)}"
        ),
        (
            "LOIO mean effect range  : "
            f"{loio_min:.6f}"
            " .. "
            f"{loio_max:.6f}"
        ),
        "",
        "=== SENSITIVITY EXCLUSIONS ===",
        sensitivity[
            [
                "scenario",
                "remaining_machines",
                "true_minus_false_mean",
                "paired_day_ci95_low",
                "paired_day_ci95_high",
            ]
        ].to_string(
            index=False
        ),
        "",
        "=== TIME BLOCKS ===",
        time_blocks[
            [
                "method",
                "time_block",
                "day_count",
                "true_minus_false_mean",
                "paired_day_ci95_low",
                "paired_day_ci95_high",
                "wf_sign_match_rate",
                "wf_mean_future_excess",
            ]
        ].to_string(
            index=False
        ),
        "",
        "=== EDGE SENSITIVITY ===",
        edge_sensitivity.to_string(
            index=False
        ),
        "",
        "=== QA ===",
        qa.to_string(
            index=False
        ),
        "",
        "Phase1B does not create a Challenger.",
        "No production ranking, Forward, Champion, fingerprint, or weight is changed.",
    ]

    return "\n".join(
        lines
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    safe_output_guard()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        phase1a,
        stage1,
        stage2a,
    ) = load_runtime()

    (
        panel,
        floor_attr,
        floor_static_qa,
        target_dates,
    ) = build_phase1b_panel(
        phase1a,
        stage1,
        stage2a,
    )

    rect = (
        panel.loc[
            panel["is_rect"]
        ]
        .copy()
    )

    if (
        rect[
            "machine_no"
        ].nunique()
        != EXPECTED_RECT_MACHINES
    ):
        raise RuntimeError(
            "RECT machine count mismatch."
        )

    # 1
    corner_segment_summary = (
        build_corner_segment_summary(
            rect,
            phase1a,
        )
    )

    # 2-3
    (
        corner_day_segment_adjusted,
        corner_day_segment_adjusted_summary,
    ) = build_day_segment_adjusted(
        rect,
        phase1a,
    )

    # 4-5
    (
        corner_machine_name_matched,
        corner_machine_name_matched_summary,
    ) = build_machine_name_matched(
        rect,
        phase1a,
    )

    # 6-7
    (
        corner_within_island,
        corner_within_island_summary,
    ) = build_within_island(
        rect,
        phase1a,
    )

    # 8
    corner_leave_one_island_out = (
        build_leave_one_island_out(
            corner_within_island,
            phase1a,
        )
    )

    # 9
    corner_sensitivity_exclusions = (
        build_sensitivity_exclusions(
            rect,
            phase1a,
        )
    )

    # 10
    corner_time_blocks = (
        build_time_blocks(
            rect,
            target_dates,
            stage1,
            phase1a,
        )
    )

    # 11
    edge_sensitivity_summary = (
        build_edge_sensitivity(
            rect,
            floor_attr,
            target_dates,
            phase1a,
        )
    )

    # 12
    phase1b_qa = build_phase1b_qa(
        panel,
        rect,
        floor_attr,
        floor_static_qa,
        target_dates,
        corner_day_segment_adjusted,
        corner_machine_name_matched,
        corner_machine_name_matched_summary,
        corner_within_island,
        corner_leave_one_island_out,
    )

    # 13
    phase1b_summary = build_summary_text(
        corner_segment_summary,
        corner_day_segment_adjusted_summary,
        corner_machine_name_matched_summary,
        corner_within_island_summary,
        corner_leave_one_island_out,
        corner_sensitivity_exclusions,
        corner_time_blocks,
        edge_sensitivity_summary,
        phase1b_qa,
    )

    outputs = {
        "corner_segment_summary.csv":
            corner_segment_summary,
        "corner_day_segment_adjusted.csv":
            corner_day_segment_adjusted,
        "corner_day_segment_adjusted_summary.csv":
            corner_day_segment_adjusted_summary,
        "corner_machine_name_matched.csv":
            corner_machine_name_matched,
        "corner_machine_name_matched_summary.csv":
            corner_machine_name_matched_summary,
        "corner_within_island.csv":
            corner_within_island,
        "corner_within_island_summary.csv":
            corner_within_island_summary,
        "corner_leave_one_island_out.csv":
            corner_leave_one_island_out,
        "corner_sensitivity_exclusions.csv":
            corner_sensitivity_exclusions,
        "corner_time_blocks.csv":
            corner_time_blocks,
        "edge_sensitivity_summary.csv":
            edge_sensitivity_summary,
        "phase1b_qa.csv":
            phase1b_qa,
    }

    for filename, df in outputs.items():
        df.to_csv(
            OUTPUT_DIR / filename,
            index=False,
            encoding="utf-8-sig",
        )

    (
        OUTPUT_DIR
        / "phase1b_summary.txt"
    ).write_text(
        phase1b_summary,
        encoding="utf-8-sig",
    )

    print(
        "Floor Phase1B completed."
    )
    print(
        f"output_dir       : {OUTPUT_DIR}"
    )
    print(
        f"target_days      : {len(target_dates)}"
    )
    print(
        f"panel_rows       : {len(panel)}"
    )
    print(
        f"rect_rows        : {len(rect)}"
    )
    print(
        f"machines         : {EXPECTED_MACHINES}"
    )
    print(
        f"corner_machines  : "
        f"{int(floor_attr['is_corner'].sum())}"
    )
    print(
        f"edge_machines    : "
        f"{int(floor_attr['is_edge'].sum())}"
    )
    print(
        f"expected_fp      : "
        f"{EXPECTED_WEIGHT_FINGERPRINT}"
    )
    print(
        "phase1b_QA       : PASS"
    )
    print(
        "production_write : NONE"
    )
    print(
        "formal_write     : NONE"
    )
    print(
        "champion_change  : NONE"
    )
    print(
        "weight_tuning    : NONE"
    )

    print(
        "\n=== CORNER SEGMENT SUMMARY ==="
    )
    print(
        corner_segment_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== SEGMENT ADJUSTED SUMMARY ==="
    )
    print(
        corner_day_segment_adjusted_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== MACHINE NAME MATCHED SUMMARY ==="
    )
    print(
        corner_machine_name_matched_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== WITHIN ISLAND SUMMARY ==="
    )
    print(
        corner_within_island_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== LEAVE ONE ISLAND OUT ==="
    )
    print(
        corner_leave_one_island_out.to_string(
            index=False
        )
    )

    print(
        "\n=== EDGE SENSITIVITY ==="
    )
    print(
        edge_sensitivity_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
