from __future__ import annotations

"""
STORE_BEHAVIOR_RESEARCH
Phase 1C-1 WEEKDAY vs WEEKEND_HOLIDAY

OFFLINE RESEARCH ONLY

Purpose
-------
Compare store-wide STRONG_PRIMARY rate between:

WEEKDAY:
- Monday-Friday
- excluding Japanese public holidays

WEEKEND_HOLIDAY:
- Saturday
- Sunday
- Japanese public holidays

PRIMARY endpoint:
- STRONG_PRIMARY = actual_diff >= +2000
- Daily store-wide strong rate
- Day is the statistical unit

This script does NOT:
- run weekday 7-category analysis
- test date endings
- test repeated-digit dates
- test month-start / month-end
- test consecutive-holiday position
- test machine-model interactions
- test Floor interactions
- modify production
- modify CHAMPION_V4.2_C

Protected systems remain unchanged:
- CHAMPION_V4.2_C
- Champion weights
- Forward Guard
- Formal Forward
- production ranking
- neighbor_avg
- auto promotion
- Floor research PAUSED
"""

from pathlib import Path
import hashlib
import json
import math
import platform
import sys

import numpy as np
import pandas as pd


# ============================================================
# Project paths
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

STAGE1_PANEL = (
    ANALYSIS_DIR
    / "offline_stage1_prevday_dependency_v1"
    / "03_ab_all_machine_scores.csv"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "offline_store_behavior_phase1c1_weekday_weekend_v1"
)


# ============================================================
# Fixed research specification
# ============================================================

RESEARCH_NAME = "STORE_BEHAVIOR_RESEARCH"

RESEARCH_PHASE = (
    "PHASE1C1_WEEKDAY_VS_WEEKEND_HOLIDAY"
)

RESEARCH_SCOPE = "OFFLINE_RESEARCH_ONLY"

STORE = "MARUHAN_MEGA_CITY_MAEBASHI_INTER"

CHAMPION_MODEL = "CHAMPION_V4.2_C"

CHAMPION_FINGERPRINT = (
    "a1eaf45d71ded209"
)

EXPECTED_TARGET_DAYS = 55

EXPECTED_MACHINES_PER_DAY = 514

EXPECTED_PANEL_ROWS = (
    EXPECTED_TARGET_DAYS
    * EXPECTED_MACHINES_PER_DAY
)

EXPECTED_DEV_DAYS = 28

EXPECTED_LATER_DAYS = 27

EXPECTED_DEV_MIN = pd.Timestamp(
    "2026-07-13"
)

EXPECTED_DEV_MAX = pd.Timestamp(
    "2026-08-09"
)

EXPECTED_LATER_MIN = pd.Timestamp(
    "2026-08-10"
)

EXPECTED_LATER_MAX = pd.Timestamp(
    "2026-09-13"
)

PRIMARY_THRESHOLD = 2000

BOOTSTRAP_REPLICATES = 10_000

PERMUTATION_REPLICATES = 10_000

RANDOM_SEED_BOOTSTRAP = 2026092104

RANDOM_SEED_PERMUTATION = 2026092105


# ============================================================
# Fixed Japanese holiday calendar
# ============================================================

HOLIDAY_SOURCE_NAME = (
    "Cabinet Office, Government of Japan "
    "National Holidays 2026"
)

HOLIDAY_SOURCE_URL = (
    "https://www8.cao.go.jp/chosei/shukujitsu/gaiyou.html"
)

HOLIDAY_SOURCE_RETRIEVED = "2026-09-21"

# Only holidays inside the fixed analysis range are required.
# These dates are copied from the Cabinet Office 2026 holiday list.
FIXED_HOLIDAYS = {
    pd.Timestamp(
        "2026-07-20"
    ): "Marine Day / 海の日",
    pd.Timestamp(
        "2026-08-11"
    ): "Mountain Day / 山の日",
}


# ============================================================
# Generic helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing columns: "
            f"{missing}"
        )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return math.nan

    return (
        float(numerator)
        / float(denominator)
    )


def effect_direction(
    value: float,
    atol: float = 1e-15,
) -> str:
    if value > atol:
        return (
            "WEEKEND_HOLIDAY_HIGHER"
        )

    if value < -atol:
        return (
            "WEEKDAY_HIGHER"
        )

    return "NO_DIFFERENCE"


def empirical_two_sided_p(
    null_values: np.ndarray,
    observed: float,
) -> float:
    extreme = int(
        np.count_nonzero(
            np.abs(
                null_values
            )
            >=
            abs(
                observed
            )
        )
    )

    return (
        1.0
        + extreme
    ) / (
        len(
            null_values
        )
        + 1.0
    )


# ============================================================
# Input loading
# ============================================================

def load_stage1_panel() -> pd.DataFrame:
    if not STAGE1_PANEL.exists():
        raise RuntimeError(
            "Stage1 panel not found: "
            f"{STAGE1_PANEL}"
        )

    df = pd.read_csv(
        STAGE1_PANEL,
        encoding="utf-8-sig",
    )

    require_columns(
        df,
        [
            "target_date",
            "machine_no",
            "actual_machine_name",
            "actual_diff",
        ],
        "Stage1 panel",
    )

    df = df[
        [
            "target_date",
            "machine_no",
            "actual_machine_name",
            "actual_diff",
        ]
    ].copy()

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    ).dt.normalize()

    df["machine_no"] = pd.to_numeric(
        df["machine_no"],
        errors="raise",
    ).astype(
        int
    )

    df["actual_diff"] = pd.to_numeric(
        df["actual_diff"],
        errors="raise",
    )

    if (
        df[
            "actual_machine_name"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "actual_machine_name contains missing values."
        )

    df[
        "actual_machine_name"
    ] = (
        df[
            "actual_machine_name"
        ]
        .astype(
            str
        )
    )

    return df


# ============================================================
# Input QA
# ============================================================

def run_input_qa(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[pd.Timestamp],
]:
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
                    bool(
                        passed
                    ),
            }
        )

    target_dates = sorted(
        df[
            "target_date"
        ]
        .drop_duplicates()
        .tolist()
    )

    machine_counts = (
        df
        .groupby(
            "target_date",
            sort=True,
        )[
            "machine_no"
        ]
        .nunique()
    )

    duplicate_count = int(
        df.duplicated(
            [
                "target_date",
                "machine_no",
            ]
        ).sum()
    )

    add(
        "panel_rows",
        len(
            df
        ),
        EXPECTED_PANEL_ROWS,
        len(
            df
        )
        == EXPECTED_PANEL_ROWS,
    )

    add(
        "target_days",
        len(
            target_dates
        ),
        EXPECTED_TARGET_DAYS,
        len(
            target_dates
        )
        == EXPECTED_TARGET_DAYS,
    )

    add(
        "unique_machine_no",
        df[
            "machine_no"
        ].nunique(),
        EXPECTED_MACHINES_PER_DAY,
        df[
            "machine_no"
        ].nunique()
        == EXPECTED_MACHINES_PER_DAY,
    )

    add(
        "duplicate_target_date_machine_no",
        duplicate_count,
        0,
        duplicate_count
        == 0,
    )

    add(
        "actual_diff_missing",
        int(
            df[
                "actual_diff"
            ]
            .isna()
            .sum()
        ),
        0,
        not df[
            "actual_diff"
        ]
        .isna()
        .any(),
    )

    add(
        "actual_machine_name_missing",
        int(
            df[
                "actual_machine_name"
            ]
            .isna()
            .sum()
        ),
        0,
        not df[
            "actual_machine_name"
        ]
        .isna()
        .any(),
    )

    add(
        "machines_per_day_min",
        int(
            machine_counts.min()
        ),
        EXPECTED_MACHINES_PER_DAY,
        int(
            machine_counts.min()
        )
        == EXPECTED_MACHINES_PER_DAY,
    )

    add(
        "machines_per_day_max",
        int(
            machine_counts.max()
        ),
        EXPECTED_MACHINES_PER_DAY,
        int(
            machine_counts.max()
        )
        == EXPECTED_MACHINES_PER_DAY,
    )

    if (
        len(
            target_dates
        )
        == EXPECTED_TARGET_DAYS
    ):
        dev_dates = (
            target_dates[
                :EXPECTED_DEV_DAYS
            ]
        )

        later_dates = (
            target_dates[
                EXPECTED_DEV_DAYS:
            ]
        )

        add(
            "dev_days",
            len(
                dev_dates
            ),
            EXPECTED_DEV_DAYS,
            len(
                dev_dates
            )
            == EXPECTED_DEV_DAYS,
        )

        add(
            "later_days",
            len(
                later_dates
            ),
            EXPECTED_LATER_DAYS,
            len(
                later_dates
            )
            == EXPECTED_LATER_DAYS,
        )

        add(
            "dev_min_date",
            dev_dates[
                0
            ].date(),
            EXPECTED_DEV_MIN.date(),
            dev_dates[
                0
            ]
            == EXPECTED_DEV_MIN,
        )

        add(
            "dev_max_date",
            dev_dates[
                -1
            ].date(),
            EXPECTED_DEV_MAX.date(),
            dev_dates[
                -1
            ]
            == EXPECTED_DEV_MAX,
        )

        add(
            "later_min_date",
            later_dates[
                0
            ].date(),
            EXPECTED_LATER_MIN.date(),
            later_dates[
                0
            ]
            == EXPECTED_LATER_MIN,
        )

        add(
            "later_max_date",
            later_dates[
                -1
            ].date(),
            EXPECTED_LATER_MAX.date(),
            later_dates[
                -1
            ]
            == EXPECTED_LATER_MAX,
        )

    qa = pd.DataFrame(
        rows
    )

    failed = qa.loc[
        ~qa[
            "pass"
        ]
    ]

    if not failed.empty:
        raise RuntimeError(
            "Input QA failed:\n"
            + failed.to_string(
                index=False
            )
        )

    return (
        qa,
        target_dates,
    )


# ============================================================
# Calendar classification
# ============================================================

def classify_calendar_day(
    target_date: pd.Timestamp,
) -> tuple[
    str,
    bool,
    str,
]:
    target_date = (
        pd.Timestamp(
            target_date
        )
        .normalize()
    )

    weekday_number = int(
        target_date.weekday()
    )

    is_weekend = (
        weekday_number
        >= 5
    )

    is_holiday = (
        target_date
        in FIXED_HOLIDAYS
    )

    holiday_name = (
        FIXED_HOLIDAYS.get(
            target_date,
            "",
        )
    )

    if (
        is_weekend
        or is_holiday
    ):
        group = (
            "WEEKEND_HOLIDAY"
        )
    else:
        group = "WEEKDAY"

    return (
        group,
        is_holiday,
        holiday_name,
    )


def build_calendar_qa(
    target_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows = []

    dev_set = set(
        target_dates[
            :EXPECTED_DEV_DAYS
        ]
    )

    for target_date in target_dates:
        (
            calendar_group,
            is_holiday,
            holiday_name,
        ) = classify_calendar_day(
            target_date
        )

        rows.append(
            {
                "target_date":
                    target_date.date(),
                "weekday_number":
                    int(
                        target_date.weekday()
                    ),
                "weekday_name":
                    target_date.day_name(),
                "is_weekend":
                    bool(
                        target_date.weekday()
                        >= 5
                    ),
                "is_japanese_holiday":
                    is_holiday,
                "holiday_name":
                    holiday_name,
                "calendar_group":
                    calendar_group,
                "period":
                    (
                        "DEV"
                        if target_date
                        in dev_set
                        else "LATER"
                    ),
            }
        )

    calendar = pd.DataFrame(
        rows
    )

    if (
        len(
            calendar
        )
        != EXPECTED_TARGET_DAYS
    ):
        raise RuntimeError(
            "Calendar row count mismatch."
        )

    if (
        calendar[
            "target_date"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate target_date in calendar QA."
        )

    holiday_dates_in_panel = set(
        pd.to_datetime(
            calendar.loc[
                calendar[
                    "is_japanese_holiday"
                ],
                "target_date",
            ]
        )
        .dt.normalize()
        .tolist()
    )

    expected_holiday_dates = (
        set(
            FIXED_HOLIDAYS.keys()
        )
        &
        set(
            target_dates
        )
    )

    if (
        holiday_dates_in_panel
        != expected_holiday_dates
    ):
        raise RuntimeError(
            "Holiday classification mismatch."
        )

    return calendar


# ============================================================
# Daily store-strength table
# ============================================================

def build_daily_store_strength(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    x = panel.copy()

    x[
        "is_strong_primary"
    ] = (
        x[
            "actual_diff"
        ]
        >= PRIMARY_THRESHOLD
    )

    daily = (
        x
        .groupby(
            "target_date",
            as_index=False,
            sort=True,
        )
        .agg(
            total_machine_n=(
                "machine_no",
                "size",
            ),
            unique_machine_n=(
                "machine_no",
                "nunique",
            ),
            strong_count=(
                "is_strong_primary",
                "sum",
            ),
            mean_actual_diff=(
                "actual_diff",
                "mean",
            ),
            median_actual_diff=(
                "actual_diff",
                "median",
            ),
        )
    )

    daily[
        "strong_rate"
    ] = (
        daily[
            "strong_count"
        ]
        /
        daily[
            "total_machine_n"
        ]
    )

    daily[
        "strong_rate_pct"
    ] = (
        daily[
            "strong_rate"
        ]
        * 100.0
    )

    calendar_x = (
        calendar.copy()
    )

    calendar_x[
        "target_date"
    ] = pd.to_datetime(
        calendar_x[
            "target_date"
        ],
        errors="raise",
    ).dt.normalize()

    daily = daily.merge(
        calendar_x,
        on="target_date",
        how="left",
        validate="one_to_one",
    )

    if (
        daily[
            "calendar_group"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Daily calendar merge failed."
        )

    if not daily[
        "total_machine_n"
    ].eq(
        EXPECTED_MACHINES_PER_DAY
    ).all():
        raise RuntimeError(
            "Daily total machine count mismatch."
        )

    if not daily[
        "unique_machine_n"
    ].eq(
        EXPECTED_MACHINES_PER_DAY
    ).all():
        raise RuntimeError(
            "Daily unique machine count mismatch."
        )

    return daily


# ============================================================
# Statistical helpers
# ============================================================

def bootstrap_mean_difference_ci(
    weekday_values: np.ndarray,
    weekend_values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    float,
    float,
]:
    if (
        len(
            weekday_values
        )
        == 0
        or len(
            weekend_values
        )
        == 0
    ):
        return (
            math.nan,
            math.nan,
        )

    weekday_indices = rng.integers(
        0,
        len(
            weekday_values
        ),
        size=(
            BOOTSTRAP_REPLICATES,
            len(
                weekday_values
            ),
        ),
    )

    weekend_indices = rng.integers(
        0,
        len(
            weekend_values
        ),
        size=(
            BOOTSTRAP_REPLICATES,
            len(
                weekend_values
            ),
        ),
    )

    weekday_means = (
        weekday_values[
            weekday_indices
        ]
        .mean(
            axis=1
        )
    )

    weekend_means = (
        weekend_values[
            weekend_indices
        ]
        .mean(
            axis=1
        )
    )

    differences = (
        weekend_means
        - weekday_means
    )

    low, high = np.quantile(
        differences,
        [
            0.025,
            0.975,
        ],
        method="linear",
    )

    return (
        float(
            low
        ),
        float(
            high
        ),
    )


def permutation_difference_test(
    values: np.ndarray,
    weekend_n: int,
    observed_difference: float,
    rng: np.random.Generator,
) -> tuple[
    float,
    float,
    float,
]:
    total_n = len(
        values
    )

    weekday_n = (
        total_n
        - weekend_n
    )

    if (
        weekend_n <= 0
        or weekday_n <= 0
    ):
        return (
            math.nan,
            math.nan,
            math.nan,
        )

    null_differences = np.empty(
        PERMUTATION_REPLICATES,
        dtype=float,
    )

    for i in range(
        PERMUTATION_REPLICATES
    ):
        permuted = rng.permutation(
            values
        )

        weekend_values = (
            permuted[
                :weekend_n
            ]
        )

        weekday_values = (
            permuted[
                weekend_n:
            ]
        )

        null_differences[
            i
        ] = (
            float(
                weekend_values.mean()
            )
            -
            float(
                weekday_values.mean()
            )
        )

    p_raw = empirical_two_sided_p(
        null_differences,
        observed_difference,
    )

    low, high = np.quantile(
        null_differences,
        [
            0.025,
            0.975,
        ],
        method="linear",
    )

    return (
        p_raw,
        float(
            low
        ),
        float(
            high
        ),
    )


# ============================================================
# Group summaries
# ============================================================

def summarize_period(
    daily: pd.DataFrame,
    period: str,
    bootstrap_rng: np.random.Generator,
    permutation_rng: np.random.Generator,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    if period == "FULL":
        x = daily.copy()
    else:
        x = (
            daily.loc[
                daily[
                    "period"
                ].eq(
                    period
                )
            ]
            .copy()
        )

    weekday = (
        x.loc[
            x[
                "calendar_group"
            ].eq(
                "WEEKDAY"
            )
        ]
        .copy()
    )

    weekend = (
        x.loc[
            x[
                "calendar_group"
            ].eq(
                "WEEKEND_HOLIDAY"
            )
        ]
        .copy()
    )

    weekday_values = (
        weekday[
            "strong_rate"
        ]
        .to_numpy(
            dtype=float
        )
    )

    weekend_values = (
        weekend[
            "strong_rate"
        ]
        .to_numpy(
            dtype=float
        )
    )

    weekday_mean = float(
        weekday_values.mean()
    )

    weekend_mean = float(
        weekend_values.mean()
    )

    difference = (
        weekend_mean
        - weekday_mean
    )

    ci_low, ci_high = (
        bootstrap_mean_difference_ci(
            weekday_values,
            weekend_values,
            bootstrap_rng,
        )
    )

    (
        permutation_p_raw,
        permutation_null_ci_low,
        permutation_null_ci_high,
    ) = permutation_difference_test(
        values=x[
            "strong_rate"
        ].to_numpy(
            dtype=float
        ),
        weekend_n=len(
            weekend_values
        ),
        observed_difference=difference,
        rng=permutation_rng,
    )

    group_rows = []

    for group_name, g in (
        (
            "WEEKDAY",
            weekday,
        ),
        (
            "WEEKEND_HOLIDAY",
            weekend,
        ),
    ):
        values = (
            g[
                "strong_rate"
            ]
            .to_numpy(
                dtype=float
            )
        )

        group_rows.append(
            {
                "period":
                    period,
                "calendar_group":
                    group_name,
                "day_n":
                    len(
                        g
                    ),
                "period_min_date":
                    x[
                        "target_date"
                    ].min(),
                "period_max_date":
                    x[
                        "target_date"
                    ].max(),
                "mean_daily_strong_rate":
                    float(
                        values.mean()
                    ),
                "median_daily_strong_rate":
                    float(
                        np.median(
                            values
                        )
                    ),
                "min_daily_strong_rate":
                    float(
                        values.min()
                    ),
                "max_daily_strong_rate":
                    float(
                        values.max()
                    ),
                "mean_daily_strong_rate_pct":
                    float(
                        values.mean()
                        * 100.0
                    ),
                "median_daily_strong_rate_pct":
                    float(
                        np.median(
                            values
                        )
                        * 100.0
                    ),
                "total_strong_machine_days":
                    int(
                        g[
                            "strong_count"
                        ].sum()
                    ),
                "total_machine_days":
                    int(
                        g[
                            "total_machine_n"
                        ].sum()
                    ),
            }
        )

    difference_row = {
        "period":
            period,
        "weekday_day_n":
            len(
                weekday
            ),
        "weekend_holiday_day_n":
            len(
                weekend
            ),
        "weekday_mean_daily_strong_rate":
            weekday_mean,
        "weekend_holiday_mean_daily_strong_rate":
            weekend_mean,
        "weekend_minus_weekday_difference":
            difference,
        "weekend_minus_weekday_difference_pct_points":
            difference
            * 100.0,
        "weekend_div_weekday_ratio":
            safe_ratio(
                weekend_mean,
                weekday_mean,
            ),
        "bootstrap_ci95_low":
            ci_low,
        "bootstrap_ci95_high":
            ci_high,
        "bootstrap_ci95_low_pct_points":
            ci_low
            * 100.0,
        "bootstrap_ci95_high_pct_points":
            ci_high
            * 100.0,
        "permutation_p_raw_two_sided":
            permutation_p_raw,
        "permutation_null_ci95_low":
            permutation_null_ci_low,
        "permutation_null_ci95_high":
            permutation_null_ci_high,
        "effect_direction":
            effect_direction(
                difference
            ),
    }

    return (
        pd.DataFrame(
            group_rows
        ),
        difference_row,
    )


def build_summaries(
    daily: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    bootstrap_rng = np.random.default_rng(
        RANDOM_SEED_BOOTSTRAP
    )

    permutation_rng = np.random.default_rng(
        RANDOM_SEED_PERMUTATION
    )

    group_parts = []

    difference_rows = []

    for period in (
        "FULL",
        "DEV",
        "LATER",
    ):
        (
            group_summary,
            difference_row,
        ) = summarize_period(
            daily,
            period,
            bootstrap_rng,
            permutation_rng,
        )

        group_parts.append(
            group_summary
        )

        difference_rows.append(
            difference_row
        )

    return (
        pd.concat(
            group_parts,
            ignore_index=True,
        ),
        pd.DataFrame(
            difference_rows
        ),
    )


# ============================================================
# Leave-one-day-out robustness
# ============================================================

def build_leave_one_day_out(
    daily: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    detail_rows = []

    summary_rows = []

    for period in (
        "FULL",
        "DEV",
        "LATER",
    ):
        if period == "FULL":
            x = daily.copy()
        else:
            x = (
                daily.loc[
                    daily[
                        "period"
                    ].eq(
                        period
                    )
                ]
                .copy()
            )

        weekday = (
            x.loc[
                x[
                    "calendar_group"
                ].eq(
                    "WEEKDAY"
                ),
                "strong_rate",
            ]
            .to_numpy(
                dtype=float
            )
        )

        weekend = (
            x.loc[
                x[
                    "calendar_group"
                ].eq(
                    "WEEKEND_HOLIDAY"
                ),
                "strong_rate",
            ]
            .to_numpy(
                dtype=float
            )
        )

        overall_difference = (
            float(
                weekend.mean()
            )
            -
            float(
                weekday.mean()
            )
        )

        overall_direction = (
            effect_direction(
                overall_difference
            )
        )

        period_rows = []

        for excluded_index, row in x.iterrows():
            loo = x.drop(
                index=excluded_index
            )

            loo_weekday = (
                loo.loc[
                    loo[
                        "calendar_group"
                    ].eq(
                        "WEEKDAY"
                    ),
                    "strong_rate",
                ]
                .to_numpy(
                    dtype=float
                )
            )

            loo_weekend = (
                loo.loc[
                    loo[
                        "calendar_group"
                    ].eq(
                        "WEEKEND_HOLIDAY"
                    ),
                    "strong_rate",
                ]
                .to_numpy(
                    dtype=float
                )
            )

            if (
                len(
                    loo_weekday
                )
                == 0
                or len(
                    loo_weekend
                )
                == 0
            ):
                raise RuntimeError(
                    "LOO created an empty calendar group."
                )

            loo_difference = (
                float(
                    loo_weekend.mean()
                )
                -
                float(
                    loo_weekday.mean()
                )
            )

            loo_direction = (
                effect_direction(
                    loo_difference
                )
            )

            direction_flip = bool(
                loo_direction
                != overall_direction
            )

            detail_row = {
                "period":
                    period,
                "overall_difference":
                    overall_difference,
                "overall_direction":
                    overall_direction,
                "excluded_target_date":
                    pd.Timestamp(
                        row[
                            "target_date"
                        ]
                    ).date(),
                "excluded_calendar_group":
                    row[
                        "calendar_group"
                    ],
                "excluded_strong_rate":
                    float(
                        row[
                            "strong_rate"
                        ]
                    ),
                "leave_one_day_out_difference":
                    loo_difference,
                "leave_one_day_out_direction":
                    loo_direction,
                "direction_flip":
                    direction_flip,
            }

            detail_rows.append(
                detail_row
            )

            period_rows.append(
                detail_row
            )

        period_df = pd.DataFrame(
            period_rows
        )

        summary_rows.append(
            {
                "period":
                    period,
                "overall_difference":
                    overall_difference,
                "overall_difference_pct_points":
                    overall_difference
                    * 100.0,
                "overall_direction":
                    overall_direction,
                "leave_one_day_out_min_difference":
                    float(
                        period_df[
                            "leave_one_day_out_difference"
                        ].min()
                    ),
                "leave_one_day_out_max_difference":
                    float(
                        period_df[
                            "leave_one_day_out_difference"
                        ].max()
                    ),
                "direction_flip_count":
                    int(
                        period_df[
                            "direction_flip"
                        ].sum()
                    ),
                "direction_flip_any":
                    bool(
                        period_df[
                            "direction_flip"
                        ].any()
                    ),
            }
        )

    return (
        pd.DataFrame(
            detail_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


# ============================================================
# Calendar count QA
# ============================================================

def build_calendar_count_qa(
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for period in (
        "FULL",
        "DEV",
        "LATER",
    ):
        if period == "FULL":
            x = (
                calendar.copy()
            )
        else:
            x = (
                calendar.loc[
                    calendar[
                        "period"
                    ].eq(
                        period
                    )
                ]
                .copy()
            )

        weekday_n = int(
            x[
                "calendar_group"
            ]
            .eq(
                "WEEKDAY"
            )
            .sum()
        )

        weekend_n = int(
            x[
                "calendar_group"
            ]
            .eq(
                "WEEKEND_HOLIDAY"
            )
            .sum()
        )

        holiday_n = int(
            x[
                "is_japanese_holiday"
            ]
            .sum()
        )

        rows.append(
            {
                "period":
                    period,
                "target_day_n":
                    len(
                        x
                    ),
                "weekday_day_n":
                    weekday_n,
                "weekend_holiday_day_n":
                    weekend_n,
                "japanese_holiday_day_n":
                    holiday_n,
                "group_total_check":
                    (
                        weekday_n
                        + weekend_n
                    ),
                "pass":
                    bool(
                        weekday_n
                        > 0
                        and weekend_n
                        > 0
                        and (
                            weekday_n
                            + weekend_n
                            ==
                            len(
                                x
                            )
                        )
                    ),
            }
        )

    qa = pd.DataFrame(
        rows
    )

    failed = qa.loc[
        ~qa[
            "pass"
        ]
    ]

    if not failed.empty:
        raise RuntimeError(
            "Calendar count QA failed:\n"
            + failed.to_string(
                index=False
            )
        )

    return qa


# ============================================================
# Metadata
# ============================================================

def build_metadata(
    target_dates: list[pd.Timestamp],
    calendar_count_qa: pd.DataFrame,
) -> pd.DataFrame:
    holiday_text = "|".join(
        (
            f"{date.date()}="
            f"{name}"
        )
        for date, name
        in sorted(
            FIXED_HOLIDAYS.items()
        )
    )

    rows = [
        (
            "research_name",
            RESEARCH_NAME,
        ),
        (
            "research_phase",
            RESEARCH_PHASE,
        ),
        (
            "scope",
            RESEARCH_SCOPE,
        ),
        (
            "store",
            STORE,
        ),
        (
            "script_name",
            Path(
                __file__
            ).name,
        ),
        (
            "stage1_panel_path",
            str(
                STAGE1_PANEL
            ),
        ),
        (
            "stage1_panel_sha256",
            sha256_file(
                STAGE1_PANEL
            ),
        ),
        (
            "target_days",
            len(
                target_dates
            ),
        ),
        (
            "target_date_min",
            target_dates[
                0
            ].date(),
        ),
        (
            "target_date_max",
            target_dates[
                -1
            ].date(),
        ),
        (
            "machines_per_day",
            EXPECTED_MACHINES_PER_DAY,
        ),
        (
            "primary_threshold",
            PRIMARY_THRESHOLD,
        ),
        (
            "primary_endpoint",
            (
                "Daily store-wide rate of "
                "actual_diff >= +2000"
            ),
        ),
        (
            "statistical_unit",
            "TARGET_DATE",
        ),
        (
            "primary_comparison",
            (
                "WEEKEND_HOLIDAY "
                "versus WEEKDAY"
            ),
        ),
        (
            "difference_definition",
            (
                "mean_daily_strong_rate("
                "WEEKEND_HOLIDAY) - "
                "mean_daily_strong_rate("
                "WEEKDAY)"
            ),
        ),
        (
            "direction_precommitted",
            "NO_TWO_SIDED_COMPARISON",
        ),
        (
            "weekday_definition",
            (
                "Monday-Friday and not "
                "Japanese public holiday"
            ),
        ),
        (
            "weekend_holiday_definition",
            (
                "Saturday or Sunday or "
                "Japanese public holiday"
            ),
        ),
        (
            "holiday_source",
            HOLIDAY_SOURCE_NAME,
        ),
        (
            "holiday_source_url",
            HOLIDAY_SOURCE_URL,
        ),
        (
            "holiday_source_retrieved",
            HOLIDAY_SOURCE_RETRIEVED,
        ),
        (
            "fixed_holidays_in_analysis_range",
            holiday_text,
        ),
        (
            "bootstrap_replicates",
            BOOTSTRAP_REPLICATES,
        ),
        (
            "bootstrap_seed",
            RANDOM_SEED_BOOTSTRAP,
        ),
        (
            "bootstrap_method",
            (
                "Independent day-level "
                "resampling within each "
                "calendar group"
            ),
        ),
        (
            "permutation_replicates",
            PERMUTATION_REPLICATES,
        ),
        (
            "permutation_seed",
            RANDOM_SEED_PERMUTATION,
        ),
        (
            "permutation_method",
            (
                "Two-sided day-level "
                "calendar-label permutation "
                "with observed group sizes fixed"
            ),
        ),
        (
            "dev_days",
            EXPECTED_DEV_DAYS,
        ),
        (
            "dev_min",
            EXPECTED_DEV_MIN.date(),
        ),
        (
            "dev_max",
            EXPECTED_DEV_MAX.date(),
        ),
        (
            "later_days",
            EXPECTED_LATER_DAYS,
        ),
        (
            "later_min",
            EXPECTED_LATER_MIN.date(),
        ),
        (
            "later_max",
            EXPECTED_LATER_MAX.date(),
        ),
        (
            "later_interpretation",
            "INTERNAL_TEMPORAL_CONFIRMATION",
        ),
        (
            "external_validation",
            "NO",
        ),
        (
            "secondary_analysis",
            "NONE",
        ),
        (
            "weekday_7_category_analysis",
            "NOT_IMPLEMENTED",
        ),
        (
            "date_ending_analysis",
            "NOT_IMPLEMENTED",
        ),
        (
            "repeated_digit_date_analysis",
            "NOT_IMPLEMENTED",
        ),
        (
            "month_start_end_analysis",
            "NOT_IMPLEMENTED",
        ),
        (
            "interaction_analysis",
            "NOT_IMPLEMENTED",
        ),
        (
            "production_write",
            "NONE",
        ),
        (
            "champion_model_protected",
            CHAMPION_MODEL,
        ),
        (
            "champion_fingerprint_protected",
            CHAMPION_FINGERPRINT,
        ),
        (
            "champion_weights_change",
            "NONE",
        ),
        (
            "forward_guard_change",
            "NONE",
        ),
        (
            "formal_forward_change",
            "NONE",
        ),
        (
            "production_ranking_change",
            "NONE",
        ),
        (
            "neighbor_avg_change",
            "NONE",
        ),
        (
            "auto_promotion",
            "NONE",
        ),
        (
            "floor_research_status",
            "PAUSED_UNCHANGED",
        ),
        (
            "final_research_classification",
            "NOT_ASSIGNED_BY_02_2_RETURN_TO_03_2",
        ),
        (
            "calendar_full_weekday_n",
            int(
                calendar_count_qa.loc[
                    calendar_count_qa[
                        "period"
                    ].eq(
                        "FULL"
                    ),
                    "weekday_day_n",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "calendar_full_weekend_holiday_n",
            int(
                calendar_count_qa.loc[
                    calendar_count_qa[
                        "period"
                    ].eq(
                        "FULL"
                    ),
                    "weekend_holiday_day_n",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "python_version",
            sys.version.replace(
                "\n",
                " ",
            ),
        ),
        (
            "platform",
            platform.platform(),
        ),
        (
            "numpy_version",
            np.__version__,
        ),
        (
            "pandas_version",
            pd.__version__,
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "item",
            "value",
        ],
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 78
    )

    print(
        "STORE_BEHAVIOR_RESEARCH "
        "Phase 1C-1 WEEKDAY vs WEEKEND_HOLIDAY"
    )

    print(
        "OFFLINE RESEARCH ONLY"
    )

    print(
        "=" * 78
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "[1/7] Loading Stage1 panel..."
    )

    panel = load_stage1_panel()

    (
        input_qa,
        target_dates,
    ) = run_input_qa(
        panel
    )

    print(
        "Input QA: PASS"
    )

    print(
        f"rows        : {len(panel):,}"
    )

    print(
        f"target days : {len(target_dates)}"
    )

    print()
    print(
        "[2/7] Building fixed calendar classification..."
    )

    calendar = build_calendar_qa(
        target_dates
    )

    calendar_count_qa = (
        build_calendar_count_qa(
            calendar
        )
    )

    print(
        "Calendar QA: PASS"
    )

    print(
        f"holiday source: {HOLIDAY_SOURCE_NAME}"
    )

    print(
        "fixed holidays in range:"
    )

    for date, name in sorted(
        FIXED_HOLIDAYS.items()
    ):
        if date in set(
            target_dates
        ):
            print(
                f"  {date.date()} "
                f"{name}"
            )

    print()
    print(
        "[3/7] Building daily store-strength table..."
    )

    daily = build_daily_store_strength(
        panel,
        calendar,
    )

    print(
        "Daily store-strength QA: PASS"
    )

    print()
    print(
        "[4/7] Building group summaries..."
    )

    (
        group_summary,
        difference_summary,
    ) = build_summaries(
        daily
    )

    print()
    print(
        "[5/7] Running leave-one-day-out..."
    )

    (
        loo_detail,
        loo_summary,
    ) = build_leave_one_day_out(
        daily
    )

    print()
    print(
        "[6/7] Building metadata..."
    )

    metadata = build_metadata(
        target_dates,
        calendar_count_qa,
    )

    print()
    print(
        "[7/7] Writing outputs..."
    )

    outputs: dict[
        str,
        pd.DataFrame,
    ] = {
        "01_input_qa.csv":
            input_qa,
        "02_calendar_qa.csv":
            calendar,
        "03_daily_store_strength.csv":
            daily,
        "04_group_summary.csv":
            group_summary,
        "05_dev_later_summary.csv":
            difference_summary,
        "06_leave_one_day_out_summary.csv":
            loo_summary,
        "07_metadata.csv":
            metadata,
        "09_calendar_count_qa.csv":
            calendar_count_qa,
        "10_leave_one_day_out_detail.csv":
            loo_detail,
    }

    for filename, df in outputs.items():
        path = (
            OUTPUT_DIR
            / filename
        )

        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"WROTE: {path} "
            f"({len(df):,} rows)"
        )

    manifest = {
        "research_name":
            RESEARCH_NAME,
        "research_phase":
            RESEARCH_PHASE,
        "scope":
            RESEARCH_SCOPE,
        "primary_threshold":
            PRIMARY_THRESHOLD,
        "calendar_source":
            HOLIDAY_SOURCE_NAME,
        "calendar_source_url":
            HOLIDAY_SOURCE_URL,
        "fixed_holidays": {
            str(
                date.date()
            ): name
            for date, name
            in sorted(
                FIXED_HOLIDAYS.items()
            )
        },
        "bootstrap_replicates":
            BOOTSTRAP_REPLICATES,
        "bootstrap_seed":
            RANDOM_SEED_BOOTSTRAP,
        "permutation_replicates":
            PERMUTATION_REPLICATES,
        "permutation_seed":
            RANDOM_SEED_PERMUTATION,
        "outputs": {
            filename: {
                "rows":
                    int(
                        len(
                            df
                        )
                    ),
                "sha256":
                    sha256_file(
                        OUTPUT_DIR
                        / filename
                    ),
            }
            for filename, df
            in outputs.items()
        },
    }

    manifest_path = (
        OUTPUT_DIR
        / "08_output_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"WROTE: {manifest_path}"
    )

    print()
    print(
        "=" * 78
    )

    print(
        "Phase 1C-1 WEEKDAY vs WEEKEND_HOLIDAY completed."
    )

    print(
        "=" * 78
    )

    print(
        f"output_dir       : {OUTPUT_DIR}"
    )

    print(
        "input_QA         : PASS"
    )

    print(
        "calendar_QA      : PASS"
    )

    print(
        "statistical_unit : TARGET_DATE"
    )

    print(
        "PRIMARY threshold: actual_diff >= +2000"
    )

    print(
        "comparison       : WEEKEND_HOLIDAY - WEEKDAY"
    )

    print(
        f"bootstrap        : {BOOTSTRAP_REPLICATES}"
    )

    print(
        f"bootstrap_seed   : {RANDOM_SEED_BOOTSTRAP}"
    )

    print(
        f"permutation      : {PERMUTATION_REPLICATES}"
    )

    print(
        f"permutation_seed : {RANDOM_SEED_PERMUTATION}"
    )

    print(
        "production_write : NONE"
    )

    print(
        "Champion change  : NONE"
    )

    print(
        "Formal change    : NONE"
    )

    print(
        "Phase1C extra    : NOT IMPLEMENTED"
    )

    print(
        "git_action       : NONE"
    )

    print()
    print(
        "CALENDAR COUNTS:"
    )

    print(
        calendar_count_qa[
            [
                "period",
                "target_day_n",
                "weekday_day_n",
                "weekend_holiday_day_n",
                "japanese_holiday_day_n",
                "pass",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "GROUP SUMMARY:"
    )

    print(
        group_summary[
            [
                "period",
                "calendar_group",
                "day_n",
                "mean_daily_strong_rate_pct",
                "median_daily_strong_rate_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "WEEKEND_HOLIDAY - WEEKDAY:"
    )

    print(
        difference_summary[
            [
                "period",
                "weekday_day_n",
                "weekend_holiday_day_n",
                "weekday_mean_daily_strong_rate",
                "weekend_holiday_mean_daily_strong_rate",
                "weekend_minus_weekday_difference",
                "weekend_minus_weekday_difference_pct_points",
                "bootstrap_ci95_low_pct_points",
                "bootstrap_ci95_high_pct_points",
                "permutation_p_raw_two_sided",
                "effect_direction",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "LEAVE-ONE-DAY-OUT:"
    )

    print(
        loo_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do not assign final research classification here."
    )

    print(
        "Return Phase1C-1 outputs to 03_2."
    )

    print(
        "Do not start date-ending / repeated-digit / "
        "weekday-7-category research before 03_2 review."
    )


if __name__ == "__main__":
    main()