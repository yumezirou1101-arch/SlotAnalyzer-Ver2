from __future__ import annotations

"""
STORE_BEHAVIOR_RESEARCH
Phase 1B MACHINE-GROUP CONCENTRATION

OFFLINE RESEARCH ONLY

Purpose
-------
Test whether high-difference machines concentrate within
actual_machine_name groups more often than expected after conditioning on
the same day's store-wide number of strong machines.

PRIMARY:
- Machine groups with install size >= 3 only
- STRONG_PRIMARY: actual_diff >= +2000
- SOFT HIGH-ALLOCATION event:
    strong_count >= ceil(group_size * 2 / 3)

Fixed install-size strata:
- 3-4
- 5-9
- 10+

SECONDARY:
- Same SOFT event with thresholds +1000 / +4000
- STRICT ALL-POSITIVE:
    all machines in the machine group have actual_diff > 0

Null:
- For each target_date independently
- Store-wide qualifying count is fixed
- Labels are randomly reassigned across all 514 machine positions
- actual_machine_name group structure is fixed
- 10,000 permutations

This is OFFLINE RESEARCH ONLY.

Protected systems remain unchanged:
- CHAMPION_V4.2_C
- Champion weights
- Forward Guard
- Formal Forward
- production ranking
- neighbor_avg
- auto promotion
- Floor research PAUSED

No production write.
No Phase1C implementation.
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

MACHINE_DIR = (
    PROJECT_ROOT
    / "machine_number"
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
    / "offline_store_behavior_phase1b_machine_group_v1"
)


# ============================================================
# Fixed research specification
# ============================================================

RESEARCH_NAME = "STORE_BEHAVIOR_RESEARCH"

RESEARCH_PHASE = (
    "PHASE1B_MACHINE_GROUP_CONCENTRATION"
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

PRIMARY_MIN_INSTALL = 3

PRIMARY_THRESHOLD = 2000

SECONDARY_THRESHOLDS = (
    1000,
    4000,
)

PERMUTATIONS = 10_000

RANDOM_SEED = 2026092103

PERMUTATION_BATCH_SIZE = 1000


# ============================================================
# Scenario specification
# ============================================================

SCENARIOS = (
    {
        "scenario":
            "SOFT_1000",
        "scenario_class":
            "SECONDARY",
        "label_rule":
            "actual_diff >= +1000",
        "event_rule":
            "strong_count >= ceil(n*2/3)",
        "threshold":
            1000,
        "mode":
            "SOFT",
    },
    {
        "scenario":
            "SOFT_2000_PRIMARY",
        "scenario_class":
            "PRIMARY",
        "label_rule":
            "actual_diff >= +2000",
        "event_rule":
            "strong_count >= ceil(n*2/3)",
        "threshold":
            2000,
        "mode":
            "SOFT",
    },
    {
        "scenario":
            "SOFT_4000",
        "scenario_class":
            "SECONDARY",
        "label_rule":
            "actual_diff >= +4000",
        "event_rule":
            "strong_count >= ceil(n*2/3)",
        "threshold":
            4000,
        "mode":
            "SOFT",
    },
    {
        "scenario":
            "STRICT_ALL_POSITIVE",
        "scenario_class":
            "SECONDARY",
        "label_rule":
            "actual_diff > 0",
        "event_rule":
            "all machines positive",
        "threshold":
            None,
        "mode":
            "ALL_POSITIVE",
    },
)

PRIMARY_SCENARIO = (
    "SOFT_2000_PRIMARY"
)

INSTALL_STRATA = (
    "3-4",
    "5-9",
    "10+",
)

SCOPES = (
    "OVERALL",
    *INSTALL_STRATA,
)


# ============================================================
# Generic helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

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


def empirical_upper_p(
    null_values: np.ndarray,
    observed: float,
) -> float:
    ge_count = int(
        np.count_nonzero(
            null_values >= observed
        )
    )

    return (
        1.0 + ge_count
    ) / (
        len(null_values) + 1.0
    )


def percentile_ci95(
    values: np.ndarray,
) -> tuple[float, float]:
    low, high = np.quantile(
        values,
        [0.025, 0.975],
        method="linear",
    )

    return (
        float(low),
        float(high),
    )


def install_stratum(
    n: int,
) -> str:
    if 3 <= n <= 4:
        return "3-4"

    if 5 <= n <= 9:
        return "5-9"

    if n >= 10:
        return "10+"

    return "REFERENCE_LT3"


def period_for_date(
    target_date: pd.Timestamp,
    dev_dates: set[pd.Timestamp],
) -> str:
    if target_date in dev_dates:
        return "DEV"

    return "LATER"


def scenario_labels(
    actual_diff: np.ndarray,
    scenario: dict,
) -> np.ndarray:
    if (
        scenario["mode"]
        == "SOFT"
    ):
        return (
            actual_diff
            >= int(
                scenario["threshold"]
            )
        )

    if (
        scenario["mode"]
        == "ALL_POSITIVE"
    ):
        return (
            actual_diff > 0
        )

    raise RuntimeError(
        "Unknown scenario mode: "
        f"{scenario['mode']}"
    )


def required_event_count(
    group_size: int,
    scenario: dict,
) -> int:
    if (
        scenario["mode"]
        == "SOFT"
    ):
        return int(
            math.ceil(
                group_size
                * 2.0
                / 3.0
            )
        )

    if (
        scenario["mode"]
        == "ALL_POSITIVE"
    ):
        return group_size

    raise RuntimeError(
        "Unknown scenario mode."
    )


# ============================================================
# Input
# ============================================================

def load_stage1_panel() -> pd.DataFrame:
    if not STAGE1_PANEL.exists():
        raise RuntimeError(
            "Stage1 input not found: "
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
    ).astype(int)

    df["actual_diff"] = pd.to_numeric(
        df["actual_diff"],
        errors="raise",
    )

    if (
        df["actual_machine_name"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "actual_machine_name contains missing values."
        )

    df["actual_machine_name"] = (
        df["actual_machine_name"]
        .astype(str)
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
                "qa_item": item,
                "actual": actual,
                "expected": expected,
                "pass": bool(passed),
            }
        )

    target_dates = sorted(
        df["target_date"]
        .drop_duplicates()
        .tolist()
    )

    duplicate_count = int(
        df.duplicated(
            [
                "target_date",
                "machine_no",
            ]
        ).sum()
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

    add(
        "rows",
        len(df),
        EXPECTED_PANEL_ROWS,
        len(df)
        == EXPECTED_PANEL_ROWS,
    )

    add(
        "target_days",
        len(target_dates),
        EXPECTED_TARGET_DAYS,
        len(target_dates)
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
        duplicate_count == 0,
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
        len(target_dates)
        == EXPECTED_TARGET_DAYS
    ):
        dev = target_dates[
            :EXPECTED_DEV_DAYS
        ]

        later = target_dates[
            EXPECTED_DEV_DAYS:
        ]

        add(
            "dev_days",
            len(dev),
            EXPECTED_DEV_DAYS,
            len(dev)
            == EXPECTED_DEV_DAYS,
        )

        add(
            "later_days",
            len(later),
            EXPECTED_LATER_DAYS,
            len(later)
            == EXPECTED_LATER_DAYS,
        )

        add(
            "dev_min",
            dev[0].date(),
            EXPECTED_DEV_MIN.date(),
            dev[0]
            == EXPECTED_DEV_MIN,
        )

        add(
            "dev_max",
            dev[-1].date(),
            EXPECTED_DEV_MAX.date(),
            dev[-1]
            == EXPECTED_DEV_MAX,
        )

        add(
            "later_min",
            later[0].date(),
            EXPECTED_LATER_MIN.date(),
            later[0]
            == EXPECTED_LATER_MIN,
        )

        add(
            "later_max",
            later[-1].date(),
            EXPECTED_LATER_MAX.date(),
            later[-1]
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
# Machine-group structure
# ============================================================

def build_daily_groups(
    day: pd.DataFrame,
) -> tuple[
    list[dict],
    pd.DataFrame,
]:
    day = (
        day
        .sort_values(
            "machine_no"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    day[
        "position_index"
    ] = np.arange(
        len(day),
        dtype=np.int32,
    )

    groups = []

    qa_rows = []

    for (
        machine_name,
        g,
    ) in day.groupby(
        "actual_machine_name",
        sort=True,
        dropna=False,
    ):
        positions = (
            g[
                "position_index"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        n = len(
            positions
        )

        stratum = (
            install_stratum(
                n
            )
        )

        primary_eligible = (
            n
            >= PRIMARY_MIN_INSTALL
        )

        groups.append(
            {
                "actual_machine_name":
                    str(
                        machine_name
                    ),
                "positions":
                    positions,
                "install_size":
                    n,
                "install_stratum":
                    stratum,
                "primary_eligible":
                    primary_eligible,
            }
        )

        qa_rows.append(
            {
                "actual_machine_name":
                    str(
                        machine_name
                    ),
                "install_size":
                    n,
                "install_stratum":
                    stratum,
                "primary_eligible":
                    primary_eligible,
                "machine_nos":
                    "|".join(
                        map(
                            str,
                            sorted(
                                g[
                                    "machine_no"
                                ]
                                .astype(int)
                                .tolist()
                            ),
                        )
                    ),
            }
        )

    total_positions = sum(
        int(
            group[
                "install_size"
            ]
        )
        for group in groups
    )

    if (
        total_positions
        != EXPECTED_MACHINES_PER_DAY
    ):
        raise RuntimeError(
            "Machine-group total position mismatch: "
            f"{total_positions}"
        )

    return (
        groups,
        pd.DataFrame(
            qa_rows
        ),
    )


# ============================================================
# Observed event counts
# ============================================================

def observed_group_events(
    labels: np.ndarray,
    groups: list[dict],
    scenario: dict,
) -> dict:
    counts = {
        "OVERALL": 0,
        "3-4": 0,
        "5-9": 0,
        "10+": 0,
    }

    reference_two_machine_events = 0

    for group in groups:
        n = int(
            group[
                "install_size"
            ]
        )

        positions = group[
            "positions"
        ]

        qualifying_n = int(
            labels[
                positions
            ].sum()
        )

        needed = (
            required_event_count(
                n,
                scenario,
            )
        )

        event = (
            qualifying_n
            >= needed
        )

        if n == 2:
            if event:
                reference_two_machine_events += 1

            continue

        if (
            n
            < PRIMARY_MIN_INSTALL
        ):
            continue

        if event:
            counts[
                "OVERALL"
            ] += 1

            counts[
                group[
                    "install_stratum"
                ]
            ] += 1

    counts[
        "REFERENCE_2_MACHINE"
    ] = (
        reference_two_machine_events
    )

    return counts


# ============================================================
# Permutation
# ============================================================

def run_permutations(
    qualifying_n: int,
    groups: list[dict],
    scenario: dict,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    result = {
        scope: np.zeros(
            PERMUTATIONS,
            dtype=np.int32,
        )
        for scope in SCOPES
    }

    completed = 0

    while (
        completed
        < PERMUTATIONS
    ):
        batch_n = min(
            PERMUTATION_BATCH_SIZE,
            PERMUTATIONS
            - completed,
        )

        labels = np.zeros(
            (
                batch_n,
                EXPECTED_MACHINES_PER_DAY,
            ),
            dtype=bool,
        )

        if qualifying_n > 0:
            if (
                qualifying_n
                == EXPECTED_MACHINES_PER_DAY
            ):
                labels[
                    :,
                    :,
                ] = True

            else:
                random_scores = rng.random(
                    (
                        batch_n,
                        EXPECTED_MACHINES_PER_DAY,
                    ),
                    dtype=np.float64,
                )

                selected = np.argpartition(
                    random_scores,
                    kth=qualifying_n - 1,
                    axis=1,
                )[
                    :,
                    :qualifying_n,
                ]

                row_index = np.arange(
                    batch_n
                )[
                    :,
                    None
                ]

                labels[
                    row_index,
                    selected,
                ] = True

        batch_counts = {
            scope: np.zeros(
                batch_n,
                dtype=np.int32,
            )
            for scope in SCOPES
        }

        for group in groups:
            n = int(
                group[
                    "install_size"
                ]
            )

            if (
                n
                < PRIMARY_MIN_INSTALL
            ):
                continue

            positions = group[
                "positions"
            ]

            qualifying_counts = (
                labels[
                    :,
                    positions,
                ]
                .sum(
                    axis=1
                )
            )

            needed = (
                required_event_count(
                    n,
                    scenario,
                )
            )

            events = (
                qualifying_counts
                >= needed
            )

            batch_counts[
                "OVERALL"
            ] += events

            batch_counts[
                group[
                    "install_stratum"
                ]
            ] += events

        sl = slice(
            completed,
            completed + batch_n,
        )

        for scope in SCOPES:
            result[
                scope
            ][
                sl
            ] = batch_counts[
                scope
            ]

        completed += batch_n

    return result


# ============================================================
# Daily analysis
# ============================================================

def run_daily_analysis(
    panel: pd.DataFrame,
    target_dates: list[pd.Timestamp],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    dev_set = set(
        target_dates[
            :EXPECTED_DEV_DAYS
        ]
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    group_qa_parts = []

    daily_observed_rows = []

    daily_expected_rows = []

    permutation_store = {}

    total_jobs = (
        len(target_dates)
        * len(
            SCENARIOS
        )
    )

    job_no = 0

    for target_date in target_dates:
        day = (
            panel.loc[
                panel[
                    "target_date"
                ].eq(
                    target_date
                )
            ]
            .sort_values(
                "machine_no"
            )
            .reset_index(
                drop=True
            )
            .copy()
        )

        if (
            len(day)
            != EXPECTED_MACHINES_PER_DAY
        ):
            raise RuntimeError(
                "Daily machine count mismatch: "
                f"{target_date.date()}"
            )

        period = period_for_date(
            target_date,
            dev_set,
        )

        (
            groups,
            group_qa,
        ) = build_daily_groups(
            day
        )

        group_qa.insert(
            0,
            "period",
            period,
        )

        group_qa.insert(
            0,
            "target_date",
            target_date.date(),
        )

        group_qa_parts.append(
            group_qa
        )

        actual_diff = (
            day[
                "actual_diff"
            ]
            .to_numpy(
                dtype=float
            )
        )

        primary_group_count = int(
            sum(
                bool(
                    group[
                        "primary_eligible"
                    ]
                )
                for group in groups
            )
        )

        stratum_group_counts = {
            stratum: int(
                sum(
                    (
                        group[
                            "primary_eligible"
                        ]
                        and
                        group[
                            "install_stratum"
                        ]
                        == stratum
                    )
                    for group in groups
                )
            )
            for stratum in INSTALL_STRATA
        }

        two_machine_group_count = int(
            sum(
                int(
                    group[
                        "install_size"
                    ]
                )
                == 2
                for group in groups
            )
        )

        for scenario in SCENARIOS:
            job_no += 1

            labels = scenario_labels(
                actual_diff,
                scenario,
            )

            qualifying_n = int(
                labels.sum()
            )

            observed = (
                observed_group_events(
                    labels,
                    groups,
                    scenario,
                )
            )

            perm = run_permutations(
                qualifying_n=qualifying_n,
                groups=groups,
                scenario=scenario,
                rng=rng,
            )

            permutation_store[
                (
                    target_date,
                    scenario[
                        "scenario"
                    ],
                )
            ] = perm

            observed_row = {
                "target_date":
                    target_date.date(),
                "period":
                    period,
                "scenario":
                    scenario[
                        "scenario"
                    ],
                "scenario_class":
                    scenario[
                        "scenario_class"
                    ],
                "label_rule":
                    scenario[
                        "label_rule"
                    ],
                "event_rule":
                    scenario[
                        "event_rule"
                    ],
                "qualifying_machine_n":
                    qualifying_n,
                "qualifying_machine_rate":
                    (
                        qualifying_n
                        / EXPECTED_MACHINES_PER_DAY
                    ),
                "all_machine_group_count":
                    len(
                        groups
                    ),
                "primary_group_count":
                    primary_group_count,
                "groups_3_4":
                    stratum_group_counts[
                        "3-4"
                    ],
                "groups_5_9":
                    stratum_group_counts[
                        "5-9"
                    ],
                "groups_10_plus":
                    stratum_group_counts[
                        "10+"
                    ],
                "two_machine_reference_group_count":
                    two_machine_group_count,
                "observed_event_count":
                    observed[
                        "OVERALL"
                    ],
                "observed_event_3_4":
                    observed[
                        "3-4"
                    ],
                "observed_event_5_9":
                    observed[
                        "5-9"
                    ],
                "observed_event_10_plus":
                    observed[
                        "10+"
                    ],
                "reference_2_machine_event_count":
                    observed[
                        "REFERENCE_2_MACHINE"
                    ],
            }

            daily_observed_rows.append(
                observed_row
            )

            for scope in SCOPES:
                observed_value = (
                    observed[
                        scope
                    ]
                )

                null_values = perm[
                    scope
                ]

                null_mean = float(
                    null_values.mean()
                )

                daily_expected_rows.append(
                    {
                        "target_date":
                            target_date.date(),
                        "period":
                            period,
                        "scenario":
                            scenario[
                                "scenario"
                            ],
                        "scenario_class":
                            scenario[
                                "scenario_class"
                            ],
                        "scope":
                            scope,
                        "qualifying_machine_n":
                            qualifying_n,
                        "observed_event_count":
                            observed_value,
                        "null_mean_event_count":
                            null_mean,
                        "observed_minus_expected":
                            (
                                observed_value
                                - null_mean
                            ),
                        "observed_div_expected":
                            safe_ratio(
                                observed_value,
                                null_mean,
                            ),
                    }
                )

            print(
                "[PERM] "
                f"{job_no}/{total_jobs} "
                f"date={target_date.date()} "
                f"period={period} "
                f"scenario={scenario['scenario']} "
                f"qualifying_n={qualifying_n} "
                f"groups_n3plus={primary_group_count} "
                f"observed={observed['OVERALL']} "
                f"null_mean="
                f"{perm['OVERALL'].mean():.4f}"
            )

    group_qa_df = pd.concat(
        group_qa_parts,
        ignore_index=True,
    )

    daily_observed_df = (
        pd.DataFrame(
            daily_observed_rows
        )
    )

    daily_expected_df = (
        pd.DataFrame(
            daily_expected_rows
        )
    )

    return (
        group_qa_df,
        daily_observed_df,
        daily_expected_df,
        permutation_store,
    )


# ============================================================
# Summary helpers
# ============================================================

def dates_for_period(
    target_dates: list[pd.Timestamp],
    period: str,
) -> list[pd.Timestamp]:
    if period == "FULL":
        return target_dates

    if period == "DEV":
        return target_dates[
            :EXPECTED_DEV_DAYS
        ]

    if period == "LATER":
        return target_dates[
            EXPECTED_DEV_DAYS:
        ]

    raise RuntimeError(
        f"Unknown period: {period}"
    )


def observed_column_for_scope(
    scope: str,
) -> str:
    mapping = {
        "OVERALL":
            "observed_event_count",
        "3-4":
            "observed_event_3_4",
        "5-9":
            "observed_event_5_9",
        "10+":
            "observed_event_10_plus",
    }

    return mapping[
        scope
    ]


def summarize_scenario_scope(
    daily_observed: pd.DataFrame,
    permutation_store: dict,
    target_dates: list[pd.Timestamp],
    scenario_name: str,
    period: str,
    scope: str,
) -> dict:
    period_dates = dates_for_period(
        target_dates,
        period,
    )

    period_date_set = set(
        period_dates
    )

    obs = (
        daily_observed.loc[
            daily_observed[
                "scenario"
            ].eq(
                scenario_name
            )
        ]
        .copy()
    )

    obs[
        "target_date_ts"
    ] = pd.to_datetime(
        obs[
            "target_date"
        ],
        errors="raise",
    ).dt.normalize()

    obs = obs.loc[
        obs[
            "target_date_ts"
        ].isin(
            period_date_set
        )
    ].copy()

    observed_col = (
        observed_column_for_scope(
            scope
        )
    )

    observed_count = float(
        obs[
            observed_col
        ].sum()
    )

    null_total = np.zeros(
        PERMUTATIONS,
        dtype=np.int32,
    )

    for target_date in period_dates:
        null_total += (
            permutation_store[
                (
                    target_date,
                    scenario_name,
                )
            ][
                scope
            ]
        )

    null_mean = float(
        null_total.mean()
    )

    ci_low, ci_high = (
        percentile_ci95(
            null_total
        )
    )

    excess = (
        observed_count
        - null_mean
    )

    return {
        "period":
            period,
        "scenario":
            scenario_name,
        "scope":
            scope,
        "target_days":
            len(
                period_dates
            ),
        "period_min_date":
            period_dates[
                0
            ].date(),
        "period_max_date":
            period_dates[
                -1
            ].date(),
        "observed_event_count":
            observed_count,
        "null_mean_event_count":
            null_mean,
        "observed_minus_expected":
            excess,
        "observed_div_expected":
            safe_ratio(
                observed_count,
                null_mean,
            ),
        "null_ci95_low":
            ci_low,
        "null_ci95_high":
            ci_high,
        "empirical_p_raw":
            empirical_upper_p(
                null_total,
                observed_count,
            ),
        "effect_direction":
            (
                "ABOVE_RANDOM"
                if excess > 0
                else (
                    "BELOW_RANDOM"
                    if excess < 0
                    else "EQUAL_RANDOM"
                )
            ),
    }


def build_all_summary(
    daily_observed: pd.DataFrame,
    permutation_store: dict,
    target_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows = []

    for scenario in SCENARIOS:
        scenario_name = (
            scenario[
                "scenario"
            ]
        )

        for period in (
            "FULL",
            "DEV",
            "LATER",
        ):
            for scope in SCOPES:
                row = (
                    summarize_scenario_scope(
                        daily_observed,
                        permutation_store,
                        target_dates,
                        scenario_name,
                        period,
                        scope,
                    )
                )

                row[
                    "scenario_class"
                ] = scenario[
                    "scenario_class"
                ]

                row[
                    "label_rule"
                ] = scenario[
                    "label_rule"
                ]

                row[
                    "event_rule"
                ] = scenario[
                    "event_rule"
                ]

                rows.append(
                    row
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Leave-one-day-out
# ============================================================

def build_primary_loo(
    daily_expected: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    x = (
        daily_expected.loc[
            daily_expected[
                "scenario"
            ].eq(
                PRIMARY_SCENARIO
            )
            &
            daily_expected[
                "scope"
            ].eq(
                "OVERALL"
            )
        ]
        .copy()
    )

    x[
        "target_date"
    ] = pd.to_datetime(
        x[
            "target_date"
        ],
        errors="raise",
    ).dt.normalize()

    detail_rows = []

    summary_rows = []

    for period in (
        "FULL",
        "DEV",
        "LATER",
    ):
        if period == "FULL":
            p = x.copy()
        else:
            p = x.loc[
                x[
                    "period"
                ].eq(
                    period
                )
            ].copy()

        overall_effect = float(
            p[
                "observed_minus_expected"
            ].sum()
        )

        if overall_effect > 0:
            overall_direction = (
                "ABOVE_RANDOM"
            )
        elif overall_effect < 0:
            overall_direction = (
                "BELOW_RANDOM"
            )
        else:
            overall_direction = (
                "EQUAL_RANDOM"
            )

        period_detail = []

        for _, row in p.iterrows():
            excluded_effect = float(
                row[
                    "observed_minus_expected"
                ]
            )

            loo_effect = (
                overall_effect
                - excluded_effect
            )

            if loo_effect > 0:
                loo_direction = (
                    "ABOVE_RANDOM"
                )
            elif loo_effect < 0:
                loo_direction = (
                    "BELOW_RANDOM"
                )
            else:
                loo_direction = (
                    "EQUAL_RANDOM"
                )

            direction_flip = bool(
                loo_direction
                != overall_direction
            )

            detail_row = {
                "period":
                    period,
                "scenario":
                    PRIMARY_SCENARIO,
                "scope":
                    "OVERALL",
                "overall_effect":
                    overall_effect,
                "overall_direction":
                    overall_direction,
                "excluded_target_date":
                    row[
                        "target_date"
                    ].date(),
                "excluded_day_effect":
                    excluded_effect,
                "leave_one_day_out_effect":
                    loo_effect,
                "leave_one_day_out_direction":
                    loo_direction,
                "direction_flip":
                    direction_flip,
            }

            detail_rows.append(
                detail_row
            )

            period_detail.append(
                detail_row
            )

        detail_df = pd.DataFrame(
            period_detail
        )

        summary_rows.append(
            {
                "period":
                    period,
                "scenario":
                    PRIMARY_SCENARIO,
                "scope":
                    "OVERALL",
                "overall_effect":
                    overall_effect,
                "overall_direction":
                    overall_direction,
                "leave_one_day_out_min_effect":
                    float(
                        detail_df[
                            "leave_one_day_out_effect"
                        ].min()
                    ),
                "leave_one_day_out_max_effect":
                    float(
                        detail_df[
                            "leave_one_day_out_effect"
                        ].max()
                    ),
                "direction_flip_count":
                    int(
                        detail_df[
                            "direction_flip"
                        ].sum()
                    ),
                "direction_flip_any":
                    bool(
                        detail_df[
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
# Output tables
# ============================================================

def build_primary_outputs(
    all_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    overall = (
        all_summary.loc[
            all_summary[
                "scenario"
            ].eq(
                PRIMARY_SCENARIO
            )
            &
            all_summary[
                "scope"
            ].eq(
                "OVERALL"
            )
        ]
        .copy()
    )

    overall = overall.merge(
        loo_summary[
            [
                "period",
                "direction_flip_count",
                "direction_flip_any",
                "leave_one_day_out_min_effect",
                "leave_one_day_out_max_effect",
            ]
        ],
        on="period",
        how="left",
        validate="one_to_one",
    )

    primary_full = (
        overall.loc[
            overall[
                "period"
            ].eq(
                "FULL"
            )
        ]
        .copy()
    )

    primary_dev_later = (
        overall.loc[
            overall[
                "period"
            ].isin(
                [
                    "DEV",
                    "LATER",
                ]
            )
        ]
        .copy()
    )

    return (
        primary_full,
        primary_dev_later,
    )


def build_install_size_summary(
    all_summary: pd.DataFrame,
) -> pd.DataFrame:
    return (
        all_summary.loc[
            all_summary[
                "scenario"
            ].eq(
                PRIMARY_SCENARIO
            )
            &
            all_summary[
                "scope"
            ].isin(
                INSTALL_STRATA
            )
        ]
        .copy()
    )


def build_secondary_summary(
    all_summary: pd.DataFrame,
) -> pd.DataFrame:
    return (
        all_summary.loc[
            all_summary[
                "scenario_class"
            ].eq(
                "SECONDARY"
            )
            &
            all_summary[
                "scope"
            ].eq(
                "OVERALL"
            )
        ]
        .copy()
    )


# ============================================================
# Machine-group QA summary
# ============================================================

def build_machine_group_summary(
    group_qa: pd.DataFrame,
) -> pd.DataFrame:
    x = group_qa.copy()

    summary = (
        x
        .groupby(
            [
                "target_date",
                "period",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            all_machine_groups=(
                "actual_machine_name",
                "size",
            ),
            primary_groups_n3plus=(
                "primary_eligible",
                "sum",
            ),
            min_install_size=(
                "install_size",
                "min",
            ),
            max_install_size=(
                "install_size",
                "max",
            ),
        )
    )

    for stratum in INSTALL_STRATA:
        counts = (
            x.loc[
                x[
                    "install_stratum"
                ].eq(
                    stratum
                )
            ]
            .groupby(
                [
                    "target_date",
                    "period",
                ],
                sort=True,
            )
            .size()
            .rename(
                f"groups_{stratum}"
            )
            .reset_index()
        )

        summary = summary.merge(
            counts,
            on=[
                "target_date",
                "period",
            ],
            how="left",
            validate="one_to_one",
        )

    two_counts = (
        x.loc[
            x[
                "install_size"
            ].eq(
                2
            )
        ]
        .groupby(
            [
                "target_date",
                "period",
            ],
            sort=True,
        )
        .size()
        .rename(
            "groups_2_reference"
        )
        .reset_index()
    )

    summary = summary.merge(
        two_counts,
        on=[
            "target_date",
            "period",
        ],
        how="left",
        validate="one_to_one",
    )

    count_cols = [
        col
        for col in summary.columns
        if (
            col.startswith(
                "groups_"
            )
        )
    ]

    summary[
        count_cols
    ] = (
        summary[
            count_cols
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    return summary


# ============================================================
# Metadata
# ============================================================

def build_metadata(
    target_dates: list[pd.Timestamp],
) -> pd.DataFrame:
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
            "primary_min_install_size",
            PRIMARY_MIN_INSTALL,
        ),
        (
            "primary_threshold",
            PRIMARY_THRESHOLD,
        ),
        (
            "primary_event",
            (
                "Within actual_machine_name "
                "group, actual_diff>=2000 "
                "count >= ceil(n*2/3)"
            ),
        ),
        (
            "install_strata",
            "3-4|5-9|10+",
        ),
        (
            "secondary_thresholds",
            "1000|4000",
        ),
        (
            "secondary_strict_metric",
            "ALL actual_diff > 0",
        ),
        (
            "null_model",
            (
                "Per target_date, store-wide "
                "qualifying-machine count fixed; "
                "labels shuffled across all 514 "
                "machine positions; "
                "actual_machine_name group "
                "structure fixed"
            ),
        ),
        (
            "permutations_per_day_scenario",
            PERMUTATIONS,
        ),
        (
            "random_seed",
            RANDOM_SEED,
        ),
        (
            "empirical_p_formula",
            (
                "(1 + count(null >= observed)) "
                "/ (10000 + 1)"
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
            "two_machine_primary",
            "NO_REFERENCE_ONLY",
        ),
        (
            "production_write",
            "NONE",
        ),
        (
            "phase1c_implemented",
            "NO",
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
        "Phase 1B MACHINE-GROUP CONCENTRATION"
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

    print(
        f"machines/day: {EXPECTED_MACHINES_PER_DAY}"
    )

    print()
    print(
        "[2/7] Running daily machine-group analysis..."
    )

    print(
        "PRIMARY:"
    )

    print(
        "  install size >= 3"
    )

    print(
        "  actual_diff >= +2000"
    )

    print(
        "  event = strong_count >= ceil(n*2/3)"
    )

    print(
        f"permutations: {PERMUTATIONS:,}"
    )

    print(
        f"random seed : {RANDOM_SEED}"
    )

    (
        group_qa,
        daily_observed,
        daily_expected,
        permutation_store,
    ) = run_daily_analysis(
        panel,
        target_dates,
    )

    print()
    print(
        "[3/7] Building machine-group QA..."
    )

    machine_group_summary = (
        build_machine_group_summary(
            group_qa
        )
    )

    if (
        machine_group_summary[
            "primary_groups_n3plus"
        ].le(
            0
        ).any()
    ):
        raise RuntimeError(
            "At least one date has zero PRIMARY "
            "machine groups."
        )

    print(
        "Machine-group QA: PASS"
    )

    print()
    print(
        "[4/7] Building permutation summaries..."
    )

    all_summary = build_all_summary(
        daily_observed,
        permutation_store,
        target_dates,
    )

    print()
    print(
        "[5/7] Running PRIMARY leave-one-day-out..."
    )

    (
        loo_detail,
        loo_summary,
    ) = build_primary_loo(
        daily_expected
    )

    (
        primary_full,
        primary_dev_later,
    ) = build_primary_outputs(
        all_summary,
        loo_summary,
    )

    install_size_summary = (
        build_install_size_summary(
            all_summary
        )
    )

    secondary_summary = (
        build_secondary_summary(
            all_summary
        )
    )

    print()
    print(
        "[6/7] Building metadata..."
    )

    metadata = build_metadata(
        target_dates
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
        "02_machine_group_qa.csv":
            machine_group_summary,
        "03_daily_observed.csv":
            daily_observed,
        "04_daily_permutation_expected.csv":
            daily_expected,
        "05_primary_summary.csv":
            primary_full,
        "06_dev_later_summary.csv":
            primary_dev_later,
        "07_install_size_summary.csv":
            install_size_summary,
        "08_secondary_summary.csv":
            secondary_summary,
        "09_leave_one_day_out_summary.csv":
            loo_summary,
        "10_metadata.csv":
            metadata,
        "12_machine_group_detail.csv":
            group_qa,
        "13_leave_one_day_out_detail.csv":
            loo_detail,
        "14_all_scenario_summary.csv":
            all_summary,
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
        "primary_scenario":
            PRIMARY_SCENARIO,
        "primary_threshold":
            PRIMARY_THRESHOLD,
        "primary_min_install_size":
            PRIMARY_MIN_INSTALL,
        "primary_event":
            "strong_count >= ceil(n*2/3)",
        "install_strata":
            list(
                INSTALL_STRATA
            ),
        "secondary_thresholds":
            list(
                SECONDARY_THRESHOLDS
            ),
        "strict_secondary":
            "ALL actual_diff > 0",
        "permutations":
            PERMUTATIONS,
        "random_seed":
            RANDOM_SEED,
        "outputs": {
            filename: {
                "rows":
                    int(
                        len(df)
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
        / "11_output_manifest.json"
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
        "Phase 1B MACHINE-GROUP CONCENTRATION completed."
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
        "machine_group_QA : PASS"
    )

    print(
        "PRIMARY          : SOFT HIGH-ALLOCATION"
    )

    print(
        "PRIMARY install  : >=3 machines"
    )

    print(
        "PRIMARY threshold: actual_diff >= +2000"
    )

    print(
        "PRIMARY rule     : >= ceil(n*2/3)"
    )

    print(
        f"permutations     : {PERMUTATIONS}"
    )

    print(
        f"random_seed      : {RANDOM_SEED}"
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
        "Phase1C          : NOT IMPLEMENTED"
    )

    print(
        "git_action       : NONE"
    )

    display_cols = [
        "period",
        "observed_event_count",
        "null_mean_event_count",
        "observed_minus_expected",
        "observed_div_expected",
        "null_ci95_low",
        "null_ci95_high",
        "empirical_p_raw",
        "direction_flip_any",
    ]

    print()
    print(
        "PRIMARY FULL:"
    )

    print(
        primary_full[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "PRIMARY DEV/LATER:"
    )

    print(
        primary_dev_later[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "INSTALL-SIZE STRATA:"
    )

    print(
        install_size_summary[
            [
                "period",
                "scope",
                "observed_event_count",
                "null_mean_event_count",
                "observed_minus_expected",
                "observed_div_expected",
                "empirical_p_raw",
            ]
        ].to_string(
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
        "Return Phase1B outputs to 03_2."
    )

    print(
        "Do not start Phase1C before 03_2 review."
    )


if __name__ == "__main__":
    main()