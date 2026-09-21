from __future__ import annotations

"""
STORE_BEHAVIOR_RESEARCH
Phase 1A-R MACHINE-COMPOSITION-CONDITIONED ROBUSTNESS

OFFLINE RESEARCH ONLY

Purpose
-------
Re-evaluate the Phase 1A spatial concentration signal while preserving
same-day machine composition.

For every target_date and actual_machine_name:

- eligible installed positions are fixed
- number of strong machines is fixed
- strong / non-strong labels are shuffled only among positions occupied
  by that same actual_machine_name on that same day

This preserves:
- machine installation count
- machine-specific strong count
- machine-specific strong rate
- physical locations occupied by each machine model

Only within-machine placement of strong labels is randomized.

This is a confounder robustness check for Phase 1A.
It is NOT a new pattern search.

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
"""

from pathlib import Path
import hashlib
import json
import math
import platform
import sys
from typing import Iterable

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

FLOOR_MAP = (
    MACHINE_DIR
    / "maruhan_maebashi_floor_map_layout.csv"
)

PHASE1A_DIR = (
    ANALYSIS_DIR
    / "offline_store_behavior_phase1a_spatial_v1"
)

GLOBAL_DAILY_OBSERVED = (
    PHASE1A_DIR
    / "03_daily_spatial_observed.csv"
)

GLOBAL_ALL_SUMMARY = (
    PHASE1A_DIR
    / "12_all_threshold_period_summary.csv"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "offline_store_behavior_phase1a_r_machine_composition_v1"
)


# ============================================================
# Fixed research specification
# ============================================================

RESEARCH_NAME = "STORE_BEHAVIOR_RESEARCH"

RESEARCH_PHASE = (
    "PHASE1A_R_MACHINE_COMPOSITION_CONDITIONED_ROBUSTNESS"
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

EXPECTED_FLOOR_ROWS = 514

EXPECTED_ELIGIBLE_POSITIONS = 497

EXPECTED_EXCLUDED_ARC_POSITIONS = 17

EXPECTED_LINEAR_GROUPS = 33

EXPECTED_LINEAR_SEGMENTS = 35

EXPECTED_H1_EDGES = 462

EXPECTED_H2_WINDOWS = 427

EXPECTED_H3_WINDOWS = 358

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

SECONDARY_THRESHOLDS = (
    1000,
    4000,
)

ALL_THRESHOLDS = (
    PRIMARY_THRESHOLD,
    *SECONDARY_THRESHOLDS,
)

THRESHOLD_LABELS = {
    1000: "STRONG_LOOSE",
    2000: "STRONG_PRIMARY",
    4000: "STRONG_STRICT",
}

PERMUTATIONS = 10_000

# Dedicated fixed seed for Phase 1A-R.
RANDOM_SEED = 2026092102

PERMUTATION_BATCH_SIZE = 1000


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
    required: Iterable[str],
    label: str,
) -> None:
    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing required columns: "
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


def holm_adjust(
    p_values: dict[str, float],
) -> dict[str, float]:
    ordered = sorted(
        p_values.items(),
        key=lambda kv: kv[1],
    )

    m = len(ordered)

    adjusted_sorted = []

    running_max = 0.0

    for rank, (
        key,
        raw_p,
    ) in enumerate(
        ordered,
        start=1,
    ):
        multiplier = (
            m - rank + 1
        )

        adjusted = min(
            1.0,
            raw_p * multiplier,
        )

        running_max = max(
            running_max,
            adjusted,
        )

        adjusted_sorted.append(
            (
                key,
                min(
                    1.0,
                    running_max,
                ),
            )
        )

    return dict(
        adjusted_sorted
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
            "actual_diff",
            "actual_machine_name",
        ],
        "Stage1 panel",
    )

    df = df[
        [
            "target_date",
            "machine_no",
            "actual_diff",
            "actual_machine_name",
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


def load_floor_map() -> pd.DataFrame:
    if not FLOOR_MAP.exists():
        raise RuntimeError(
            "Floor Map not found: "
            f"{FLOOR_MAP}"
        )

    floor = pd.read_csv(
        FLOOR_MAP,
        encoding="utf-8-sig",
    )

    require_columns(
        floor,
        [
            "machine_no",
            "island_id",
            "side",
            "position_order",
            "x",
            "y",
            "rotation",
            "slot_index",
            "shape",
        ],
        "Floor Map",
    )

    floor = floor.copy()

    floor["machine_no"] = pd.to_numeric(
        floor["machine_no"],
        errors="raise",
    ).astype(int)

    floor["slot_index"] = pd.to_numeric(
        floor["slot_index"],
        errors="raise",
    ).astype(int)

    floor["position_order"] = pd.to_numeric(
        floor["position_order"],
        errors="raise",
    ).astype(int)

    return floor


def load_global_phase1a() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not GLOBAL_DAILY_OBSERVED.exists():
        raise RuntimeError(
            "Phase1A global daily output not found: "
            f"{GLOBAL_DAILY_OBSERVED}"
        )

    if not GLOBAL_ALL_SUMMARY.exists():
        raise RuntimeError(
            "Phase1A global summary not found: "
            f"{GLOBAL_ALL_SUMMARY}"
        )

    daily = pd.read_csv(
        GLOBAL_DAILY_OBSERVED,
        encoding="utf-8-sig",
    )

    summary = pd.read_csv(
        GLOBAL_ALL_SUMMARY,
        encoding="utf-8-sig",
    )

    daily["target_date"] = pd.to_datetime(
        daily["target_date"],
        errors="raise",
    ).dt.normalize()

    return (
        daily,
        summary,
    )


# ============================================================
# Input QA
# ============================================================

def stage1_input_qa(
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

    diff_missing = int(
        df["actual_diff"]
        .isna()
        .sum()
    )

    name_missing = int(
        df["actual_machine_name"]
        .isna()
        .sum()
    )

    machine_counts = (
        df
        .groupby(
            "target_date",
            sort=True,
        )
        ["machine_no"]
        .nunique()
    )

    add(
        "panel_rows",
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
        "unique_machines",
        df["machine_no"].nunique(),
        EXPECTED_MACHINES_PER_DAY,
        df["machine_no"].nunique()
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
        diff_missing,
        0,
        diff_missing == 0,
    )

    add(
        "actual_machine_name_missing",
        name_missing,
        0,
        name_missing == 0,
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
        dev_dates = target_dates[
            :EXPECTED_DEV_DAYS
        ]

        later_dates = target_dates[
            EXPECTED_DEV_DAYS:
        ]

        add(
            "dev_days",
            len(dev_dates),
            EXPECTED_DEV_DAYS,
            len(dev_dates)
            == EXPECTED_DEV_DAYS,
        )

        add(
            "later_days",
            len(later_dates),
            EXPECTED_LATER_DAYS,
            len(later_dates)
            == EXPECTED_LATER_DAYS,
        )

        add(
            "dev_min_date",
            dev_dates[0].date(),
            EXPECTED_DEV_MIN.date(),
            dev_dates[0]
            == EXPECTED_DEV_MIN,
        )

        add(
            "dev_max_date",
            dev_dates[-1].date(),
            EXPECTED_DEV_MAX.date(),
            dev_dates[-1]
            == EXPECTED_DEV_MAX,
        )

        add(
            "later_min_date",
            later_dates[0].date(),
            EXPECTED_LATER_MIN.date(),
            later_dates[0]
            == EXPECTED_LATER_MIN,
        )

        add(
            "later_max_date",
            later_dates[-1].date(),
            EXPECTED_LATER_MAX.date(),
            later_dates[-1]
            == EXPECTED_LATER_MAX,
        )

    qa = pd.DataFrame(
        rows
    )

    bad = qa.loc[
        ~qa["pass"]
    ]

    if not bad.empty:
        raise RuntimeError(
            "Stage1 input QA failed:\n"
            + bad.to_string(
                index=False
            )
        )

    return (
        qa,
        target_dates,
    )


# ============================================================
# Fixed Phase 1A linear geometry
# ============================================================

def build_linear_geometry(
    floor: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    eligible = (
        floor.loc[
            floor["shape"]
            .astype(str)
            .eq("rect")
        ]
        .copy()
    )

    eligible = (
        eligible
        .sort_values(
            [
                "island_id",
                "side",
                "slot_index",
                "machine_no",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    eligible[
        "eligible_position_index"
    ] = np.arange(
        len(eligible),
        dtype=int,
    )

    machine_to_pos = dict(
        zip(
            eligible["machine_no"],
            eligible[
                "eligible_position_index"
            ],
        )
    )

    segment_rows = []

    edges: list[
        tuple[int, int]
    ] = []

    triples: list[
        tuple[int, int, int]
    ] = []

    fives: list[
        tuple[int, int, int, int, int]
    ] = []

    segment_id = 0

    for (
        island_id,
        side,
    ), group in eligible.groupby(
        [
            "island_id",
            "side",
        ],
        sort=True,
    ):
        group = (
            group
            .sort_values(
                "slot_index"
            )
            .reset_index(
                drop=True
            )
        )

        start = 0

        boundaries = []

        for i in range(
            1,
            len(group),
        ):
            prev_slot = int(
                group.loc[
                    i - 1,
                    "slot_index",
                ]
            )

            curr_slot = int(
                group.loc[
                    i,
                    "slot_index",
                ]
            )

            if (
                curr_slot
                - prev_slot
                != 1
            ):
                boundaries.append(
                    i
                )

        boundaries.append(
            len(group)
        )

        for end in boundaries:
            segment = (
                group.iloc[
                    start:end
                ]
                .copy()
            )

            start = end

            if segment.empty:
                continue

            segment_id += 1

            machines = (
                segment[
                    "machine_no"
                ]
                .astype(int)
                .tolist()
            )

            positions = [
                int(
                    machine_to_pos[m]
                )
                for m in machines
            ]

            segment_rows.append(
                {
                    "segment_id":
                        segment_id,
                    "island_id":
                        island_id,
                    "side":
                        side,
                    "machine_count":
                        len(machines),
                    "first_machine_no":
                        machines[0],
                    "last_machine_no":
                        machines[-1],
                    "first_slot_index":
                        int(
                            segment.iloc[
                                0
                            ][
                                "slot_index"
                            ]
                        ),
                    "last_slot_index":
                        int(
                            segment.iloc[
                                -1
                            ][
                                "slot_index"
                            ]
                        ),
                    "machine_nos":
                        "|".join(
                            map(
                                str,
                                machines,
                            )
                        ),
                }
            )

            for i in range(
                len(positions) - 1
            ):
                edges.append(
                    (
                        positions[i],
                        positions[
                            i + 1
                        ],
                    )
                )

            for i in range(
                len(positions) - 2
            ):
                triples.append(
                    (
                        positions[i],
                        positions[
                            i + 1
                        ],
                        positions[
                            i + 2
                        ],
                    )
                )

            for i in range(
                len(positions) - 4
            ):
                fives.append(
                    (
                        positions[i],
                        positions[
                            i + 1
                        ],
                        positions[
                            i + 2
                        ],
                        positions[
                            i + 3
                        ],
                        positions[
                            i + 4
                        ],
                    )
                )

    segments = pd.DataFrame(
        segment_rows
    )

    edges_array = np.asarray(
        edges,
        dtype=np.int32,
    )

    triples_array = np.asarray(
        triples,
        dtype=np.int32,
    )

    fives_array = np.asarray(
        fives,
        dtype=np.int32,
    )

    return (
        eligible,
        segments,
        edges_array,
        triples_array,
        fives_array,
    )


def floor_geometry_qa(
    floor: pd.DataFrame,
    eligible: pd.DataFrame,
    segments: pd.DataFrame,
    edges: np.ndarray,
    triples: np.ndarray,
    fives: np.ndarray,
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
                "qa_item": item,
                "actual": actual,
                "expected": expected,
                "pass": bool(passed),
            }
        )

    rect_count = int(
        floor[
            "shape"
        ]
        .astype(str)
        .eq("rect")
        .sum()
    )

    arc = floor.loc[
        floor[
            "shape"
        ]
        .astype(str)
        .eq("arc")
    ].copy()

    linear_groups = int(
        eligible.groupby(
            [
                "island_id",
                "side",
            ]
        ).ngroups
    )

    arc_is_c05_ring = bool(
        len(arc)
        == EXPECTED_EXCLUDED_ARC_POSITIONS
        and
        arc[
            "island_id"
        ]
        .astype(str)
        .eq("C05")
        .all()
        and
        arc[
            "side"
        ]
        .astype(str)
        .eq("RING")
        .all()
    )

    add(
        "floor_rows",
        len(floor),
        EXPECTED_FLOOR_ROWS,
        len(floor)
        == EXPECTED_FLOOR_ROWS,
    )

    add(
        "floor_unique_machine_no",
        floor[
            "machine_no"
        ].nunique(),
        EXPECTED_FLOOR_ROWS,
        floor[
            "machine_no"
        ].nunique()
        == EXPECTED_FLOOR_ROWS,
    )

    add(
        "floor_duplicate_machine_no",
        int(
            floor[
                "machine_no"
            ]
            .duplicated()
            .sum()
        ),
        0,
        not floor[
            "machine_no"
        ]
        .duplicated()
        .any(),
    )

    add(
        "eligible_rect_positions",
        rect_count,
        EXPECTED_ELIGIBLE_POSITIONS,
        rect_count
        == EXPECTED_ELIGIBLE_POSITIONS,
    )

    add(
        "excluded_arc_positions",
        len(arc),
        EXPECTED_EXCLUDED_ARC_POSITIONS,
        len(arc)
        == EXPECTED_EXCLUDED_ARC_POSITIONS,
    )

    add(
        "arc_only_c05_ring",
        int(
            arc_is_c05_ring
        ),
        1,
        arc_is_c05_ring,
    )

    add(
        "linear_groups",
        linear_groups,
        EXPECTED_LINEAR_GROUPS,
        linear_groups
        == EXPECTED_LINEAR_GROUPS,
    )

    add(
        "linear_segments",
        len(segments),
        EXPECTED_LINEAR_SEGMENTS,
        len(segments)
        == EXPECTED_LINEAR_SEGMENTS,
    )

    add(
        "h1_edges",
        len(edges),
        EXPECTED_H1_EDGES,
        len(edges)
        == EXPECTED_H1_EDGES,
    )

    add(
        "h2_windows",
        len(triples),
        EXPECTED_H2_WINDOWS,
        len(triples)
        == EXPECTED_H2_WINDOWS,
    )

    add(
        "h3_windows",
        len(fives),
        EXPECTED_H3_WINDOWS,
        len(fives)
        == EXPECTED_H3_WINDOWS,
    )

    qa = pd.DataFrame(
        rows
    )

    bad = qa.loc[
        ~qa["pass"]
    ]

    if not bad.empty:
        raise RuntimeError(
            "Floor geometry QA failed:\n"
            + bad.to_string(
                index=False
            )
        )

    return qa


# ============================================================
# Analysis panel
# ============================================================

def build_analysis_panel(
    stage1: pd.DataFrame,
    floor: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    floor_cols = floor[
        [
            "machine_no",
            "island_id",
            "side",
            "position_order",
            "slot_index",
            "shape",
        ]
    ].copy()

    panel = stage1.merge(
        floor_cols,
        on="machine_no",
        how="left",
        validate="many_to_one",
    )

    if len(panel) != len(stage1):
        raise RuntimeError(
            "Panel/Floor merge row count changed."
        )

    if panel[
        [
            "island_id",
            "side",
            "slot_index",
            "shape",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Panel/Floor merge contains missing mapping."
        )

    eligible_set = set(
        eligible[
            "machine_no"
        ]
        .astype(int)
        .tolist()
    )

    panel[
        "phase1a_linear_eligible"
    ] = (
        panel[
            "machine_no"
        ]
        .isin(
            eligible_set
        )
    )

    counts = (
        panel
        .groupby(
            "target_date",
            sort=True,
        )[
            "phase1a_linear_eligible"
        ]
        .sum()
    )

    if not counts.eq(
        EXPECTED_ELIGIBLE_POSITIONS
    ).all():
        raise RuntimeError(
            "Eligible position count mismatch by date."
        )

    return panel


# ============================================================
# Spatial statistics
# ============================================================

def observed_statistics(
    strong: np.ndarray,
    edges: np.ndarray,
    triples: np.ndarray,
    fives: np.ndarray,
) -> tuple[int, int, int]:
    h1 = int(
        np.sum(
            strong[
                edges[:, 0]
            ]
            &
            strong[
                edges[:, 1]
            ]
        )
    )

    h2 = int(
        np.sum(
            strong[
                triples[:, 0]
            ]
            &
            strong[
                triples[:, 1]
            ]
            &
            strong[
                triples[:, 2]
            ]
        )
    )

    five_counts = (
        strong[
            fives
        ]
        .sum(
            axis=1
        )
    )

    h3 = int(
        np.sum(
            five_counts >= 3
        )
    )

    return (
        h1,
        h2,
        h3,
    )


# ============================================================
# Machine-composition-conditioned permutation
# ============================================================

def build_machine_groups(
    day: pd.DataFrame,
    threshold: int,
) -> tuple[
    list[
        tuple[
            str,
            np.ndarray,
            int,
        ]
    ],
    pd.DataFrame,
]:
    groups = []

    qa_rows = []

    for machine_name, g in day.groupby(
        "actual_machine_name",
        sort=True,
        dropna=False,
    ):
        positions = (
            g[
                "eligible_position_index"
            ]
            .to_numpy(
                dtype=np.int32
            )
        )

        strong_n = int(
            (
                g[
                    "actual_diff"
                ]
                >= threshold
            ).sum()
        )

        n = len(
            positions
        )

        groups.append(
            (
                str(
                    machine_name
                ),
                positions,
                strong_n,
            )
        )

        qa_rows.append(
            {
                "actual_machine_name":
                    str(
                        machine_name
                    ),
                "eligible_machine_count":
                    n,
                "strong_n":
                    strong_n,
                "strong_rate_pct":
                    (
                        100.0
                        * strong_n
                        / n
                    ),
                "permutation_variable":
                    bool(
                        0
                        < strong_n
                        < n
                    ),
            }
        )

    total_positions = sum(
        len(positions)
        for _, positions, _
        in groups
    )

    total_strong = sum(
        strong_n
        for _, _, strong_n
        in groups
    )

    if (
        total_positions
        != EXPECTED_ELIGIBLE_POSITIONS
    ):
        raise RuntimeError(
            "Machine-group position total mismatch: "
            f"{total_positions}"
        )

    return (
        groups,
        pd.DataFrame(
            qa_rows
        ),
    )


def conditioned_permutation_statistics(
    groups: list[
        tuple[
            str,
            np.ndarray,
            int,
        ]
    ],
    n_positions: int,
    edges: np.ndarray,
    triples: np.ndarray,
    fives: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    h1_all = np.empty(
        PERMUTATIONS,
        dtype=np.int32,
    )

    h2_all = np.empty(
        PERMUTATIONS,
        dtype=np.int32,
    )

    h3_all = np.empty(
        PERMUTATIONS,
        dtype=np.int32,
    )

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
                n_positions,
            ),
            dtype=bool,
        )

        for (
            _machine_name,
            positions,
            strong_n,
        ) in groups:
            group_n = len(
                positions
            )

            if strong_n == 0:
                continue

            if (
                strong_n
                == group_n
            ):
                labels[
                    :,
                    positions,
                ] = True

                continue

            random_scores = rng.random(
                (
                    batch_n,
                    group_n,
                ),
                dtype=np.float64,
            )

            selected_local = (
                np.argpartition(
                    random_scores,
                    kth=strong_n - 1,
                    axis=1,
                )[
                    :,
                    :strong_n,
                ]
            )

            selected_global = positions[
                selected_local
            ]

            row_index = np.arange(
                batch_n
            )[
                :,
                None
            ]

            labels[
                row_index,
                selected_global,
            ] = True

        h1 = np.sum(
            labels[
                :,
                edges[:, 0]
            ]
            &
            labels[
                :,
                edges[:, 1]
            ],
            axis=1,
        )

        h2 = np.sum(
            labels[
                :,
                triples[:, 0]
            ]
            &
            labels[
                :,
                triples[:, 1]
            ]
            &
            labels[
                :,
                triples[:, 2]
            ],
            axis=1,
        )

        five_counts = (
            labels[
                :,
                fives
            ]
            .sum(
                axis=2
            )
        )

        h3 = np.sum(
            five_counts >= 3,
            axis=1,
        )

        sl = slice(
            completed,
            completed + batch_n,
        )

        h1_all[
            sl
        ] = h1

        h2_all[
            sl
        ] = h2

        h3_all[
            sl
        ] = h3

        completed += batch_n

    return (
        h1_all,
        h2_all,
        h3_all,
    )


# ============================================================
# Daily conditioned analysis
# ============================================================

def run_daily_analysis(
    panel: pd.DataFrame,
    eligible: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    edges: np.ndarray,
    triples: np.ndarray,
    fives: np.ndarray,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    eligible_order = (
        eligible
        .sort_values(
            "eligible_position_index"
        )
        [
            [
                "machine_no",
                "eligible_position_index",
            ]
        ]
        .copy()
    )

    machine_to_position = dict(
        zip(
            eligible_order[
                "machine_no"
            ].astype(int),
            eligible_order[
                "eligible_position_index"
            ].astype(int),
        )
    )

    n_positions = len(
        eligible_order
    )

    dev_set = set(
        target_dates[
            :EXPECTED_DEV_DAYS
        ]
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    observed_rows = []

    permutation_parts = []

    group_qa_parts = []

    total_jobs = (
        len(target_dates)
        * len(
            ALL_THRESHOLDS
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
                &
                panel[
                    "phase1a_linear_eligible"
                ]
            ]
            .copy()
        )

        day[
            "eligible_position_index"
        ] = (
            day[
                "machine_no"
            ]
            .map(
                machine_to_position
            )
        )

        if (
            day[
                "eligible_position_index"
            ].isna().any()
        ):
            raise RuntimeError(
                "Eligible position mapping missing: "
                f"{target_date.date()}"
            )

        day[
            "eligible_position_index"
        ] = (
            day[
                "eligible_position_index"
            ]
            .astype(int)
        )

        day = (
            day
            .sort_values(
                "eligible_position_index"
            )
            .reset_index(
                drop=True
            )
        )

        if (
            len(day)
            != EXPECTED_ELIGIBLE_POSITIONS
        ):
            raise RuntimeError(
                "Daily eligible count mismatch: "
                f"{target_date.date()}"
            )

        period = (
            "DEV"
            if target_date
            in dev_set
            else "LATER"
        )

        actual_diff = (
            day[
                "actual_diff"
            ]
            .to_numpy(
                dtype=float
            )
        )

        for threshold in ALL_THRESHOLDS:
            job_no += 1

            strong = (
                actual_diff
                >= threshold
            )

            strong_n = int(
                strong.sum()
            )

            (
                obs_h1,
                obs_h2,
                obs_h3,
            ) = observed_statistics(
                strong,
                edges,
                triples,
                fives,
            )

            (
                groups,
                group_qa,
            ) = build_machine_groups(
                day,
                threshold,
            )

            group_qa.insert(
                0,
                "threshold_label",
                THRESHOLD_LABELS[
                    threshold
                ],
            )

            group_qa.insert(
                0,
                "threshold",
                threshold,
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

            (
                null_h1,
                null_h2,
                null_h3,
            ) = conditioned_permutation_statistics(
                groups=groups,
                n_positions=n_positions,
                edges=edges,
                triples=triples,
                fives=fives,
                rng=rng,
            )

            h1_mean = float(
                null_h1.mean()
            )

            h2_mean = float(
                null_h2.mean()
            )

            h3_mean = float(
                null_h3.mean()
            )

            observed_rows.append(
                {
                    "target_date":
                        target_date.date(),
                    "period":
                        period,
                    "threshold":
                        threshold,
                    "threshold_label":
                        THRESHOLD_LABELS[
                            threshold
                        ],
                    "eligible_positions":
                        n_positions,
                    "machine_group_count":
                        len(groups),
                    "strong_n":
                        strong_n,
                    "strong_rate_pct":
                        (
                            100.0
                            * strong_n
                            / n_positions
                        ),
                    "h1_observed":
                        obs_h1,
                    "h1_conditioned_null_mean":
                        h1_mean,
                    "h1_observed_minus_conditioned_expected":
                        (
                            obs_h1
                            - h1_mean
                        ),
                    "h2_observed":
                        obs_h2,
                    "h2_conditioned_null_mean":
                        h2_mean,
                    "h2_observed_minus_conditioned_expected":
                        (
                            obs_h2
                            - h2_mean
                        ),
                    "h3_observed":
                        obs_h3,
                    "h3_conditioned_null_mean":
                        h3_mean,
                    "h3_observed_minus_conditioned_expected":
                        (
                            obs_h3
                            - h3_mean
                        ),
                }
            )

            permutation_parts.append(
                pd.DataFrame(
                    {
                        "target_date":
                            np.repeat(
                                str(
                                    target_date.date()
                                ),
                                PERMUTATIONS,
                            ),
                        "period":
                            np.repeat(
                                period,
                                PERMUTATIONS,
                            ),
                        "threshold":
                            np.repeat(
                                threshold,
                                PERMUTATIONS,
                            ),
                        "threshold_label":
                            np.repeat(
                                THRESHOLD_LABELS[
                                    threshold
                                ],
                                PERMUTATIONS,
                            ),
                        "permutation_id":
                            np.arange(
                                1,
                                PERMUTATIONS + 1,
                                dtype=np.int32,
                            ),
                        "eligible_positions":
                            np.repeat(
                                n_positions,
                                PERMUTATIONS,
                            ),
                        "strong_n":
                            np.repeat(
                                strong_n,
                                PERMUTATIONS,
                            ),
                        "machine_group_count":
                            np.repeat(
                                len(groups),
                                PERMUTATIONS,
                            ),
                        "h1_conditioned_null_count":
                            null_h1,
                        "h2_conditioned_null_count":
                            null_h2,
                        "h3_conditioned_null_count":
                            null_h3,
                    }
                )
            )

            print(
                "[COND-PERM] "
                f"{job_no}/{total_jobs} "
                f"date={target_date.date()} "
                f"period={period} "
                f"threshold={threshold} "
                f"groups={len(groups)} "
                f"strong_n={strong_n} "
                f"H1={obs_h1} "
                f"H2={obs_h2} "
                f"H3={obs_h3}"
            )

    observed_df = pd.DataFrame(
        observed_rows
    )

    permutation_df = pd.concat(
        permutation_parts,
        ignore_index=True,
    )

    group_qa_df = pd.concat(
        group_qa_parts,
        ignore_index=True,
    )

    expected_rows = (
        EXPECTED_TARGET_DAYS
        * len(
            ALL_THRESHOLDS
        )
        * PERMUTATIONS
    )

    if (
        len(
            permutation_df
        )
        != expected_rows
    ):
        raise RuntimeError(
            "Conditioned permutation row mismatch: "
            f"{len(permutation_df)} "
            f"!= {expected_rows}"
        )

    return (
        observed_df,
        permutation_df,
        group_qa_df,
    )


# ============================================================
# Global Phase1A compatibility QA
# ============================================================

def verify_global_observed_match(
    conditioned_daily: pd.DataFrame,
    global_daily: pd.DataFrame,
) -> pd.DataFrame:
    required_global = [
        "target_date",
        "threshold",
        "strong_n",
        "h1_observed",
        "h2_observed",
        "h3_observed",
    ]

    require_columns(
        global_daily,
        required_global,
        "Global Phase1A daily",
    )

    left = conditioned_daily[
        [
            "target_date",
            "threshold",
            "strong_n",
            "h1_observed",
            "h2_observed",
            "h3_observed",
        ]
    ].copy()

    left[
        "target_date"
    ] = pd.to_datetime(
        left[
            "target_date"
        ],
        errors="raise",
    ).dt.normalize()

    right = global_daily[
        required_global
    ].copy()

    merged = left.merge(
        right,
        on=[
            "target_date",
            "threshold",
        ],
        how="outer",
        suffixes=(
            "_conditioned_run",
            "_global_phase1a",
        ),
        indicator=True,
        validate="one_to_one",
    )

    if not merged[
        "_merge"
    ].eq(
        "both"
    ).all():
        raise RuntimeError(
            "Global Phase1A daily key mismatch."
        )

    comparisons = [
        "strong_n",
        "h1_observed",
        "h2_observed",
        "h3_observed",
    ]

    rows = []

    for col in comparisons:
        mismatch = int(
            (
                merged[
                    f"{col}_conditioned_run"
                ]
                !=
                merged[
                    f"{col}_global_phase1a"
                ]
            ).sum()
        )

        rows.append(
            {
                "qa_item":
                    f"global_match_{col}",
                "mismatch_count":
                    mismatch,
                "expected":
                    0,
                "pass":
                    mismatch == 0,
            }
        )

    qa = pd.DataFrame(
        rows
    )

    bad = qa.loc[
        ~qa[
            "pass"
        ]
    ]

    if not bad.empty:
        raise RuntimeError(
            "Global Phase1A observed compatibility QA failed:\n"
            + bad.to_string(
                index=False
            )
        )

    return qa


# ============================================================
# Period summaries
# ============================================================

METRICS = {
    "H1_PHYSICAL_STRONG_STRONG_PAIR":
        (
            "h1_observed",
            "h1_conditioned_null_count",
            "h1_conditioned_null_mean",
        ),
    "H2_PHYSICAL_STRONG_TRIPLE":
        (
            "h2_observed",
            "h2_conditioned_null_count",
            "h2_conditioned_null_mean",
        ),
    "H3_5SLOT_GE3_STRONG_CLUSTER":
        (
            "h3_observed",
            "h3_conditioned_null_count",
            "h3_conditioned_null_mean",
        ),
}


def period_filter(
    df: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    if period == "FULL":
        return df.copy()

    return df.loc[
        df[
            "period"
        ].eq(
            period
        )
    ].copy()


def summarize_period(
    observed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    threshold: int,
    period: str,
) -> pd.DataFrame:
    obs = period_filter(
        observed_df.loc[
            observed_df[
                "threshold"
            ].eq(
                threshold
            )
        ],
        period,
    )

    perm = period_filter(
        permutation_df.loc[
            permutation_df[
                "threshold"
            ].eq(
                threshold
            )
        ],
        period,
    )

    if obs.empty:
        raise RuntimeError(
            "Observed period is empty: "
            f"{threshold}, {period}"
        )

    rows = []

    for metric, (
        observed_col,
        null_col,
        daily_null_mean_col,
    ) in METRICS.items():
        observed_count = float(
            obs[
                observed_col
            ].sum()
        )

        null_totals = (
            perm
            .groupby(
                "permutation_id",
                sort=True,
            )[
                null_col
            ]
            .sum()
            .to_numpy(
                dtype=float
            )
        )

        if (
            len(null_totals)
            != PERMUTATIONS
        ):
            raise RuntimeError(
                "Conditioned period null replicate mismatch."
            )

        conditioned_null_mean = float(
            null_totals.mean()
        )

        ci_low, ci_high = (
            percentile_ci95(
                null_totals
            )
        )

        raw_p = empirical_upper_p(
            null_totals,
            observed_count,
        )

        expected_from_daily = float(
            obs[
                daily_null_mean_col
            ].sum()
        )

        if not np.isclose(
            conditioned_null_mean,
            expected_from_daily,
            atol=1e-10,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Conditioned period mean mismatch."
            )

        excess = (
            observed_count
            - conditioned_null_mean
        )

        rows.append(
            {
                "period":
                    period,
                "period_min_date":
                    obs[
                        "target_date"
                    ].min(),
                "period_max_date":
                    obs[
                        "target_date"
                    ].max(),
                "observed_target_days":
                    int(
                        obs[
                            "target_date"
                        ].nunique()
                    ),
                "threshold":
                    threshold,
                "threshold_label":
                    THRESHOLD_LABELS[
                        threshold
                    ],
                "metric":
                    metric,
                "strong_machine_days":
                    int(
                        obs[
                            "strong_n"
                        ].sum()
                    ),
                "observed_count":
                    observed_count,
                "conditioned_null_mean":
                    conditioned_null_mean,
                "observed_minus_conditioned_expected":
                    excess,
                "observed_div_conditioned_expected":
                    safe_ratio(
                        observed_count,
                        conditioned_null_mean,
                    ),
                "conditioned_null_ci95_low":
                    ci_low,
                "conditioned_null_ci95_high":
                    ci_high,
                "empirical_p_raw":
                    raw_p,
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
        )

    return pd.DataFrame(
        rows
    )


def build_all_summaries(
    observed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
) -> pd.DataFrame:
    parts = []

    for threshold in ALL_THRESHOLDS:
        for period in (
            "FULL",
            "DEV",
            "LATER",
        ):
            parts.append(
                summarize_period(
                    observed_df,
                    permutation_df,
                    threshold,
                    period,
                )
            )

    summary = pd.concat(
        parts,
        ignore_index=True,
    )

    summary[
        "holm_p_later_primary"
    ] = np.nan

    later_primary_mask = (
        summary[
            "threshold"
        ].eq(
            PRIMARY_THRESHOLD
        )
        &
        summary[
            "period"
        ].eq(
            "LATER"
        )
    )

    later_primary = (
        summary.loc[
            later_primary_mask
        ]
        .copy()
    )

    raw_map = dict(
        zip(
            later_primary[
                "metric"
            ],
            later_primary[
                "empirical_p_raw"
            ],
        )
    )

    adjusted = holm_adjust(
        raw_map
    )

    for metric, adj_p in adjusted.items():
        mask = (
            later_primary_mask
            &
            summary[
                "metric"
            ].eq(
                metric
            )
        )

        summary.loc[
            mask,
            "holm_p_later_primary",
        ] = adj_p

    return summary


# ============================================================
# Leave-one-day-out robustness
# ============================================================

def leave_one_day_out_robustness(
    observed_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for threshold in ALL_THRESHOLDS:
        threshold_df = (
            observed_df.loc[
                observed_df[
                    "threshold"
                ].eq(
                    threshold
                )
            ]
            .copy()
        )

        for period in (
            "FULL",
            "DEV",
            "LATER",
        ):
            x = period_filter(
                threshold_df,
                period,
            )

            for metric, (
                observed_col,
                _null_col,
                daily_null_mean_col,
            ) in METRICS.items():
                overall_effect = float(
                    (
                        x[
                            observed_col
                        ]
                        -
                        x[
                            daily_null_mean_col
                        ]
                    ).sum()
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

                for _, row in x.iterrows():
                    daily_effect = float(
                        row[
                            observed_col
                        ]
                        -
                        row[
                            daily_null_mean_col
                        ]
                    )

                    loo_effect = (
                        overall_effect
                        - daily_effect
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

                    rows.append(
                        {
                            "period":
                                period,
                            "threshold":
                                threshold,
                            "threshold_label":
                                THRESHOLD_LABELS[
                                    threshold
                                ],
                            "metric":
                                metric,
                            "overall_effect":
                                overall_effect,
                            "overall_direction":
                                overall_direction,
                            "excluded_target_date":
                                row[
                                    "target_date"
                                ],
                            "excluded_day_effect":
                                daily_effect,
                            "leave_one_day_out_effect":
                                loo_effect,
                            "leave_one_day_out_direction":
                                loo_direction,
                            "direction_flip":
                                bool(
                                    loo_direction
                                    != overall_direction
                                ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def build_loo_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouped = detail.groupby(
        [
            "period",
            "threshold",
            "threshold_label",
            "metric",
        ],
        sort=True,
    )

    for (
        period,
        threshold,
        threshold_label,
        metric,
    ), g in grouped:
        rows.append(
            {
                "period":
                    period,
                "threshold":
                    threshold,
                "threshold_label":
                    threshold_label,
                "metric":
                    metric,
                "overall_effect":
                    float(
                        g[
                            "overall_effect"
                        ].iloc[0]
                    ),
                "overall_direction":
                    g[
                        "overall_direction"
                    ].iloc[0],
                "leave_one_day_out_min_effect":
                    float(
                        g[
                            "leave_one_day_out_effect"
                        ].min()
                    ),
                "leave_one_day_out_max_effect":
                    float(
                        g[
                            "leave_one_day_out_effect"
                        ].max()
                    ),
                "direction_flip_count":
                    int(
                        g[
                            "direction_flip"
                        ].sum()
                    ),
                "direction_flip_any":
                    bool(
                        g[
                            "direction_flip"
                        ].any()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Global null vs conditioned null comparison
# ============================================================

def compare_with_global_null(
    conditioned_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        global_summary,
        [
            "period",
            "threshold",
            "metric",
            "observed_count",
            "null_mean",
            "observed_minus_expected",
            "observed_div_expected",
            "empirical_p_raw",
        ],
        "Global Phase1A summary",
    )

    global_x = global_summary[
        [
            "period",
            "threshold",
            "metric",
            "observed_count",
            "null_mean",
            "observed_minus_expected",
            "observed_div_expected",
            "empirical_p_raw",
        ]
    ].copy()

    global_x = global_x.rename(
        columns={
            "observed_count":
                "global_observed_count",
            "null_mean":
                "global_null_mean",
            "observed_minus_expected":
                "global_null_excess",
            "observed_div_expected":
                "global_null_ratio",
            "empirical_p_raw":
                "global_empirical_p_raw",
        }
    )

    conditioned_x = conditioned_summary.copy()

    merged = conditioned_x.merge(
        global_x,
        on=[
            "period",
            "threshold",
            "metric",
        ],
        how="left",
        validate="one_to_one",
    )

    if (
        merged[
            "global_null_mean"
        ].isna().any()
    ):
        raise RuntimeError(
            "Global summary comparison merge missing rows."
        )

    observed_match = np.isclose(
        merged[
            "observed_count"
        ].to_numpy(
            dtype=float
        ),
        merged[
            "global_observed_count"
        ].to_numpy(
            dtype=float
        ),
        atol=0.0,
        rtol=0.0,
    )

    if not observed_match.all():
        raise RuntimeError(
            "Observed counts differ between "
            "global and conditioned analyses."
        )

    merged[
        "conditioned_null_excess"
    ] = (
        merged[
            "observed_minus_conditioned_expected"
        ]
    )

    merged[
        "conditioned_null_ratio"
    ] = (
        merged[
            "observed_div_conditioned_expected"
        ]
    )

    # Ratio effect is measured relative to 1.0.
    global_ratio_effect = (
        merged[
            "global_null_ratio"
        ]
        - 1.0
    )

    conditioned_ratio_effect = (
        merged[
            "conditioned_null_ratio"
        ]
        - 1.0
    )

    merged[
        "ratio_effect_retained_fraction"
    ] = np.where(
        global_ratio_effect
        != 0,
        conditioned_ratio_effect
        / global_ratio_effect,
        np.nan,
    )

    merged[
        "ratio_attenuation_fraction"
    ] = (
        1.0
        -
        merged[
            "ratio_effect_retained_fraction"
        ]
    )

    merged[
        "excess_effect_retained_fraction"
    ] = np.where(
        merged[
            "global_null_excess"
        ]
        != 0,
        merged[
            "conditioned_null_excess"
        ]
        /
        merged[
            "global_null_excess"
        ],
        np.nan,
    )

    merged[
        "excess_attenuation_fraction"
    ] = (
        1.0
        -
        merged[
            "excess_effect_retained_fraction"
        ]
    )

    return merged


# ============================================================
# Primary / sensitivity outputs
# ============================================================

def add_loo_columns(
    summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
) -> pd.DataFrame:
    loo = loo_summary[
        [
            "period",
            "threshold",
            "metric",
            "direction_flip_count",
            "direction_flip_any",
            "leave_one_day_out_min_effect",
            "leave_one_day_out_max_effect",
        ]
    ].copy()

    return summary.merge(
        loo,
        on=[
            "period",
            "threshold",
            "metric",
        ],
        how="left",
        validate="one_to_one",
    )


def build_primary_outputs(
    comparison: pd.DataFrame,
    loo_summary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    with_loo = add_loo_columns(
        comparison,
        loo_summary,
    )

    full = (
        with_loo.loc[
            with_loo[
                "threshold"
            ].eq(
                PRIMARY_THRESHOLD
            )
            &
            with_loo[
                "period"
            ].eq(
                "FULL"
            )
        ]
        .copy()
    )

    dev_later = (
        with_loo.loc[
            with_loo[
                "threshold"
            ].eq(
                PRIMARY_THRESHOLD
            )
            &
            with_loo[
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
        full,
        dev_later,
    )


def build_sensitivity_output(
    comparison: pd.DataFrame,
    loo_summary: pd.DataFrame,
) -> pd.DataFrame:
    with_loo = add_loo_columns(
        comparison,
        loo_summary,
    )

    return (
        with_loo.loc[
            with_loo[
                "threshold"
            ].isin(
                SECONDARY_THRESHOLDS
            )
        ]
        .copy()
    )


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
            "purpose",
            (
                "Machine-composition-conditioned "
                "robustness check for Phase1A "
                "spatial concentration"
            ),
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
            "production_write",
            "NONE",
        ),
        (
            "formal_forward_write",
            "NONE",
        ),
        (
            "ranking_change",
            "NONE",
        ),
        (
            "weight_tuning",
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
            "phase1b_implemented",
            "NO",
        ),
        (
            "phase1c_implemented",
            "NO",
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
            "floor_map_path",
            str(
                FLOOR_MAP
            ),
        ),
        (
            "floor_map_sha256",
            sha256_file(
                FLOOR_MAP
            ),
        ),
        (
            "global_phase1a_daily_path",
            str(
                GLOBAL_DAILY_OBSERVED
            ),
        ),
        (
            "global_phase1a_daily_sha256",
            sha256_file(
                GLOBAL_DAILY_OBSERVED
            ),
        ),
        (
            "global_phase1a_summary_path",
            str(
                GLOBAL_ALL_SUMMARY
            ),
        ),
        (
            "global_phase1a_summary_sha256",
            sha256_file(
                GLOBAL_ALL_SUMMARY
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
            "external_holdout",
            "NO",
        ),
        (
            "eligible_geometry",
            "RECT_LINEAR_ONLY",
        ),
        (
            "eligible_positions",
            EXPECTED_ELIGIBLE_POSITIONS,
        ),
        (
            "excluded_geometry",
            "C05_RING_ARC",
        ),
        (
            "excluded_positions",
            EXPECTED_EXCLUDED_ARC_POSITIONS,
        ),
        (
            "linear_groups",
            EXPECTED_LINEAR_GROUPS,
        ),
        (
            "linear_segments",
            EXPECTED_LINEAR_SEGMENTS,
        ),
        (
            "h1_edges",
            EXPECTED_H1_EDGES,
        ),
        (
            "h2_windows",
            EXPECTED_H2_WINDOWS,
        ),
        (
            "h3_windows",
            EXPECTED_H3_WINDOWS,
        ),
        (
            "primary_threshold",
            PRIMARY_THRESHOLD,
        ),
        (
            "secondary_thresholds",
            "|".join(
                map(
                    str,
                    SECONDARY_THRESHOLDS,
                )
            ),
        ),
        (
            "conditioning_variable",
            "actual_machine_name",
        ),
        (
            "conditioned_randomization",
            (
                "Within each target_date x "
                "actual_machine_name group, "
                "eligible positions and strong_n "
                "are fixed; strong labels are "
                "shuffled only within that group"
            ),
        ),
        (
            "permutations_per_day_threshold",
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
            "multiple_testing",
            (
                "Holm adjustment for LATER "
                "PRIMARY H1/H2/H3"
            ),
        ),
        (
            "single_day_robustness",
            (
                "leave-one-day-out conditioned "
                "effect direction check"
            ),
        ),
        (
            "final_research_classification",
            (
                "NOT_ASSIGNED_BY_02_2; "
                "RETURN_TO_03_2"
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
        "Phase 1A-R MACHINE-COMPOSITION ROBUSTNESS"
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
        "[1/8] Loading Stage1 panel..."
    )

    stage1 = load_stage1_panel()

    (
        input_qa,
        target_dates,
    ) = stage1_input_qa(
        stage1
    )

    print(
        "Stage1 input QA: PASS"
    )

    print()
    print(
        "[2/8] Loading fixed Phase1A geometry..."
    )

    floor = load_floor_map()

    (
        eligible,
        segments,
        edges,
        triples,
        fives,
    ) = build_linear_geometry(
        floor
    )

    floor_qa = floor_geometry_qa(
        floor,
        eligible,
        segments,
        edges,
        triples,
        fives,
    )

    print(
        "Floor geometry QA: PASS"
    )

    print(
        f"eligible positions : {len(eligible)}"
    )

    print(
        f"linear segments    : {len(segments)}"
    )

    print(
        f"H1 edges           : {len(edges)}"
    )

    print(
        f"H2 windows         : {len(triples)}"
    )

    print(
        f"H3 windows         : {len(fives)}"
    )

    print()
    print(
        "[3/8] Building analysis panel..."
    )

    panel = build_analysis_panel(
        stage1,
        floor,
        eligible,
    )

    print(
        "Panel/Floor QA: PASS"
    )

    print()
    print(
        "[4/8] Loading existing global Phase1A results..."
    )

    (
        global_daily,
        global_summary,
    ) = load_global_phase1a()

    print(
        "Global Phase1A inputs loaded."
    )

    print()
    print(
        "[5/8] Running machine-composition-conditioned "
        "permutations..."
    )

    print(
        f"permutations per date/threshold: "
        f"{PERMUTATIONS:,}"
    )

    print(
        f"random seed: {RANDOM_SEED}"
    )

    (
        daily_observed,
        daily_permutations,
        machine_group_qa,
    ) = run_daily_analysis(
        panel=panel,
        eligible=eligible,
        target_dates=target_dates,
        edges=edges,
        triples=triples,
        fives=fives,
    )

    print()
    print(
        "[6/8] Verifying observed counts against "
        "original Phase1A..."
    )

    global_match_qa = (
        verify_global_observed_match(
            daily_observed,
            global_daily,
        )
    )

    print(
        "Global observed compatibility QA: PASS"
    )

    print()
    print(
        "[7/8] Building summaries and robustness checks..."
    )

    conditioned_summary = (
        build_all_summaries(
            daily_observed,
            daily_permutations,
        )
    )

    loo_detail = (
        leave_one_day_out_robustness(
            daily_observed
        )
    )

    loo_summary = (
        build_loo_summary(
            loo_detail
        )
    )

    comparison = (
        compare_with_global_null(
            conditioned_summary,
            global_summary,
        )
    )

    (
        primary_full,
        primary_dev_later,
    ) = build_primary_outputs(
        comparison,
        loo_summary,
    )

    sensitivity = (
        build_sensitivity_output(
            comparison,
            loo_summary,
        )
    )

    metadata = build_metadata(
        target_dates
    )

    print()
    print(
        "[8/8] Writing outputs..."
    )

    outputs: dict[
        str,
        pd.DataFrame,
    ] = {
        "01_input_qa.csv":
            input_qa,
        "02_floor_geometry_qa.csv":
            floor_qa,
        "03_global_observed_match_qa.csv":
            global_match_qa,
        "04_machine_group_qa.csv":
            machine_group_qa,
        "05_daily_conditioned_observed.csv":
            daily_observed,
        "06_daily_conditioned_permutations.csv":
            daily_permutations,
        "07_primary_full_comparison.csv":
            primary_full,
        "08_primary_dev_later_comparison.csv":
            primary_dev_later,
        "09_sensitivity_comparison.csv":
            sensitivity,
        "10_all_conditioned_summary.csv":
            conditioned_summary,
        "11_global_vs_conditioned_comparison.csv":
            comparison,
        "12_leave_one_day_out_detail.csv":
            loo_detail,
        "13_leave_one_day_out_summary.csv":
            loo_summary,
        "14_linear_segments.csv":
            segments,
        "15_metadata.csv":
            metadata,
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
        "random_seed":
            RANDOM_SEED,
        "permutations":
            PERMUTATIONS,
        "primary_threshold":
            PRIMARY_THRESHOLD,
        "secondary_thresholds":
            list(
                SECONDARY_THRESHOLDS
            ),
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
        / "16_output_manifest.json"
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
        "Phase 1A-R MACHINE-COMPOSITION ROBUSTNESS completed."
    )

    print(
        "=" * 78
    )

    print(
        f"output_dir        : {OUTPUT_DIR}"
    )

    print(
        "input_QA          : PASS"
    )

    print(
        "floor_QA          : PASS"
    )

    print(
        "global_match_QA   : PASS"
    )

    print(
        "conditioning      : target_date x actual_machine_name"
    )

    print(
        f"eligible_positions: {EXPECTED_ELIGIBLE_POSITIONS}"
    )

    print(
        f"H1_edges          : {EXPECTED_H1_EDGES}"
    )

    print(
        f"H2_windows        : {EXPECTED_H2_WINDOWS}"
    )

    print(
        f"H3_windows        : {EXPECTED_H3_WINDOWS}"
    )

    print(
        f"permutations      : {PERMUTATIONS}"
    )

    print(
        f"random_seed       : {RANDOM_SEED}"
    )

    print(
        "production_write  : NONE"
    )

    print(
        "Champion change   : NONE"
    )

    print(
        "Formal change     : NONE"
    )

    print(
        "auto_promotion    : NONE"
    )

    print(
        "git_action        : NONE"
    )

    display_cols = [
        "period",
        "metric",
        "observed_count",
        "global_null_mean",
        "conditioned_null_mean",
        "global_null_ratio",
        "conditioned_null_ratio",
        "global_null_excess",
        "conditioned_null_excess",
        "ratio_attenuation_fraction",
        "excess_attenuation_fraction",
        "empirical_p_raw",
        "holm_p_later_primary",
        "direction_flip_any",
    ]

    print()
    print(
        "PRIMARY FULL global vs conditioned:"
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
        "PRIMARY DEV/LATER global vs conditioned:"
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
        "IMPORTANT:"
    )

    print(
        "02_2 does not assign final research classification."
    )

    print(
        "Return global-vs-conditioned results to 03_2."
    )

    print(
        "Do not start Phase1B before 03_2 review."
    )


if __name__ == "__main__":
    main()