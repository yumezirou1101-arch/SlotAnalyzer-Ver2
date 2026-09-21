from __future__ import annotations

"""
STORE_BEHAVIOR_RESEARCH
Phase 1A SPATIAL CONCENTRATION

OFFLINE RESEARCH ONLY

Purpose
-------
Evaluate whether same-day high-difference machines are spatially
concentrated on the fixed physical floor layout more often than expected
under same-day random reassignment.

This script does NOT:
- write to production
- change Champion weights
- change Forward Guard
- change Formal Forward
- change production ranking
- change neighbor_avg
- auto-promote any result
- reopen paused Floor research

Research design is pre-defined by 03_2.

PRIMARY threshold
-----------------
actual_diff >= +2000

SECONDARY sensitivity thresholds
--------------------------------
actual_diff >= +1000
actual_diff >= +4000

PRIMARY spatial hypotheses
--------------------------
H1:
    physical strong-strong adjacent edge count

H2:
    3-slot consecutive physical window where all 3 are strong

H3:
    fixed 5-slot consecutive physical window where >= 3 are strong

Eligible geometry
-----------------
Only fixed LINEAR physical geometry is eligible.

The canonical Floor Map contains:
- 497 rect positions
- 17 arc positions in C05 / RING

C05 / RING / arc is excluded from this Phase 1A analysis because the
pre-defined PRIMARY scope is linear geometry only.

Known slot-index gaps split physical segments:
- C03 / S : 842 <-> 843
- C08 / N : 1015 <-> 1014

Expected fixed geometry QA:
- eligible rect positions = 497
- linear groups = 33
- linear segments = 35
- H1 edges = 462
- H2 triple windows = 427
- H3 five-slot windows = 358

Randomization
-------------
For each target day and each threshold:

1. Count strong machines among eligible linear positions.
2. Fix that strong count exactly.
3. Randomly reassign strong/non-strong labels across the same 497
   eligible positions.
4. Repeat 10,000 times.
5. Save every permutation's daily strong count and H1/H2/H3 statistics.

This conditions on each day's overall strength and tests spatial
placement only.

DEV / LATER
-----------
55 target days sorted ascending.

DEV:
    first 28 target days
    2026-07-13 through 2026-08-09

LATER:
    remaining 27 target days
    2026-08-10 through 2026-09-13

LATER is INTERNAL TEMPORAL CONFIRMATION only.
It is NOT an external holdout.

No calendar gap is filled.

Multiple testing
----------------
For LATER PRIMARY H1/H2/H3:
- raw empirical p is saved
- Holm-adjusted p is saved

Research STOP support
---------------------
Leave-one-day-out direction stability is also calculated.
No condition is changed in response to the result.
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
# Fixed project paths
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

STAGE1_PANEL = (
    DATA_DIR
    / "analysis_31days_deep"
    / "offline_stage1_prevday_dependency_v1"
    / "03_ab_all_machine_scores.csv"
)

FLOOR_MAP = (
    MACHINE_DIR
    / "maruhan_maebashi_floor_map_layout.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "offline_store_behavior_phase1a_spatial_v1"
)


# ============================================================
# Fixed research specification
# ============================================================

RESEARCH_NAME = "STORE_BEHAVIOR_RESEARCH"
RESEARCH_PHASE = "PHASE1A_SPATIAL_CONCENTRATION"
RESEARCH_SCOPE = "OFFLINE_RESEARCH_ONLY"

STORE = "MARUHAN_MEGA_CITY_MAEBASHI_INTER"

CHAMPION_MODEL = "CHAMPION_V4.2_C"
CHAMPION_FINGERPRINT = "a1eaf45d71ded209"

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

EXPECTED_DEV_MIN = pd.Timestamp("2026-07-13")
EXPECTED_DEV_MAX = pd.Timestamp("2026-08-09")

EXPECTED_LATER_MIN = pd.Timestamp("2026-08-10")
EXPECTED_LATER_MAX = pd.Timestamp("2026-09-13")

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

# Fixed before observing Phase 1A results.
RANDOM_SEED = 20260921

# Batch size controls memory only.
# It does not change the randomization design.
PERMUTATION_BATCH_SIZE = 1000


# ============================================================
# Helpers
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
    """
    Holm step-down adjusted p-values.

    Input:
        {
            "H1": raw_p,
            "H2": raw_p,
            "H3": raw_p,
        }

    Returns:
        same keys with adjusted p-values.
    """

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
# Input loading and QA
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

    return df


def stage1_input_qa(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
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
        int(machine_counts.min()),
        EXPECTED_MACHINES_PER_DAY,
        int(machine_counts.min())
        == EXPECTED_MACHINES_PER_DAY,
    )

    add(
        "machines_per_day_max",
        int(machine_counts.max()),
        EXPECTED_MACHINES_PER_DAY,
        int(machine_counts.max())
        == EXPECTED_MACHINES_PER_DAY,
    )

    if len(target_dates) == EXPECTED_TARGET_DAYS:
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
# Floor Map and fixed linear geometry
# ============================================================

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


def build_linear_geometry(
    floor: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build fixed Phase 1A eligible geometry.

    Eligibility:
        shape == "rect"

    Each island_id + side group is ordered by slot_index.
    A new segment begins whenever slot_index does not increase by exactly 1.

    No island boundary crossing.
    No gap crossing.
    No C05 / RING / arc positions.
    """

    eligible = (
        floor.loc[
            floor["shape"].astype(str)
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

    group_count = 0

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
        group_count += 1

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
                    "shape":
                        "rect",
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

    edge_array = np.asarray(
        edges,
        dtype=np.int32,
    )

    triple_array = np.asarray(
        triples,
        dtype=np.int32,
    )

    five_array = np.asarray(
        fives,
        dtype=np.int32,
    )

    if (
        group_count
        != EXPECTED_LINEAR_GROUPS
    ):
        raise RuntimeError(
            "Linear group count mismatch: "
            f"{group_count} != "
            f"{EXPECTED_LINEAR_GROUPS}"
        )

    return (
        eligible,
        segments,
        edge_array,
        triple_array,
        five_array,
    )


def floor_adjacency_qa(
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

    duplicate_floor = int(
        floor[
            "machine_no"
        ]
        .duplicated()
        .sum()
    )

    arc_count = int(
        floor[
            "shape"
        ]
        .astype(str)
        .eq("arc")
        .sum()
    )

    rect_count = int(
        floor[
            "shape"
        ]
        .astype(str)
        .eq("rect")
        .sum()
    )

    eligible_unique = int(
        eligible[
            "machine_no"
        ]
        .nunique()
    )

    linear_groups = int(
        eligible
        .groupby(
            [
                "island_id",
                "side",
            ]
        )
        .ngroups
    )

    c05_ring_arc = floor.loc[
        floor[
            "shape"
        ]
        .astype(str)
        .eq("arc"),
        [
            "island_id",
            "side",
        ],
    ]

    arc_only_c05_ring = bool(
        len(
            c05_ring_arc
        )
        == EXPECTED_EXCLUDED_ARC_POSITIONS
        and
        c05_ring_arc[
            "island_id"
        ]
        .astype(str)
        .eq("C05")
        .all()
        and
        c05_ring_arc[
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
        ]
        .nunique(),
        EXPECTED_FLOOR_ROWS,
        floor[
            "machine_no"
        ]
        .nunique()
        == EXPECTED_FLOOR_ROWS,
    )

    add(
        "floor_duplicate_machine_no",
        duplicate_floor,
        0,
        duplicate_floor == 0,
    )

    add(
        "rect_positions",
        rect_count,
        EXPECTED_ELIGIBLE_POSITIONS,
        rect_count
        == EXPECTED_ELIGIBLE_POSITIONS,
    )

    add(
        "arc_positions",
        arc_count,
        EXPECTED_EXCLUDED_ARC_POSITIONS,
        arc_count
        == EXPECTED_EXCLUDED_ARC_POSITIONS,
    )

    add(
        "arc_only_c05_ring",
        int(
            arc_only_c05_ring
        ),
        1,
        arc_only_c05_ring,
    )

    add(
        "eligible_unique_machine_no",
        eligible_unique,
        EXPECTED_ELIGIBLE_POSITIONS,
        eligible_unique
        == EXPECTED_ELIGIBLE_POSITIONS,
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
        "h2_triple_windows",
        len(triples),
        EXPECTED_H2_WINDOWS,
        len(triples)
        == EXPECTED_H2_WINDOWS,
    )

    add(
        "h3_five_windows",
        len(fives),
        EXPECTED_H3_WINDOWS,
        len(fives)
        == EXPECTED_H3_WINDOWS,
    )

    # Explicitly verify known physical gaps
    segment_machine_sets = [
        set(
            map(
                int,
                text.split("|"),
            )
        )
        for text in segments[
            "machine_nos"
        ]
    ]

    def same_segment(
        a: int,
        b: int,
    ) -> bool:
        return any(
            a in s
            and b in s
            for s in segment_machine_sets
        )

    add(
        "gap_842_843_disconnected",
        int(
            not same_segment(
                842,
                843,
            )
        ),
        1,
        not same_segment(
            842,
            843,
        ),
    )

    add(
        "gap_1015_1014_disconnected",
        int(
            not same_segment(
                1015,
                1014,
            )
        ),
        1,
        not same_segment(
            1015,
            1014,
        ),
    )

    qa = pd.DataFrame(
        rows
    )

    bad = qa.loc[
        ~qa["pass"]
    ]

    if not bad.empty:
        raise RuntimeError(
            "Floor adjacency QA failed:\n"
            + bad.to_string(
                index=False
            )
        )

    return qa


# ============================================================
# Panel / Floor merge QA
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

    eligible_counts = (
        panel
        .groupby(
            "target_date",
            sort=True,
        )[
            "phase1a_linear_eligible"
        ]
        .sum()
    )

    if not eligible_counts.eq(
        EXPECTED_ELIGIBLE_POSITIONS
    ).all():
        bad = eligible_counts.loc[
            ~eligible_counts.eq(
                EXPECTED_ELIGIBLE_POSITIONS
            )
        ]

        raise RuntimeError(
            "Eligible floor positions per day mismatch:\n"
            + bad.to_string()
        )

    return panel


# ============================================================
# Spatial statistic
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


def permutation_statistics(
    strong_n: int,
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
    """
    Generate exactly PERMUTATIONS random label assignments.

    Each replicate has exactly strong_n strong labels.
    """

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

        if strong_n == 0:
            labels = np.zeros(
                (
                    batch_n,
                    n_positions,
                ),
                dtype=bool,
            )

        elif (
            strong_n
            == n_positions
        ):
            labels = np.ones(
                (
                    batch_n,
                    n_positions,
                ),
                dtype=bool,
            )

        else:
            random_scores = rng.random(
                (
                    batch_n,
                    n_positions,
                ),
                dtype=np.float64,
            )

            selected = np.argpartition(
                random_scores,
                kth=strong_n - 1,
                axis=1,
            )[
                :,
                :strong_n,
            ]

            labels = np.zeros(
                (
                    batch_n,
                    n_positions,
                ),
                dtype=bool,
            )

            rows = np.arange(
                batch_n
            )[
                :,
                None
            ]

            labels[
                rows,
                selected,
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
            completed
            + batch_n,
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
# Daily observed + permutation analysis
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
]:
    eligible_order = (
        eligible
        .sort_values(
            "eligible_position_index"
        )
        [
            "machine_no"
        ]
        .astype(int)
        .tolist()
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
            .set_index(
                "machine_no"
            )
            .loc[
                eligible_order
            ]
            .reset_index()
        )

        if (
            len(day)
            != EXPECTED_ELIGIBLE_POSITIONS
        ):
            raise RuntimeError(
                "Daily eligible position count mismatch: "
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
                null_h1,
                null_h2,
                null_h3,
            ) = permutation_statistics(
                strong_n=strong_n,
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
                    "h1_null_mean":
                        h1_mean,
                    "h1_observed_minus_expected":
                        (
                            obs_h1
                            - h1_mean
                        ),
                    "h2_observed":
                        obs_h2,
                    "h2_null_mean":
                        h2_mean,
                    "h2_observed_minus_expected":
                        (
                            obs_h2
                            - h2_mean
                        ),
                    "h3_observed":
                        obs_h3,
                    "h3_null_mean":
                        h3_mean,
                    "h3_observed_minus_expected":
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
                        "h1_null_count":
                            null_h1,
                        "h2_null_count":
                            null_h2,
                        "h3_null_count":
                            null_h3,
                    }
                )
            )

            print(
                "[PERM] "
                f"{job_no}/{total_jobs} "
                f"date={target_date.date()} "
                f"period={period} "
                f"threshold={threshold} "
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

    expected_perm_rows = (
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
        != expected_perm_rows
    ):
        raise RuntimeError(
            "Permutation row count mismatch: "
            f"{len(permutation_df)} != "
            f"{expected_perm_rows}"
        )

    # Explicit proof that each randomization preserved each day's
    # eligible strong count exactly.
    check = (
        permutation_df
        .groupby(
            [
                "target_date",
                "threshold",
            ],
            sort=True,
        )
        ["strong_n"]
        .nunique()
    )

    if not check.eq(1).all():
        raise RuntimeError(
            "Permutation strong_n preservation QA failed."
        )

    return (
        observed_df,
        permutation_df,
    )


# ============================================================
# Period-level summaries
# ============================================================

METRICS = {
    "H1_PHYSICAL_STRONG_STRONG_PAIR":
        (
            "h1_observed",
            "h1_null_count",
            "h1_null_mean",
        ),
    "H2_PHYSICAL_STRONG_TRIPLE":
        (
            "h2_observed",
            "h2_null_count",
            "h2_null_mean",
        ),
    "H3_5SLOT_GE3_STRONG_CLUSTER":
        (
            "h3_observed",
            "h3_null_count",
            "h3_null_mean",
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
            f"threshold={threshold}, "
            f"period={period}"
        )

    observed_days = int(
        obs[
            "target_date"
        ]
        .nunique()
    )

    strong_machine_days = int(
        obs[
            "strong_n"
        ]
        .sum()
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
            ]
            .sum()
        )

        # Aggregate same permutation_id across all days.
        # Day-level random assignments are independent because the RNG
        # advances continuously, while permutation IDs provide aligned
        # replicate totals for the period null distribution.
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
                "Period null replicate count mismatch: "
                f"{threshold}, {period}, {metric}"
            )

        null_mean = float(
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

        daily_expected_total = float(
            obs[
                daily_null_mean_col
            ]
            .sum()
        )

        if not np.isclose(
            null_mean,
            daily_expected_total,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Period null mean does not equal sum of daily means: "
                f"{threshold}, {period}, {metric}"
            )

        effect = (
            observed_count
            - null_mean
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
                    observed_days,
                "threshold":
                    threshold,
                "threshold_label":
                    THRESHOLD_LABELS[
                        threshold
                    ],
                "metric":
                    metric,
                "strong_machine_days":
                    strong_machine_days,
                "observed_count":
                    observed_count,
                "null_mean":
                    null_mean,
                "observed_minus_expected":
                    effect,
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
                    raw_p,
                "effect_direction":
                    (
                        "ABOVE_RANDOM"
                        if effect > 0
                        else (
                            "BELOW_RANDOM"
                            if effect < 0
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
# Leave-one-day-out temporal fragility check
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
                    )
                    .sum()
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

                    direction_flip = (
                        loo_direction
                        != overall_direction
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
                                    direction_flip
                                ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def loo_summary(
    loo_detail: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouped = loo_detail.groupby(
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
# Output assembly
# ============================================================

def build_primary_summary(
    all_summary: pd.DataFrame,
    loo_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    primary = (
        all_summary.loc[
            all_summary[
                "threshold"
            ].eq(
                PRIMARY_THRESHOLD
            )
            &
            all_summary[
                "period"
            ].eq(
                "FULL"
            )
        ]
        .copy()
    )

    loo = (
        loo_summary_df.loc[
            loo_summary_df[
                "threshold"
            ].eq(
                PRIMARY_THRESHOLD
            )
            &
            loo_summary_df[
                "period"
            ].eq(
                "FULL"
            ),
            [
                "metric",
                "direction_flip_count",
                "direction_flip_any",
                "leave_one_day_out_min_effect",
                "leave_one_day_out_max_effect",
            ],
        ]
        .copy()
    )

    return primary.merge(
        loo,
        on="metric",
        how="left",
        validate="one_to_one",
    )


def build_dev_later_summary(
    all_summary: pd.DataFrame,
    loo_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    x = (
        all_summary.loc[
            all_summary[
                "threshold"
            ].eq(
                PRIMARY_THRESHOLD
            )
            &
            all_summary[
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

    loo = (
        loo_summary_df.loc[
            loo_summary_df[
                "threshold"
            ].eq(
                PRIMARY_THRESHOLD
            )
            &
            loo_summary_df[
                "period"
            ].isin(
                [
                    "DEV",
                    "LATER",
                ]
            ),
            [
                "period",
                "metric",
                "direction_flip_count",
                "direction_flip_any",
                "leave_one_day_out_min_effect",
                "leave_one_day_out_max_effect",
            ],
        ]
        .copy()
    )

    return x.merge(
        loo,
        on=[
            "period",
            "metric",
        ],
        how="left",
        validate="one_to_one",
    )


def build_sensitivity_summary(
    all_summary: pd.DataFrame,
    loo_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    x = (
        all_summary.loc[
            all_summary[
                "threshold"
            ].isin(
                SECONDARY_THRESHOLDS
            )
        ]
        .copy()
    )

    loo = (
        loo_summary_df.loc[
            loo_summary_df[
                "threshold"
            ].isin(
                SECONDARY_THRESHOLDS
            ),
            [
                "period",
                "threshold",
                "metric",
                "direction_flip_count",
                "direction_flip_any",
                "leave_one_day_out_min_effect",
                "leave_one_day_out_max_effect",
            ],
        ]
        .copy()
    )

    return x.merge(
        loo,
        on=[
            "period",
            "threshold",
            "metric",
        ],
        how="left",
        validate="one_to_one",
    )


def build_metadata(
    target_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    metadata = [
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
            "h1_definition",
            (
                "physical adjacent edge with "
                "both endpoints strong"
            ),
        ),
        (
            "h1_edges",
            EXPECTED_H1_EDGES,
        ),
        (
            "h2_definition",
            (
                "3-slot consecutive linear "
                "window with all 3 strong"
            ),
        ),
        (
            "h2_windows",
            EXPECTED_H2_WINDOWS,
        ),
        (
            "h3_definition",
            (
                "fixed 5-slot consecutive "
                "linear window with >=3 strong"
            ),
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
            "permutations_per_day_threshold",
            PERMUTATIONS,
        ),
        (
            "random_seed",
            RANDOM_SEED,
        ),
        (
            "permutation_design",
            (
                "same-day fixed strong_n "
                "random reassignment across "
                "497 eligible linear positions"
            ),
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
                "leave-one-day-out direction "
                "flip check"
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
        metadata,
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
        "=" * 72
    )
    print(
        "STORE_BEHAVIOR_RESEARCH "
        "Phase 1A SPATIAL CONCENTRATION"
    )
    print(
        "OFFLINE RESEARCH ONLY"
    )
    print(
        "=" * 72
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "[1/7] Loading and validating Stage1 panel..."
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
        "[2/7] Loading canonical Floor Map..."
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

    floor_qa = floor_adjacency_qa(
        floor,
        eligible,
        segments,
        edges,
        triples,
        fives,
    )

    print(
        "Floor / linear geometry QA: PASS"
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
        "[3/7] Building fixed analysis panel..."
    )

    panel = build_analysis_panel(
        stage1,
        floor,
        eligible,
    )

    print(
        "Panel/Floor merge QA: PASS"
    )

    print()
    print(
        "[4/7] Running observed spatial counts "
        "+ same-day permutations..."
    )
    print(
        f"permutations per date/threshold: {PERMUTATIONS:,}"
    )
    print(
        f"random seed: {RANDOM_SEED}"
    )

    (
        daily_observed,
        daily_permutations,
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
        "[5/7] Building FULL / DEV / LATER summaries..."
    )

    all_summary = build_all_summaries(
        daily_observed,
        daily_permutations,
    )

    print()
    print(
        "[6/7] Running leave-one-day-out robustness..."
    )

    loo_detail = (
        leave_one_day_out_robustness(
            daily_observed
        )
    )

    loo_summary_df = (
        loo_summary(
            loo_detail
        )
    )

    primary_summary = (
        build_primary_summary(
            all_summary,
            loo_summary_df,
        )
    )

    dev_later_summary = (
        build_dev_later_summary(
            all_summary,
            loo_summary_df,
        )
    )

    sensitivity_summary = (
        build_sensitivity_summary(
            all_summary,
            loo_summary_df,
        )
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
        "02_floor_adjacency_qa.csv":
            floor_qa,
        "03_daily_spatial_observed.csv":
            daily_observed,
        "04_daily_permutation_expected.csv":
            daily_permutations,
        "05_primary_spatial_summary.csv":
            primary_summary,
        "06_dev_later_summary.csv":
            dev_later_summary,
        "07_sensitivity_summary.csv":
            sensitivity_summary,
        "08_metadata.csv":
            metadata,
        "09_linear_segments.csv":
            segments,
        "10_leave_one_day_out_detail.csv":
            loo_detail,
        "11_leave_one_day_out_summary.csv":
            loo_summary_df,
        "12_all_threshold_period_summary.csv":
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
        / "13_output_manifest.json"
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
        "=" * 72
    )
    print(
        "Phase 1A SPATIAL CONCENTRATION completed."
    )
    print(
        "=" * 72
    )

    print(
        f"output_dir       : {OUTPUT_DIR}"
    )
    print(
        "input_QA         : PASS"
    )
    print(
        "floor_QA         : PASS"
    )
    print(
        "eligible_geometry: RECT_LINEAR_ONLY"
    )
    print(
        f"eligible_positions: {EXPECTED_ELIGIBLE_POSITIONS}"
    )
    print(
        f"H1_edges         : {EXPECTED_H1_EDGES}"
    )
    print(
        f"H2_windows       : {EXPECTED_H2_WINDOWS}"
    )
    print(
        f"H3_windows       : {EXPECTED_H3_WINDOWS}"
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
        "auto_promotion   : NONE"
    )
    print(
        "git_action       : NONE"
    )

    print()
    print(
        "PRIMARY FULL summary:"
    )

    print(
        primary_summary[
            [
                "metric",
                "observed_count",
                "null_mean",
                "observed_minus_expected",
                "observed_div_expected",
                "null_ci95_low",
                "null_ci95_high",
                "empirical_p_raw",
                "direction_flip_any",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "PRIMARY DEV/LATER summary:"
    )

    print(
        dev_later_summary[
            [
                "period",
                "metric",
                "observed_count",
                "null_mean",
                "observed_minus_expected",
                "observed_div_expected",
                "empirical_p_raw",
                "holm_p_later_primary",
                "direction_flip_any",
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
        "Do not interpret or promote results here."
    )
    print(
        "Return outputs to 03_2 for Phase 1A review."
    )


if __name__ == "__main__":
    main()