from __future__ import annotations

"""
W4 TEMPORAL ROBUSTNESS CHECK
OFFLINE RESEARCH ONLY

Research:
    WIN_LOSS_SEQUENCE_RESEARCH

Purpose:
    Evaluate the pre-selected STREAK_EXACT W4 candidate on a fixed
    28-target-day / 27-target-day chronological split.

Important:
    - INTERNAL TEMPORAL ROBUSTNESS CHECK only.
    - This is NOT an external validation.
    - W4 was selected after inspecting the full 55-target-day dataset.
    - No additional pattern search is performed.
    - No Champion / Formal Forward / Forward Guard / production changes.
    - No automatic promotion.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import sys

import pandas as pd


# ============================================================
# Constants
# ============================================================

RESEARCH_NAME = "WIN_LOSS_SEQUENCE_RESEARCH"
CHECK_NAME = "W4_INTERNAL_TEMPORAL_ROBUSTNESS_CHECK"
RESEARCH_MODE = "OFFLINE_RESEARCH_ONLY"
VERSION = "v1"

STORE = "MARUHAN_MAEBASHI_INTER"

EXPECTED_TARGET_DAYS = 55
FIRST_HALF_DAYS = 28
SECOND_HALF_DAYS = 27

W4_LABEL = "W4"

INPUT_RELATIVE = Path(
    "data/maruhan_maebashi/machine_number/"
    "analysis_31days_deep/"
    "offline_stage1_prevday_dependency_v1/"
    "03_ab_all_machine_scores.csv"
)

OUTPUT_RELATIVE = Path(
    "data/maruhan_maebashi/machine_number/"
    "analysis_31days_deep/"
    "offline_win_loss_sequence_w4_temporal_robustness_v1"
)

SUMMARY_FILENAME = "01_w4_temporal_robustness_summary.csv"
PARTITION_QA_FILENAME = "02_w4_temporal_partition_qa.csv"
W4_DETAILS_FILENAME = "03_w4_event_details.csv"
METADATA_FILENAME = "04_metadata.csv"


# ============================================================
# Data classes
# ============================================================

@dataclass(frozen=True)
class PeriodMetrics:
    period: str
    target_date_min: str
    target_date_max: str

    w4_sample_n: int
    w4_win_count: int
    w4_next_day_win_rate: float

    baseline_sample_n: int
    baseline_win_count: int
    baseline_win_rate: float

    win_rate_diff_pp: float

    w4_next_day_mean_diff: float
    baseline_mean_diff: float
    mean_diff_lift: float

    w4_median_diff: float
    baseline_median_diff: float

    win_rate_ci95_low: float
    win_rate_ci95_high: float

    observed_target_days: int


# ============================================================
# Generic helpers
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def wilson_interval(
    wins: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)

    p = wins / n
    z2 = z * z

    denominator = 1.0 + z2 / n

    center = (
        p
        + z2 / (2.0 * n)
    ) / denominator

    half = (
        z
        * math.sqrt(
            (p * (1.0 - p) / n)
            + (z2 / (4.0 * n * n))
        )
        / denominator
    )

    return (
        max(0.0, center - half),
        min(1.0, center + half),
    )


def write_csv_atomic(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )

    tmp.replace(path)


def safe_name_series(
    s: pd.Series,
) -> pd.Series:
    x = s.astype("string")
    x = x.fillna("<MISSING_MACHINE_NAME>")
    x = x.str.strip()
    x = x.replace("", "<EMPTY_MACHINE_NAME>")
    return x


# ============================================================
# Input / sequence construction
# ============================================================

def load_input(
    input_path: Path,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        usecols=[
            "target_date",
            "machine_no",
            "actual_machine_name",
            "actual_diff",
        ],
        low_memory=False,
    )

    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    )

    df["machine_no"] = pd.to_numeric(
        df["machine_no"],
        errors="raise",
    ).astype("int64")

    df["actual_diff"] = pd.to_numeric(
        df["actual_diff"],
        errors="raise",
    )

    if df["actual_diff"].isna().any():
        raise RuntimeError(
            "actual_diff contains missing values."
        )

    duplicate_count = int(
        df.duplicated(
            ["target_date", "machine_no"]
        ).sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate target_date x machine_no rows detected: "
            f"{duplicate_count}"
        )

    df["actual_machine_name"] = safe_name_series(
        df["actual_machine_name"]
    )

    df = df.sort_values(
        ["machine_no", "target_date"],
        kind="mergesort",
    ).reset_index(drop=True)

    return df


def add_sequence_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the already-approved WIN_LOSS_SEQUENCE_RESEARCH
    sequence rules exactly.

    W:
        actual_diff > 0

    L:
        actual_diff <= 0

    Sequence continues only when:
        1) machine_no is the same
        2) target_date is calendar-consecutive
        3) actual_machine_name is unchanged
    """
    x = df.copy()

    x["wl"] = pd.Series(
        [
            "W" if value > 0 else "L"
            for value in x["actual_diff"]
        ],
        index=x.index,
        dtype="string",
    )

    grouped = x.groupby(
        "machine_no",
        sort=False,
        group_keys=False,
    )

    x["prev_date"] = grouped[
        "target_date"
    ].shift(1)

    x["prev_actual_machine_name"] = grouped[
        "actual_machine_name"
    ].shift(1)

    x["calendar_gap_days"] = (
        x["target_date"]
        - x["prev_date"]
    ).dt.days

    x["same_machine_name_as_prev"] = (
        x["actual_machine_name"]
        == x["prev_actual_machine_name"]
    )

    x["sequence_continues_from_prev"] = (
        x["calendar_gap_days"].eq(1)
        & x["same_machine_name_as_prev"]
    )

    x["new_segment"] = (
        ~x["sequence_continues_from_prev"]
    ).astype("int64")

    x["sequence_segment_id"] = (
        x.groupby(
            "machine_no",
            sort=False,
        )["new_segment"]
        .cumsum()
        .astype("int64")
    )

    segment_keys = [
        "machine_no",
        "sequence_segment_id",
    ]

    seg_group = x.groupby(
        segment_keys,
        sort=False,
        group_keys=False,
    )

    x["wl_lag1"] = seg_group["wl"].shift(1)
    x["wl_lag2"] = seg_group["wl"].shift(2)
    x["wl_lag3"] = seg_group["wl"].shift(3)
    x["wl_lag4"] = seg_group["wl"].shift(4)
    x["wl_lag5"] = seg_group["wl"].shift(5)
    x["wl_lag6"] = seg_group["wl"].shift(6)

    def exact_prior_streak_length(
        row: pd.Series,
    ) -> int:
        first = row["wl_lag1"]

        if pd.isna(first):
            return 0

        length = 1

        for lag_col in [
            "wl_lag2",
            "wl_lag3",
            "wl_lag4",
            "wl_lag5",
            "wl_lag6",
        ]:
            value = row[lag_col]

            if pd.isna(value):
                break

            if value != first:
                break

            length += 1

        return length

    x["prior_streak_length_raw"] = x.apply(
        exact_prior_streak_length,
        axis=1,
    )

    x["prior_streak_symbol"] = (
        x["wl_lag1"]
        .fillna("")
        .astype(str)
    )

    x["streak_exact_label"] = ""

    w4_mask = (
        x["prior_streak_symbol"].eq("W")
        & x["prior_streak_length_raw"].eq(4)
    )

    x.loc[
        w4_mask,
        "streak_exact_label",
    ] = W4_LABEL

    x["target_win"] = (
        x["actual_diff"] > 0
    )

    return x


# ============================================================
# Fixed 28 / 27 target-date partition
# ============================================================

def make_fixed_partition(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.Timestamp], list[pd.Timestamp]]:
    target_dates = list(
        pd.DatetimeIndex(
            sorted(
                panel["target_date"].unique()
            )
        )
    )

    if len(target_dates) != EXPECTED_TARGET_DAYS:
        raise RuntimeError(
            "Unexpected target-day count. "
            f"expected={EXPECTED_TARGET_DAYS} "
            f"actual={len(target_dates)}"
        )

    first_dates = target_dates[
        :FIRST_HALF_DAYS
    ]

    second_dates = target_dates[
        FIRST_HALF_DAYS:
    ]

    if len(first_dates) != FIRST_HALF_DAYS:
        raise RuntimeError(
            "First-half partition size mismatch."
        )

    if len(second_dates) != SECOND_HALF_DAYS:
        raise RuntimeError(
            "Second-half partition size mismatch."
        )

    first_set = set(first_dates)
    second_set = set(second_dates)

    if first_set & second_set:
        raise RuntimeError(
            "Partition overlap detected."
        )

    x = panel.copy()

    x["period"] = ""

    x.loc[
        x["target_date"].isin(first_set),
        "period",
    ] = "FIRST_28"

    x.loc[
        x["target_date"].isin(second_set),
        "period",
    ] = "SECOND_27"

    if x["period"].eq("").any():
        raise RuntimeError(
            "Some rows were not assigned to a partition."
        )

    return (
        x,
        first_dates,
        second_dates,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_period_metrics(
    panel: pd.DataFrame,
    period: str,
) -> PeriodMetrics:
    period_rows = panel.loc[
        panel["period"].eq(period)
    ].copy()

    if period_rows.empty:
        raise RuntimeError(
            f"No rows for period: {period}"
        )

    # Baseline is recalculated within each half using
    # STREAK-eligible target rows only.
    baseline = period_rows.loc[
        period_rows["wl_lag1"].notna()
    ].copy()

    w4 = period_rows.loc[
        period_rows[
            "streak_exact_label"
        ].eq(W4_LABEL)
    ].copy()

    baseline_n = len(baseline)
    baseline_wins = int(
        baseline["target_win"].sum()
    )

    w4_n = len(w4)
    w4_wins = int(
        w4["target_win"].sum()
    )

    if baseline_n <= 0:
        raise RuntimeError(
            f"Baseline is empty for period: {period}"
        )

    if w4_n <= 0:
        raise RuntimeError(
            f"W4 is empty for period: {period}"
        )

    baseline_win_rate = (
        baseline_wins / baseline_n
    )

    w4_win_rate = (
        w4_wins / w4_n
    )

    baseline_mean = float(
        baseline["actual_diff"].mean()
    )

    w4_mean = float(
        w4["actual_diff"].mean()
    )

    baseline_median = float(
        baseline["actual_diff"].median()
    )

    w4_median = float(
        w4["actual_diff"].median()
    )

    ci_low, ci_high = wilson_interval(
        w4_wins,
        w4_n,
    )

    return PeriodMetrics(
        period=period,
        target_date_min=(
            period_rows["target_date"]
            .min()
            .date()
            .isoformat()
        ),
        target_date_max=(
            period_rows["target_date"]
            .max()
            .date()
            .isoformat()
        ),
        w4_sample_n=w4_n,
        w4_win_count=w4_wins,
        w4_next_day_win_rate=(
            w4_win_rate * 100.0
        ),
        baseline_sample_n=baseline_n,
        baseline_win_count=baseline_wins,
        baseline_win_rate=(
            baseline_win_rate * 100.0
        ),
        win_rate_diff_pp=(
            (w4_win_rate - baseline_win_rate)
            * 100.0
        ),
        w4_next_day_mean_diff=w4_mean,
        baseline_mean_diff=baseline_mean,
        mean_diff_lift=(
            w4_mean - baseline_mean
        ),
        w4_median_diff=w4_median,
        baseline_median_diff=baseline_median,
        win_rate_ci95_low=(
            ci_low * 100.0
        ),
        win_rate_ci95_high=(
            ci_high * 100.0
        ),
        observed_target_days=int(
            w4["target_date"].nunique()
        ),
    )


def classify_result(
    first: PeriodMetrics,
    second: PeriodMetrics,
) -> str:
    """
    Pre-defined interpretation rule.

    ROBUST_DIRECTION:
        both periods have:
            win_rate_diff_pp > 0
            mean_diff_lift > 0

    MIXED:
        - within either period, win-rate and mean-diff directions disagree,
          OR
        - one period is jointly positive and the other jointly negative.

    NO_TEMPORAL_REPLICATION:
        improvement direction is not reproduced in both periods, without
        the explicit mixed-direction situation above.
    """
    first_win_pos = (
        first.win_rate_diff_pp > 0
    )
    first_mean_pos = (
        first.mean_diff_lift > 0
    )

    second_win_pos = (
        second.win_rate_diff_pp > 0
    )
    second_mean_pos = (
        second.mean_diff_lift > 0
    )

    if (
        first_win_pos
        and first_mean_pos
        and second_win_pos
        and second_mean_pos
    ):
        return "ROBUST_DIRECTION"

    first_discordant = (
        first_win_pos
        != first_mean_pos
    )

    second_discordant = (
        second_win_pos
        != second_mean_pos
    )

    first_joint_negative = (
        first.win_rate_diff_pp < 0
        and first.mean_diff_lift < 0
    )

    second_joint_negative = (
        second.win_rate_diff_pp < 0
        and second.mean_diff_lift < 0
    )

    first_joint_positive = (
        first_win_pos
        and first_mean_pos
    )

    second_joint_positive = (
        second_win_pos
        and second_mean_pos
    )

    if (
        first_discordant
        or second_discordant
        or (
            first_joint_positive
            and second_joint_negative
        )
        or (
            first_joint_negative
            and second_joint_positive
        )
    ):
        return "MIXED"

    return "NO_TEMPORAL_REPLICATION"


# ============================================================
# Output builders
# ============================================================

def metrics_to_row(
    metrics: PeriodMetrics,
    overall_classification: str,
) -> dict:
    return {
        "research_name": RESEARCH_NAME,
        "check_name": CHECK_NAME,
        "research_mode": RESEARCH_MODE,
        "store": STORE,
        "pattern": W4_LABEL,
        "period": metrics.period,
        "target_date_min": metrics.target_date_min,
        "target_date_max": metrics.target_date_max,
        "w4_sample_n": metrics.w4_sample_n,
        "w4_win_count": metrics.w4_win_count,
        "w4_next_day_win_rate": metrics.w4_next_day_win_rate,
        "baseline_sample_n": metrics.baseline_sample_n,
        "baseline_win_count": metrics.baseline_win_count,
        "baseline_win_rate": metrics.baseline_win_rate,
        "win_rate_diff_pp": metrics.win_rate_diff_pp,
        "w4_next_day_mean_diff": metrics.w4_next_day_mean_diff,
        "baseline_mean_diff": metrics.baseline_mean_diff,
        "mean_diff_lift": metrics.mean_diff_lift,
        "w4_median_diff": metrics.w4_median_diff,
        "baseline_median_diff": metrics.baseline_median_diff,
        "win_rate_ci95_low": metrics.win_rate_ci95_low,
        "win_rate_ci95_high": metrics.win_rate_ci95_high,
        "observed_target_days": metrics.observed_target_days,
        "overall_classification": overall_classification,
    }


def build_partition_qa(
    panel: pd.DataFrame,
    first_dates: list[pd.Timestamp],
    second_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows: list[dict] = []

    for period, dates in [
        ("FIRST_28", first_dates),
        ("SECOND_27", second_dates),
    ]:
        period_rows = panel.loc[
            panel["period"].eq(period)
        ].copy()

        rows.append(
            {
                "period": period,
                "target_day_count": len(dates),
                "target_date_min": (
                    min(dates).date().isoformat()
                ),
                "target_date_max": (
                    max(dates).date().isoformat()
                ),
                "row_count": len(period_rows),
                "unique_machines": int(
                    period_rows[
                        "machine_no"
                    ].nunique()
                ),
                "streak_eligible_rows": int(
                    period_rows[
                        "wl_lag1"
                    ].notna().sum()
                ),
                "w4_rows": int(
                    period_rows[
                        "streak_exact_label"
                    ].eq(W4_LABEL).sum()
                ),
                "target_dates": "|".join(
                    d.date().isoformat()
                    for d in dates
                ),
            }
        )

    return pd.DataFrame(rows)


def build_w4_details(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    details = panel.loc[
        panel[
            "streak_exact_label"
        ].eq(W4_LABEL)
    ].copy()

    columns = [
        "period",
        "target_date",
        "machine_no",
        "actual_machine_name",
        "actual_diff",
        "target_win",
        "sequence_segment_id",
        "wl_lag1",
        "wl_lag2",
        "wl_lag3",
        "wl_lag4",
        "wl_lag5",
        "prior_streak_length_raw",
        "streak_exact_label",
    ]

    details = details[columns].copy()

    details["target_date"] = (
        details["target_date"]
        .dt.date
        .astype(str)
    )

    return details


def build_metadata(
    *,
    input_path: Path,
    output_dir: Path,
    first_dates: list[pd.Timestamp],
    second_dates: list[pd.Timestamp],
    classification: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "research_name": RESEARCH_NAME,
                "check_name": CHECK_NAME,
                "research_mode": RESEARCH_MODE,
                "version": VERSION,
                "store": STORE,
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "candidate": W4_LABEL,
                "candidate_selection_note": (
                    "W4 was selected after reviewing the full "
                    "55-target-day basic sequence analysis. "
                    "Therefore this split is an internal temporal "
                    "robustness check, not an independent validation."
                ),
                "partition_rule": (
                    "Sort the 55 Stage1 target dates ascending; "
                    "FIRST_28 = first 28 target dates; "
                    "SECOND_27 = remaining 27 target dates."
                ),
                "first_half_target_days": len(first_dates),
                "second_half_target_days": len(second_dates),
                "first_half_min": (
                    min(first_dates).date().isoformat()
                ),
                "first_half_max": (
                    max(first_dates).date().isoformat()
                ),
                "second_half_min": (
                    min(second_dates).date().isoformat()
                ),
                "second_half_max": (
                    max(second_dates).date().isoformat()
                ),
                "sequence_rule": (
                    "same machine_no + calendar-consecutive target_date "
                    "+ same actual_machine_name"
                ),
                "win_definition": "actual_diff > 0",
                "lose_definition": "actual_diff <= 0",
                "w4_definition": (
                    "STREAK_EXACT W4; exact trailing four-win streak; "
                    "not overlapping W1/W2/W3 and not 5+"
                ),
                "baseline_rule": (
                    "Within each period, use only STREAK-eligible "
                    "target rows where wl_lag1 is available."
                ),
                "ci_method": "Wilson 95%",
                "classification": classification,
                "classification_rule": (
                    "ROBUST_DIRECTION if both periods have "
                    "win_rate_diff_pp > 0 and mean_diff_lift > 0; "
                    "MIXED for discordant/reversed directions; "
                    "otherwise NO_TEMPORAL_REPLICATION."
                ),
                "sequence_history_partition_note": (
                    "Sequence features are constructed on the full "
                    "chronological dataset before target-date partitioning. "
                    "A SECOND_27 target may use only its genuinely prior "
                    "calendar-consecutive history, including prior dates "
                    "that fall in FIRST_28. No future information is used."
                ),
                "automatic_promotion": False,
                "champion_changed": False,
                "formal_forward_changed": False,
                "forward_guard_changed": False,
                "production_ranking_changed": False,
                "neighbor_avg_changed": False,
                "floor_status": "PAUSED",
            }
        ]
    )


# ============================================================
# Main
# ============================================================

def main() -> int:
    root = project_root()

    input_path = (
        root / INPUT_RELATIVE
    )

    output_dir = (
        root / OUTPUT_RELATIVE
    )

    print("=" * 72)
    print(CHECK_NAME)
    print(RESEARCH_MODE)
    print("=" * 72)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print()

    df = load_input(
        input_path
    )

    panel = add_sequence_columns(
        df
    )

    (
        panel,
        first_dates,
        second_dates,
    ) = make_fixed_partition(
        panel
    )

    first = calculate_period_metrics(
        panel,
        "FIRST_28",
    )

    second = calculate_period_metrics(
        panel,
        "SECOND_27",
    )

    classification = classify_result(
        first,
        second,
    )

    summary = pd.DataFrame(
        [
            metrics_to_row(
                first,
                classification,
            ),
            metrics_to_row(
                second,
                classification,
            ),
        ]
    )

    partition_qa = build_partition_qa(
        panel,
        first_dates,
        second_dates,
    )

    w4_details = build_w4_details(
        panel
    )

    metadata = build_metadata(
        input_path=input_path,
        output_dir=output_dir,
        first_dates=first_dates,
        second_dates=second_dates,
        classification=classification,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_atomic(
        summary,
        output_dir / SUMMARY_FILENAME,
    )

    write_csv_atomic(
        partition_qa,
        output_dir / PARTITION_QA_FILENAME,
    )

    write_csv_atomic(
        w4_details,
        output_dir / W4_DETAILS_FILENAME,
    )

    write_csv_atomic(
        metadata,
        output_dir / METADATA_FILENAME,
    )

    print("[PARTITION QA]")
    print(
        f"FIRST_28 : "
        f"{len(first_dates)} target days / "
        f"{min(first_dates).date()} -> "
        f"{max(first_dates).date()}"
    )
    print(
        f"SECOND_27: "
        f"{len(second_dates)} target days / "
        f"{min(second_dates).date()} -> "
        f"{max(second_dates).date()}"
    )
    print()

    print("[W4 TEMPORAL ROBUSTNESS]")
    display_cols = [
        "period",
        "target_date_min",
        "target_date_max",
        "w4_sample_n",
        "w4_win_count",
        "w4_next_day_win_rate",
        "baseline_sample_n",
        "baseline_win_count",
        "baseline_win_rate",
        "win_rate_diff_pp",
        "w4_next_day_mean_diff",
        "baseline_mean_diff",
        "mean_diff_lift",
        "w4_median_diff",
        "baseline_median_diff",
        "win_rate_ci95_low",
        "win_rate_ci95_high",
        "observed_target_days",
    ]

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        260,
    ):
        print(
            summary[
                display_cols
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}",
            )
        )

    print()
    print(
        f"CLASSIFICATION: {classification}"
    )

    print()
    print("[OUTPUT FILES]")

    for filename in [
        SUMMARY_FILENAME,
        PARTITION_QA_FILENAME,
        W4_DETAILS_FILENAME,
        METADATA_FILENAME,
    ]:
        path = output_dir / filename

        print(
            f"{filename}: "
            f"{path.stat().st_size:,} bytes"
        )

    print()
    print(
        "No additional pattern search was performed."
    )
    print(
        "No Champion / Formal Forward / Forward Guard / "
        "production ranking changes were made."
    )
    print(
        "Return these OFFLINE results to 03 for review."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
