from __future__ import annotations

"""
WIN_LOSS_SEQUENCE_RESEARCH
OFFLINE RESEARCH ONLY

Purpose
-------
Use the closed Stage1 artifact for Maruhan Mega City Maebashi Inter and
summarize next-day performance after pre-defined WIN/LOSE sequences.

This script does not change Champion, Champion weights, Forward Guard,
Formal Forward, production rankings, neighbor_avg, Floor, or other stores.
Stage1 / Stage1.5 are not rerun.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math
import sys

import pandas as pd


RESEARCH_NAME = "WIN_LOSS_SEQUENCE_RESEARCH"
RESEARCH_MODE = "OFFLINE_RESEARCH_ONLY"
VERSION = "v1"
STORE = "MARUHAN_MAEBASHI_INTER"

INPUT_RELATIVE = Path(
    "data/maruhan_maebashi/machine_number/"
    "analysis_31days_deep/"
    "offline_stage1_prevday_dependency_v1/"
    "03_ab_all_machine_scores.csv"
)

OUTPUT_RELATIVE = Path(
    "data/maruhan_maebashi/machine_number/"
    "analysis_31days_deep/"
    "offline_win_loss_sequence_research_v1"
)

SUMMARY_FILENAME = "01_win_loss_sequence_summary.csv"
QA_FILENAME = "02_win_loss_sequence_qa.csv"
BREAKS_FILENAME = "03_sequence_breaks.csv"
DETAIL_FILENAME = "04_pattern_event_details.csv"
METADATA_FILENAME = "05_metadata.csv"


@dataclass(frozen=True)
class BaselineStats:
    n: int
    wins: int
    win_rate: float
    mean_diff: float
    median_diff: float
    observed_target_days: int


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
    center = (p + z2 / (2.0 * n)) / denominator
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


def require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
) -> None:
    missing = [
        col
        for col in required
        if col not in df.columns
    ]
    if missing:
        raise RuntimeError(
            "Required columns are missing: "
            + ", ".join(missing)
        )


def safe_name_series(s: pd.Series) -> pd.Series:
    x = s.astype("string")
    x = x.fillna("<MISSING_MACHINE_NAME>")
    x = x.str.strip()
    x = x.replace("", "<EMPTY_MACHINE_NAME>")
    return x


def write_csv_atomic(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        encoding="utf-8-sig",
    )
    tmp.replace(path)


def load_input(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    require_columns(
        df,
        [
            "target_date",
            "machine_no",
            "actual_diff",
            "actual_machine_name",
        ],
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


def build_qa_table(df: pd.DataFrame) -> pd.DataFrame:
    per_day = (
        df
        .groupby(
            "target_date",
            sort=True,
        )["machine_no"]
        .nunique()
    )

    all_dates = pd.date_range(
        start=df["target_date"].min(),
        end=df["target_date"].max(),
        freq="D",
    )

    observed_dates = pd.DatetimeIndex(
        sorted(df["target_date"].unique())
    )

    missing_dates = all_dates.difference(
        observed_dates
    )

    qa_rows = [
        {"metric": "rows", "value": len(df)},
        {
            "metric": "target_days",
            "value": int(df["target_date"].nunique()),
        },
        {
            "metric": "date_min",
            "value": df["target_date"].min().date().isoformat(),
        },
        {
            "metric": "date_max",
            "value": df["target_date"].max().date().isoformat(),
        },
        {
            "metric": "unique_machines",
            "value": int(df["machine_no"].nunique()),
        },
        {
            "metric": "duplicate_target_machine",
            "value": int(
                df.duplicated(
                    ["target_date", "machine_no"]
                ).sum()
            ),
        },
        {
            "metric": "actual_diff_missing",
            "value": int(df["actual_diff"].isna().sum()),
        },
        {
            "metric": "actual_machine_name_missing_or_empty",
            "value": int(
                df["actual_machine_name"].isin(
                    [
                        "<MISSING_MACHINE_NAME>",
                        "<EMPTY_MACHINE_NAME>",
                    ]
                ).sum()
            ),
        },
        {
            "metric": "machines_per_day_min",
            "value": int(per_day.min()),
        },
        {
            "metric": "machines_per_day_max",
            "value": int(per_day.max()),
        },
        {
            "metric": "missing_calendar_days_count",
            "value": len(missing_dates),
        },
        {
            "metric": "missing_calendar_days",
            "value": "|".join(
                d.date().isoformat()
                for d in missing_dates
            ),
        },
    ]

    return pd.DataFrame(qa_rows)


def add_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    x["prev_date"] = grouped["target_date"].shift(1)
    x["prev_actual_machine_name"] = grouped[
        "actual_machine_name"
    ].shift(1)

    x["calendar_gap_days"] = (
        x["target_date"] - x["prev_date"]
    ).dt.days

    x["same_machine_name_as_prev"] = (
        x["actual_machine_name"]
        == x["prev_actual_machine_name"]
    )

    x["sequence_continues_from_prev"] = (
        x["calendar_gap_days"].eq(1)
        & x["same_machine_name_as_prev"]
    )

    x["sequence_break_reason"] = ""

    first_mask = x["prev_date"].isna()
    x.loc[
        first_mask,
        "sequence_break_reason",
    ] = "FIRST_OBSERVATION"

    gap_mask = (
        ~first_mask
        & ~x["calendar_gap_days"].eq(1)
    )
    x.loc[
        gap_mask,
        "sequence_break_reason",
    ] = "CALENDAR_GAP"

    name_change_mask = (
        ~first_mask
        & x["calendar_gap_days"].eq(1)
        & ~x["same_machine_name_as_prev"]
    )
    x.loc[
        name_change_mask,
        "sequence_break_reason",
    ] = "MACHINE_NAME_CHANGE"

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

    for lag in range(1, 7):
        x[f"wl_lag{lag}"] = seg_group["wl"].shift(lag)

    x["prior_rows_in_segment"] = seg_group.cumcount()

    def exact_prior_streak_length(row: pd.Series) -> int:
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

    for length in range(1, 6):
        win_mask = (
            x["prior_streak_symbol"].eq("W")
            & x["prior_streak_length_raw"].eq(length)
        )
        lose_mask = (
            x["prior_streak_symbol"].eq("L")
            & x["prior_streak_length_raw"].eq(length)
        )

        x.loc[
            win_mask,
            "streak_exact_label",
        ] = f"W{length}"

        x.loc[
            lose_mask,
            "streak_exact_label",
        ] = f"L{length}"

    two_valid = (
        x["wl_lag1"].notna()
        & x["wl_lag2"].notna()
    )

    x["pattern_2d"] = ""
    x.loc[
        two_valid,
        "pattern_2d",
    ] = (
        x.loc[two_valid, "wl_lag2"].astype(str)
        + x.loc[two_valid, "wl_lag1"].astype(str)
    )

    three_valid = (
        x["wl_lag1"].notna()
        & x["wl_lag2"].notna()
        & x["wl_lag3"].notna()
    )

    x["pattern_3d"] = ""
    x.loc[
        three_valid,
        "pattern_3d",
    ] = (
        x.loc[three_valid, "wl_lag3"].astype(str)
        + x.loc[three_valid, "wl_lag2"].astype(str)
        + x.loc[three_valid, "wl_lag1"].astype(str)
    )

    x["target_win"] = x["actual_diff"] > 0

    return x


def baseline_stats(scope: pd.DataFrame) -> BaselineStats:
    n = len(scope)

    if n == 0:
        return BaselineStats(
            n=0,
            wins=0,
            win_rate=math.nan,
            mean_diff=math.nan,
            median_diff=math.nan,
            observed_target_days=0,
        )

    wins = int(scope["target_win"].sum())

    return BaselineStats(
        n=n,
        wins=wins,
        win_rate=wins / n,
        mean_diff=float(scope["actual_diff"].mean()),
        median_diff=float(scope["actual_diff"].median()),
        observed_target_days=int(scope["target_date"].nunique()),
    )


def metric_row(
    *,
    family: str,
    pattern: str,
    scope: pd.DataFrame,
    baseline: BaselineStats,
    description: str,
) -> dict:
    n = len(scope)
    wins = int(scope["target_win"].sum())

    if n > 0:
        win_rate = wins / n
        mean_diff = float(scope["actual_diff"].mean())
        median_diff = float(scope["actual_diff"].median())
        observed_target_days = int(scope["target_date"].nunique())
        ci_low, ci_high = wilson_interval(wins, n)
    else:
        win_rate = math.nan
        mean_diff = math.nan
        median_diff = math.nan
        observed_target_days = 0
        ci_low = math.nan
        ci_high = math.nan

    if (
        not math.isnan(win_rate)
        and not math.isnan(baseline.win_rate)
    ):
        win_rate_diff_pp = (
            win_rate - baseline.win_rate
        ) * 100.0
    else:
        win_rate_diff_pp = math.nan

    return {
        "research_name": RESEARCH_NAME,
        "research_mode": RESEARCH_MODE,
        "store": STORE,
        "family": family,
        "pattern": pattern,
        "description": description,
        "n": n,
        "wins": wins,
        "losses": n - wins,
        "next_day_win_rate": (
            win_rate * 100.0
            if not math.isnan(win_rate)
            else math.nan
        ),
        "win_rate_ci95_low": (
            ci_low * 100.0
            if not math.isnan(ci_low)
            else math.nan
        ),
        "win_rate_ci95_high": (
            ci_high * 100.0
            if not math.isnan(ci_high)
            else math.nan
        ),
        "baseline_n": baseline.n,
        "baseline_wins": baseline.wins,
        "baseline_win_rate": (
            baseline.win_rate * 100.0
            if not math.isnan(baseline.win_rate)
            else math.nan
        ),
        "win_rate_diff_pp": win_rate_diff_pp,
        "next_day_mean_diff": mean_diff,
        "next_day_median_diff": median_diff,
        "baseline_mean_diff": baseline.mean_diff,
        "baseline_median_diff": baseline.median_diff,
        "observed_target_days": observed_target_days,
        "baseline_observed_target_days": baseline.observed_target_days,
    }


def build_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    streak_baseline_scope = panel.loc[
        panel["wl_lag1"].notna()
    ].copy()
    streak_baseline = baseline_stats(streak_baseline_scope)

    for length in range(1, 6):
        for symbol in ["W", "L"]:
            label = f"{symbol}{length}"
            subset = panel.loc[
                panel["streak_exact_label"].eq(label)
            ].copy()

            word = "win" if symbol == "W" else "loss"

            rows.append(
                metric_row(
                    family="STREAK_EXACT",
                    pattern=label,
                    scope=subset,
                    baseline=streak_baseline,
                    description=(
                        f"Exact {length}-{word} streak immediately "
                        "before target day"
                    ),
                )
            )

    two_patterns = [
        "WW",
        "WL",
        "LW",
        "LL",
    ]

    two_day_scope = panel.loc[
        panel["pattern_2d"].isin(two_patterns)
    ].copy()
    two_day_baseline = baseline_stats(two_day_scope)

    for pattern in two_patterns:
        subset = two_day_scope.loc[
            two_day_scope["pattern_2d"].eq(pattern)
        ].copy()

        rows.append(
            metric_row(
                family="PATTERN_2D",
                pattern=pattern,
                scope=subset,
                baseline=two_day_baseline,
                description=f"Exact trailing 2-day pattern {pattern}",
            )
        )

    three_patterns = [
        "WWW",
        "WWL",
        "WLW",
        "WLL",
        "LWW",
        "LWL",
        "LLW",
        "LLL",
    ]

    three_day_scope = panel.loc[
        panel["pattern_3d"].isin(three_patterns)
    ].copy()
    three_day_baseline = baseline_stats(three_day_scope)

    for pattern in three_patterns:
        subset = three_day_scope.loc[
            three_day_scope["pattern_3d"].eq(pattern)
        ].copy()

        rows.append(
            metric_row(
                family="PATTERN_3D",
                pattern=pattern,
                scope=subset,
                baseline=three_day_baseline,
                description=f"Exact trailing 3-day pattern {pattern}",
            )
        )

    summary = pd.DataFrame(rows)

    order = (
        [f"W{i}" for i in range(1, 6)]
        + [f"L{i}" for i in range(1, 6)]
        + two_patterns
        + three_patterns
    )
    order_map = {
        pattern: i
        for i, pattern in enumerate(order)
    }

    family_order = {
        "STREAK_EXACT": 0,
        "PATTERN_2D": 1,
        "PATTERN_3D": 2,
    }

    summary["_family_order"] = summary["family"].map(family_order)
    summary["_pattern_order"] = summary["pattern"].map(order_map)

    summary = summary.sort_values(
        ["_family_order", "_pattern_order"],
        kind="mergesort",
    ).drop(
        columns=["_family_order", "_pattern_order"]
    )

    return summary.reset_index(drop=True)


def build_detail_export(panel: pd.DataFrame) -> pd.DataFrame:
    detail = panel.loc[
        panel["streak_exact_label"].ne("")
        | panel["pattern_2d"].ne("")
        | panel["pattern_3d"].ne("")
    ].copy()

    columns = [
        "target_date",
        "machine_no",
        "actual_machine_name",
        "actual_diff",
        "target_win",
        "sequence_segment_id",
        "prior_rows_in_segment",
        "wl_lag1",
        "wl_lag2",
        "wl_lag3",
        "wl_lag4",
        "wl_lag5",
        "wl_lag6",
        "prior_streak_symbol",
        "prior_streak_length_raw",
        "streak_exact_label",
        "pattern_2d",
        "pattern_3d",
    ]

    detail = detail[columns].copy()
    detail["target_date"] = detail["target_date"].dt.date.astype(str)

    return detail


def build_breaks_export(panel: pd.DataFrame) -> pd.DataFrame:
    breaks = panel.loc[
        panel["sequence_break_reason"].ne("")
    ].copy()

    columns = [
        "target_date",
        "machine_no",
        "actual_machine_name",
        "prev_date",
        "prev_actual_machine_name",
        "calendar_gap_days",
        "sequence_break_reason",
        "sequence_segment_id",
    ]

    breaks = breaks[columns].copy()
    breaks["target_date"] = breaks["target_date"].dt.date.astype(str)
    breaks["prev_date"] = (
        pd.to_datetime(
            breaks["prev_date"],
            errors="coerce",
        )
        .dt.date
        .astype("string")
    )

    return breaks


def make_metadata(
    input_path: Path,
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "research_name": RESEARCH_NAME,
                "research_mode": RESEARCH_MODE,
                "version": VERSION,
                "store": STORE,
                "input_path": str(input_path),
                "output_dir": str(output_dir),
                "rows": len(df),
                "target_days": int(df["target_date"].nunique()),
                "date_min": df["target_date"].min().date().isoformat(),
                "date_max": df["target_date"].max().date().isoformat(),
                "unique_machines": int(df["machine_no"].nunique()),
                "win_definition": "actual_diff > 0",
                "lose_definition": "actual_diff <= 0",
                "sequence_identity": (
                    "same machine_no + calendar-consecutive target_date + "
                    "same actual_machine_name"
                ),
                "streak_rule": "EXACT_LENGTH_1_TO_5",
                "two_day_patterns": "WW|WL|LW|LL",
                "three_day_patterns": (
                    "WWW|WWL|WLW|WLL|LWW|LWL|LLW|LLL"
                ),
                "streak_baseline": (
                    "all rows with at least one valid previous day in the "
                    "same contiguous same-name segment"
                ),
                "two_day_baseline": (
                    "all rows with a valid exact trailing 2-day pattern"
                ),
                "three_day_baseline": (
                    "all rows with a valid exact trailing 3-day pattern"
                ),
                "ci_method": "Wilson 95%",
                "automatic_promotion": False,
                "champion_changed": False,
                "formal_forward_changed": False,
                "forward_guard_changed": False,
                "production_ranking_changed": False,
                "floor_status": "PAUSED",
            }
        ]
    )


def main() -> int:
    root = project_root()
    input_path = root / INPUT_RELATIVE
    output_dir = root / OUTPUT_RELATIVE

    print("=" * 72)
    print(f"{RESEARCH_NAME} {VERSION}")
    print(RESEARCH_MODE)
    print("=" * 72)
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print()

    df = load_input(input_path)
    qa = build_qa_table(df)
    panel = add_sequence_columns(df)
    summary = build_summary(panel)
    breaks = build_breaks_export(panel)
    details = build_detail_export(panel)
    metadata = make_metadata(
        input_path,
        df,
        output_dir,
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
        qa,
        output_dir / QA_FILENAME,
    )
    write_csv_atomic(
        breaks,
        output_dir / BREAKS_FILENAME,
    )
    write_csv_atomic(
        details,
        output_dir / DETAIL_FILENAME,
    )
    write_csv_atomic(
        metadata,
        output_dir / METADATA_FILENAME,
    )

    print("[INPUT QA]")
    print(f"rows              : {len(df):,}")
    print(f"target days       : {df['target_date'].nunique()}")
    print(
        "date range        : "
        f"{df['target_date'].min().date()} -> "
        f"{df['target_date'].max().date()}"
    )
    print(f"unique machines   : {df['machine_no'].nunique()}")
    print()

    print("[SEQUENCE QA]")
    print(
        "calendar-gap breaks        : "
        f"{int((breaks['sequence_break_reason'] == 'CALENDAR_GAP').sum()):,}"
    )
    print(
        "machine-name-change breaks : "
        f"{int((breaks['sequence_break_reason'] == 'MACHINE_NAME_CHANGE').sum()):,}"
    )
    print()

    print("[SUMMARY]")
    display_cols = [
        "family",
        "pattern",
        "n",
        "wins",
        "next_day_win_rate",
        "baseline_win_rate",
        "win_rate_diff_pp",
        "next_day_mean_diff",
        "next_day_median_diff",
        "win_rate_ci95_low",
        "win_rate_ci95_high",
        "observed_target_days",
    ]

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
    ):
        print(
            summary[display_cols].to_string(
                index=False,
                float_format=lambda value: f"{value:.2f}",
            )
        )

    print()
    print("[OUTPUT FILES]")
    for filename in [
        SUMMARY_FILENAME,
        QA_FILENAME,
        BREAKS_FILENAME,
        DETAIL_FILENAME,
        METADATA_FILENAME,
    ]:
        path = output_dir / filename
        print(f"{filename}: {path.stat().st_size:,} bytes")

    print()
    print(
        "No Champion / Formal Forward / Forward Guard / production "
        "ranking changes were made."
    )
    print(
        "Review these OFFLINE results in 03 before any further research."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
