from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from slotanalyzer_derived_prediction_evidence import (
    verify_derived_prediction,
    verify_source_64,
)


# ============================================================
# V4.2_C Top15 DISPLAY Artifacts v1
# ============================================================
#
# Purpose
# -------
# Create DISPLAY-ONLY Top15 CSV files for:
#
#   - NORMAL
#   - A-TYPE
#   - JUGGLER
#
# from the already-created frozen prediction artifacts.
#
# Important safety policy
# -----------------------
# - Champion score is NOT recalculated.
# - Ranking is NOT recalculated.
# - Existing Top10 files are NOT modified.
# - Existing metadata files are NOT modified.
# - Existing Formal Forward artifacts are NOT modified.
# - Forward Guard is NOT modified.
# - Formal evaluation remains Top1 / Top3 / Top5 / Top10.
# - Top15 is DISPLAY / research convenience only.
#
# The source ranking columns already exist:
#
#   NORMAL  : prediction_rank
#   A-TYPE  : a_type_rank
#   JUGGLER : juggler_rank
#
# This script only selects existing ranks 1-15.
#
# Before saving, ranks 1-10 are compared against the frozen
# official Top10 artifacts.  machine_no, machine_name, score
# and rank must match exactly.
#
# Existing Top15 files are never overwritten.  If a Top15 file
# already exists, it is verified against the expected content.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
)

DIR_64 = (
    ANALYSIS_DIR
    / "64_Ver4_2_future_top10"
)

DIR_74 = (
    ANALYSIS_DIR
    / "74_Ver4_2_A_type_prediction"
)

DIR_75 = (
    ANALYSIS_DIR
    / "75_Ver4_2_Juggler_prediction"
)

DISPLAY_N = 15
FORMAL_TOP10_N = 10


# ============================================================
# Helpers
# ============================================================


def header(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create DISPLAY-ONLY Top15 artifacts from frozen "
            "V4.2_C NORMAL / A-TYPE / JUGGLER rankings."
        )
    )

    parser.add_argument(
        "--target-date",
        required=True,
        help="Prediction target date YYYY-MM-DD.",
    )

    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required CSV not found:\n{path}"
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Required CSV is empty:\n{path}"
        )

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )


def normalize_target_date(
    value: str,
) -> pd.Timestamp:
    target = pd.to_datetime(
        value,
        format="%Y-%m-%d",
        errors="raise",
    )

    return pd.Timestamp(
        target
    ).normalize()


def validate_rank_column(
    frame: pd.DataFrame,
    rank_col: str,
    label: str,
) -> pd.Series:
    if rank_col not in frame.columns:
        raise RuntimeError(
            f"{label}: missing rank column: {rank_col}"
        )

    ranks = pd.to_numeric(
        frame[rank_col],
        errors="coerce",
    )

    if ranks.isna().any():
        raise RuntimeError(
            f"{label}: invalid/non-numeric rank detected."
        )

    ranks = ranks.astype(int)

    if ranks.duplicated().any():
        duplicated = sorted(
            ranks[
                ranks.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"{label}: duplicated ranks detected: "
            f"{duplicated}"
        )

    expected = set(
        range(
            1,
            len(frame) + 1,
        )
    )

    actual = set(
        ranks.tolist()
    )

    if actual != expected:
        missing = sorted(
            expected - actual
        )

        extra = sorted(
            actual - expected
        )

        raise RuntimeError(
            f"{label}: rank sequence is not consecutive. "
            f"missing={missing}, extra={extra}"
        )

    return ranks


def select_existing_top15(
    frame: pd.DataFrame,
    rank_col: str,
    label: str,
) -> pd.DataFrame:
    ranks = validate_rank_column(
        frame,
        rank_col,
        label,
    )

    if len(frame) < DISPLAY_N:
        raise RuntimeError(
            f"{label}: fewer than {DISPLAY_N} ranked rows. "
            f"rows={len(frame)}"
        )

    selected = frame.loc[
        ranks <= DISPLAY_N
    ].copy()

    selected["_display_rank"] = pd.to_numeric(
        selected[rank_col],
        errors="raise",
    ).astype(int)

    selected = (
        selected
        .sort_values(
            "_display_rank",
            kind="stable",
        )
        .drop(
            columns=[
                "_display_rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    selected_ranks = (
        pd.to_numeric(
            selected[rank_col],
            errors="raise",
        )
        .astype(int)
        .tolist()
    )

    expected_ranks = list(
        range(
            1,
            DISPLAY_N + 1,
        )
    )

    if selected_ranks != expected_ranks:
        raise RuntimeError(
            f"{label}: Top15 ranks are invalid. "
            f"actual={selected_ranks}"
        )

    return selected


def compare_top10_exact(
    expected_top15: pd.DataFrame,
    frozen_top10: pd.DataFrame,
    rank_col: str,
    label: str,
) -> None:
    required = [
        rank_col,
        "machine_no",
        "machine_name",
        "score",
    ]

    missing_top15 = [
        col
        for col in required
        if col not in expected_top15.columns
    ]

    missing_top10 = [
        col
        for col in required
        if col not in frozen_top10.columns
    ]

    if missing_top15:
        raise RuntimeError(
            f"{label}: Top15 source missing columns: "
            f"{missing_top15}"
        )

    if missing_top10:
        raise RuntimeError(
            f"{label}: frozen Top10 missing columns: "
            f"{missing_top10}"
        )

    if len(frozen_top10) != FORMAL_TOP10_N:
        raise RuntimeError(
            f"{label}: frozen Top10 must contain exactly "
            f"{FORMAL_TOP10_N} rows. "
            f"rows={len(frozen_top10)}"
        )

    top15_first10 = (
        expected_top15[
            required
        ]
        .head(
            FORMAL_TOP10_N
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    frozen = (
        frozen_top10[
            required
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    for frame in (
        top15_first10,
        frozen,
    ):
        frame[rank_col] = pd.to_numeric(
            frame[rank_col],
            errors="raise",
        ).astype(int)

        frame["machine_no"] = pd.to_numeric(
            frame["machine_no"],
            errors="raise",
        ).astype(int)

        frame["machine_name"] = (
            frame["machine_name"]
            .astype(str)
        )

        frame["score"] = pd.to_numeric(
            frame["score"],
            errors="raise",
        )

    try:
        assert_frame_equal(
            top15_first10,
            frozen,
            check_dtype=True,
            check_exact=True,
        )

    except AssertionError as exc:
        raise RuntimeError(
            f"{label}: Top10 exact-match QA FAILED.\n"
            f"{exc}"
        ) from exc


def validate_target_dates(
    frame: pd.DataFrame,
    target_date: pd.Timestamp,
    label: str,
) -> None:
    if "target_date" not in frame.columns:
        raise RuntimeError(
            f"{label}: target_date column is missing."
        )

    targets = pd.to_datetime(
        frame["target_date"],
        errors="coerce",
    )

    if targets.isna().any():
        raise RuntimeError(
            f"{label}: invalid target_date detected."
        )

    targets = targets.dt.normalize()

    unique_targets = set(
        targets.tolist()
    )

    if unique_targets != {
        target_date
    }:
        raise RuntimeError(
            f"{label}: target_date mismatch. "
            f"expected={target_date.date()}, "
            f"actual={sorted(unique_targets)}"
        )


def compare_existing_top15(
    expected: pd.DataFrame,
    existing: pd.DataFrame,
    rank_col: str,
    label: str,
) -> None:
    if len(existing) != DISPLAY_N:
        raise RuntimeError(
            f"{label}: existing Top15 row count mismatch. "
            f"expected={DISPLAY_N}, "
            f"actual={len(existing)}"
        )

    if list(existing.columns) != list(expected.columns):
        raise RuntimeError(
            f"{label}: existing Top15 column layout mismatch."
        )

    required = [
        rank_col,
        "machine_no",
        "machine_name",
        "score",
        "target_date",
        "latest_data_date",
    ]

    missing = [
        col
        for col in required
        if col not in existing.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label}: existing Top15 missing columns: "
            f"{missing}"
        )

    left = (
        expected[
            required
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    right = (
        existing[
            required
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    for frame in (
        left,
        right,
    ):
        frame[rank_col] = pd.to_numeric(
            frame[rank_col],
            errors="raise",
        ).astype(int)

        frame["machine_no"] = pd.to_numeric(
            frame["machine_no"],
            errors="raise",
        ).astype(int)

        frame["machine_name"] = (
            frame["machine_name"]
            .astype(str)
        )

        frame["score"] = pd.to_numeric(
            frame["score"],
            errors="raise",
        )

        frame["target_date"] = (
            pd.to_datetime(
                frame["target_date"],
                errors="raise",
            )
            .dt
            .strftime(
                "%Y-%m-%d"
            )
        )

        frame["latest_data_date"] = (
            pd.to_datetime(
                frame["latest_data_date"],
                errors="raise",
            )
            .dt
            .strftime(
                "%Y-%m-%d"
            )
        )

    try:
        assert_frame_equal(
            left,
            right,
            check_dtype=True,
            check_exact=True,
        )

    except AssertionError as exc:
        raise RuntimeError(
            f"{label}: existing Top15 verification FAILED.\n"
            f"{exc}"
        ) from exc


def save_or_verify_top15(
    frame: pd.DataFrame,
    output_path: Path,
    rank_col: str,
    label: str,
) -> str:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        existing = read_csv(
            output_path
        )

        compare_existing_top15(
            frame,
            existing,
            rank_col,
            label,
        )

        print(
            f"{label:<18}: ALREADY_EXISTS / VERIFIED"
        )

        return "ALREADY_EXISTS_VERIFIED"

    frame.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        mode="x",
    )

    print(
        f"{label:<18}: CREATED"
    )

    return "CREATED"


def process_category(
    *,
    label: str,
    all_path: Path,
    top10_path: Path,
    top15_path: Path,
    rank_col: str,
    target_date: pd.Timestamp,
) -> dict:
    all_frame = read_csv(
        all_path
    )

    frozen_top10 = read_csv(
        top10_path
    )

    validate_target_dates(
        all_frame,
        target_date,
        f"{label} all",
    )

    validate_target_dates(
        frozen_top10,
        target_date,
        f"{label} frozen Top10",
    )

    top15 = select_existing_top15(
        all_frame,
        rank_col,
        label,
    )

    compare_top10_exact(
        top15,
        frozen_top10,
        rank_col,
        label,
    )

    print(
        f"{label:<18}: Top10 exact QA PASS"
    )

    status = save_or_verify_top15(
        top15,
        top15_path,
        rank_col,
        label,
    )

    first_rank = int(
        pd.to_numeric(
            top15.iloc[0][rank_col],
            errors="raise",
        )
    )

    last_rank = int(
        pd.to_numeric(
            top15.iloc[-1][rank_col],
            errors="raise",
        )
    )

    return {
        "label": label,
        "status": status,
        "rows": len(top15),
        "first_rank": first_rank,
        "last_rank": last_rank,
        "path": top15_path,
    }


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = parse_args()

    target_date = normalize_target_date(
        args.target_date
    )

    target_day = (
        target_date.date()
    )

    ymd = target_date.strftime(
        "%Y%m%d"
    )

    header(
        "V4.2_C TOP15 DISPLAY ARTIFACTS v1"
    )

    print(
        f"target date           : {target_day}"
    )

    print(
        "policy                : DISPLAY ONLY"
    )

    print(
        "score recalculation   : NO"
    )

    print(
        "rank recalculation    : NO"
    )

    print(
        "formal Top10 modified : NO"
    )

    print(
        "Formal Forward changed: NO"
    )

    # --------------------------------------------------------
    # Verify frozen formal source sets first.
    # --------------------------------------------------------

    header(
        "VERIFY FROZEN SOURCE ARTIFACTS"
    )

    source64 = verify_source_64(
        DIR_64,
        target_day,
    )

    verify_derived_prediction(
        DIR_74,
        DIR_64,
        target_day,
        "A_TYPE",
    )

    verify_derived_prediction(
        DIR_75,
        DIR_64,
        target_day,
        "JUGGLER",
    )

    print(
        "64 NORMAL             : VERIFIED"
    )

    print(
        "74 A-TYPE             : VERIFIED"
    )

    print(
        "75 JUGGLER            : VERIFIED"
    )

    print(
        f"model                 : {source64.model}"
    )

    print(
        f"weight fingerprint    : {source64.weight_fingerprint}"
    )

    print(
        f"latest data date      : {source64.latest_data_date}"
    )

    # --------------------------------------------------------
    # Source and output paths.
    # --------------------------------------------------------

    normal_all = (
        DIR_64
        / f"64_prediction_{ymd}_all514.csv"
    )

    normal_top10 = (
        DIR_64
        / f"64_prediction_{ymd}_top10.csv"
    )

    normal_top15 = (
        DIR_64
        / f"64_prediction_{ymd}_top15.csv"
    )

    a_all = (
        DIR_74
        / f"74_A_type_prediction_{ymd}_all.csv"
    )

    a_top10 = (
        DIR_74
        / f"74_A_type_prediction_{ymd}_top10.csv"
    )

    a_top15 = (
        DIR_74
        / f"74_A_type_prediction_{ymd}_top15.csv"
    )

    j_all = (
        DIR_75
        / f"75_Juggler_prediction_{ymd}_all.csv"
    )

    j_top10 = (
        DIR_75
        / f"75_Juggler_prediction_{ymd}_top10.csv"
    )

    j_top15 = (
        DIR_75
        / f"75_Juggler_prediction_{ymd}_top15.csv"
    )

    # --------------------------------------------------------
    # Create / verify DISPLAY Top15 artifacts.
    # --------------------------------------------------------

    header(
        "CREATE / VERIFY DISPLAY TOP15"
    )

    results = []

    results.append(
        process_category(
            label="NORMAL",
            all_path=normal_all,
            top10_path=normal_top10,
            top15_path=normal_top15,
            rank_col="prediction_rank",
            target_date=target_date,
        )
    )

    results.append(
        process_category(
            label="A-TYPE",
            all_path=a_all,
            top10_path=a_top10,
            top15_path=a_top15,
            rank_col="a_type_rank",
            target_date=target_date,
        )
    )

    results.append(
        process_category(
            label="JUGGLER",
            all_path=j_all,
            top10_path=j_top10,
            top15_path=j_top15,
            rank_col="juggler_rank",
            target_date=target_date,
        )
    )

    # --------------------------------------------------------
    # Completion QA.
    # --------------------------------------------------------

    header(
        "TOP15 DISPLAY QA"
    )

    for result in results:
        print(
            f"{result['label']:<18}: "
            f"rows={result['rows']} "
            f"ranks={result['first_rank']}-{result['last_rank']} "
            f"status={result['status']}"
        )

    if any(
        result["rows"] != DISPLAY_N
        or result["first_rank"] != 1
        or result["last_rank"] != DISPLAY_N
        for result in results
    ):
        raise RuntimeError(
            "Top15 DISPLAY QA FAILED."
        )

    header(
        "FILES"
    )

    for result in results:
        print(
            result["path"]
        )

    header(
        "COMPLETE"
    )

    print(
        "Top15 DISPLAY artifacts completed successfully."
    )

    print(
        "Existing Formal Top10 artifacts were not modified."
    )

    print(
        "Champion score / ranking logic was not modified."
    )

    print(
        "Formal evaluation remains Top1 / Top3 / Top5 / Top10."
    )

    print(
        "Recent-3-day actual differences are intentionally "
        "NOT added here; they belong to the notification "
        "DISPLAY layer."
    )


if __name__ == "__main__":
    main()