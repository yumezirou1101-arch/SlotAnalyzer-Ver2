from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# SlotAnalyzer - Min-Repo Collection Status
# ============================================================
#
# Purpose:
#   Compare Ana-Slo daily CSV availability with saved Min-Repo HTMLs
#   and show which dates are still missing.
#
# This script DOES NOT download anything and DOES NOT modify
# existing Ana-Slo / prediction-model data.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

ANA_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "external_features"
    / "minrepo"
)

DEFAULT_START = "2026-08-01"
DEFAULT_END = "2026-08-18"


# ============================================================
# HELPERS
# ============================================================

def header(
    title: str,
) -> None:

    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Check Min-Repo collection coverage against "
            "Ana-Slo daily CSV availability."
        )
    )

    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=(
            "Start date YYYY-MM-DD "
            f"(default: {DEFAULT_START})"
        ),
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=(
            "End date YYYY-MM-DD "
            f"(default: {DEFAULT_END})"
        ),
    )

    return parser.parse_args()


def normalize_date(
    value: str,
) -> pd.Timestamp:

    date = pd.to_datetime(
        value,
        format="%Y-%m-%d",
        errors="raise",
    )

    return pd.Timestamp(
        date
    )


def find_ana_dates() -> set[pd.Timestamp]:

    dates: set[pd.Timestamp] = set()

    pattern = re.compile(
        r"^ana_slo_(\d{8})\.csv$",
        re.IGNORECASE,
    )

    for path in ANA_DIR.glob(
        "ana_slo_????????.csv"
    ):

        match = pattern.match(
            path.name
        )

        if not match:
            continue

        date = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.notna(
            date
        ):
            dates.add(
                pd.Timestamp(
                    date
                )
            )

    return dates


def find_minrepo_dates(
    suffix: str,
) -> set[pd.Timestamp]:

    dates: set[pd.Timestamp] = set()

    pattern = re.compile(
        rf"^minrepo_(\d{{8}})_{re.escape(suffix)}\.html$",
        re.IGNORECASE,
    )

    for path in PROJECT_ROOT.glob(
        f"minrepo_*_{suffix}.html"
    ):

        match = pattern.match(
            path.name
        )

        if not match:
            continue

        date = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.notna(
            date
        ):
            dates.add(
                pd.Timestamp(
                    date
                )
            )

    return dates


def mark(
    value: bool,
) -> str:

    return (
        "OK"
        if value
        else "MISSING"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    start = normalize_date(
        args.start
    )

    end = normalize_date(
        args.end
    )

    if start > end:
        raise SystemExit(
            "start date must be <= end date"
        )

    ana_dates = find_ana_dates()

    allmachines_dates = (
        find_minrepo_dates(
            "allmachines"
        )
    )

    full_dates = find_minrepo_dates(
        "full"
    )

    dates = pd.date_range(
        start,
        end,
        freq="D",
    )

    rows = []

    for date in dates:

        ymd = date.strftime(
            "%Y%m%d"
        )

        ana_ok = (
            date in ana_dates
        )

        all_ok = (
            date in allmachines_dates
        )

        full_ok = (
            date in full_dates
        )

        complete = (
            ana_ok
            and all_ok
            and full_ok
        )

        rows.append(
            {
                "date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "weekday":
                    [
                        "月",
                        "火",
                        "水",
                        "木",
                        "金",
                        "土",
                        "日",
                    ][
                        date.dayofweek
                    ],

                "ana_slo_csv":
                    mark(
                        ana_ok
                    ),

                "minrepo_allmachines":
                    mark(
                        all_ok
                    ),

                "minrepo_full":
                    mark(
                        full_ok
                    ),

                "complete":
                    mark(
                        complete
                    ),

                "expected_allmachines_file":
                    (
                        f"minrepo_{ymd}_allmachines.html"
                    ),

                "expected_full_file":
                    (
                        f"minrepo_{ymd}_full.html"
                    ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    header(
        "Min-Repo Collection Status"
    )

    print(
        f"range                 : "
        f"{start.date()} to {end.date()}"
    )

    print(
        f"target days           : {len(df)}"
    )

    print(
        f"Ana-Slo CSV available : "
        f"{(df['ana_slo_csv'] == 'OK').sum()}"
    )

    print(
        f"MinRepo allmachines   : "
        f"{(df['minrepo_allmachines'] == 'OK').sum()}"
    )

    print(
        f"MinRepo full          : "
        f"{(df['minrepo_full'] == 'OK').sum()}"
    )

    print(
        f"complete dates        : "
        f"{(df['complete'] == 'OK').sum()}"
    )

    # --------------------------------------------------------
    # Dates where Ana-Slo exists:
    # these are the practical Min-Repo collection targets.
    # --------------------------------------------------------

    collectable = df[
        df[
            "ana_slo_csv"
        ]
        == "OK"
    ].copy()

    missing = collectable[
        (
            collectable[
                "minrepo_allmachines"
            ]
            != "OK"
        )
        | (
            collectable[
                "minrepo_full"
            ]
            != "OK"
        )
    ].copy()

    header(
        "STATUS BY DATE"
    )

    print(
        df[
            [
                "date",
                "weekday",
                "ana_slo_csv",
                "minrepo_allmachines",
                "minrepo_full",
                "complete",
            ]
        ].to_string(
            index=False
        )
    )

    header(
        "NEXT COLLECTION TARGETS"
    )

    if missing.empty:

        print(
            "No missing Min-Repo files for dates "
            "that already have Ana-Slo daily CSVs."
        )

    else:

        target_cols = [
            "date",
            "weekday",
            "minrepo_allmachines",
            "minrepo_full",
            "expected_allmachines_file",
            "expected_full_file",
        ]

        print(
            missing[
                target_cols
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save status
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        OUTPUT_DIR
        / "minrepo_collection_status.csv"
    )

    missing_path = (
        OUTPUT_DIR
        / "minrepo_collection_missing.csv"
    )

    df.to_csv(
        status_path,
        index=False,
        encoding="utf-8-sig",
    )

    missing.to_csv(
        missing_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    print(
        status_path
    )

    print(
        missing_path
    )

    print()
    print(
        "This script only checks collection status. "
        "No existing data was modified."
    )


if __name__ == "__main__":
    main()
