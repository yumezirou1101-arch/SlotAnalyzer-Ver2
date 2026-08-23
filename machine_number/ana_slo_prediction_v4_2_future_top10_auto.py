from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

MACHINE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

FIXED_SCRIPT = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_prediction_v4_2_future_top10_fixed.py"
)

EXPECTED_MACHINES = 514

DAILY_FILE_RE = re.compile(
    r"^ana_slo_(\d{8})\.csv$",
    re.IGNORECASE,
)


def header(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically detect the latest validated daily CSV "
            "and run the existing V4.2_C future TOP10 prediction "
            "for the next day."
        )
    )

    parser.add_argument(
        "--target-date",
        default=None,
        help=(
            "Optional target date YYYY-MM-DD. "
            "Default: latest validated daily CSV + 1 day."
        ),
    )

    parser.add_argument(
        "--allow-gap",
        action="store_true",
        help=(
            "Allow an explicit target date more than one day after "
            "the latest validated daily CSV."
        ),
    )

    return parser.parse_args()


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import script: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp932",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not read CSV: {path}\n"
        f"last error: {last_error}"
    )


def detect_daily_files() -> list[tuple[pd.Timestamp, Path]]:
    found: list[tuple[pd.Timestamp, Path]] = []

    for path in MACHINE_DATA_DIR.glob(
        "ana_slo_????????.csv"
    ):
        match = DAILY_FILE_RE.match(
            path.name
        )

        if not match:
            continue

        date = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(date):
            continue

        found.append(
            (
                pd.Timestamp(date),
                path,
            )
        )

    return sorted(
        found,
        key=lambda x: x[0],
    )


def validate_latest_daily_csv(
    path: Path,
    expected_date: pd.Timestamp,
) -> dict:
    df = read_csv_flexible(
        path
    )

    result = {
        "rows": len(df),
        "unique_machines": None,
        "duplicates": None,
        "date_ok": None,
    }

    machine_col = None

    for candidate in (
        "台番号",
        "machine_no",
    ):
        if candidate in df.columns:
            machine_col = candidate
            break

    if machine_col is None:
        raise RuntimeError(
            f"Machine-number column not found in {path}"
        )

    machine_numeric = pd.to_numeric(
        df[machine_col],
        errors="coerce",
    )

    result[
        "unique_machines"
    ] = int(
        machine_numeric.nunique(
            dropna=True
        )
    )

    result[
        "duplicates"
    ] = int(
        machine_numeric.duplicated(
            keep=False
        ).sum()
    )

    date_col = None

    for candidate in (
        "日付",
        "date",
    ):
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is not None:
        dates = pd.to_datetime(
            df[date_col],
            errors="coerce",
        ).dropna()

        result[
            "date_ok"
        ] = (
            len(dates) > 0
            and dates.nunique() == 1
            and pd.Timestamp(
                dates.iloc[0]
            ).normalize()
            == expected_date.normalize()
        )

    if result["rows"] != EXPECTED_MACHINES:
        raise RuntimeError(
            f"Latest daily CSV has {result['rows']} rows; "
            f"expected {EXPECTED_MACHINES}."
        )

    if result["unique_machines"] != EXPECTED_MACHINES:
        raise RuntimeError(
            f"Latest daily CSV has "
            f"{result['unique_machines']} unique machines; "
            f"expected {EXPECTED_MACHINES}."
        )

    if result["duplicates"] != 0:
        raise RuntimeError(
            f"Latest daily CSV has duplicate machine numbers: "
            f"{result['duplicates']}"
        )

    if result["date_ok"] is False:
        raise RuntimeError(
            "Latest daily CSV internal date does not match filename date."
        )

    return result


def main() -> None:
    args = parse_args()

    header(
        "V4.2_C Future TOP10 - Automatic Date Launcher"
    )

    daily_files = detect_daily_files()

    if not daily_files:
        raise RuntimeError(
            f"No ana_slo_YYYYMMDD.csv files found in:\n"
            f"{MACHINE_DATA_DIR}"
        )

    latest_date, latest_path = daily_files[-1]

    validation = validate_latest_daily_csv(
        latest_path,
        latest_date,
    )

    if args.target_date:
        target_date = pd.Timestamp(
            pd.to_datetime(
                args.target_date,
                format="%Y-%m-%d",
                errors="raise",
            )
        )

        if target_date <= latest_date:
            raise RuntimeError(
                "Target date must be later than latest data date."
            )

        expected_next = (
            latest_date
            + pd.Timedelta(
                days=1
            )
        )

        if (
            target_date != expected_next
            and not args.allow_gap
        ):
            raise RuntimeError(
                "Target date is not exactly one day after latest data date. "
                "Use --allow-gap only if this is intentional."
            )

    else:
        target_date = (
            latest_date
            + pd.Timedelta(
                days=1
            )
        )

    print(
        f"latest daily CSV      : {latest_path}"
    )

    print(
        f"latest data date      : {latest_date.date()}"
    )

    print(
        f"prediction target     : {target_date.date()}"
    )

    print(
        f"rows                  : {validation['rows']}"
    )

    print(
        f"unique machines       : {validation['unique_machines']}"
    )

    print(
        f"duplicates            : {validation['duplicates']}"
    )

    print(
        f"internal date check   : {validation['date_ok']}"
    )

    fixed = load_module(
        FIXED_SCRIPT,
        "slotanalyzer_future_top10_fixed",
    )

    fixed.TARGET_DATE = pd.Timestamp(
        target_date
    )

    fixed.EXPECTED_LATEST_DATA_DATE = pd.Timestamp(
        latest_date
    )

    header(
        "RUN EXISTING V4.2_C FUTURE PREDICTION"
    )

    fixed.main()


if __name__ == "__main__":
    main()
