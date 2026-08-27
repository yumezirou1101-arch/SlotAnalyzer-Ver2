from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = PROJECT_ROOT / "data" / "maruhan_maebashi" / "machine_number"
CONVERTER = PROJECT_ROOT / "ana_slo_source_html_to_daily_csv_auto.py"
FORWARD = PROJECT_ROOT / "machine_number" / "ana_slo_prediction_v4_2_forward_champion_challenger.py"
FUTURE = PROJECT_ROOT / "machine_number" / "ana_slo_prediction_v4_2_future_top10_auto.py"

EXPECTED_MACHINES = 514
SOURCE_RE = re.compile(r"^ana_slo_(\d{8})_source\.html$", re.IGNORECASE)


def header(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the daily SlotAnalyzer update after Ana-Slo HTML is saved."
    )
    parser.add_argument(
        "source_html",
        nargs="?",
        default=None,
        help="Optional ana_slo_YYYYMMDD_source.html path.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def detect_source_html() -> tuple[Path, pd.Timestamp]:
    candidates = []

    for path in PROJECT_ROOT.glob("ana_slo_????????_source.html"):
        match = SOURCE_RE.match(path.name)
        if not match:
            continue

        date = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(date):
            continue

        candidates.append((pd.Timestamp(date), path))

    if not candidates:
        raise RuntimeError("No ana_slo_YYYYMMDD_source.html found.")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1], candidates[-1][0]


def resolve_source(source_arg: str | None) -> tuple[Path, pd.Timestamp]:
    if source_arg is None:
        return detect_source_html()

    path = Path(source_arg)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()
    require_file(path)

    match = SOURCE_RE.match(path.name)

    if not match:
        raise RuntimeError(
            "Source filename must be ana_slo_YYYYMMDD_source.html"
        )

    date = pd.Timestamp(
        pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="raise",
        )
    )

    return path, date


def run_step(title: str, command: list[str]) -> None:
    header(title)

    print("COMMAND:")
    print(" ".join(f'"{x}"' if " " in x else x for x in command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{title} failed (exit code {result.returncode})."
        )


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not read CSV: {path}\nlast error: {last_error}"
    )


def validate_daily_csv(
    csv_path: Path,
    expected_date: pd.Timestamp,
) -> dict:
    require_file(csv_path)

    df = read_csv_flexible(csv_path)

    machine_col = next(
        (c for c in ("台番号", "machine_no") if c in df.columns),
        None,
    )

    if machine_col is None:
        raise RuntimeError("Machine-number column not found.")

    machine_numeric = pd.to_numeric(
        df[machine_col],
        errors="coerce",
    )

    rows = len(df)
    unique_machines = int(
        machine_numeric.nunique(dropna=True)
    )
    duplicates = int(
        machine_numeric.duplicated(keep=False).sum()
    )
    missing_machine = int(
        machine_numeric.isna().sum()
    )

    date_col = next(
        (c for c in ("日付", "date") if c in df.columns),
        None,
    )

    if date_col is None:
        raise RuntimeError("Date column not found.")

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    valid_dates = dates.dropna()

    date_ok = (
        dates.isna().sum() == 0
        and valid_dates.nunique() == 1
        and pd.Timestamp(valid_dates.iloc[0]).normalize()
        == expected_date.normalize()
    )

    name_col = next(
        (c for c in ("機種名", "machine_name") if c in df.columns),
        None,
    )

    missing_name = None

    if name_col is not None:
        missing_name = int(
            df[name_col]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .isna()
            .sum()
        )

    if rows != EXPECTED_MACHINES:
        raise RuntimeError(
            f"CSV rows={rows}; expected {EXPECTED_MACHINES}."
        )

    if unique_machines != EXPECTED_MACHINES:
        raise RuntimeError(
            f"Unique machines={unique_machines}; expected {EXPECTED_MACHINES}."
        )

    if duplicates != 0:
        raise RuntimeError(
            f"Duplicate machine rows={duplicates}."
        )

    if missing_machine != 0:
        raise RuntimeError(
            f"Missing machine numbers={missing_machine}."
        )

    if not date_ok:
        raise RuntimeError(
            "CSV internal date check failed."
        )

    if missing_name not in (None, 0):
        raise RuntimeError(
            f"Missing machine names={missing_name}."
        )

    return {
        "rows": rows,
        "unique_machines": unique_machines,
        "duplicates": duplicates,
        "missing_machine": missing_machine,
        "missing_name": missing_name,
        "date_ok": date_ok,
    }


def main() -> None:
    args = parse_args()

    header("SlotAnalyzer - Daily Forward Update Runner")

    for path in (CONVERTER, FORWARD, FUTURE):
        require_file(path)

    source_html, source_date = resolve_source(
        args.source_html
    )

    csv_path = (
        DATA_DIR
        / f"ana_slo_{source_date:%Y%m%d}.csv"
    )

    print(f"source HTML  : {source_html}")
    print(f"source date  : {source_date.date()}")
    print(f"daily CSV    : {csv_path}")

    run_step(
        "STEP 1 / 3 - HTML -> DAILY CSV",
        [
            sys.executable,
            str(CONVERTER),
            str(source_html),
        ],
    )

    header("INDEPENDENT DAILY CSV CHECK")

    quality = validate_daily_csv(
        csv_path,
        source_date,
    )

    for key, value in quality.items():
        print(f"{key:20}: {value}")

    print()
    print("RESULT: DAILY CSV VALIDATION OK")

    run_step(
        "STEP 2 / 3 - FORWARD CHAMPION / CHALLENGER",
        [
            sys.executable,
            str(FORWARD),
        ],
    )

    run_step(
        "STEP 3 / 3 - NEXT-DAY V4.2_C TOP10",
        [
            sys.executable,
            str(FUTURE),
        ],
    )

    header("DAILY UPDATE COMPLETE")

    target_date = (
        source_date
        + pd.Timedelta(days=1)
    )

    top10_path = (
        DATA_DIR
        / "analysis_31days_deep"
        / "64_Ver4_2_future_top10"
        / f"64_prediction_{target_date:%Y%m%d}_top10.csv"
    )

    print(f"source date       : {source_date.date()}")
    print(f"prediction target : {target_date.date()}")
    print(f"expected TOP10    : {top10_path}")
    print(f"TOP10 exists      : {top10_path.exists()}")
    print()
    print(
        "No existing model weights or development-period data were modified."
    )
    print(
        f"completed at      : {datetime.now():%Y-%m-%d %H:%M:%S}"
    )


if __name__ == "__main__":
    main()
