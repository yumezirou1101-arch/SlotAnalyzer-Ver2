from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STORE_NAME = "\u3084\u3059\u3060\u524d\u6a4b\u5e97"

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "yasuda_maebashi"
    / "source_html"
)

CSV_DIR = (
    PROJECT_ROOT
    / "data"
    / "yasuda_maebashi"
    / "machine_number"
)

FETCH_SCRIPT = (
    PROJECT_ROOT
    / "machine_number"
    / "ana_slo_yasuda_maebashi_click_fetch_v1.py"
)

CONVERTER_SCRIPT = (
    PROJECT_ROOT
    / "ana_slo_yasuda_maebashi_source_html_to_daily_csv_v1.py"
)

SOURCE_RE = re.compile(
    r"^ana_slo_(\d{8})_source\.html$"
)

CSV_RE = re.compile(
    r"^ana_slo_(\d{8})\.csv$"
)


def banner(text: str) -> None:
    print()
    print("=" * 92)
    print(text)
    print("=" * 92)


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must be YYYY-MM-DD."
        ) from exc


def date_from_source(path: Path) -> date:
    match = SOURCE_RE.match(path.name)

    if not match:
        raise RuntimeError(
            f"Invalid source filename: {path.name}"
        )

    return datetime.strptime(
        match.group(1),
        "%Y%m%d",
    ).date()


def expected_csv_path(source_date: date) -> Path:
    return (
        CSV_DIR
        / f"ana_slo_{source_date:%Y%m%d}.csv"
    )


def newest_source() -> Path:
    files = []

    if SOURCE_DIR.exists():
        for path in SOURCE_DIR.glob(
            "ana_slo_*_source.html"
        ):
            if SOURCE_RE.match(path.name):
                files.append(path)

    if not files:
        raise RuntimeError(
            "No Yasuda Maebashi source HTML found."
        )

    return max(
        files,
        key=date_from_source,
    )


def run_child(
    label: str,
    command: list[str],
) -> None:
    banner(label)

    print(
        "COMMAND:",
        subprocess.list2cmdline(command),
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed "
            f"with exit code {result.returncode}."
        )


def validate_paths() -> None:
    missing = []

    for path in (
        FETCH_SCRIPT,
        CONVERTER_SCRIPT,
    ):
        if not path.is_file():
            missing.append(path)

    if missing:
        lines = [
            "Required script not found:"
        ]

        lines.extend(
            f"  {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "\n".join(lines)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-click daily update for "
            "Yasuda Maebashi Ana-Slo data."
        )
    )

    parser.add_argument(
        "--fetch-days",
        type=int,
        default=1,
        help=(
            "Number of newest visible Ana-Slo "
            "date links to inspect. Default: 1."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Pass --overwrite to the fetch script."
        ),
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help=(
            "Skip Ana-Slo fetch and use the "
            "newest existing Yasuda source HTML."
        ),
    )

    parser.add_argument(
        "--target-date",
        type=parse_date,
        default=None,
        help=(
            "Operational target date YYYY-MM-DD. "
            "Default: today. The expected actual "
            "source date is the previous calendar day."
        ),
    )

    parser.add_argument(
        "--allow-gap",
        action="store_true",
        help=(
            "Allow the newest source date to be older "
            "than target-date minus one day. "
            "Use only intentionally."
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.fetch_days < 1:
        raise ValueError(
            "--fetch-days must be >= 1"
        )

    validate_paths()

    target_date = (
        args.target_date
        if args.target_date is not None
        else date.today()
    )

    expected_source_date = (
        target_date
        - timedelta(days=1)
    )

    banner(
        "YASUDA MAEBASHI - ONE CLICK DAILY UPDATE V1"
    )

    print(f"store                : {STORE_NAME}")
    print(f"target date          : {target_date}")
    print(
        "expected source date : "
        f"{expected_source_date}"
    )
    print(f"fetch days           : {args.fetch_days}")
    print(f"skip fetch           : {args.skip_fetch}")
    print(f"allow gap            : {args.allow_gap}")

    if not args.skip_fetch:
        fetch_command = [
            sys.executable,
            str(FETCH_SCRIPT),
            "--max-days",
            str(args.fetch_days),
        ]

        if args.overwrite:
            fetch_command.append(
                "--overwrite"
            )

        run_child(
            "STEP 1 / 3 - FETCH NEWEST HTML",
            fetch_command,
        )
    else:
        banner(
            "STEP 1 / 3 - FETCH NEWEST HTML : SKIPPED"
        )

    source_html = newest_source()
    source_date = date_from_source(
        source_html
    )

    banner(
        "STEP 2 / 3 - SOURCE FRESHNESS CHECK"
    )

    print(f"source HTML          : {source_html}")
    print(f"source date          : {source_date}")
    print(
        "expected source date : "
        f"{expected_source_date}"
    )

    if source_date > expected_source_date:
        raise RuntimeError(
            "Source date is newer than the expected "
            "operational source date. "
            f"source={source_date}, "
            f"expected={expected_source_date}"
        )

    if (
        source_date != expected_source_date
        and not args.allow_gap
    ):
        raise RuntimeError(
            "FRESHNESS GUARD: newest Yasuda Maebashi "
            "source is not fresh enough. "
            f"source={source_date}, "
            f"expected={expected_source_date}. "
            "Daily CSV conversion was stopped."
        )

    if source_date != expected_source_date:
        print(
            "WARNING: freshness gap accepted "
            "because --allow-gap was specified."
        )
    else:
        print("FRESHNESS: OK")

    run_child(
        "STEP 3 / 3 - CONVERT AND VALIDATE DAILY CSV",
        [
            sys.executable,
            str(CONVERTER_SCRIPT),
            str(source_html),
            "--output-dir",
            str(CSV_DIR),
        ],
    )

    daily_csv = expected_csv_path(
        source_date
    )

    if not daily_csv.is_file():
        raise RuntimeError(
            "Validated daily CSV was not created: "
            f"{daily_csv}"
        )

    csv_date_match = CSV_RE.match(
        daily_csv.name
    )

    if not csv_date_match:
        raise RuntimeError(
            f"Unexpected CSV filename: {daily_csv.name}"
        )

    csv_date = datetime.strptime(
        csv_date_match.group(1),
        "%Y%m%d",
    ).date()

    if csv_date != source_date:
        raise RuntimeError(
            "CSV date does not match source date. "
            f"csv={csv_date}, "
            f"source={source_date}"
        )

    banner(
        "FINAL RESULT"
    )

    print("RESULT               : OK")
    print(f"store                : {STORE_NAME}")
    print(f"source date          : {source_date}")
    print(f"source HTML          : {source_html}")
    print(f"daily CSV            : {daily_csv}")
    print()
    print(
        "Yasuda Maebashi acquisition, freshness, "
        "conversion, and quality validation completed."
    )


if __name__ == "__main__":
    main()