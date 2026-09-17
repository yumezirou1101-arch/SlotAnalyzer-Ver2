from __future__ import annotations

from pathlib import Path
from io import StringIO
import argparse
import os
import re
import tempfile

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
SOURCE_GLOB = "ana_slo_bic_tsubame_takasaki_????????_source.html"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bic_tsubame_takasaki"
    / "machine_number"
)

SUMMARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "bic_tsubame_takasaki"
    / "batch_logs"
)

STORE_NAMES = (
    "ビックつばめ高崎店",
    "ビックつばめ高崎",
)

DEFAULT_MIN_MACHINES = 200

KNOWN_CLOSURE_START = pd.Timestamp("2026-07-21")
KNOWN_CLOSURE_END = pd.Timestamp("2026-08-07")
KNOWN_CLOSURE_REASON = "STORE_RENOVATION"


def header(title: str) -> None:
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-convert Bic Tsubame Takasaki Ana-Slo "
            "source HTML files to validated daily CSV files."
        )
    )
    parser.add_argument(
        "--min-machines",
        type=int,
        default=DEFAULT_MIN_MACHINES,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--only-date",
        type=lambda value: pd.Timestamp(value).normalize(),
        help=(
            "Convert only one YYYY-MM-DD source file. "
            "This mode is intended for daily catch-up and "
            "fails non-zero on validation/error."
        ),
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return path.read_text(encoding=encoding)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not read HTML text: {path}\n{last_error}"
    )


def detect_date_from_name(path: Path) -> pd.Timestamp:
    match = re.fullmatch(
        r"ana_slo_bic_tsubame_takasaki_(\d{8})_source\.html",
        path.name,
        re.IGNORECASE,
    )
    if not match:
        raise RuntimeError(
            f"Unexpected source filename: {path.name}"
        )

    return pd.Timestamp(
        pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="raise",
        )
    ).normalize()


def is_known_closure(page_date: pd.Timestamp) -> bool:
    return (
        KNOWN_CLOSURE_START
        <= page_date
        <= KNOWN_CLOSURE_END
    )


def validate_store_and_date(
    text: str,
    page_date: pd.Timestamp,
) -> str:
    store_name = None

    for name in STORE_NAMES:
        if name in text:
            store_name = name
            break

    if store_name is None:
        raise RuntimeError(
            "Store name validation failed."
        )

    date_text = page_date.strftime("%Y/%m/%d")
    if date_text not in text:
        raise RuntimeError(
            f"Page date validation failed: {date_text}"
        )

    return store_name


def find_main_table(
    html_text: str,
) -> pd.DataFrame:
    try:
        tables = pd.read_html(
            StringIO(html_text)
        )
    except ValueError as exc:
        raise RuntimeError(
            "No HTML tables found."
        ) from exc

    required_variants = (
        {"機種名", "台番号", "G数", "差枚"},
        {"機種名", "台番号", "総回転数", "差枚"},
    )

    candidates = []

    for index, table in enumerate(tables):
        columns = {
            str(column).strip()
            for column in table.columns
        }

        if any(
            required.issubset(columns)
            for required in required_variants
        ):
            candidates.append(
                (len(table), index, table.copy())
            )

    if not candidates:
        raise RuntimeError(
            "Main machine table not found."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][2]


def clean_table(
    df: pd.DataFrame,
    page_date: pd.Timestamp,
) -> pd.DataFrame:
    x = df.copy()
    x.columns = [
        str(column).strip()
        for column in x.columns
    ]

    rename = {
        "機種名": "machine_name",
        "台番号": "machine_no",
        "G数": "G",
        "総回転数": "G",
        "差枚": "diff",
        "BB": "BB",
        "RB": "RB",
        "ART": "ART",
        "合成確率": "combined_prob",
        "BB確率": "BB_prob",
        "RB確率": "RB_prob",
        "ART確率": "ART_prob",
    }

    x = x.rename(
        columns={
            key: value
            for key, value in rename.items()
            if key in x.columns
        }
    )

    required = {
        "machine_name",
        "machine_no",
        "G",
        "diff",
    }

    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(
            "Required columns are missing: "
            f"{sorted(missing)}"
        )

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="coerce",
    )

    for column in ("G", "diff", "BB", "RB", "ART"):
        if column not in x.columns:
            continue

        x[column] = (
            x[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.strip()
        )

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    x["date"] = page_date.date()

    first_columns = [
        "date",
        "machine_name",
        "machine_no",
        "G",
        "diff",
    ]

    rest = [
        column
        for column in x.columns
        if column not in first_columns
    ]

    return x[first_columns + rest].copy()


def validate_daily(
    df: pd.DataFrame,
    min_machines: int,
) -> dict:
    records = len(df)

    machine_no = pd.to_numeric(
        df["machine_no"],
        errors="coerce",
    )

    unique_machines = int(
        machine_no.nunique(dropna=True)
    )

    duplicate_rows = int(
        machine_no.duplicated(
            keep=False
        ).sum()
    )

    missing_machine = int(
        machine_no.isna().sum()
    )

    names = (
        df["machine_name"]
        .astype(str)
        .str.strip()
    )

    missing_name = int(
        names.isin(
            ["", "nan", "None"]
        ).sum()
    )

    diff_num = pd.to_numeric(
        df["diff"],
        errors="coerce",
    )

    g_num = pd.to_numeric(
        df["G"],
        errors="coerce",
    )

    invalid_diff = int(
        diff_num.isna().sum()
    )

    invalid_g = int(
        g_num.isna().sum()
    )

    negative_g = int(
        ((g_num < 0).fillna(False)).sum()
    )

    machine_count_ok = (
        records >= min_machines
    )

    unique_ok = (
        unique_machines == records
        and duplicate_rows == 0
        and missing_machine == 0
    )

    data_ok = (
        missing_name == 0
        and invalid_diff == 0
        and invalid_g == 0
        and negative_g == 0
    )

    return {
        "ok": (
            machine_count_ok
            and unique_ok
            and data_ok
        ),
        "records": records,
        "unique_machines": unique_machines,
        "duplicate_rows": duplicate_rows,
        "missing_machine": missing_machine,
        "missing_name": missing_name,
        "invalid_diff": invalid_diff,
        "invalid_G": invalid_g,
        "negative_G": negative_g,
        "diff_min": (
            diff_num.min()
            if len(diff_num)
            else None
        ),
        "diff_max": (
            diff_num.max()
            if len(diff_num)
            else None
        ),
        "G_min": (
            g_num.min()
            if len(g_num)
            else None
        ),
        "G_max": (
            g_num.max()
            if len(g_num)
            else None
        ),
    }


def validate_written_daily(
    path: Path,
    page_date: pd.Timestamp,
    min_machines: int,
) -> None:
    written = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    quality = validate_daily(
        written,
        min_machines,
    )

    written_dates = (
        pd.to_datetime(
            written["date"],
            errors="raise",
        )
        .dt.date
        .unique()
        .tolist()
    )

    if not quality["ok"]:
        raise RuntimeError(
            "Written CSV quality re-validation failed."
        )

    if written_dates != [page_date.date()]:
        raise RuntimeError(
            "Written CSV date re-validation failed: "
            f"{written_dates}"
        )


def write_daily_atomic(
    daily: pd.DataFrame,
    output_path: Path,
    page_date: pd.Timestamp,
    min_machines: int,
    *,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}_",
        suffix=".tmp",
        dir=output_path.parent,
    )

    os.close(fd)
    temporary_path = Path(temporary_name)

    try:
        daily.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
        )

        validate_written_daily(
            temporary_path,
            page_date,
            min_machines,
        )

        if (
            output_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                "Daily CSV appeared during conversion: "
                f"{output_path}"
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def make_summary_row(
    *,
    page_date,
    source_file: str,
    status: str,
    quality: dict | None = None,
    output_file: str = "",
    error: str = "",
) -> dict:
    quality = quality or {}

    return {
        "date": (
            page_date.date()
            if isinstance(page_date, pd.Timestamp)
            else page_date
        ),
        "source_file": source_file,
        "status": status,
        "records": quality.get("records", ""),
        "unique_machines": quality.get(
            "unique_machines",
            "",
        ),
        "duplicate_rows": quality.get(
            "duplicate_rows",
            "",
        ),
        "missing_machine": quality.get(
            "missing_machine",
            "",
        ),
        "missing_name": quality.get(
            "missing_name",
            "",
        ),
        "invalid_diff": quality.get(
            "invalid_diff",
            "",
        ),
        "invalid_G": quality.get(
            "invalid_G",
            "",
        ),
        "negative_G": quality.get(
            "negative_G",
            "",
        ),
        "diff_min": quality.get(
            "diff_min",
            "",
        ),
        "diff_max": quality.get(
            "diff_max",
            "",
        ),
        "G_min": quality.get(
            "G_min",
            "",
        ),
        "G_max": quality.get(
            "G_max",
            "",
        ),
        "output_file": output_file,
        "error": error,
    }


def main() -> None:
    args = parse_args()

    if args.min_machines < 1:
        raise ValueError(
            "--min-machines must be >= 1"
        )

    header(
        "Bic Tsubame Takasaki - "
        "Batch HTML -> Daily CSV"
    )

    if args.only_date is not None:
        if is_known_closure(args.only_date):
            raise RuntimeError(
                "Requested --only-date is a known "
                "store closure: "
                f"{args.only_date.date()} "
                f"({KNOWN_CLOSURE_REASON}). "
                "Do not treat this date as a "
                "fetch/freshness failure."
            )

        only_source = PROJECT_ROOT / (
            "ana_slo_bic_tsubame_takasaki_"
            f"{args.only_date:%Y%m%d}_source.html"
        )

        sources = (
            [only_source]
            if only_source.is_file()
            else []
        )

    else:
        sources = sorted(
            PROJECT_ROOT.glob(
                SOURCE_GLOB
            )
        )

    if not sources:
        raise FileNotFoundError(
            "No Bic Tsubame Takasaki "
            "source HTML files found."
        )

    only_date_text = (
        str(args.only_date.date())
        if args.only_date is not None
        else "-"
    )

    print(
        f"source files found    : "
        f"{len(sources)}"
    )
    print(
        f"min machines          : "
        f"{args.min_machines}"
    )
    print(
        f"overwrite             : "
        f"{args.overwrite}"
    )
    print(
        f"only date             : "
        f"{only_date_text}"
    )
    print(
        "known closure         : "
        f"{KNOWN_CLOSURE_START.date()} "
        "to "
        f"{KNOWN_CLOSURE_END.date()} "
        f"({KNOWN_CLOSURE_REASON})"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    for index, source in enumerate(
        sources,
        start=1,
    ):
        header(
            f"[{index}/{len(sources)}] "
            f"{source.name}"
        )

        page_date = ""

        try:
            page_date = detect_date_from_name(
                source
            )

            if is_known_closure(page_date):
                raise RuntimeError(
                    "Source HTML unexpectedly exists "
                    "inside known closure period: "
                    f"{page_date.date()}"
                )

            output_path = (
                OUTPUT_DIR
                / (
                    "ana_slo_bic_tsubame_takasaki_"
                    f"{page_date:%Y%m%d}.csv"
                )
            )

            if (
                output_path.exists()
                and not args.overwrite
            ):
                print(
                    "existing CSV          : "
                    f"{output_path}"
                )
                print(
                    "RESULT                : "
                    "SKIPPED_EXISTING"
                )

                summary_rows.append(
                    make_summary_row(
                        page_date=page_date,
                        source_file=source.name,
                        status="SKIPPED_EXISTING",
                        output_file=str(
                            output_path
                        ),
                    )
                )
                continue

            text = read_text(source)

            store_name = (
                validate_store_and_date(
                    text,
                    page_date,
                )
            )

            table = find_main_table(text)

            daily = clean_table(
                table,
                page_date,
            )

            quality = validate_daily(
                daily,
                args.min_machines,
            )

            print(
                f"page date             : "
                f"{page_date.date()}"
            )
            print(
                f"store name            : "
                f"{store_name}"
            )
            print(
                f"records               : "
                f"{quality['records']}"
            )
            print(
                f"unique machines       : "
                f"{quality['unique_machines']}"
            )
            print(
                f"duplicate rows        : "
                f"{quality['duplicate_rows']}"
            )
            print(
                f"missing machine       : "
                f"{quality['missing_machine']}"
            )
            print(
                f"missing name          : "
                f"{quality['missing_name']}"
            )
            print(
                f"invalid diff          : "
                f"{quality['invalid_diff']}"
            )
            print(
                f"invalid G             : "
                f"{quality['invalid_G']}"
            )
            print(
                f"negative G            : "
                f"{quality['negative_G']}"
            )
            print(
                f"diff min/max          : "
                f"{quality['diff_min']} / "
                f"{quality['diff_max']}"
            )
            print(
                f"G min/max             : "
                f"{quality['G_min']} / "
                f"{quality['G_max']}"
            )

            if not quality["ok"]:
                print(
                    "RESULT                : "
                    "FAILED_VALIDATION"
                )

                summary_rows.append(
                    make_summary_row(
                        page_date=page_date,
                        source_file=source.name,
                        status="FAILED_VALIDATION",
                        quality=quality,
                        error=(
                            "daily data quality failed"
                        ),
                    )
                )
                continue

            daily["machine_no"] = (
                pd.to_numeric(
                    daily["machine_no"],
                    errors="raise",
                )
                .astype(int)
            )

            write_daily_atomic(
                daily,
                output_path,
                page_date,
                args.min_machines,
                overwrite=args.overwrite,
            )

            print(
                f"saved                 : "
                f"{output_path}"
            )
            print(
                "RESULT                : OK"
            )

            summary_rows.append(
                make_summary_row(
                    page_date=page_date,
                    source_file=source.name,
                    status="OK",
                    quality=quality,
                    output_file=str(
                        output_path
                    ),
                )
            )

        except Exception as exc:
            print(
                f"ERROR                 : "
                f"{exc}"
            )

            summary_rows.append(
                make_summary_row(
                    page_date=page_date,
                    source_file=source.name,
                    status="ERROR",
                    error=repr(exc),
                )
            )

    summary = pd.DataFrame(summary_rows)

    stamp = (
        pd.Timestamp.now()
        .strftime("%Y%m%d_%H%M%S")
    )

    summary_path = (
        SUMMARY_DIR
        / (
            "bic_tsubame_takasaki_"
            "batch_convert_"
            f"{stamp}.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    counts = (
        summary["status"]
        .value_counts()
        .to_dict()
    )

    header("SUMMARY")

    print(
        f"processed             : "
        f"{len(summary)}"
    )

    for status, count in sorted(
        counts.items()
    ):
        print(
            f"{status:<22}: {count}"
        )

    print(
        f"summary log           : "
        f"{summary_path}"
    )

    print()
    print(
        "Batch conversion complete."
    )
    print(
        "Known closure dates are not "
        "treated as missing data."
    )
    print(
        "Historical machine-count changes "
        "are allowed if quality checks pass."
    )
    print(
        "No other store CSV files were modified."
    )

    statuses = set(
        summary["status"].astype(str)
    )

    bad_statuses = statuses - {
        "OK",
        "SKIPPED_EXISTING",
    }

    if bad_statuses:
        raise RuntimeError(
            "Bic Tsubame Takasaki batch "
            "conversion failed: "
            + ", ".join(
                sorted(bad_statuses)
            )
        )


if __name__ == "__main__":
    main()
