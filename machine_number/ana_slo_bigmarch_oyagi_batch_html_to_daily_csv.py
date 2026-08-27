from __future__ import annotations

from pathlib import Path
from io import StringIO
import argparse
import re
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
SOURCE_GLOB = "ana_slo_bigmarch_oyagi_????????_source.html"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
)

SUMMARY_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "batch_logs"
)

STORE_NAMES = (
    "ビックマーチ高崎おおやぎ店",
    "ビッグマーチ高崎おおやぎ店",
)

DEFAULT_MIN_MACHINES = 200


def header(title: str) -> None:
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-convert Big March Takasaki Oyagi Ana-Slo source HTML files to daily CSV."
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
    return parser.parse_args()


def read_text(path: Path) -> str:
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return path.read_text(encoding=enc)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read HTML text: {path}\n{last_error}")


def detect_date_from_name(path: Path) -> pd.Timestamp:
    m = re.fullmatch(
        r"ana_slo_bigmarch_oyagi_(\d{8})_source\.html",
        path.name,
        re.IGNORECASE,
    )
    if not m:
        raise RuntimeError(f"Unexpected source filename: {path.name}")
    return pd.Timestamp(
        pd.to_datetime(m.group(1), format="%Y%m%d", errors="raise")
    ).normalize()


def validate_store_and_date(text: str, page_date: pd.Timestamp) -> str:
    store_name = None
    for name in STORE_NAMES:
        if name in text:
            store_name = name
            break
    if store_name is None:
        raise RuntimeError("Store name validation failed.")

    date_text = page_date.strftime("%Y/%m/%d")
    if date_text not in text:
        raise RuntimeError(f"Page date validation failed: {date_text}")

    return store_name


def find_main_table(html_text: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html_text))
    required = {"機種名", "台番号", "G数", "差枚"}
    candidates = []

    for idx, table in enumerate(tables):
        cols = {str(c).strip() for c in table.columns}
        if required.issubset(cols):
            candidates.append((len(table), idx, table.copy()))

    if not candidates:
        raise RuntimeError("Main machine table not found.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


def clean_table(df: pd.DataFrame, page_date: pd.Timestamp) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]

    rename = {
        "機種名": "machine_name",
        "台番号": "machine_no",
        "G数": "G",
        "差枚": "diff",
        "BB": "BB",
        "RB": "RB",
        "ART": "ART",
        "合成確率": "combined_prob",
        "BB確率": "BB_prob",
        "RB確率": "RB_prob",
        "ART確率": "ART_prob",
    }
    x = x.rename(columns={k: v for k, v in rename.items() if k in x.columns})

    x["machine_name"] = x["machine_name"].astype(str).str.strip()
    x["machine_no"] = pd.to_numeric(x["machine_no"], errors="coerce")

    for col in ("G", "diff", "BB", "RB", "ART"):
        if col in x.columns:
            x[col] = (
                x[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("+", "", regex=False)
                .str.strip()
            )
            x[col] = pd.to_numeric(x[col], errors="coerce")

    x["date"] = page_date.date()

    first_cols = ["date", "machine_name", "machine_no", "G", "diff"]
    rest = [c for c in x.columns if c not in first_cols]
    return x[first_cols + rest].copy()


def validate_daily(df: pd.DataFrame, min_machines: int) -> dict:
    records = len(df)
    machine_no = pd.to_numeric(df["machine_no"], errors="coerce")
    unique_machines = machine_no.nunique(dropna=True)
    duplicate_rows = int(machine_no.duplicated(keep=False).sum())
    missing_machine = int(machine_no.isna().sum())

    names = df["machine_name"].astype(str).str.strip()
    missing_name = int(names.isin(["", "nan", "None"]).sum())

    diff_num = pd.to_numeric(df["diff"], errors="coerce")
    g_num = pd.to_numeric(df["G"], errors="coerce")

    invalid_diff = int(diff_num.isna().sum())
    invalid_g = int(g_num.isna().sum())
    negative_g = int(((g_num < 0).fillna(False)).sum())

    machine_count_ok = records >= min_machines
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
        "ok": machine_count_ok and unique_ok and data_ok,
        "records": records,
        "unique_machines": unique_machines,
        "duplicate_rows": duplicate_rows,
        "missing_machine": missing_machine,
        "missing_name": missing_name,
        "invalid_diff": invalid_diff,
        "invalid_G": invalid_g,
        "negative_G": negative_g,
        "diff_min": diff_num.min() if len(diff_num) else None,
        "diff_max": diff_num.max() if len(diff_num) else None,
        "G_min": g_num.min() if len(g_num) else None,
        "G_max": g_num.max() if len(g_num) else None,
    }


def main() -> None:
    args = parse_args()

    if args.min_machines < 1:
        raise ValueError("--min-machines must be >= 1")

    header("Big March Takasaki Oyagi - Batch HTML -> Daily CSV")

    sources = sorted(PROJECT_ROOT.glob(SOURCE_GLOB))

    if not sources:
        raise FileNotFoundError("No Big March Oyagi source HTML files found.")

    print(f"source files found    : {len(sources)}")
    print(f"min machines          : {args.min_machines}")
    print(f"overwrite             : {args.overwrite}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for idx, source in enumerate(sources, start=1):
        header(f"[{idx}/{len(sources)}] {source.name}")

        try:
            page_date = detect_date_from_name(source)
            out = OUTPUT_DIR / f"ana_slo_bigmarch_oyagi_{page_date.strftime('%Y%m%d')}.csv"

            if out.exists() and not args.overwrite:
                print(f"existing CSV          : {out}")
                print("RESULT                : SKIPPED_EXISTING")

                summary_rows.append({
                    "date": page_date.date(),
                    "source_file": source.name,
                    "status": "SKIPPED_EXISTING",
                    "records": "",
                    "unique_machines": "",
                    "duplicate_rows": "",
                    "missing_machine": "",
                    "missing_name": "",
                    "invalid_diff": "",
                    "invalid_G": "",
                    "negative_G": "",
                    "diff_min": "",
                    "diff_max": "",
                    "G_min": "",
                    "G_max": "",
                    "output_file": str(out),
                    "error": "",
                })
                continue

            text = read_text(source)
            store_name = validate_store_and_date(text, page_date)
            table = find_main_table(text)
            daily = clean_table(table, page_date)
            quality = validate_daily(daily, args.min_machines)

            print(f"page date             : {page_date.date()}")
            print(f"store name            : {store_name}")
            print(f"records               : {quality['records']}")
            print(f"unique machines       : {quality['unique_machines']}")
            print(f"duplicate rows        : {quality['duplicate_rows']}")
            print(f"missing machine       : {quality['missing_machine']}")
            print(f"missing name          : {quality['missing_name']}")
            print(f"invalid diff          : {quality['invalid_diff']}")
            print(f"invalid G             : {quality['invalid_G']}")
            print(f"negative G            : {quality['negative_G']}")

            if not quality["ok"]:
                print("RESULT                : FAILED_VALIDATION")
                status = "FAILED_VALIDATION"
                output_file = ""
                error = "daily data quality failed"
            else:
                daily["machine_no"] = pd.to_numeric(
                    daily["machine_no"], errors="raise"
                ).astype(int)

                daily.to_csv(
                    out,
                    index=False,
                    encoding="utf-8-sig",
                )
                print(f"saved                 : {out}")
                print("RESULT                : OK")

                status = "OK"
                output_file = str(out)
                error = ""

            summary_rows.append({
                "date": page_date.date(),
                "source_file": source.name,
                "status": status,
                "records": quality["records"],
                "unique_machines": quality["unique_machines"],
                "duplicate_rows": quality["duplicate_rows"],
                "missing_machine": quality["missing_machine"],
                "missing_name": quality["missing_name"],
                "invalid_diff": quality["invalid_diff"],
                "invalid_G": quality["invalid_G"],
                "negative_G": quality["negative_G"],
                "diff_min": quality["diff_min"],
                "diff_max": quality["diff_max"],
                "G_min": quality["G_min"],
                "G_max": quality["G_max"],
                "output_file": output_file,
                "error": error,
            })

        except Exception as exc:
            print(f"ERROR                 : {exc}")

            summary_rows.append({
                "date": "",
                "source_file": source.name,
                "status": "ERROR",
                "records": "",
                "unique_machines": "",
                "duplicate_rows": "",
                "missing_machine": "",
                "missing_name": "",
                "invalid_diff": "",
                "invalid_G": "",
                "negative_G": "",
                "diff_min": "",
                "diff_max": "",
                "G_min": "",
                "G_max": "",
                "output_file": "",
                "error": repr(exc),
            })

    summary = pd.DataFrame(summary_rows)

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_path = SUMMARY_DIR / f"bigmarch_oyagi_batch_convert_{stamp}.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    counts = summary["status"].value_counts().to_dict()

    header("SUMMARY")
    print(f"processed             : {len(summary)}")

    for status, count in sorted(counts.items()):
        print(f"{status:<22}: {count}")

    print(f"summary log           : {summary_path}")
    print()
    print("Batch conversion complete.")
    print("Historical machine-count changes are allowed.")
    print("No Maruhan Maebashi files were modified.")


if __name__ == "__main__":
    main()
