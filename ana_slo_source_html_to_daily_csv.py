from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

SOURCE_HTML = PROJECT_ROOT / "ana_slo_20260819_source.html"
TARGET_DATE = "2026-08-19"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

OUTPUT_CSV = OUTPUT_DIR / "ana_slo_20260819.csv"

EXPECTED_MACHINES = 514

EXPECTED_COLUMNS = [
    "日付",
    "台番号",
    "機種名",
    "G数",
    "差枚",
    "BB",
    "RB",
    "合成確率",
    "BB確率",
    "RB確率",
]


def header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def clean_text(value: str) -> str:
    return (
        str(value)
        .replace("\u3000", " ")
        .replace("\xa0", " ")
        .strip()
    )


def clean_number_text(value: str) -> str:
    return (
        clean_text(value)
        .replace(",", "")
        .replace("+", "")
    )


def parse_int(value: str):
    s = clean_number_text(value)

    if s in {"", "-", "－", "―", "—"}:
        return None

    m = re.search(r"-?\d+", s)

    if not m:
        return None

    return int(m.group(0))


def normalize_header(value: str) -> str:
    s = clean_text(value)

    replacements = {
        "ゲーム数": "G数",
        "総回転数": "G数",
        "回転数": "G数",
        "差枚数": "差枚",
        "BB回数": "BB",
        "RB回数": "RB",
    }

    return replacements.get(s, s)


def extract_page_date(
    soup: BeautifulSoup,
) -> str | None:

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else ""
    )

    match = re.search(
        r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})",
        title,
    )

    if not match:
        return None

    y, m, d = match.groups()

    return (
        f"{int(y):04d}-"
        f"{int(m):02d}-"
        f"{int(d):02d}"
    )


def get_main_table(
    soup: BeautifulSoup,
):
    table = soup.find(
        "table",
        id="all_data_table",
    )

    if table is not None:
        return table

    for candidate in soup.find_all("table"):
        text = candidate.get_text(" ", strip=True)

        required = [
            "台番号",
            "機種名",
            "G数",
            "差枚",
        ]

        if all(
            item in text
            for item in required
        ):
            return candidate

    return None


def extract_table_rows(
    table,
) -> tuple[
    list[str],
    list[list[str]],
]:

    rows = table.find_all("tr")

    if not rows:
        raise RuntimeError(
            "Table has no rows."
        )

    header_cells = rows[0].find_all(
        ["th", "td"]
    )

    headers = [
        normalize_header(
            cell.get_text(
                " ",
                strip=True,
            )
        )
        for cell in header_cells
    ]

    data_rows = []

    for tr in rows[1:]:
        cells = tr.find_all(
            ["td", "th"]
        )

        values = [
            clean_text(
                cell.get_text(
                    " ",
                    strip=True,
                )
            )
            for cell in cells
        ]

        if values:
            data_rows.append(values)

    return headers, data_rows


def locate_columns(
    headers: list[str],
) -> dict[str, int]:

    aliases = {
        "機種名": ["機種名"],
        "台番号": ["台番号"],
        "G数": ["G数", "ゲーム数", "総回転数", "回転数"],
        "差枚": ["差枚", "差枚数"],
        "BB": ["BB", "BB回数"],
        "RB": ["RB", "RB回数"],
        "合成確率": ["合成確率"],
        "BB確率": ["BB確率"],
        "RB確率": ["RB確率"],
    }

    result = {}

    for canonical, candidates in aliases.items():
        found = None

        for candidate in candidates:
            candidate = normalize_header(
                candidate
            )

            if candidate in headers:
                found = headers.index(
                    candidate
                )
                break

        if found is not None:
            result[canonical] = found

    mandatory = [
        "機種名",
        "台番号",
        "G数",
        "差枚",
    ]

    missing = [
        item
        for item in mandatory
        if item not in result
    ]

    if missing:
        raise RuntimeError(
            "Required table columns not found: "
            f"{missing}\n"
            f"Detected headers: {headers}"
        )

    return result


def safe_get(
    row: list[str],
    index_map: dict[str, int],
    name: str,
) -> str:

    idx = index_map.get(name)

    if idx is None:
        return ""

    if idx >= len(row):
        return ""

    return clean_text(row[idx])


def main() -> None:

    header(
        "Ana-Slo Saved Source HTML -> Daily CSV"
    )

    if not SOURCE_HTML.exists():
        raise FileNotFoundError(
            f"Source HTML not found: {SOURCE_HTML}"
        )

    raw = SOURCE_HTML.read_text(
        encoding="utf-8",
        errors="replace",
    )

    print(
        f"source_html          : {SOURCE_HTML}"
    )
    print(
        f"html_chars           : {len(raw):,}"
    )

    soup = BeautifulSoup(
        raw,
        "html.parser",
    )

    page_date = extract_page_date(
        soup
    )

    print(
        f"page_date            : {page_date}"
    )
    print(
        f"expected_date        : {TARGET_DATE}"
    )

    if page_date != TARGET_DATE:
        raise RuntimeError(
            "Page date mismatch."
        )

    table = get_main_table(
        soup
    )

    if table is None:
        raise RuntimeError(
            "all_data_table not found."
        )

    headers, data_rows = (
        extract_table_rows(
            table
        )
    )

    print(
        f"detected_headers     : {headers}"
    )
    print(
        f"raw_table_rows       : {len(data_rows):,}"
    )

    index_map = locate_columns(
        headers
    )

    records = []

    for row in data_rows:
        machine_no = parse_int(
            safe_get(
                row,
                index_map,
                "台番号",
            )
        )

        machine_name = safe_get(
            row,
            index_map,
            "機種名",
        )

        if (
            machine_no is None
            or not machine_name
        ):
            continue

        records.append(
            {
                "日付":
                    TARGET_DATE,

                "台番号":
                    machine_no,

                "機種名":
                    machine_name,

                "G数":
                    parse_int(
                        safe_get(
                            row,
                            index_map,
                            "G数",
                        )
                    ),

                "差枚":
                    parse_int(
                        safe_get(
                            row,
                            index_map,
                            "差枚",
                        )
                    ),

                "BB":
                    parse_int(
                        safe_get(
                            row,
                            index_map,
                            "BB",
                        )
                    ),

                "RB":
                    parse_int(
                        safe_get(
                            row,
                            index_map,
                            "RB",
                        )
                    ),

                "合成確率":
                    safe_get(
                        row,
                        index_map,
                        "合成確率",
                    ),

                "BB確率":
                    safe_get(
                        row,
                        index_map,
                        "BB確率",
                    ),

                "RB確率":
                    safe_get(
                        row,
                        index_map,
                        "RB確率",
                    ),
            }
        )

    df = pd.DataFrame(
        records,
        columns=EXPECTED_COLUMNS,
    )

    header(
        "QUALITY CHECK"
    )

    duplicates = int(
        df.duplicated(
            subset=[
                "日付",
                "台番号",
            ]
        ).sum()
    )

    missing_machine = int(
        df["台番号"].isna().sum()
    )

    missing_name = int(
        df["機種名"]
        .replace("", pd.NA)
        .isna()
        .sum()
    )

    invalid_diff = int(
        df["差枚"].isna().sum()
    )

    invalid_games = int(
        df["G数"].isna().sum()
    )

    negative_games = int(
        (
            pd.to_numeric(
                df["G数"],
                errors="coerce",
            )
            < 0
        ).sum()
    )

    print(
        f"records              : {len(df):,}"
    )
    print(
        f"unique machines      : {df['台番号'].nunique():,}"
    )
    print(
        f"duplicates           : {duplicates}"
    )
    print(
        f"missing machine      : {missing_machine}"
    )
    print(
        f"missing name         : {missing_name}"
    )
    print(
        f"invalid diff         : {invalid_diff}"
    )
    print(
        f"invalid G            : {invalid_games}"
    )
    print(
        f"negative G           : {negative_games}"
    )

    if not df.empty:
        print(
            f"diff min/max         : "
            f"{df['差枚'].min()} / "
            f"{df['差枚'].max()}"
        )

        print(
            f"G min/max            : "
            f"{df['G数'].min()} / "
            f"{df['G数'].max()}"
        )

    critical_errors = []

    if len(df) != EXPECTED_MACHINES:
        critical_errors.append(
            f"machine count {len(df)} != {EXPECTED_MACHINES}"
        )

    if duplicates != 0:
        critical_errors.append(
            f"duplicates={duplicates}"
        )

    if missing_machine != 0:
        critical_errors.append(
            f"missing_machine={missing_machine}"
        )

    if missing_name != 0:
        critical_errors.append(
            f"missing_name={missing_name}"
        )

    if invalid_diff != 0:
        critical_errors.append(
            f"invalid_diff={invalid_diff}"
        )

    if invalid_games != 0:
        critical_errors.append(
            f"invalid_G={invalid_games}"
        )

    if negative_games != 0:
        critical_errors.append(
            f"negative_G={negative_games}"
        )

    if critical_errors:
        print()
        print(
            "RESULT: QUALITY CHECK FAILED"
        )

        for error in critical_errors:
            print(
                f"  - {error}"
            )

        print()
        print(
            "CSV WILL NOT BE SAVED."
        )

        raise RuntimeError(
            "Quality check failed."
        )

    print()
    print(
        "RESULT: DAILY DATA QUALITY OK"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = (
        df.sort_values("台番号")
        .reset_index(drop=True)
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )

    header(
        "SAVED"
    )

    print(
        OUTPUT_CSV
    )
    print(
        f"saved records        : {len(df):,}"
    )

    print()
    print(
        "Next:"
    )
    print(
        "python .\\machine_number\\"
        "ana_slo_prediction_v4_2_forward_champion_challenger.py"
    )


if __name__ == "__main__":
    main()
