from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# SlotAnalyzer - Min-Repo Feature Extractor
#
# Input:
#   C:\Users\user\Desktop\Documents\SlotAnalyzer\
#       minrepo_YYYYMMDD_allmachines.html
#
# Optional input:
#   minrepo_YYYYMMDD_full.html
#   minrepo_YYYYMMDD_source.html
#
# Output:
#   data\maruhan_maebashi\external_features\minrepo\
#
# Purpose:
#   Extract Min-Repo data into stable CSVs WITHOUT changing
#   the existing V4.2_C model or existing Ana-Slo source data.
# ============================================================


ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

OUT_DIR = (
    ROOT
    / "data"
    / "maruhan_maebashi"
    / "external_features"
    / "minrepo"
)

ALL_MACHINE_PATTERN = re.compile(
    r"^minrepo_(\d{8})_allmachines\.html$",
    re.IGNORECASE,
)

AUX_PATTERNS = (
    "minrepo_{date}_full.html",
    "minrepo_{date}_source.html",
)


def parse_number(value) -> float:
    if pd.isna(value):
        return np.nan

    text = (
        str(value)
        .strip()
        .replace(",", "")
        .replace("+", "")
        .replace("−", "-")
        .replace("－", "-")
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return np.nan

    return float(match.group(0))


def parse_percent(value) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if not match:
        return np.nan

    return float(match.group(0))


def parse_win_rate(value) -> tuple[float, float, float]:
    """
    Examples:
      217/514 -> wins=217, total=514, pct=42.218...
      3/5     -> wins=3, total=5, pct=60
    """
    if pd.isna(value):
        return np.nan, np.nan, np.nan

    text = str(value).strip()

    match = re.search(
        r"(\d+)\s*/\s*(\d+)",
        text,
    )

    if not match:
        return np.nan, np.nan, np.nan

    wins = float(match.group(1))
    total = float(match.group(2))

    pct = (
        wins / total * 100.0
        if total > 0
        else np.nan
    )

    return wins, total, pct


def read_tables(path: Path) -> list[pd.DataFrame]:
    try:
        return pd.read_html(path)
    except Exception as exc:
        raise RuntimeError(
            f"HTML table read failed: {path}\n{exc}"
        ) from exc


def has_columns(
    table: pd.DataFrame,
    required: Iterable[str],
) -> bool:
    cols = {
        str(col).strip()
        for col in table.columns
    }

    return all(
        item in cols
        for item in required
    )


def find_machine_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame | None:
    candidates = [
        table
        for table in tables
        if has_columns(
            table,
            ("機種", "台番", "差枚", "G数"),
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=len,
    ).copy()


def find_machine_summary_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame | None:
    candidates = [
        table
        for table in tables
        if has_columns(
            table,
            ("機種", "平均差枚", "平均G数", "勝率", "出率"),
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=len,
    ).copy()


def find_tail_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame | None:
    candidates = [
        table
        for table in tables
        if has_columns(
            table,
            ("末尾", "平均差枚", "平均G数", "勝率", "出率"),
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=len,
    ).copy()


def clean_machine_table(
    raw: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    df = raw[
        ["機種", "台番", "差枚", "G数", "出率"]
    ].copy()

    df.columns = [
        "machine_name",
        "machine_no",
        "diff",
        "games",
        "payout_rate",
    ]

    df["machine_no"] = df["machine_no"].map(parse_number)
    df["diff"] = df["diff"].map(parse_number)
    df["games"] = df["games"].map(parse_number)
    df["payout_rate"] = df["payout_rate"].map(parse_percent)

    # Repeated headers/non-machine rows become NaN here.
    df = df.dropna(
        subset=["machine_no"]
    ).copy()

    df["machine_no"] = df["machine_no"].astype(int)
    df.insert(0, "date", date)

    return df


def clean_machine_summary(
    raw: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    df = raw[
        ["機種", "平均差枚", "平均G数", "勝率", "出率"]
    ].copy()

    df.columns = [
        "machine_name",
        "avg_diff",
        "avg_games",
        "win_rate_raw",
        "payout_rate",
    ]

    df["avg_diff"] = df["avg_diff"].map(parse_number)
    df["avg_games"] = df["avg_games"].map(parse_number)
    df["payout_rate"] = df["payout_rate"].map(parse_percent)

    parsed = df["win_rate_raw"].map(parse_win_rate)

    df["wins"] = [
        item[0]
        for item in parsed
    ]

    df["machine_count"] = [
        item[1]
        for item in parsed
    ]

    df["win_rate_percent"] = [
        item[2]
        for item in parsed
    ]

    df.insert(0, "date", date)

    return df[
        [
            "date",
            "machine_name",
            "avg_diff",
            "avg_games",
            "wins",
            "machine_count",
            "win_rate_percent",
            "payout_rate",
            "win_rate_raw",
        ]
    ]


def clean_tail_table(
    raw: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    df = raw[
        ["末尾", "平均差枚", "平均G数", "勝率", "出率"]
    ].copy()

    df.columns = [
        "tail",
        "avg_diff",
        "avg_games",
        "win_rate_raw",
        "payout_rate",
    ]

    df["tail"] = df["tail"].map(parse_number)
    df["avg_diff"] = df["avg_diff"].map(parse_number)
    df["avg_games"] = df["avg_games"].map(parse_number)
    df["payout_rate"] = df["payout_rate"].map(parse_percent)

    parsed = df["win_rate_raw"].map(parse_win_rate)

    df["wins"] = [
        item[0]
        for item in parsed
    ]

    df["machine_count"] = [
        item[1]
        for item in parsed
    ]

    df["win_rate_percent"] = [
        item[2]
        for item in parsed
    ]

    df = df.dropna(
        subset=["tail"]
    ).copy()

    df["tail"] = df["tail"].astype(int)
    df.insert(0, "date", date)

    return df[
        [
            "date",
            "tail",
            "avg_diff",
            "avg_games",
            "wins",
            "machine_count",
            "win_rate_percent",
            "payout_rate",
            "win_rate_raw",
        ]
    ]


def derive_store_summary(
    machine_df: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    """
    Derived strictly from the all-machine table.
    This avoids depending on prose/header parsing.
    """
    count = int(
        machine_df["machine_no"].nunique()
    )

    total_diff = float(
        machine_df["diff"].sum()
    )

    avg_diff = float(
        machine_df["diff"].mean()
    )

    avg_games = float(
        machine_df["games"].mean()
    )

    wins = int(
        (machine_df["diff"] > 0).sum()
    )

    win_rate = (
        wins / count * 100.0
        if count
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "date": date,
                "machine_count": count,
                "total_diff": total_diff,
                "avg_diff": avg_diff,
                "avg_games": avg_games,
                "positive_diff_machines": wins,
                "positive_diff_rate_percent": win_rate,
                "source": "derived_from_minrepo_allmachines",
            }
        ]
    )


def load_aux_tables(
    date: str,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    list[str],
]:
    machine_summary = None
    tail_summary = None
    used_files: list[str] = []

    for template in AUX_PATTERNS:
        path = ROOT / template.format(date=date)

        if not path.exists():
            continue

        tables = read_tables(path)
        used_files.append(path.name)

        if machine_summary is None:
            raw = find_machine_summary_table(tables)
            if raw is not None:
                machine_summary = clean_machine_summary(
                    raw,
                    date,
                )

        if tail_summary is None:
            raw = find_tail_table(tables)
            if raw is not None:
                tail_summary = clean_tail_table(
                    raw,
                    date,
                )

    return (
        machine_summary,
        tail_summary,
        used_files,
    )


def main() -> None:
    files = []

    for path in ROOT.glob(
        "minrepo_*_allmachines.html"
    ):
        match = ALL_MACHINE_PATTERN.match(
            path.name
        )

        if match:
            files.append(
                (
                    match.group(1),
                    path,
                )
            )

    files.sort()

    if not files:
        raise SystemExit(
            "minrepo_YYYYMMDD_allmachines.html が見つかりません。"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_machine_rows = []
    all_store_rows = []
    all_machine_summary_rows = []
    all_tail_rows = []
    diagnostics = []

    print("=" * 104)
    print("SlotAnalyzer - Min-Repo Feature Extractor")
    print("=" * 104)

    for date, path in files:
        print()
        print("-" * 104)
        print(f"DATE: {date}")
        print("-" * 104)

        tables = read_tables(path)
        raw_machine = find_machine_table(tables)

        if raw_machine is None:
            print(
                f"[SKIP] 全台データ表なし: {path.name}"
            )
            continue

        machine_df = clean_machine_table(
            raw_machine,
            date,
        )

        duplicate_count = int(
            machine_df.duplicated(
                subset=["machine_no"]
            ).sum()
        )

        store_df = derive_store_summary(
            machine_df,
            date,
        )

        (
            machine_summary_df,
            tail_df,
            aux_files,
        ) = load_aux_tables(date)

        # Some all-machine pages may themselves contain these tables.
        if machine_summary_df is None:
            raw = find_machine_summary_table(
                tables
            )
            if raw is not None:
                machine_summary_df = clean_machine_summary(
                    raw,
                    date,
                )

        if tail_df is None:
            raw = find_tail_table(
                tables
            )
            if raw is not None:
                tail_df = clean_tail_table(
                    raw,
                    date,
                )

        print(
            f"all-machine raw rows       : {len(raw_machine)}"
        )
        print(
            f"valid machine rows         : {len(machine_df)}"
        )
        print(
            f"unique machine numbers     : {machine_df['machine_no'].nunique()}"
        )
        print(
            f"duplicate machine numbers  : {duplicate_count}"
        )
        print(
            f"store total diff           : {store_df.iloc[0]['total_diff']:+,.0f}"
        )
        print(
            f"store avg diff             : {store_df.iloc[0]['avg_diff']:+,.2f}"
        )
        print(
            f"positive diff rate         : {store_df.iloc[0]['positive_diff_rate_percent']:.2f}%"
        )

        if machine_summary_df is not None:
            print(
                f"machine summary rows       : {len(machine_summary_df)}"
            )
        else:
            print(
                "machine summary rows       : NOT AVAILABLE"
            )

        if tail_df is not None:
            print(
                f"tail summary rows          : {len(tail_df)}"
            )
        else:
            print(
                "tail summary rows          : NOT AVAILABLE"
            )

        if aux_files:
            print(
                f"auxiliary HTML             : {', '.join(aux_files)}"
            )
        else:
            print(
                "auxiliary HTML             : none"
            )

        date_dir = OUT_DIR / date
        date_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        machine_df.to_csv(
            date_dir / "machines.csv",
            index=False,
            encoding="utf-8-sig",
        )

        store_df.to_csv(
            date_dir / "store_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if machine_summary_df is not None:
            machine_summary_df.to_csv(
                date_dir / "machine_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

        if tail_df is not None:
            tail_df.to_csv(
                date_dir / "tail_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

        all_machine_rows.append(
            machine_df
        )

        all_store_rows.append(
            store_df
        )

        if machine_summary_df is not None:
            all_machine_summary_rows.append(
                machine_summary_df
            )

        if tail_df is not None:
            all_tail_rows.append(
                tail_df
            )

        diagnostics.append(
            {
                "date": date,
                "allmachines_file": path.name,
                "raw_machine_rows": len(raw_machine),
                "valid_machine_rows": len(machine_df),
                "unique_machine_numbers": machine_df["machine_no"].nunique(),
                "duplicate_machine_numbers": duplicate_count,
                "machine_summary_available": machine_summary_df is not None,
                "tail_summary_available": tail_df is not None,
                "auxiliary_files": "|".join(aux_files),
            }
        )

    if not all_machine_rows:
        raise SystemExit(
            "有効な全台データを抽出できませんでした。"
        )

    machines_all = pd.concat(
        all_machine_rows,
        ignore_index=True,
    )

    stores_all = pd.concat(
        all_store_rows,
        ignore_index=True,
    )

    machines_all.to_csv(
        OUT_DIR / "machines_all_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stores_all.to_csv(
        OUT_DIR / "store_summary_all_dates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if all_machine_summary_rows:
        pd.concat(
            all_machine_summary_rows,
            ignore_index=True,
        ).to_csv(
            OUT_DIR / "machine_summary_all_dates.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if all_tail_rows:
        pd.concat(
            all_tail_rows,
            ignore_index=True,
        ).to_csv(
            OUT_DIR / "tail_summary_all_dates.csv",
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(
        diagnostics
    ).to_csv(
        OUT_DIR / "extraction_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 104)
    print("SUMMARY")
    print("=" * 104)
    print(
        f"dates extracted            : {stores_all['date'].nunique()}"
    )
    print(
        f"machine rows total         : {len(machines_all)}"
    )
    print(
        f"machine-summary dates      : "
        f"{pd.concat(all_machine_summary_rows)['date'].nunique() if all_machine_summary_rows else 0}"
    )
    print(
        f"tail-summary dates         : "
        f"{pd.concat(all_tail_rows)['date'].nunique() if all_tail_rows else 0}"
    )
    print()
    print(
        stores_all.to_string(
            index=False,
        )
    )
    print()
    print(f"saved: {OUT_DIR}")
    print()
    print(
        "NOTE: Existing Ana-Slo data and V4.2_C files were not modified."
    )


if __name__ == "__main__":
    main()
