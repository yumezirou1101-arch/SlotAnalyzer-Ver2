from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")
DATA_DIR = PROJECT_ROOT / "data" / "session_results"
MASTER_CSV = DATA_DIR / "session_results.csv"
SUMMARY_DIR = DATA_DIR / "summary"
ENCODING = "utf-8-sig"

COLUMNS = [
    "date",
    "weekday",
    "store",
    "player",
    "draw_no_self",
    "draw_no_partner",
    "draw_no_used",
    "machine_no",
    "machine_name",
    "prediction_model",
    "prediction_rank",
    "investment_yen",
    "recovery_yen",
    "profit_yen",
    "roi_percent",
    "win_flag",
    "rainbow7_chain",
    "bell_focus",
    "memo",
    "created_at",
]

WEEKDAY_JP = {
    0: "月",
    1: "火",
    2: "水",
    3: "木",
    4: "金",
    5: "土",
    6: "日",
}


def header(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTER_CSV.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(
            MASTER_CSV,
            index=False,
            encoding=ENCODING,
        )


def load_master() -> pd.DataFrame:
    ensure_storage()

    df = pd.read_csv(
        MASTER_CSV,
        encoding=ENCODING,
    )

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[COLUMNS].copy()


def save_master(df: pd.DataFrame) -> None:
    ensure_storage()

    df[COLUMNS].to_csv(
        MASTER_CSV,
        index=False,
        encoding=ENCODING,
        quoting=csv.QUOTE_MINIMAL,
    )


def normalize_date(value: str) -> str:
    dt = pd.to_datetime(value, errors="raise")
    return dt.strftime("%Y-%m-%d")


def weekday_jp(date_str: str) -> str:
    dt = pd.to_datetime(date_str)
    return WEEKDAY_JP[int(dt.dayofweek)]


def to_int_or_none(value):
    if value is None:
        return None

    s = str(value).strip()

    if s == "":
        return None

    return int(float(s.replace(",", "")))


def build_record(
    *,
    date: str,
    store: str,
    player: str,
    draw_no_self=None,
    draw_no_partner=None,
    draw_no_used=None,
    machine_no=None,
    machine_name: str,
    prediction_model: str = "",
    prediction_rank=None,
    investment_yen=0,
    recovery_yen=0,
    rainbow7_chain=0,
    bell_focus: str = "",
    memo: str = "",
) -> dict:

    date = normalize_date(date)
    investment = int(investment_yen)
    recovery = int(recovery_yen)
    profit = recovery - investment

    roi = (
        recovery / investment * 100.0
        if investment > 0
        else pd.NA
    )

    return {
        "date": date,
        "weekday": weekday_jp(date),
        "store": str(store).strip(),
        "player": str(player).strip(),
        "draw_no_self": to_int_or_none(draw_no_self),
        "draw_no_partner": to_int_or_none(draw_no_partner),
        "draw_no_used": to_int_or_none(draw_no_used),
        "machine_no": to_int_or_none(machine_no),
        "machine_name": str(machine_name).strip(),
        "prediction_model": str(prediction_model).strip(),
        "prediction_rank": to_int_or_none(prediction_rank),
        "investment_yen": investment,
        "recovery_yen": recovery,
        "profit_yen": profit,
        "roi_percent": (
            round(float(roi), 2)
            if pd.notna(roi)
            else pd.NA
        ),
        "win_flag": int(profit > 0),
        "rainbow7_chain": int(rainbow7_chain),
        "bell_focus": str(bell_focus).strip(),
        "memo": str(memo).strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def duplicate_mask(
    df: pd.DataFrame,
    record: dict,
) -> pd.Series:

    if df.empty:
        return pd.Series([], dtype=bool)

    return (
        (df["date"].astype(str) == str(record["date"]))
        & (df["store"].astype(str) == str(record["store"]))
        & (df["player"].astype(str) == str(record["player"]))
        & (
            pd.to_numeric(
                df["machine_no"],
                errors="coerce",
            )
            == record["machine_no"]
        )
    )


def add_record(
    record: dict,
    replace: bool = False,
) -> None:

    df = load_master()
    mask = duplicate_mask(df, record)

    duplicate_count = int(mask.sum()) if len(mask) else 0

    if duplicate_count:
        if not replace:
            raise RuntimeError(
                "Duplicate session detected. "
                "Use --replace only when intentionally correcting the same session."
            )

        df = df.loc[~mask].copy()

    df = pd.concat(
        [
            df,
            pd.DataFrame([record]),
        ],
        ignore_index=True,
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = (
        df.sort_values(
            [
                "date",
                "store",
                "player",
                "machine_no",
            ]
        )
        .reset_index(drop=True)
    )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    save_master(df)


def numeric_series(
    df: pd.DataFrame,
    col: str,
) -> pd.Series:
    return pd.to_numeric(
        df[col],
        errors="coerce",
    )


def make_group_summary(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:

    work = df.copy()

    for col in [
        "investment_yen",
        "recovery_yen",
        "profit_yen",
        "win_flag",
    ]:
        work[col] = numeric_series(work, col)

    result = (
        work.groupby(
            group_col,
            dropna=False,
            as_index=False,
        )
        .agg(
            sessions=("date", "size"),
            investment_yen=("investment_yen", "sum"),
            recovery_yen=("recovery_yen", "sum"),
            profit_yen=("profit_yen", "sum"),
            wins=("win_flag", "sum"),
            avg_profit_yen=("profit_yen", "mean"),
        )
    )

    result["win_rate_percent"] = (
        result["wins"]
        / result["sessions"]
        * 100.0
    )

    result["roi_percent"] = (
        result["recovery_yen"]
        / result["investment_yen"]
        * 100.0
    )

    return result


def save_summaries(df: pd.DataFrame) -> None:
    if df.empty:
        return

    targets = {
        "summary_by_store.csv": "store",
        "summary_by_machine.csv": "machine_name",
        "summary_by_weekday.csv": "weekday",
        "summary_by_prediction_rank.csv": "prediction_rank",
        "summary_by_draw_no_used.csv": "draw_no_used",
    }

    for filename, group_col in targets.items():
        make_group_summary(
            df,
            group_col,
        ).to_csv(
            SUMMARY_DIR / filename,
            index=False,
            encoding=ENCODING,
        )


def print_summary() -> None:
    df = load_master()

    header("SlotAnalyzer Session Summary")

    if df.empty:
        print("No session data.")
        print(MASTER_CSV)
        return

    investment = numeric_series(df, "investment_yen").sum()
    recovery = numeric_series(df, "recovery_yen").sum()
    profit = numeric_series(df, "profit_yen").sum()

    wins = int(
        (
            numeric_series(df, "profit_yen")
            > 0
        ).sum()
    )

    sessions = len(df)

    win_rate = (
        wins / sessions * 100.0
        if sessions
        else 0.0
    )

    roi = (
        recovery / investment * 100.0
        if investment > 0
        else 0.0
    )

    print(f"sessions             : {sessions}")
    print(f"wins                 : {wins}")
    print(f"win_rate             : {win_rate:.2f}%")
    print(f"investment           : {investment:,.0f} yen")
    print(f"recovery             : {recovery:,.0f} yen")
    print(f"profit               : {profit:+,.0f} yen")
    print(f"roi                  : {roi:.2f}%")

    header("LATEST SESSIONS")

    cols = [
        "date",
        "store",
        "machine_no",
        "machine_name",
        "prediction_rank",
        "investment_yen",
        "recovery_yen",
        "profit_yen",
    ]

    print(
        df.tail(10)[cols].to_string(
            index=False
        )
    )

    save_summaries(df)

    print()
    print(f"master               : {MASTER_CSV}")
    print(f"summary dir          : {SUMMARY_DIR}")


def add_20260822_session(
    replace: bool,
) -> None:

    record = build_record(
        date="2026-08-22",
        store="マルハンメガシティ前橋インター",
        player="ゆめじろう",
        draw_no_self=360,
        draw_no_partner=265,
        draw_no_used=265,
        machine_no=912,
        machine_name="ヤバチバ",
        prediction_model="CHAMPION_V4.2_C",
        prediction_rank=1,
        investment_yen=41000,
        recovery_yen=93000,
        rainbow7_chain=6,
        bell_focus="あり（虹7後、約200G）",
        memo=(
            "8/22事前予測1位を実戦。"
            "虹7モード6連。"
            "虹7部分の実戦記録合計4815枚。"
            "通常時ベル集中を確認。"
        ),
    )

    add_record(
        record,
        replace=replace,
    )

    header("SESSION SAVED")
    print("2026-08-22 / 912 / ヤバチバ")
    print("investment           : 41,000 yen")
    print("recovery             : 93,000 yen")
    print("profit               : +52,000 yen")
    print("prediction rank      : 1")
    print(MASTER_CSV)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SlotAnalyzer session / profit management V1"
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser(
        "init",
        help="Create the master CSV if it does not exist.",
    )

    today = sub.add_parser(
        "add-20260822",
        help="Add the known 2026-08-22 session.",
    )

    today.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing duplicate session intentionally.",
    )

    sub.add_parser(
        "summary",
        help="Show overall results and refresh summary CSV files.",
    )

    sub.add_parser(
        "list",
        help="Show all stored sessions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = args.command or "summary"

    if command == "init":
        ensure_storage()
        header("INITIALIZED")
        print(MASTER_CSV)

    elif command == "add-20260822":
        add_20260822_session(
            replace=bool(args.replace)
        )
        print_summary()

    elif command == "list":
        df = load_master()
        header("ALL SESSIONS")

        if df.empty:
            print("No session data.")
        else:
            print(
                df.to_string(
                    index=False
                )
            )

    elif command == "summary":
        print_summary()

    else:
        raise RuntimeError(
            f"Unknown command: {command}"
        )


if __name__ == "__main__":
    main()
