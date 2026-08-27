from __future__ import annotations

from pathlib import Path
import argparse
import re

import pandas as pd


# ============================================================
# 75 - Juggler Separated Prediction
# ============================================================
#
# Purpose
# -------
# Read the already-created 64 all-514 prediction snapshot and
# create a SEPARATE Juggler-only ranking without changing 64 or 74.
#
# Important
# -------
# - 64 = official normal prediction
# - 74 = A-type overall prediction
# - 75 = Juggler-only prediction
# - V4.2_C score is NOT recalculated here.
# - This is NOT yet a BB/RB/combined-probability Juggler model.
# ============================================================


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

ANALYSIS_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
)

SOURCE_64_DIR = ANALYSIS_DIR / "64_Ver4_2_future_top10"

OUTPUT_DIR = ANALYSIS_DIR / "75_Ver4_2_Juggler_prediction"

TOP_N = 10
PRIMARY_N = 5


def header(title: str) -> None:
    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a Juggler-only ranking from a frozen 64 all514 prediction."
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="YYYY-MM-DD. If omitted, newest 64 all514 prediction is used.",
    )
    return parser.parse_args()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"CSV read failed: {path}\nlast_error={last_error}")


def discover_all514_files():
    rx = re.compile(r"^64_prediction_(\d{8})_all514\.csv$", re.IGNORECASE)
    found = []

    for path in SOURCE_64_DIR.glob("64_prediction_????????_all514.csv"):
        m = rx.match(path.name)
        if not m:
            continue
        date = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
        if pd.isna(date):
            continue
        found.append((pd.Timestamp(date), path))

    return sorted(found, key=lambda x: x[0])


def resolve_source_file(target_date):
    files = discover_all514_files()

    if not files:
        raise RuntimeError(f"No 64 all514 files found in:\n{SOURCE_64_DIR}")

    if target_date is None:
        return files[-1]

    target_date = pd.Timestamp(target_date).normalize()

    for date, path in files:
        if date.normalize() == target_date:
            return date, path

    raise FileNotFoundError(
        f"No 64 all514 prediction found for {target_date.date()}."
    )


def validate_source(df: pd.DataFrame, target_date: pd.Timestamp) -> None:
    required = [
        "machine_no",
        "machine_name",
        "score",
        "prediction_rank",
        "tier",
        "target_date",
        "latest_data_date",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"64 source columns missing: {missing}")

    if len(df) != 514:
        raise RuntimeError(f"64 source rows={len(df)}; expected 514.")

    machine_no = pd.to_numeric(df["machine_no"], errors="coerce")

    if machine_no.nunique(dropna=True) != 514:
        raise RuntimeError("64 source does not contain 514 unique machine numbers.")

    if machine_no.duplicated(keep=False).sum() != 0:
        raise RuntimeError("64 source contains duplicate machine numbers.")

    dates = pd.to_datetime(df["target_date"], errors="coerce").dropna()

    if (
        len(dates) == 0
        or dates.nunique() != 1
        or pd.Timestamp(dates.iloc[0]).normalize() != target_date.normalize()
    ):
        raise RuntimeError("64 source target_date does not match filename target date.")


def is_juggler(machine_name) -> bool:
    if pd.isna(machine_name):
        return False
    return "ジャグラー" in str(machine_name).strip()


def main() -> None:
    args = parse_args()

    header("75 - V4.2_C Juggler Separated Prediction")

    requested_target = None
    if args.target_date:
        requested_target = pd.Timestamp(
            pd.to_datetime(args.target_date, format="%Y-%m-%d", errors="raise")
        )

    target_date, source_path = resolve_source_file(requested_target)

    print(f"source 64 file        : {source_path}")
    print(f"prediction target     : {target_date.date()}")

    source = read_csv_flexible(source_path)
    validate_source(source, target_date)

    print(f"source rows           : {len(source)}")
    print(f"source unique machines: {source['machine_no'].nunique()}")

    juggler = source[source["machine_name"].apply(is_juggler)].copy()

    if juggler.empty:
        raise RuntimeError("No Juggler machines found.")

    juggler = (
        juggler.sort_values(
            ["score", "prediction_rank", "machine_no"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    juggler["juggler_rank"] = range(1, len(juggler) + 1)
    juggler["juggler_tier"] = "OUTSIDE_TOP10"
    juggler.loc[juggler["juggler_rank"] <= TOP_N, "juggler_tier"] = "NEXT"
    juggler.loc[juggler["juggler_rank"] <= PRIMARY_N, "juggler_tier"] = "PRIMARY"

    top10 = juggler.head(TOP_N).copy()

    print(f"Juggler machines      : {len(juggler)}")

    header(f"{target_date.date()} JUGGLER PREDICTION TOP10")

    display_cols = [
        "juggler_rank",
        "juggler_tier",
        "machine_no",
        "machine_name",
        "score",
        "prediction_rank",
    ]

    optional_cols = [
        "avg31",
        "recent7_avg",
        "recent7_win",
        "last_diff",
        "prev_change",
        "weekday_avg",
        "type_avg",
        "plus1000_rate",
        "plus2000_rate",
        "neighbor_avg",
    ]

    display_cols += [c for c in optional_cols if c in top10.columns]

    print(top10[display_cols].to_string(index=False))

    print()
    print("PRIMARY = Juggler ranks 1-5")
    print("NEXT    = Juggler ranks 6-10")
    print()
    print("Important:")
    print("- Frozen 64 V4.2_C scores are reused.")
    print("- 64 normal prediction is not modified.")
    print("- 74 A-type prediction is not modified.")
    print("- This is not yet a Juggler-specific BB/RB model.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ymd = target_date.strftime("%Y%m%d")

    all_path = OUTPUT_DIR / f"75_Juggler_prediction_{ymd}_all.csv"
    top10_path = OUTPUT_DIR / f"75_Juggler_prediction_{ymd}_top10.csv"
    metadata_path = OUTPUT_DIR / f"75_Juggler_prediction_{ymd}_metadata.csv"

    juggler.to_csv(all_path, index=False, encoding="utf-8-sig")
    top10.to_csv(top10_path, index=False, encoding="utf-8-sig")

    metadata = pd.DataFrame(
        [
            {
                "target_date": target_date.date(),
                "source_64_file": source_path.name,
                "source_rows": len(source),
                "juggler_machines": len(juggler),
                "top_n": TOP_N,
                "primary_n": PRIMARY_N,
                "score_recalculated": False,
                "normal_prediction_modified": False,
                "a_type_prediction_modified": False,
                "classification_policy": "machine_name_contains_ジャグラー",
                "juggler_specific_bb_rb_model": False,
            }
        ]
    )

    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    header("FILES SAVED")
    for path in (all_path, top10_path, metadata_path):
        print(path)

    print()
    print("75 Juggler separated prediction complete.")


if __name__ == "__main__":
    main()
