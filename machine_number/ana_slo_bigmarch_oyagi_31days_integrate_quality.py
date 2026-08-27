from __future__ import annotations

from pathlib import Path
import argparse
import re
import pandas as pd


# ============================================================
# Big March Takasaki Oyagi
# 31-day integration + whole-period data quality check
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
)

OUTPUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "01_data_quality"
)

FILE_RE = re.compile(
    r"^ana_slo_bigmarch_oyagi_(\d{8})\.csv$",
    re.IGNORECASE,
)


def header(title: str) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Integrate Big March Takasaki Oyagi daily CSV files "
            "and run whole-period quality checks."
        )
    )
    p.add_argument(
        "--days",
        type=int,
        default=31,
        help="Newest N daily CSV files to use. Default: 31.",
    )
    p.add_argument(
        "--min-machines",
        type=int,
        default=200,
        help="Minimum acceptable machines per day. Default: 200.",
    )
    return p.parse_args()


def discover_daily_files():
    found = []

    for path in DATA_DIR.glob("ana_slo_bigmarch_oyagi_*.csv"):
        m = FILE_RE.fullmatch(path.name)
        if not m:
            continue

        date = pd.to_datetime(
            m.group(1),
            format="%Y%m%d",
            errors="raise",
        ).normalize()

        found.append((date, path))

    return sorted(found, key=lambda x: x[0])


def normalize_daily(path: Path, expected_date: pd.Timestamp):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    required = {
        "date",
        "machine_name",
        "machine_no",
        "G",
        "diff",
    }

    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise RuntimeError(
            f"Required columns missing: {missing_cols}"
        )

    x = df.copy()

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).dt.normalize()

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="coerce",
    )

    x["G"] = pd.to_numeric(
        x["G"],
        errors="coerce",
    )

    x["diff"] = pd.to_numeric(
        x["diff"],
        errors="coerce",
    )

    internal_dates = sorted(
        x["date"].dropna().unique()
    )

    internal_date_ok = (
        len(internal_dates) == 1
        and pd.Timestamp(internal_dates[0]).normalize() == expected_date
    )

    return x, internal_date_ok


def main():
    args = parse_args()

    if args.days < 1:
        raise ValueError("--days must be >= 1")

    if args.min_machines < 1:
        raise ValueError("--min-machines must be >= 1")

    header(
        "Big March Takasaki Oyagi - 31 Day Integration / Whole-Period Quality Check"
    )

    all_files = discover_daily_files()

    if not all_files:
        raise FileNotFoundError(
            f"No daily CSV files found in {DATA_DIR}"
        )

    selected = all_files[-args.days:]

    print(f"daily files found     : {len(all_files)}")
    print(f"days requested        : {args.days}")
    print(f"days selected         : {len(selected)}")
    print(f"min machines/day      : {args.min_machines}")
    print(
        f"selected range        : "
        f"{selected[0][0].date()} to {selected[-1][0].date()}"
    )

    if len(selected) < args.days:
        print(
            f"WARNING               : only {len(selected)} daily files are available."
        )

    daily_quality = []
    frames = []

    for date, path in selected:
        try:
            x, internal_date_ok = normalize_daily(path, date)

            rows = len(x)
            unique_machines = int(
                x["machine_no"].nunique(dropna=True)
            )
            duplicates = int(
                x["machine_no"].duplicated(keep=False).sum()
            )
            missing_machine = int(
                x["machine_no"].isna().sum()
            )
            missing_name = int(
                x["machine_name"]
                .isin(["", "nan", "None"])
                .sum()
            )
            missing_diff = int(
                x["diff"].isna().sum()
            )
            missing_g = int(
                x["G"].isna().sum()
            )
            negative_g = int(
                ((x["G"] < 0).fillna(False)).sum()
            )

            machine_count_ok = rows >= args.min_machines

            basic_ok = all(
                (
                    machine_count_ok,
                    internal_date_ok,
                    unique_machines == rows,
                    duplicates == 0,
                    missing_machine == 0,
                    missing_name == 0,
                    missing_diff == 0,
                    missing_g == 0,
                    negative_g == 0,
                )
            )

            daily_quality.append({
                "date": date.date(),
                "file": path.name,
                "rows": rows,
                "machines": unique_machines,
                "duplicates": duplicates,
                "missing_machine": missing_machine,
                "missing_name": missing_name,
                "missing_diff": missing_diff,
                "missing_G": missing_g,
                "negative_G": negative_g,
                "internal_date_ok": internal_date_ok,
                "machine_count_ok": machine_count_ok,
                "basic_ok": basic_ok,
                "diff_min": x["diff"].min(),
                "diff_max": x["diff"].max(),
                "G_min": x["G"].min(),
                "G_max": x["G"].max(),
                "error": "",
            })

            x["source_file"] = path.name
            frames.append(x)

        except Exception as exc:
            daily_quality.append({
                "date": date.date(),
                "file": path.name,
                "rows": "",
                "machines": "",
                "duplicates": "",
                "missing_machine": "",
                "missing_name": "",
                "missing_diff": "",
                "missing_G": "",
                "negative_G": "",
                "internal_date_ok": False,
                "machine_count_ok": False,
                "basic_ok": False,
                "diff_min": "",
                "diff_max": "",
                "G_min": "",
                "G_max": "",
                "error": repr(exc),
            })

    quality = pd.DataFrame(daily_quality)

    header("DAILY QUALITY")
    display_cols = [
        "date",
        "rows",
        "machines",
        "duplicates",
        "missing_name",
        "missing_diff",
        "missing_G",
        "internal_date_ok",
        "machine_count_ok",
        "basic_ok",
    ]
    print(quality[display_cols].to_string(index=False))

    if not frames:
        raise RuntimeError(
            "No valid daily CSV could be loaded."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined["date"] = pd.to_datetime(
        combined["date"]
    ).dt.normalize()

    # --------------------------------------------------------
    # Whole-period checks
    # --------------------------------------------------------
    total_rows = len(combined)
    total_days = combined["date"].nunique()

    pair_duplicates = int(
        combined.duplicated(
            subset=["date", "machine_no"],
            keep=False,
        ).sum()
    )

    machine_name_by_no = (
        combined
        .dropna(subset=["machine_no"])
        .groupby("machine_no")["machine_name"]
        .nunique()
        .sort_values(ascending=False)
    )

    renamed_machine_numbers = (
        machine_name_by_no[
            machine_name_by_no > 1
        ]
    )

    machine_presence = (
        combined
        .groupby("machine_no")
        .agg(
            observed_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            machine_names=("machine_name", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["observed_days", "machine_no"],
            ascending=[False, True],
        )
    )

    daily_counts = (
        combined
        .groupby("date")
        .agg(
            rows=("machine_no", "size"),
            machines=("machine_no", "nunique"),
            total_diff=("diff", "sum"),
            avg_diff=("diff", "mean"),
            avg_G=("G", "mean"),
            zero_G=("G", lambda s: int((s == 0).sum())),
        )
        .reset_index()
        .sort_values("date")
    )

    # Detect calendar gaps only as information.
    min_date = combined["date"].min()
    max_date = combined["date"].max()

    calendar_dates = pd.date_range(
        min_date,
        max_date,
        freq="D",
    )

    observed_dates = set(
        combined["date"].drop_duplicates()
    )

    missing_calendar_dates = [
        d
        for d in calendar_dates
        if d not in observed_dates
    ]

    header("WHOLE-PERIOD QUALITY")

    print(f"records              : {total_rows:,}")
    print(f"days                 : {total_days}")
    print(f"date range           : {min_date.date()} to {max_date.date()}")
    print(f"date-machine duplicates: {pair_duplicates}")
    print(
        f"machine numbers with name changes: "
        f"{len(renamed_machine_numbers)}"
    )
    print(
        f"calendar dates missing inside range: "
        f"{len(missing_calendar_dates)}"
    )

    if missing_calendar_dates:
        print(
            "missing dates         : "
            + ", ".join(
                d.strftime("%Y-%m-%d")
                for d in missing_calendar_dates
            )
        )

    bad_days = quality[
        quality["basic_ok"] != True
    ]

    overall_ok = (
        len(bad_days) == 0
        and pair_duplicates == 0
    )

    print()
    print(
        "RESULT               : "
        + (
            "WHOLE-PERIOD DATA QUALITY OK"
            if overall_ok
            else "REVIEW REQUIRED"
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_tag = min_date.strftime("%Y%m%d")
    end_tag = max_date.strftime("%Y%m%d")

    combined_path = (
        OUTPUT_DIR
        / f"01_bigmarch_oyagi_integrated_{start_tag}_{end_tag}.csv"
    )

    quality_path = (
        OUTPUT_DIR
        / f"01_bigmarch_oyagi_daily_quality_{start_tag}_{end_tag}.csv"
    )

    counts_path = (
        OUTPUT_DIR
        / f"01_bigmarch_oyagi_daily_counts_{start_tag}_{end_tag}.csv"
    )

    presence_path = (
        OUTPUT_DIR
        / f"01_bigmarch_oyagi_machine_presence_{start_tag}_{end_tag}.csv"
    )

    rename_path = (
        OUTPUT_DIR
        / f"01_bigmarch_oyagi_machine_name_changes_{start_tag}_{end_tag}.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / f"01_bigmarch_oyagi_quality_summary_{start_tag}_{end_tag}.csv"
    )

    combined_save = combined.copy()
    combined_save["date"] = combined_save["date"].dt.strftime("%Y-%m-%d")

    daily_counts_save = daily_counts.copy()
    daily_counts_save["date"] = daily_counts_save["date"].dt.strftime("%Y-%m-%d")

    presence_save = machine_presence.copy()
    presence_save["first_date"] = pd.to_datetime(
        presence_save["first_date"]
    ).dt.strftime("%Y-%m-%d")
    presence_save["last_date"] = pd.to_datetime(
        presence_save["last_date"]
    ).dt.strftime("%Y-%m-%d")

    rename_df = (
        renamed_machine_numbers
        .rename("machine_name_count")
        .reset_index()
    )

    summary = pd.DataFrame([
        {
            "start_date": min_date.date(),
            "end_date": max_date.date(),
            "days": total_days,
            "records": total_rows,
            "bad_days": len(bad_days),
            "date_machine_duplicates": pair_duplicates,
            "machine_numbers_with_name_changes": len(
                renamed_machine_numbers
            ),
            "missing_calendar_dates": len(
                missing_calendar_dates
            ),
            "overall_quality_ok": overall_ok,
        }
    ])

    combined_save.to_csv(
        combined_path,
        index=False,
        encoding="utf-8-sig",
    )
    quality.to_csv(
        quality_path,
        index=False,
        encoding="utf-8-sig",
    )
    daily_counts_save.to_csv(
        counts_path,
        index=False,
        encoding="utf-8-sig",
    )
    presence_save.to_csv(
        presence_path,
        index=False,
        encoding="utf-8-sig",
    )
    rename_df.to_csv(
        rename_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    header("FILES SAVED")
    for p in (
        combined_path,
        quality_path,
        counts_path,
        presence_path,
        rename_path,
        summary_path,
    ):
        print(p)

    print()
    print(
        "Next step: review this quality result before building/backtesting "
        "the Big March Oyagi prediction model."
    )
    print(
        "No Maruhan Maebashi files were modified."
    )


if __name__ == "__main__":
    main()
