from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

SOURCE_DIR = (
    PROJECT_ROOT / "data" / "bigmarch_takasaki_oyagi" / "machine_number"
    / "analysis_31days_deep" / "05_juggler_nonjuggler_recent_win"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "bigmarch_takasaki_oyagi" / "machine_number"
    / "analysis_31days_deep" / "07_juggler_recent7_top3_diagnostics"
)

SOURCE_FILE = SOURCE_DIR / "05_segment_recent_win_picks.csv"


def header(title: str) -> None:
    print()
    print("=" * 122)
    print(title)
    print("=" * 122)


def pct(x: pd.Series) -> float:
    return float(x.mean() * 100) if len(x) else np.nan


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(SOURCE_FILE)

    df = pd.read_csv(SOURCE_FILE, encoding="utf-8-sig")
    df["target_date"] = pd.to_datetime(df["target_date"], errors="raise")

    required = {
        "target_date", "segment", "model", "prediction_rank",
        "machine_no", "machine_name", "recent_win",
        "actual_diff", "actual_win", "actual_plus1000", "actual_plus2000",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    x = df[
        (df["segment"] == "JUGGLER")
        & (df["model"] == "JUGGLER_RECENT7_WIN")
        & (df["prediction_rank"] <= 3)
    ].copy()

    if x.empty:
        raise RuntimeError("No JUGGLER_RECENT7_WIN Top3 records found.")

    x["weekday_no"] = x["target_date"].dt.weekday
    jp_weekdays = {
        0: "月", 1: "火", 2: "水", 3: "木",
        4: "金", 5: "土", 6: "日",
    }
    x["weekday"] = x["weekday_no"].map(jp_weekdays)
    x["is_weekend"] = x["weekday_no"].isin([5, 6])

    header("07 - Big March Takasaki Oyagi JUGGLER RECENT7 Top3 Diagnostics")
    print(f"source                : {SOURCE_FILE}")
    print(f"records               : {len(x)}")
    print(f"evaluation days       : {x['target_date'].nunique()}")
    print(f"date range            : {x['target_date'].min().date()} to {x['target_date'].max().date()}")
    print(f"unique machines       : {x['machine_no'].nunique()}")
    print(f"unique machine names  : {x['machine_name'].nunique()}")
    print(f"avg actual diff       : {x['actual_diff'].mean():.2f}")
    print(f"total actual diff     : {x['actual_diff'].sum():.0f}")
    print(f"win rate              : {pct(x['actual_win']):.2f}%")

    machine_name = (
        x.groupby("machine_name")
        .agg(
            selections=("actual_diff", "size"),
            unique_machines=("machine_no", "nunique"),
            avg_actual_diff=("actual_diff", "mean"),
            median_actual_diff=("actual_diff", "median"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
            plus1000_rate=("actual_plus1000", "mean"),
            plus2000_rate=("actual_plus2000", "mean"),
        )
        .reset_index()
        .sort_values(["selections", "total_actual_diff"], ascending=[False, False])
    )
    for c in ["win_rate", "plus1000_rate", "plus2000_rate"]:
        machine_name[c] *= 100

    header("BY MACHINE NAME")
    print(machine_name.to_string(index=False))

    machine_no = (
        x.groupby(["machine_no", "machine_name"])
        .agg(
            selections=("actual_diff", "size"),
            avg_rank=("prediction_rank", "mean"),
            avg_actual_diff=("actual_diff", "mean"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
        )
        .reset_index()
        .sort_values(["selections", "total_actual_diff"], ascending=[False, False])
    )
    machine_no["win_rate"] *= 100

    header("BY MACHINE NUMBER")
    print(machine_no.to_string(index=False))

    weekday = (
        x.groupby(["weekday_no", "weekday"])
        .agg(
            selections=("actual_diff", "size"),
            days=("target_date", "nunique"),
            avg_actual_diff=("actual_diff", "mean"),
            median_actual_diff=("actual_diff", "median"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
            plus1000_rate=("actual_plus1000", "mean"),
        )
        .reset_index()
        .sort_values("weekday_no")
    )
    weekday["win_rate"] *= 100
    weekday["plus1000_rate"] *= 100

    header("BY WEEKDAY")
    print(weekday.to_string(index=False))

    weekend = (
        x.groupby("is_weekend")
        .agg(
            selections=("actual_diff", "size"),
            days=("target_date", "nunique"),
            avg_actual_diff=("actual_diff", "mean"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
        )
        .reset_index()
    )
    weekend["day_type"] = np.where(weekend["is_weekend"], "SAT_SUN", "WEEKDAY")
    weekend["win_rate"] *= 100

    header("WEEKDAY VS SAT/SUN")
    print(
        weekend[
            ["day_type", "selections", "days", "avg_actual_diff", "total_actual_diff", "win_rate"]
        ].to_string(index=False)
    )

    concentration = (
        x.groupby("machine_no")
        .size()
        .sort_values(ascending=False)
        .rename("selections")
        .reset_index()
    )
    concentration["share_pct"] = concentration["selections"] / len(x) * 100

    header("SELECTION CONCENTRATION")
    print(f"Top1 machine share    : {concentration['share_pct'].iloc[0]:.2f}%")
    print(f"Top3 machines share   : {concentration.head(3)['share_pct'].sum():.2f}%")
    print(f"Top5 machines share   : {concentration.head(5)['share_pct'].sum():.2f}%")
    print(f"Machines selected >=2 : {(concentration['selections'] >= 2).sum()}")
    print()
    print(concentration.head(15).to_string(index=False))

    # Leave-one-machine-name-out robustness.
    loo_rows = []
    for name in sorted(x["machine_name"].unique()):
        y = x[x["machine_name"] != name]
        loo_rows.append({
            "removed_machine_name": name,
            "remaining_n": len(y),
            "remaining_avg_diff": y["actual_diff"].mean(),
            "remaining_total_diff": y["actual_diff"].sum(),
            "remaining_win_rate": pct(y["actual_win"]),
        })
    loo = pd.DataFrame(loo_rows).sort_values("remaining_avg_diff")

    header("LEAVE-ONE-MACHINE-NAME-OUT ROBUSTNESS")
    print(loo.to_string(index=False))

    # Leave-one-machine-number-out robustness for machines selected >= 2 times.
    repeated = concentration.loc[concentration["selections"] >= 2, "machine_no"].tolist()
    loo_machine_rows = []
    for no in repeated:
        y = x[x["machine_no"] != no]
        loo_machine_rows.append({
            "removed_machine_no": no,
            "remaining_n": len(y),
            "remaining_avg_diff": y["actual_diff"].mean(),
            "remaining_total_diff": y["actual_diff"].sum(),
            "remaining_win_rate": pct(y["actual_win"]),
        })
    loo_machine = pd.DataFrame(loo_machine_rows)
    if not loo_machine.empty:
        loo_machine = loo_machine.sort_values("remaining_avg_diff")

    header("LEAVE-ONE-REPEATED-MACHINE-OUT ROBUSTNESS")
    if loo_machine.empty:
        print("No machine was selected two or more times.")
    else:
        print(loo_machine.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "07_top3_picks.csv": x,
        "07_by_machine_name.csv": machine_name,
        "07_by_machine_number.csv": machine_no,
        "07_by_weekday.csv": weekday,
        "07_weekday_vs_weekend.csv": weekend,
        "07_selection_concentration.csv": concentration,
        "07_leave_one_machine_name_out.csv": loo,
        "07_leave_one_repeated_machine_out.csv": loo_machine,
    }

    header("FILES SAVED")
    for filename, frame in files.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(path)

    print()
    print("07 Top3 diagnostics complete.")
    print("No production model was changed.")
    print("No Maruhan Maebashi files were modified.")


if __name__ == "__main__":
    main()
