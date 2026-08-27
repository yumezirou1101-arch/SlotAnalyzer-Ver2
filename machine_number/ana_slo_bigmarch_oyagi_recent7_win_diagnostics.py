from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

BACKTEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
    / "analysis_31days_deep"
    / "02_initial_walk_forward_backtest"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
    / "analysis_31days_deep"
    / "03_recent7_win_diagnostics"
)


def header(title: str) -> None:
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


def parse_args():
    p = argparse.ArgumentParser(
        description="Detailed diagnostics for RECENT7_WIN baseline."
    )
    p.add_argument(
        "--model",
        default="RECENT7_WIN",
        help="Model name to inspect. Default: RECENT7_WIN.",
    )
    return p.parse_args()


def load_files():
    picks_path = BACKTEST_DIR / "02_initial_walk_forward_picks.csv"
    daily_path = BACKTEST_DIR / "02_initial_walk_forward_daily.csv"

    if not picks_path.exists():
        raise FileNotFoundError(picks_path)
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)

    picks = pd.read_csv(picks_path, encoding="utf-8-sig")
    daily = pd.read_csv(daily_path, encoding="utf-8-sig")

    picks["target_date"] = pd.to_datetime(
        picks["target_date"],
        errors="raise",
    ).dt.normalize()

    daily["target_date"] = pd.to_datetime(
        daily["target_date"],
        errors="raise",
    ).dt.normalize()

    return picks, daily


def add_machine_family(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    def family(name: str) -> str:
        s = str(name)

        if "ジャグラー" in s:
            return "JUGGLER"
        if "沖ドキ" in s:
            return "OKIDOKI"
        if "モンキーターン" in s:
            return "MONKEY"
        if "カバネリ" in s:
            return "KABANERI"
        if "北斗" in s:
            return "HOKUTO"
        if "ヴァルヴレイヴ" in s:
            return "VALVRAVE"
        return "OTHER"

    x["machine_family"] = x["machine_name"].map(family)
    return x


def bootstrap_mean_ci(values, n_boot=10000, seed=42):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    samples = rng.choice(
        arr,
        size=(n_boot, len(arr)),
        replace=True,
    )
    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def main():
    args = parse_args()

    picks, daily = load_files()

    picks = picks[
        picks["model"] == args.model
    ].copy()

    daily = daily[
        daily["model"] == args.model
    ].copy()

    if picks.empty:
        raise RuntimeError(
            f"No picks found for model: {args.model}"
        )

    picks = add_machine_family(picks)

    header(
        f"Big March Takasaki Oyagi - {args.model} Diagnostics"
    )

    print(
        f"evaluated days        : {picks['target_date'].nunique()}"
    )
    print(
        f"selected rows         : {len(picks)}"
    )
    print(
        f"avg actual diff       : {picks['actual_diff'].mean():.2f}"
    )
    print(
        f"median actual diff    : {picks['actual_diff'].median():.2f}"
    )
    print(
        f"total actual diff     : {picks['actual_diff'].sum():.0f}"
    )
    print(
        f"win rate             : {(picks['actual_win'].mean()*100):.2f}%"
    )

    ci_low, ci_high = bootstrap_mean_ci(
        picks["actual_diff"]
    )
    print(
        f"avg diff bootstrap CI95: {ci_low:.1f} to {ci_high:.1f}"
    )

    # --------------------------------------------------------
    # Rank diagnostics
    # --------------------------------------------------------
    rank = (
        picks.groupby("prediction_rank")
        .agg(
            n=("actual_diff", "size"),
            avg_actual_diff=("actual_diff", "mean"),
            median_actual_diff=("actual_diff", "median"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
            plus1000_rate=("actual_plus1000", "mean"),
            plus2000_rate=("actual_plus2000", "mean"),
        )
        .reset_index()
    )

    for c in ("win_rate", "plus1000_rate", "plus2000_rate"):
        rank[c] = rank[c] * 100

    header("RANK-BY-RANK")
    print(rank.to_string(index=False))

    # --------------------------------------------------------
    # Daily diagnostics
    # --------------------------------------------------------
    daily_detail = (
        picks.groupby("target_date")
        .agg(
            avg_actual_diff=("actual_diff", "mean"),
            median_actual_diff=("actual_diff", "median"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
            plus1000_rate=("actual_plus1000", "mean"),
            plus2000_rate=("actual_plus2000", "mean"),
            best_machine_diff=("actual_diff", "max"),
            worst_machine_diff=("actual_diff", "min"),
        )
        .reset_index()
        .sort_values("target_date")
    )

    for c in ("win_rate", "plus1000_rate", "plus2000_rate"):
        daily_detail[c] = daily_detail[c] * 100

    daily_detail["positive_day"] = (
        daily_detail["avg_actual_diff"] > 0
    )

    header("BEST / WORST DAYS")
    print("BEST 5")
    print(
        daily_detail.sort_values(
            "avg_actual_diff",
            ascending=False,
        ).head(5).to_string(index=False)
    )

    print()
    print("WORST 5")
    print(
        daily_detail.sort_values(
            "avg_actual_diff",
            ascending=True,
        ).head(5).to_string(index=False)
    )

    # --------------------------------------------------------
    # Machine-name diagnostics
    # --------------------------------------------------------
    machine_name = (
        picks.groupby("machine_name")
        .agg(
            n=("actual_diff", "size"),
            avg_actual_diff=("actual_diff", "mean"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
            plus1000_rate=("actual_plus1000", "mean"),
            avg_signal=("recent7_win", "mean"),
        )
        .reset_index()
    )

    machine_name["win_rate"] *= 100
    machine_name["plus1000_rate"] *= 100

    machine_name = machine_name.sort_values(
        ["n", "avg_actual_diff"],
        ascending=[False, False],
    )

    header("MOST FREQUENT MACHINE NAMES")
    print(
        machine_name.head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # Family diagnostics
    # --------------------------------------------------------
    family = (
        picks.groupby("machine_family")
        .agg(
            n=("actual_diff", "size"),
            avg_actual_diff=("actual_diff", "mean"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
            plus1000_rate=("actual_plus1000", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    family["win_rate"] *= 100
    family["plus1000_rate"] *= 100

    header("MACHINE FAMILY")
    print(family.to_string(index=False))

    # --------------------------------------------------------
    # Signal bucket diagnostics
    # --------------------------------------------------------
    bins = [-0.001, 0.25, 0.50, 0.75, 1.001]
    labels = [
        "0-25%",
        "25-50%",
        "50-75%",
        "75-100%",
    ]

    picks["recent7_win_bucket"] = pd.cut(
        picks["recent7_win"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    bucket = (
        picks.groupby(
            "recent7_win_bucket",
            observed=False,
        )
        .agg(
            n=("actual_diff", "size"),
            avg_actual_diff=("actual_diff", "mean"),
            total_actual_diff=("actual_diff", "sum"),
            win_rate=("actual_win", "mean"),
            plus1000_rate=("actual_plus1000", "mean"),
            plus2000_rate=("actual_plus2000", "mean"),
        )
        .reset_index()
    )

    for c in ("win_rate", "plus1000_rate", "plus2000_rate"):
        bucket[c] *= 100

    header("RECENT7_WIN SIGNAL BUCKETS")
    print(bucket.to_string(index=False))

    # --------------------------------------------------------
    # Concentration diagnostics
    # --------------------------------------------------------
    machine_no_freq = (
        picks.groupby("machine_no")
        .agg(
            selections=("target_date", "size"),
            machine_names=("machine_name", "nunique"),
            avg_actual_diff=("actual_diff", "mean"),
            total_actual_diff=("actual_diff", "sum"),
        )
        .reset_index()
        .sort_values(
            ["selections", "total_actual_diff"],
            ascending=[False, False],
        )
    )

    total_selections = len(picks)
    top10_selection_share = (
        machine_no_freq.head(10)["selections"].sum()
        / total_selections
        * 100
    )

    header("SELECTION CONCENTRATION")
    print(
        f"unique selected machines : {picks['machine_no'].nunique()}"
    )
    print(
        f"top 10 machine-nos share : {top10_selection_share:.2f}%"
    )
    print()
    print(machine_no_freq.head(20).to_string(index=False))

    # --------------------------------------------------------
    # Robustness: remove single best day / worst day
    # --------------------------------------------------------
    best_day = daily_detail.loc[
        daily_detail["avg_actual_diff"].idxmax(),
        "target_date",
    ]
    worst_day = daily_detail.loc[
        daily_detail["avg_actual_diff"].idxmin(),
        "target_date",
    ]

    no_best = picks[
        picks["target_date"] != best_day
    ]

    no_worst = picks[
        picks["target_date"] != worst_day
    ]

    robustness = pd.DataFrame(
        [
            {
                "scenario": "ALL",
                "rows": len(picks),
                "avg_diff": picks["actual_diff"].mean(),
                "total_diff": picks["actual_diff"].sum(),
                "win_rate": picks["actual_win"].mean() * 100,
            },
            {
                "scenario": "REMOVE_BEST_DAY",
                "rows": len(no_best),
                "avg_diff": no_best["actual_diff"].mean(),
                "total_diff": no_best["actual_diff"].sum(),
                "win_rate": no_best["actual_win"].mean() * 100,
            },
            {
                "scenario": "REMOVE_WORST_DAY",
                "rows": len(no_worst),
                "avg_diff": no_worst["actual_diff"].mean(),
                "total_diff": no_worst["actual_diff"].sum(),
                "win_rate": no_worst["actual_win"].mean() * 100,
            },
        ]
    )

    header("ROBUSTNESS")
    print(
        f"best day              : {best_day.date()}"
    )
    print(
        f"worst day             : {worst_day.date()}"
    )
    print()
    print(robustness.to_string(index=False))

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rank_path = OUTPUT_DIR / "03_recent7_win_rank.csv"
    daily_path = OUTPUT_DIR / "03_recent7_win_daily.csv"
    machine_path = OUTPUT_DIR / "03_recent7_win_machine_name.csv"
    family_path = OUTPUT_DIR / "03_recent7_win_family.csv"
    bucket_path = OUTPUT_DIR / "03_recent7_win_signal_bucket.csv"
    concentration_path = OUTPUT_DIR / "03_recent7_win_machine_concentration.csv"
    robustness_path = OUTPUT_DIR / "03_recent7_win_robustness.csv"

    rank.to_csv(rank_path, index=False, encoding="utf-8-sig")
    daily_detail.to_csv(daily_path, index=False, encoding="utf-8-sig")
    machine_name.to_csv(machine_path, index=False, encoding="utf-8-sig")
    family.to_csv(family_path, index=False, encoding="utf-8-sig")
    bucket.to_csv(bucket_path, index=False, encoding="utf-8-sig")
    machine_no_freq.to_csv(
        concentration_path,
        index=False,
        encoding="utf-8-sig",
    )
    robustness.to_csv(
        robustness_path,
        index=False,
        encoding="utf-8-sig",
    )

    header("FILES SAVED")
    for p in (
        rank_path,
        daily_path,
        machine_path,
        family_path,
        bucket_path,
        concentration_path,
        robustness_path,
    ):
        print(p)

    print()
    print("RECENT7_WIN diagnostics complete.")
    print("No production model was changed.")
    print("No Maruhan Maebashi files were modified.")


if __name__ == "__main__":
    main()
