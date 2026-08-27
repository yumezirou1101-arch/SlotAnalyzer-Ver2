from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

SEGMENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
    / "analysis_31days_deep"
    / "05_juggler_nonjuggler_recent_win"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
    / "analysis_31days_deep"
    / "06_juggler_recent7_topn"
)

TOPNS = (3, 5, 10)


def header(title: str) -> None:
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare TOP3 / TOP5 / TOP10 for "
            "JUGGLER_RECENT7_WIN using the same 17 evaluation days."
        )
    )
    return p.parse_args()


def load_picks() -> pd.DataFrame:
    path = SEGMENT_DIR / "05_segment_recent_win_picks.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["target_date"] = pd.to_datetime(
        df["target_date"],
        errors="raise",
    ).dt.normalize()

    needed = {
        "target_date",
        "segment",
        "model",
        "window",
        "prediction_rank",
        "machine_no",
        "machine_name",
        "recent_win",
        "recent_n",
        "actual_diff",
        "actual_win",
        "actual_plus1000",
        "actual_plus2000",
    }

    missing = sorted(needed - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    x = df[
        (df["segment"] == "JUGGLER")
        & (df["model"] == "JUGGLER_RECENT7_WIN")
    ].copy()

    if x.empty:
        raise RuntimeError(
            "JUGGLER_RECENT7_WIN picks not found."
        )

    return x


def bootstrap_daily_mean_ci(values, n_boot=10000, seed=42):
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
    _ = parse_args()
    picks = load_picks()

    header("Big March Takasaki Oyagi - JUGGLER RECENT7 WIN TOP-N Comparison")

    print(f"evaluation days       : {picks['target_date'].nunique()}")
    print(f"available picks       : {len(picks)}")
    print(f"date range            : {picks['target_date'].min().date()} to {picks['target_date'].max().date()}")
    print(f"TOP-N candidates      : {TOPNS}")

    daily_rows = []
    pick_rows = []

    for topn in TOPNS:
        subset = picks[
            picks["prediction_rank"] <= topn
        ].copy()

        for target_date, grp in subset.groupby("target_date"):
            daily_rows.append({
                "target_date": target_date.date(),
                "topn": topn,
                "selected_n": len(grp),
                "avg_diff": grp["actual_diff"].mean(),
                "median_diff": grp["actual_diff"].median(),
                "total_diff": grp["actual_diff"].sum(),
                "win_rate": grp["actual_win"].mean() * 100,
                "plus1000_rate": grp["actual_plus1000"].mean() * 100,
                "plus2000_rate": grp["actual_plus2000"].mean() * 100,
                "positive_day": grp["actual_diff"].mean() > 0,
            })

        tmp = subset.copy()
        tmp["topn"] = topn
        pick_rows.append(tmp)

    daily = pd.DataFrame(daily_rows)
    pick_detail = pd.concat(
        pick_rows,
        ignore_index=True,
    )

    overall_rows = []

    for topn, grp in daily.groupby("topn"):
        ci_low, ci_high = bootstrap_daily_mean_ci(
            grp["avg_diff"]
        )

        overall_rows.append({
            "topn": int(topn),
            "evaluated_days": grp["target_date"].nunique(),
            "mean_daily_avg_diff": grp["avg_diff"].mean(),
            "median_daily_avg_diff": grp["avg_diff"].median(),
            "mean_win_rate": grp["win_rate"].mean(),
            "mean_plus1000_rate": grp["plus1000_rate"].mean(),
            "mean_plus2000_rate": grp["plus2000_rate"].mean(),
            "positive_day_rate": grp["positive_day"].mean() * 100,
            "total_diff": grp["total_diff"].sum(),
            "daily_mean_ci95_low": ci_low,
            "daily_mean_ci95_high": ci_high,
        })

    overall = pd.DataFrame(overall_rows).sort_values(
        ["mean_daily_avg_diff", "total_diff"],
        ascending=[False, False],
    ).reset_index(drop=True)

    header("OVERALL TOP-N COMPARISON")
    print(overall.to_string(index=False))

    # Pairwise daily comparison vs TOP10.
    ref = daily[
        daily["topn"] == 10
    ][["target_date", "avg_diff"]].rename(
        columns={"avg_diff": "top10_avg_diff"}
    )

    pairwise_rows = []

    for topn, grp in daily.groupby("topn"):
        merged = grp[
            ["target_date", "avg_diff"]
        ].merge(
            ref,
            on="target_date",
            how="inner",
        )

        change = (
            merged["avg_diff"]
            - merged["top10_avg_diff"]
        )

        pairwise_rows.append({
            "topn": int(topn),
            "days": len(merged),
            "better_than_top10_days": int((change > 0).sum()),
            "same_as_top10_days": int((change == 0).sum()),
            "worse_than_top10_days": int((change < 0).sum()),
            "mean_change_vs_top10": change.mean(),
            "total_change_vs_top10": change.sum() * topn,
        })

    pairwise = pd.DataFrame(pairwise_rows).sort_values(
        "topn"
    )

    header("PAIRWISE VS TOP10")
    print(pairwise.to_string(index=False))

    # Rank contribution table from original TOP10 picks.
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

    for c in (
        "win_rate",
        "plus1000_rate",
        "plus2000_rate",
    ):
        rank[c] *= 100

    header("RANK CONTRIBUTION")
    print(rank.to_string(index=False))

    # Robustness by removing best/worst day for each TOP-N.
    robust_rows = []

    for topn, grp in daily.groupby("topn"):
        best_idx = grp["avg_diff"].idxmax()
        worst_idx = grp["avg_diff"].idxmin()

        best_day = grp.loc[best_idx, "target_date"]
        worst_day = grp.loc[worst_idx, "target_date"]

        for scenario, use_grp in (
            ("ALL", grp),
            (
                "REMOVE_BEST_DAY",
                grp[grp["target_date"] != best_day],
            ),
            (
                "REMOVE_WORST_DAY",
                grp[grp["target_date"] != worst_day],
            ),
        ):
            robust_rows.append({
                "topn": int(topn),
                "scenario": scenario,
                "days": use_grp["target_date"].nunique(),
                "mean_daily_avg_diff": use_grp["avg_diff"].mean(),
                "positive_day_rate": use_grp["positive_day"].mean() * 100,
                "total_diff": use_grp["total_diff"].sum(),
                "best_day": best_day,
                "worst_day": worst_day,
            })

    robustness = pd.DataFrame(robust_rows)

    header("ROBUSTNESS")
    print(robustness.to_string(index=False))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_path = OUTPUT_DIR / "06_juggler_recent7_topn_daily.csv"
    picks_path = OUTPUT_DIR / "06_juggler_recent7_topn_picks.csv"
    overall_path = OUTPUT_DIR / "06_juggler_recent7_topn_overall.csv"
    pairwise_path = OUTPUT_DIR / "06_juggler_recent7_topn_vs_top10.csv"
    rank_path = OUTPUT_DIR / "06_juggler_recent7_rank_contribution.csv"
    robustness_path = OUTPUT_DIR / "06_juggler_recent7_topn_robustness.csv"

    daily.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    pick_detail.to_csv(
        picks_path,
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
        encoding="utf-8-sig",
    )

    rank.to_csv(
        rank_path,
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
        daily_path,
        picks_path,
        overall_path,
        pairwise_path,
        rank_path,
        robustness_path,
    ):
        print(p)

    print()
    print("JUGGLER RECENT7 TOP-N comparison complete.")
    print("No production model was changed.")
    print("No Maruhan Maebashi files were modified.")


if __name__ == "__main__":
    main()
