from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "53_Ver4_2_neighbor_ablation_oos"
    / "53_neighbor_ablation_daily.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
    / "analysis_31days_deep"
    / "54_Ver4_2_neighbor_pair_stability"
)

MODEL = "V4.2_C"
TOP_N = 10
MODE_CURRENT = "CURRENT_PM1"
MODE_NO = "NO_NEIGHBOR"


# ============================================================
# HELPERS
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV read failed: {path}")


def safe_pct(num: float, den: float) -> float:
    if den == 0:
        return np.nan
    return float(num / den * 100.0)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    header("54 - V4.2_C TOP10 Neighbor Pair / Stability Diagnostic")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    df = read_csv_flexible(INPUT_CSV)

    required = {
        "date",
        "split",
        "neighbor_mode",
        "model",
        "top_n",
        "avg_diff",
        "total_diff",
        "win_rate",
        "positive",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Required columns missing: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["top_n"] = pd.to_numeric(df["top_n"], errors="coerce")

    for col in (
        "avg_diff",
        "total_diff",
        "win_rate",
        "positive",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    target = df[
        (df["model"] == MODEL)
        & (df["top_n"] == TOP_N)
        & (df["neighbor_mode"].isin([MODE_CURRENT, MODE_NO]))
    ].copy()

    print(f"input rows          : {len(df):,}")
    print(f"target rows         : {len(target):,}")
    print(f"model / top_n       : {MODEL} / {TOP_N}")
    print(f"comparison          : {MODE_CURRENT} vs {MODE_NO}")

    if target.empty:
        raise RuntimeError("Target rows not found.")

    key = ["date", "split", "model", "top_n"]

    current = target[target["neighbor_mode"] == MODE_CURRENT].copy()
    no_neighbor = target[target["neighbor_mode"] == MODE_NO].copy()

    current = current.rename(
        columns={
            "avg_diff": "avg_diff_current",
            "total_diff": "total_diff_current",
            "win_rate": "win_rate_current",
            "positive": "positive_current",
        }
    )

    no_neighbor = no_neighbor.rename(
        columns={
            "avg_diff": "avg_diff_no_neighbor",
            "total_diff": "total_diff_no_neighbor",
            "win_rate": "win_rate_no_neighbor",
            "positive": "positive_no_neighbor",
        }
    )

    keep_current = key + [
        "avg_diff_current",
        "total_diff_current",
        "win_rate_current",
        "positive_current",
    ]

    keep_no = key + [
        "avg_diff_no_neighbor",
        "total_diff_no_neighbor",
        "win_rate_no_neighbor",
        "positive_no_neighbor",
    ]

    pair = current[keep_current].merge(
        no_neighbor[keep_no],
        on=key,
        how="inner",
        validate="one_to_one",
    )

    if pair.empty:
        raise RuntimeError("No paired daily rows found.")

    pair["diff_change_current_minus_no"] = (
        pair["total_diff_current"]
        - pair["total_diff_no_neighbor"]
    )

    pair["avg_change_current_minus_no"] = (
        pair["avg_diff_current"]
        - pair["avg_diff_no_neighbor"]
    )

    pair["win_rate_change_current_minus_no"] = (
        pair["win_rate_current"]
        - pair["win_rate_no_neighbor"]
    )

    pair["current_better"] = (
        pair["diff_change_current_minus_no"] > 0
    ).astype(int)

    pair["same"] = (
        pair["diff_change_current_minus_no"] == 0
    ).astype(int)

    pair["current_worse"] = (
        pair["diff_change_current_minus_no"] < 0
    ).astype(int)

    pair = pair.sort_values(["date", "split"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Overall paired statistics
    # --------------------------------------------------------

    delta = pair["diff_change_current_minus_no"]

    overall = pd.DataFrame(
        [{
            "model": MODEL,
            "top_n": TOP_N,
            "paired_days": len(pair),
            "current_better_days": int((delta > 0).sum()),
            "same_days": int((delta == 0).sum()),
            "current_worse_days": int((delta < 0).sum()),
            "current_better_rate_ex_ties": safe_pct(
                int((delta > 0).sum()),
                int((delta != 0).sum()),
            ),
            "mean_daily_advantage": float(delta.mean()),
            "median_daily_advantage": float(delta.median()),
            "total_advantage": float(delta.sum()),
            "std_daily_advantage": float(delta.std(ddof=1)),
            "min_daily_advantage": float(delta.min()),
            "max_daily_advantage": float(delta.max()),
            "current_total_diff": float(pair["total_diff_current"].sum()),
            "no_neighbor_total_diff": float(pair["total_diff_no_neighbor"].sum()),
            "current_positive_days": int((pair["total_diff_current"] > 0).sum()),
            "no_neighbor_positive_days": int((pair["total_diff_no_neighbor"] > 0).sum()),
        }]
    )

    header("OVERALL PAIRED RESULT")
    print(overall.to_string(index=False))

    # --------------------------------------------------------
    # Split stability
    # --------------------------------------------------------

    split_rows = []

    for split, g in pair.groupby("split", sort=False):
        d = g["diff_change_current_minus_no"]

        split_rows.append(
            {
                "split": split,
                "days": len(g),
                "current_better_days": int((d > 0).sum()),
                "same_days": int((d == 0).sum()),
                "current_worse_days": int((d < 0).sum()),
                "mean_advantage": float(d.mean()),
                "median_advantage": float(d.median()),
                "total_advantage": float(d.sum()),
                "current_total_diff": float(g["total_diff_current"].sum()),
                "no_neighbor_total_diff": float(g["total_diff_no_neighbor"].sum()),
            }
        )

    split_df = pd.DataFrame(split_rows)

    header("BY ROLLING SPLIT")
    print(split_df.to_string(index=False))

    # --------------------------------------------------------
    # Extreme days
    # --------------------------------------------------------

    best_days = pair.nlargest(
        min(10, len(pair)),
        "diff_change_current_minus_no",
    )[
        [
            "date",
            "split",
            "total_diff_current",
            "total_diff_no_neighbor",
            "diff_change_current_minus_no",
        ]
    ]

    worst_days = pair.nsmallest(
        min(10, len(pair)),
        "diff_change_current_minus_no",
    )[
        [
            "date",
            "split",
            "total_diff_current",
            "total_diff_no_neighbor",
            "diff_change_current_minus_no",
        ]
    ]

    header("TOP 10 DAYS FAVORING CURRENT_PM1")
    print(best_days.to_string(index=False))

    header("TOP 10 DAYS FAVORING NO_NEIGHBOR")
    print(worst_days.to_string(index=False))

    # --------------------------------------------------------
    # Leave-one-day-out robustness
    # --------------------------------------------------------

    loo_rows = []

    for idx, row in pair.iterrows():
        remaining = pair.drop(index=idx)
        d = remaining["diff_change_current_minus_no"]

        loo_rows.append(
            {
                "excluded_date": row["date"],
                "excluded_split": row["split"],
                "excluded_day_advantage": row["diff_change_current_minus_no"],
                "remaining_days": len(remaining),
                "remaining_total_advantage": float(d.sum()),
                "remaining_mean_advantage": float(d.mean()),
                "remaining_median_advantage": float(d.median()),
                "remaining_current_better_days": int((d > 0).sum()),
                "remaining_current_worse_days": int((d < 0).sum()),
            }
        )

    loo_df = pd.DataFrame(loo_rows)

    header("LEAVE-ONE-DAY-OUT ROBUSTNESS")
    print(
        loo_df.sort_values(
            "remaining_total_advantage"
        ).head(10).to_string(index=False)
    )

    # --------------------------------------------------------
    # Leave-one-split-out robustness
    # --------------------------------------------------------

    loso_rows = []

    for split in pair["split"].drop_duplicates():
        remaining = pair[pair["split"] != split].copy()
        d = remaining["diff_change_current_minus_no"]

        loso_rows.append(
            {
                "excluded_split": split,
                "remaining_days": len(remaining),
                "remaining_total_advantage": float(d.sum()),
                "remaining_mean_advantage": float(d.mean()),
                "remaining_median_advantage": float(d.median()),
                "remaining_current_better_days": int((d > 0).sum()),
                "remaining_current_worse_days": int((d < 0).sum()),
            }
        )

    loso_df = pd.DataFrame(loso_rows)

    header("LEAVE-ONE-SPLIT-OUT ROBUSTNESS")
    print(loso_df.to_string(index=False))

    # --------------------------------------------------------
    # Concentration of advantage
    # --------------------------------------------------------

    positive_delta = pair.loc[
        pair["diff_change_current_minus_no"] > 0,
        "diff_change_current_minus_no",
    ].sort_values(ascending=False)

    total_positive_adv = float(positive_delta.sum())

    top1_share = (
        safe_pct(float(positive_delta.head(1).sum()), total_positive_adv)
        if total_positive_adv > 0 else np.nan
    )
    top3_share = (
        safe_pct(float(positive_delta.head(3).sum()), total_positive_adv)
        if total_positive_adv > 0 else np.nan
    )
    top5_share = (
        safe_pct(float(positive_delta.head(5).sum()), total_positive_adv)
        if total_positive_adv > 0 else np.nan
    )

    concentration_df = pd.DataFrame(
        [{
            "positive_advantage_total": total_positive_adv,
            "top1_positive_share_pct": top1_share,
            "top3_positive_share_pct": top3_share,
            "top5_positive_share_pct": top5_share,
        }]
    )

    header("ADVANTAGE CONCENTRATION")
    print(concentration_df.to_string(index=False))

    # --------------------------------------------------------
    # Simple assessment
    # --------------------------------------------------------

    total_adv = float(delta.sum())
    median_adv = float(delta.median())
    better = int((delta > 0).sum())
    worse = int((delta < 0).sum())

    loo_all_positive = bool(
        (loo_df["remaining_total_advantage"] > 0).all()
    )

    loso_all_positive = bool(
        (loso_df["remaining_total_advantage"] > 0).all()
    )

    if (
        total_adv > 0
        and median_adv > 0
        and better > worse
        and loo_all_positive
        and loso_all_positive
    ):
        status = "ROBUST_CURRENT_PM1_ADVANTAGE"
    elif total_adv > 0 and loo_all_positive:
        status = "CURRENT_PM1_ADVANTAGE_BUT_MIXED_DAILY"
    elif total_adv > 0:
        status = "CURRENT_PM1_ADVANTAGE_CONCENTRATED_OR_UNSTABLE"
    else:
        status = "NO_STABLE_CURRENT_PM1_ADVANTAGE"

    assessment_df = pd.DataFrame(
        [{
            "status": status,
            "total_advantage": total_adv,
            "median_daily_advantage": median_adv,
            "current_better_days": better,
            "current_worse_days": worse,
            "leave_one_day_all_positive": loo_all_positive,
            "leave_one_split_all_positive": loso_all_positive,
            "note": (
                "Exploratory 39-day OOS diagnostic. "
                "Do not promote to production solely from this test."
            ),
        }]
    )

    header("ASSESSMENT")
    print(assessment_df.to_string(index=False))

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "54_neighbor_pair_daily.csv": pair,
        "54_neighbor_pair_overall.csv": overall,
        "54_neighbor_pair_by_split.csv": split_df,
        "54_neighbor_pair_leave_one_day_out.csv": loo_df,
        "54_neighbor_pair_leave_one_split_out.csv": loso_df,
        "54_neighbor_pair_concentration.csv": concentration_df,
        "54_neighbor_pair_assessment.csv": assessment_df,
    }

    header("FILES SAVED")

    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(path)

    print()
    print("54 neighbor pair / stability diagnostic complete.")


if __name__ == "__main__":
    main()
