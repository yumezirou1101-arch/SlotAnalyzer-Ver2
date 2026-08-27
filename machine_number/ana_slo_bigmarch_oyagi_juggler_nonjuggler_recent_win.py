from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

QUALITY_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
    / "analysis_31days_deep"
    / "01_data_quality"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
    / "analysis_31days_deep"
    / "05_juggler_nonjuggler_recent_win"
)

WINDOWS = (3, 5, 7, 10, 14)


def header(title: str) -> None:
    print()
    print("=" * 122)
    print(title)
    print("=" * 122)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Leakage-safe RECENT_WIN comparison split into "
            "JUGGLER and NON_JUGGLER."
        )
    )
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--min-history-days", type=int, default=14)
    return p.parse_args()


def find_integrated_file() -> Path:
    files = sorted(
        QUALITY_DIR.glob("01_bigmarch_oyagi_integrated_*.csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No integrated dataset found in {QUALITY_DIR}"
        )
    return files[-1]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    required = {"date", "machine_name", "machine_no", "G", "diff"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="raise").dt.normalize()
    x["machine_name"] = x["machine_name"].astype(str).str.strip()
    x["machine_no"] = pd.to_numeric(
        x["machine_no"], errors="raise"
    ).astype(int)
    x["G"] = pd.to_numeric(x["G"], errors="coerce")
    x["diff"] = pd.to_numeric(x["diff"], errors="coerce")

    if x["diff"].isna().any():
        raise RuntimeError("Missing/invalid diff found.")

    x["win"] = (x["diff"] > 0).astype(int)
    x["plus1000"] = (x["diff"] >= 1000).astype(int)
    x["plus2000"] = (x["diff"] >= 2000).astype(int)

    x["segment"] = np.where(
        x["machine_name"].str.contains("ジャグラー", na=False),
        "JUGGLER",
        "NON_JUGGLER",
    )

    return x.sort_values(
        ["date", "machine_no"]
    ).reset_index(drop=True)


def build_signal(
    history: pd.DataFrame,
    target_panel: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    rows = []

    for machine_no, grp in history.groupby("machine_no"):
        recent = grp.sort_values("date").tail(window)
        rows.append(
            {
                "machine_no": int(machine_no),
                "recent_win": recent["win"].mean(),
                "recent_n": recent["date"].nunique(),
            }
        )

    signal = pd.DataFrame(rows)

    out = target_panel[
        [
            "date",
            "machine_no",
            "machine_name",
            "segment",
            "diff",
            "win",
            "plus1000",
            "plus2000",
        ]
    ].merge(signal, on="machine_no", how="left")

    out["recent_n"] = out["recent_n"].fillna(0).astype(int)
    out["recent_win"] = out["recent_win"].fillna(-1.0)

    return out


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
    args = parse_args()

    if args.topn < 1:
        raise ValueError("--topn must be >= 1")
    if args.min_history_days < max(WINDOWS):
        raise ValueError(
            f"--min-history-days must be >= {max(WINDOWS)}"
        )

    source = find_integrated_file()
    data = load_data(source)

    dates = sorted(data["date"].drop_duplicates())
    eval_dates = dates[args.min_history_days:]

    if not eval_dates:
        raise RuntimeError("No evaluation dates.")

    header("Big March Takasaki Oyagi - JUGGLER / NON_JUGGLER RECENT_WIN")
    print(f"source                : {source}")
    print(f"records               : {len(data):,}")
    print(f"days                  : {len(dates)}")
    print(f"date range            : {dates[0].date()} to {dates[-1].date()}")
    print(f"windows               : {WINDOWS}")
    print(f"common warm-up        : {args.min_history_days}")
    print(f"evaluation days       : {len(eval_dates)}")
    print(f"evaluation range      : {eval_dates[0].date()} to {eval_dates[-1].date()}")
    print(f"top N per segment     : {args.topn}")

    segment_counts = (
        data.groupby(["date", "segment"])
        .size()
        .unstack(fill_value=0)
    )
    header("DAILY SEGMENT COUNTS")
    print(segment_counts.to_string())

    daily_rows = []
    pick_rows = []

    for target_date in eval_dates:
        history_all = data[data["date"] < target_date].copy()
        target_all = data[data["date"] == target_date].copy()

        header(f"TARGET {target_date.date()}")

        for segment in ("JUGGLER", "NON_JUGGLER"):
            history = history_all[
                history_all["segment"] == segment
            ].copy()
            target_panel = target_all[
                target_all["segment"] == segment
            ].copy()

            if target_panel.empty:
                print(f"{segment:<12} no target machines")
                continue

            for window in WINDOWS:
                model = f"{segment}_RECENT{window}_WIN"

                features = build_signal(
                    history,
                    target_panel,
                    window,
                )

                ranked = (
                    features.sort_values(
                        ["recent_win", "recent_n", "machine_no"],
                        ascending=[False, False, True],
                    )
                    .head(args.topn)
                    .copy()
                )

                ranked["prediction_rank"] = range(
                    1, len(ranked) + 1
                )

                avg_diff = ranked["diff"].mean()
                total_diff = ranked["diff"].sum()
                win_rate = ranked["win"].mean() * 100
                plus1000_rate = ranked["plus1000"].mean() * 100
                plus2000_rate = ranked["plus2000"].mean() * 100

                daily_rows.append(
                    {
                        "target_date": target_date.date(),
                        "segment": segment,
                        "model": model,
                        "window": window,
                        "selected_n": len(ranked),
                        "avg_diff": avg_diff,
                        "median_diff": ranked["diff"].median(),
                        "total_diff": total_diff,
                        "win_rate": win_rate,
                        "plus1000_rate": plus1000_rate,
                        "plus2000_rate": plus2000_rate,
                        "positive_day": avg_diff > 0,
                    }
                )

                for _, row in ranked.iterrows():
                    pick_rows.append(
                        {
                            "target_date": target_date.date(),
                            "segment": segment,
                            "model": model,
                            "window": window,
                            "prediction_rank": int(
                                row["prediction_rank"]
                            ),
                            "machine_no": int(row["machine_no"]),
                            "machine_name": row["machine_name"],
                            "recent_win": float(row["recent_win"]),
                            "recent_n": int(row["recent_n"]),
                            "actual_diff": float(row["diff"]),
                            "actual_win": int(row["win"]),
                            "actual_plus1000": int(
                                row["plus1000"]
                            ),
                            "actual_plus2000": int(
                                row["plus2000"]
                            ),
                        }
                    )

                print(
                    f"{model:<28} "
                    f"avg={avg_diff:>8.1f} "
                    f"win={win_rate:>5.1f}% "
                    f"+1000={plus1000_rate:>5.1f}% "
                    f"+2000={plus2000_rate:>5.1f}% "
                    f"total={total_diff:>9.0f}"
                )

    daily = pd.DataFrame(daily_rows)
    picks = pd.DataFrame(pick_rows)

    overall_rows = []

    for (segment, model), grp in daily.groupby(
        ["segment", "model"]
    ):
        ci_low, ci_high = bootstrap_daily_mean_ci(
            grp["avg_diff"]
        )

        overall_rows.append(
            {
                "segment": segment,
                "model": model,
                "window": int(grp["window"].iloc[0]),
                "evaluated_days": grp["target_date"].nunique(),
                "mean_daily_avg_diff": grp["avg_diff"].mean(),
                "median_daily_avg_diff": grp["avg_diff"].median(),
                "mean_win_rate": grp["win_rate"].mean(),
                "mean_plus1000_rate": grp["plus1000_rate"].mean(),
                "mean_plus2000_rate": grp["plus2000_rate"].mean(),
                "positive_day_rate": (
                    grp["positive_day"].mean() * 100
                ),
                "total_diff": grp["total_diff"].sum(),
                "daily_mean_ci95_low": ci_low,
                "daily_mean_ci95_high": ci_high,
            }
        )

    overall = pd.DataFrame(overall_rows).sort_values(
        ["segment", "mean_daily_avg_diff"],
        ascending=[True, False],
    )

    header("OVERALL SEGMENT COMPARISON")
    print(overall.to_string(index=False))

    best_rows = []
    for segment, grp in overall.groupby("segment"):
        best = grp.sort_values(
            ["mean_daily_avg_diff", "total_diff"],
            ascending=[False, False],
        ).iloc[0]

        best_rows.append(best.to_dict())

    best_models = pd.DataFrame(best_rows)

    header("BEST WINDOW BY SEGMENT")
    print(best_models.to_string(index=False))

    # Pairwise comparisons against RECENT7 within each segment.
    pairwise_rows = []

    for segment in ("JUGGLER", "NON_JUGGLER"):
        seg_daily = daily[daily["segment"] == segment]

        reference = seg_daily[
            seg_daily["window"] == 7
        ][["target_date", "avg_diff"]].rename(
            columns={"avg_diff": "recent7_avg_diff"}
        )

        for model, grp in seg_daily.groupby("model"):
            merged = grp[
                ["target_date", "avg_diff"]
            ].merge(
                reference,
                on="target_date",
                how="inner",
            )

            change = (
                merged["avg_diff"]
                - merged["recent7_avg_diff"]
            )

            pairwise_rows.append(
                {
                    "segment": segment,
                    "model": model,
                    "days": len(merged),
                    "better_than_recent7_days": int(
                        (change > 0).sum()
                    ),
                    "same_as_recent7_days": int(
                        (change == 0).sum()
                    ),
                    "worse_than_recent7_days": int(
                        (change < 0).sum()
                    ),
                    "mean_change_vs_recent7": change.mean(),
                    "total_change_vs_recent7": (
                        change.sum() * args.topn
                    ),
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)

    header("PAIRWISE VS RECENT7 WITHIN SEGMENT")
    print(pairwise.to_string(index=False))

    # Rank diagnostics for each segment's best model.
    best_rank_rows = []

    for _, best in best_models.iterrows():
        model = best["model"]
        grp = picks[picks["model"] == model]

        rank = (
            grp.groupby("prediction_rank")
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

        rank["segment"] = best["segment"]
        rank["model"] = model

        for c in (
            "win_rate",
            "plus1000_rate",
            "plus2000_rate",
        ):
            rank[c] *= 100

        best_rank_rows.append(rank)

    best_rank = pd.concat(
        best_rank_rows,
        ignore_index=True,
    )

    header("RANK DIAGNOSTICS FOR BEST MODELS")
    print(best_rank.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_path = OUTPUT_DIR / "05_segment_recent_win_daily.csv"
    picks_path = OUTPUT_DIR / "05_segment_recent_win_picks.csv"
    overall_path = OUTPUT_DIR / "05_segment_recent_win_overall.csv"
    best_path = OUTPUT_DIR / "05_segment_recent_win_best_models.csv"
    pairwise_path = OUTPUT_DIR / "05_segment_recent_win_vs_recent7.csv"
    rank_path = OUTPUT_DIR / "05_segment_recent_win_best_rank.csv"
    counts_path = OUTPUT_DIR / "05_segment_daily_machine_counts.csv"

    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    picks.to_csv(picks_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    best_models.to_csv(best_path, index=False, encoding="utf-8-sig")
    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
    best_rank.to_csv(rank_path, index=False, encoding="utf-8-sig")
    segment_counts.to_csv(counts_path, encoding="utf-8-sig")

    header("FILES SAVED")
    for p in (
        daily_path,
        picks_path,
        overall_path,
        best_path,
        pairwise_path,
        rank_path,
        counts_path,
    ):
        print(p)

    print()
    print("Segment RECENT_WIN comparison complete.")
    print("No production model was changed.")
    print("No Maruhan Maebashi files were modified.")


if __name__ == "__main__":
    main()
