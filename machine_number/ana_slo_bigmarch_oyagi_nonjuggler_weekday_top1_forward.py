from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


# ============================================================
# Big March Takasaki Oyagi
# NON_JUGGLER WEEKDAY_AVG TOP1 Frozen Forward Evaluation
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

MACHINE_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
)

QUALITY_DIR = (
    MACHINE_DIR
    / "analysis_31days_deep"
    / "01_data_quality"
)

OUTPUT_DIR = (
    MACHINE_DIR
    / "analysis_31days_deep"
    / "11_nonjuggler_weekday_top1_forward"
)

DEVELOPMENT_END = pd.Timestamp("2026-08-26")
FORWARD_START = pd.Timestamp("2026-08-27")

MODEL_NAME = "NON_JUGGLER_WEEKDAY_AVG_TOP1_FROZEN"
TOPN = 1
MIN_REVIEW_DAYS = 21

JUGGLER_KEYWORD = "ジャグラー"

AUTOMATIC_PROMOTION = False


def header(title: str) -> None:
    print()
    print("=" * 124)
    print(title)
    print("=" * 124)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen NON_JUGGLER WEEKDAY_AVG Top1 "
            "forward evaluation for Big March Takasaki Oyagi."
        )
    )

    parser.add_argument(
        "--min-review-days",
        type=int,
        default=MIN_REVIEW_DAYS,
        help=(
            "Minimum forward days before review. "
            f"Default: {MIN_REVIEW_DAYS}."
        ),
    )

    return parser.parse_args()


def find_development_file() -> Path:
    files = sorted(
        QUALITY_DIR.glob(
            "01_bigmarch_oyagi_integrated_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No integrated development dataset found in:\n{QUALITY_DIR}"
        )

    candidates = []

    for path in files:
        try:
            df = pd.read_csv(
                path,
                encoding="utf-8-sig",
                usecols=["date"],
            )

            dates = pd.to_datetime(
                df["date"],
                errors="raise",
            ).dt.normalize()

            if dates.max() == DEVELOPMENT_END:
                candidates.append(path)

        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            "No development dataset ending exactly on "
            f"{DEVELOPMENT_END.date()} was found."
        )

    return candidates[-1]


def normalize_frame(
    df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:

    x = df.copy()

    x.columns = [
        str(column).strip()
        for column in x.columns
    ]

    required = {
        "date",
        "machine_name",
        "machine_no",
        "G",
        "diff",
    }

    missing = sorted(
        required - set(x.columns)
    )

    if missing:
        raise RuntimeError(
            f"{source_name}: missing required columns: {missing}"
        )

    x["date"] = pd.to_datetime(
        x["date"],
        errors="raise",
    ).dt.normalize()

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="raise",
    ).astype(int)

    x["G"] = pd.to_numeric(
        x["G"],
        errors="coerce",
    )

    x["diff"] = pd.to_numeric(
        x["diff"],
        errors="raise",
    )

    duplicate_mask = x.duplicated(
        subset=[
            "date",
            "machine_no",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        rows = x.loc[
            duplicate_mask,
            [
                "date",
                "machine_no",
                "machine_name",
            ],
        ]

        raise RuntimeError(
            f"{source_name}: duplicate date-machine rows found:\n"
            + rows.head(20).to_string(index=False)
        )

    x["win"] = (
        x["diff"] > 0
    ).astype(int)

    x["plus1000"] = (
        x["diff"] >= 1000
    ).astype(int)

    x["plus2000"] = (
        x["diff"] >= 2000
    ).astype(int)

    x["weekday"] = (
        x["date"].dt.weekday
    )

    x["is_juggler"] = (
        x["machine_name"]
        .str.contains(
            JUGGLER_KEYWORD,
            na=False,
        )
    )

    return (
        x.sort_values(
            [
                "date",
                "machine_no",
            ]
        )
        .reset_index(drop=True)
    )


def load_development_data(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    x = normalize_frame(
        df,
        path.name,
    )

    latest = x["date"].max()

    if latest != DEVELOPMENT_END:
        raise RuntimeError(
            "Development dataset is not frozen correctly. "
            f"Latest date={latest.date()}, "
            f"expected={DEVELOPMENT_END.date()}."
        )

    if (x["date"] > DEVELOPMENT_END).any():
        raise RuntimeError(
            "Development dataset contains rows after DEVELOPMENT_END."
        )

    return x


def discover_forward_files():
    files = sorted(
        MACHINE_DIR.glob(
            "ana_slo_bigmarch_oyagi_*.csv"
        )
    )

    rows = []

    for path in files:
        stem = path.stem

        date_text = stem.replace(
            "ana_slo_bigmarch_oyagi_",
            "",
        )

        if len(date_text) != 8:
            continue

        try:
            file_date = pd.to_datetime(
                date_text,
                format="%Y%m%d",
                errors="raise",
            ).normalize()

        except Exception:
            continue

        if file_date < FORWARD_START:
            continue

        rows.append(
            (
                file_date,
                path,
            )
        )

    rows.sort(
        key=lambda item: item[0]
    )

    return rows


def load_forward_day(
    path: Path,
    expected_date: pd.Timestamp,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    x = normalize_frame(
        df,
        path.name,
    )

    unique_dates = sorted(
        x["date"].drop_duplicates()
    )

    if len(unique_dates) != 1:
        raise RuntimeError(
            f"{path.name}: expected exactly one date, "
            f"found {len(unique_dates)}."
        )

    actual_date = unique_dates[0]

    if actual_date != expected_date:
        raise RuntimeError(
            f"{path.name}: CSV date mismatch. "
            f"filename={expected_date.date()}, "
            f"CSV={actual_date.date()}."
        )

    return x


def build_weekday_signal(
    history: pd.DataFrame,
    target_panel: pd.DataFrame,
) -> pd.DataFrame:

    target_date = target_panel[
        "date"
    ].iloc[0]

    target_weekday = int(
        target_panel[
            "weekday"
        ].iloc[0]
    )

    if (history["date"] >= target_date).any():
        raise RuntimeError(
            "Leakage detected: history contains target/future date."
        )

    history_nonjuggler = history[
        ~history["is_juggler"]
    ].copy()

    target_nonjuggler = target_panel[
        ~target_panel["is_juggler"]
    ].copy()

    if target_nonjuggler.empty:
        raise RuntimeError(
            f"No NON_JUGGLER machines found for {target_date.date()}."
        )

    weekday_history = history_nonjuggler[
        history_nonjuggler["weekday"]
        == target_weekday
    ].copy()

    weekday_stats = (
        weekday_history
        .groupby("machine_no")
        .agg(
            weekday_avg=(
                "diff",
                "mean",
            ),
            weekday_n=(
                "date",
                "nunique",
            ),
        )
        .reset_index()
    )

    ranked = (
        target_nonjuggler[
            [
                "date",
                "machine_no",
                "machine_name",
                "diff",
                "win",
                "plus1000",
                "plus2000",
            ]
        ]
        .merge(
            weekday_stats,
            on="machine_no",
            how="left",
        )
    )

    median_score = ranked[
        "weekday_avg"
    ].median()

    if pd.isna(median_score):
        median_score = 0.0

    ranked["weekday_avg"] = (
        ranked["weekday_avg"]
        .fillna(median_score)
    )

    ranked["weekday_n"] = (
        ranked["weekday_n"]
        .fillna(0)
        .astype(int)
    )

    ranked = (
        ranked
        .sort_values(
            [
                "weekday_avg",
                "machine_no",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    ranked[
        "prediction_rank"
    ] = range(
        1,
        len(ranked) + 1,
    )

    return ranked


def bootstrap_ci(
    values,
    n_boot: int = 10000,
    seed: int = 20260829,
):
    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    samples = rng.choice(
        arr,
        size=(
            n_boot,
            len(arr),
        ),
        replace=True,
    )

    means = samples.mean(
        axis=1
    )

    return (
        float(
            np.percentile(
                means,
                2.5,
            )
        ),
        float(
            np.percentile(
                means,
                97.5,
            )
        ),
    )


def main():
    args = parse_args()

    if args.min_review_days < 1:
        raise ValueError(
            "--min-review-days must be >= 1"
        )

    development_file = (
        find_development_file()
    )

    development = (
        load_development_data(
            development_file
        )
    )

    forward_files = (
        discover_forward_files()
    )

    header(
        "Big March Takasaki Oyagi - "
        "NON_JUGGLER WEEKDAY_AVG TOP1 Frozen Forward"
    )

    print(
        f"model                 : {MODEL_NAME}"
    )
    print(
        f"development file      : {development_file}"
    )
    print(
        f"development end       : {DEVELOPMENT_END.date()}"
    )
    print(
        f"forward start         : {FORWARD_START.date()}"
    )
    print(
        f"top N                 : {TOPN}"
    )
    print(
        f"min review days       : {args.min_review_days}"
    )
    print(
        f"automatic promotion   : {AUTOMATIC_PROMOTION}"
    )
    print(
        f"forward files found   : {len(forward_files)}"
    )

    if not forward_files:
        print()
        print(
            "No forward files are available yet."
        )
        return

    history = development.copy()

    daily_rows = []
    pick_rows = []

    previous_forward_date = None

    for target_date, path in forward_files:

        if target_date <= DEVELOPMENT_END:
            continue

        if previous_forward_date is not None:
            if target_date <= previous_forward_date:
                raise RuntimeError(
                    "Forward dates are not strictly increasing."
                )

        target_panel = load_forward_day(
            path,
            target_date,
        )

        if (history["date"] >= target_date).any():
            raise RuntimeError(
                f"Leakage detected before evaluating {target_date.date()}."
            )

        ranked = build_weekday_signal(
            history,
            target_panel,
        )

        selected = (
            ranked
            .head(TOPN)
            .copy()
        )

        if selected.empty:
            raise RuntimeError(
                f"No prediction generated for {target_date.date()}."
            )

        pick = selected.iloc[0]

        avg_diff = float(
            selected["diff"].mean()
        )

        total_diff = float(
            selected["diff"].sum()
        )

        win_rate = float(
            selected["win"].mean()
            * 100
        )

        plus1000_rate = float(
            selected["plus1000"].mean()
            * 100
        )

        plus2000_rate = float(
            selected["plus2000"].mean()
            * 100
        )

        daily_rows.append(
            {
                "target_date": (
                    target_date.date()
                ),
                "model": MODEL_NAME,
                "topn": TOPN,
                "selected_n": (
                    len(selected)
                ),
                "avg_diff": (
                    avg_diff
                ),
                "total_diff": (
                    total_diff
                ),
                "win_rate": (
                    win_rate
                ),
                "plus1000_rate": (
                    plus1000_rate
                ),
                "plus2000_rate": (
                    plus2000_rate
                ),
                "positive_day": (
                    avg_diff > 0
                ),
            }
        )

        for _, row in selected.iterrows():

            pick_rows.append(
                {
                    "target_date": (
                        target_date.date()
                    ),
                    "model": (
                        MODEL_NAME
                    ),
                    "prediction_rank": int(
                        row[
                            "prediction_rank"
                        ]
                    ),
                    "machine_no": int(
                        row[
                            "machine_no"
                        ]
                    ),
                    "machine_name": (
                        row[
                            "machine_name"
                        ]
                    ),
                    "weekday_avg_score": float(
                        row[
                            "weekday_avg"
                        ]
                    ),
                    "weekday_history_n": int(
                        row[
                            "weekday_n"
                        ]
                    ),
                    "actual_diff": float(
                        row[
                            "diff"
                        ]
                    ),
                    "actual_win": int(
                        row[
                            "win"
                        ]
                    ),
                    "actual_plus1000": int(
                        row[
                            "plus1000"
                        ]
                    ),
                    "actual_plus2000": int(
                        row[
                            "plus2000"
                        ]
                    ),
                }
            )

        header(
            f"FORWARD {target_date.date()}"
        )

        print(
            f"rank 1 machine        : "
            f"{int(pick['machine_no'])} "
            f"{pick['machine_name']}"
        )
        print(
            f"weekday avg score     : "
            f"{float(pick['weekday_avg']):.3f}"
        )
        print(
            f"weekday history n     : "
            f"{int(pick['weekday_n'])}"
        )
        print(
            f"actual diff           : "
            f"{float(pick['diff']):+.0f}"
        )
        print(
            f"win                   : "
            f"{bool(pick['win'])}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Add target-day actual only AFTER prediction/evaluation.
        # This preserves leakage-safe sequential forward testing.
        # ----------------------------------------------------
        history = pd.concat(
            [
                history,
                target_panel,
            ],
            ignore_index=True,
        )

        history = (
            history
            .sort_values(
                [
                    "date",
                    "machine_no",
                ]
            )
            .reset_index(drop=True)
        )

        previous_forward_date = (
            target_date
        )

    daily = pd.DataFrame(
        daily_rows
    )

    picks = pd.DataFrame(
        pick_rows
    )

    if daily.empty:
        raise RuntimeError(
            "No forward evaluation rows were generated."
        )

    ci_low, ci_high = bootstrap_ci(
        daily["avg_diff"]
    )

    available_days = int(
        daily["target_date"]
        .nunique()
    )

    if available_days >= args.min_review_days:
        status = "READY_FOR_REVIEW"
    else:
        status = "ACCUMULATING_FORWARD_DATA"

    summary = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "development_end": (
                    DEVELOPMENT_END.date()
                ),
                "forward_start": (
                    FORWARD_START.date()
                ),
                "forward_days": (
                    available_days
                ),
                "min_review_days": (
                    args.min_review_days
                ),
                "status": status,
                "automatic_promotion": (
                    AUTOMATIC_PROMOTION
                ),
                "mean_daily_avg_diff": (
                    daily[
                        "avg_diff"
                    ].mean()
                ),
                "median_daily_avg_diff": (
                    daily[
                        "avg_diff"
                    ].median()
                ),
                "total_diff": (
                    daily[
                        "total_diff"
                    ].sum()
                ),
                "win_rate": (
                    daily[
                        "positive_day"
                    ].mean()
                    * 100
                ),
                "plus1000_rate": (
                    picks[
                        "actual_plus1000"
                    ].mean()
                    * 100
                ),
                "plus2000_rate": (
                    picks[
                        "actual_plus2000"
                    ].mean()
                    * 100
                ),
                "ci95_low": (
                    ci_low
                ),
                "ci95_high": (
                    ci_high
                ),
            }
        ]
    )

    header(
        "FORWARD SUMMARY"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    header(
        "FORWARD PICKS"
    )

    print(
        picks.to_string(
            index=False
        )
    )

    print()
    print(
        f"STATUS                : {status}"
    )
    print(
        f"AVAILABLE DAYS        : {available_days}"
    )
    print(
        f"MIN REVIEW DAYS       : {args.min_review_days}"
    )
    print(
        f"AUTO PROMOTION        : {AUTOMATIC_PROMOTION}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_path = (
        OUTPUT_DIR
        / "11_nonjuggler_weekday_top1_forward_daily.csv"
    )

    picks_path = (
        OUTPUT_DIR
        / "11_nonjuggler_weekday_top1_forward_picks.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "11_nonjuggler_weekday_top1_forward_summary.csv"
    )

    daily.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    picks.to_csv(
        picks_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    print(
        daily_path
    )
    print(
        picks_path
    )
    print(
        summary_path
    )

    print()
    print(
        "Forward evaluation complete."
    )
    print(
        "Frozen model rule was not changed."
    )
    print(
        "No automatic promotion was performed."
    )


if __name__ == "__main__":
    main()