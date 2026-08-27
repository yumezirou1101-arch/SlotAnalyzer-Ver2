from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np


# ============================================================
# 08 - Big March Takasaki Oyagi
# Frozen JUGGLER_RECENT7_WIN Top3 Forward Test
# ============================================================
#
# Frozen development period:
#   through 2026-08-26
#
# Forward period:
#   2026-08-27 onward
#
# Frozen production-candidate rule:
#   - JUGGLER only
#   - score = win rate over previous 7 observed days
#   - rank by recent7_win desc
#   - tie break: recent7_n desc, machine_no asc
#   - select Top3
#
# IMPORTANT
# ---------
# - No parameter tuning is performed here.
# - Target-day data is NEVER used to build target-day features.
# - Forward results are accumulated only.
# - No automatic promotion occurs.
# ============================================================


PROJECT_ROOT = Path(r"C:\Users\user\Desktop\Documents\SlotAnalyzer")

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
)

QUALITY_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "01_data_quality"
)

OUTPUT_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "08_juggler_recent7_top3_forward"
)

DEVELOPMENT_END = pd.Timestamp("2026-08-26")
FORWARD_START = pd.Timestamp("2026-08-27")

WINDOW = 7
TOPN = 3
MIN_REVIEW_DAYS = 21

MODEL_NAME = "JUGGLER_RECENT7_WIN_TOP3_FROZEN"

DAILY_FILE_RE = re.compile(
    r"^ana_slo_bigmarch_oyagi_(\d{8})\.csv$",
    re.IGNORECASE,
)


def header(title: str) -> None:
    print()
    print("=" * 122)
    print(title)
    print("=" * 122)


def find_locked_integrated_file() -> Path:
    files = sorted(
        QUALITY_DIR.glob(
            "01_bigmarch_oyagi_integrated_*_20260826.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Locked integrated development dataset ending 2026-08-26 "
            f"was not found in:\n{QUALITY_DIR}"
        )

    return files[-1]


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]

    required = {
        "date",
        "machine_name",
        "machine_no",
        "G",
        "diff",
    }

    missing = sorted(required - set(x.columns))

    if missing:
        raise RuntimeError(
            f"{source}: missing required columns: {missing}"
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
        errors="coerce",
    )

    if x["diff"].isna().any():
        raise RuntimeError(
            f"{source}: invalid or missing diff exists."
        )

    if x["G"].isna().any():
        raise RuntimeError(
            f"{source}: invalid or missing G exists."
        )

    if (x["G"] < 0).any():
        raise RuntimeError(
            f"{source}: negative G exists."
        )

    duplicated = x.duplicated(
        subset=["date", "machine_no"],
        keep=False,
    )

    if duplicated.any():
        raise RuntimeError(
            f"{source}: duplicate date-machine rows exist."
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

    x["is_juggler"] = (
        x["machine_name"]
        .str.contains(
            "ジャグラー",
            na=False,
        )
    )

    return x


def load_development() -> tuple[pd.DataFrame, Path]:
    path = find_locked_integrated_file()

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    x = normalize(
        df,
        source=path.name,
    )

    x = x[
        x["date"] <= DEVELOPMENT_END
    ].copy()

    if x.empty:
        raise RuntimeError(
            "Locked development dataset is empty."
        )

    return x, path


def discover_forward_daily_files():
    found = []

    for path in DATA_DIR.glob(
        "ana_slo_bigmarch_oyagi_*.csv"
    ):
        m = DAILY_FILE_RE.fullmatch(
            path.name
        )

        if not m:
            continue

        date = pd.to_datetime(
            m.group(1),
            format="%Y%m%d",
            errors="raise",
        ).normalize()

        if date >= FORWARD_START:
            found.append(
                (
                    date,
                    path,
                )
            )

    return sorted(
        found,
        key=lambda x: x[0],
    )


def quality_row(
    date: pd.Timestamp,
    path: Path,
    df: pd.DataFrame,
) -> dict:

    rows = len(df)
    machines = int(
        df["machine_no"].nunique()
    )

    duplicates = int(
        df["machine_no"]
        .duplicated(
            keep=False
        )
        .sum()
    )

    missing_name = int(
        df["machine_name"]
        .astype(str)
        .str.strip()
        .isin(
            [
                "",
                "nan",
                "None",
            ]
        )
        .sum()
    )

    missing_diff = int(
        df["diff"].isna().sum()
    )

    internal_dates = (
        df["date"]
        .drop_duplicates()
        .tolist()
    )

    internal_date_ok = (
        len(internal_dates) == 1
        and internal_dates[0] == date
    )

    basic_ok = all(
        (
            rows >= 200,
            machines == rows,
            duplicates == 0,
            missing_name == 0,
            missing_diff == 0,
            internal_date_ok,
        )
    )

    return {
        "date": date.date(),
        "file": path.name,
        "rows": rows,
        "machines": machines,
        "duplicates": duplicates,
        "missing_name": missing_name,
        "missing_diff": missing_diff,
        "internal_date_ok": internal_date_ok,
        "basic_ok": basic_ok,
    }


def build_recent7_signal(
    history: pd.DataFrame,
    target_panel: pd.DataFrame,
) -> pd.DataFrame:

    history_j = history[
        history["is_juggler"]
    ].copy()

    target_j = target_panel[
        target_panel["is_juggler"]
    ].copy()

    rows = []

    for machine_no, grp in history_j.groupby(
        "machine_no"
    ):
        recent = (
            grp.sort_values("date")
            .tail(WINDOW)
        )

        rows.append(
            {
                "machine_no": int(machine_no),
                "recent7_win": recent["win"].mean(),
                "recent7_n": recent["date"].nunique(),
                "recent7_avg_diff": recent["diff"].mean(),
            }
        )

    signal = pd.DataFrame(rows)

    if signal.empty:
        raise RuntimeError(
            "No historical Juggler signal could be built."
        )

    features = target_j[
        [
            "date",
            "machine_no",
            "machine_name",
            "diff",
            "win",
            "plus1000",
            "plus2000",
        ]
    ].merge(
        signal,
        on="machine_no",
        how="left",
    )

    features["recent7_n"] = (
        features["recent7_n"]
        .fillna(0)
        .astype(int)
    )

    features["recent7_win"] = (
        features["recent7_win"]
        .fillna(-1.0)
    )

    features["recent7_avg_diff"] = (
        features["recent7_avg_diff"]
        .fillna(0.0)
    )

    return features


def evaluate_forward(
    development: pd.DataFrame,
    forward_files,
):
    quality_rows = []
    daily_rows = []
    pick_rows = []

    history = development.copy()

    for target_date, path in forward_files:

        raw = pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

        target_panel = normalize(
            raw,
            source=path.name,
        )

        q = quality_row(
            target_date,
            path,
            target_panel,
        )

        quality_rows.append(
            q
        )

        if not q["basic_ok"]:
            print(
                f"{target_date.date()} quality failed -> skipped"
            )
            continue

        # Safety: use history strictly BEFORE target day.
        history_for_target = history[
            history["date"] < target_date
        ].copy()

        features = build_recent7_signal(
            history_for_target,
            target_panel,
        )

        ranked = (
            features.sort_values(
                [
                    "recent7_win",
                    "recent7_n",
                    "machine_no",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .head(TOPN)
            .copy()
        )

        ranked["prediction_rank"] = range(
            1,
            len(ranked) + 1,
        )

        avg_diff = ranked["diff"].mean()
        total_diff = ranked["diff"].sum()

        daily_rows.append(
            {
                "target_date": target_date.date(),
                "model": MODEL_NAME,
                "selected_n": len(ranked),
                "avg_diff": avg_diff,
                "median_diff": ranked["diff"].median(),
                "total_diff": total_diff,
                "win_rate": ranked["win"].mean() * 100,
                "plus1000_rate": ranked["plus1000"].mean() * 100,
                "plus2000_rate": ranked["plus2000"].mean() * 100,
                "positive_day": avg_diff > 0,
            }
        )

        for _, row in ranked.iterrows():

            pick_rows.append(
                {
                    "target_date": target_date.date(),
                    "model": MODEL_NAME,
                    "prediction_rank": int(
                        row["prediction_rank"]
                    ),
                    "machine_no": int(
                        row["machine_no"]
                    ),
                    "machine_name": row[
                        "machine_name"
                    ],
                    "recent7_win": float(
                        row["recent7_win"]
                    ),
                    "recent7_n": int(
                        row["recent7_n"]
                    ),
                    "recent7_avg_diff": float(
                        row["recent7_avg_diff"]
                    ),
                    "actual_diff": float(
                        row["diff"]
                    ),
                    "actual_win": int(
                        row["win"]
                    ),
                    "actual_plus1000": int(
                        row["plus1000"]
                    ),
                    "actual_plus2000": int(
                        row["plus2000"]
                    ),
                }
            )

        print(
            f"{target_date.date()} "
            f"avg={avg_diff:>8.1f} "
            f"total={total_diff:>8.0f} "
            f"win={ranked['win'].mean()*100:>5.1f}%"
        )

        # After evaluation, today's data becomes available
        # for subsequent forward dates.
        history = pd.concat(
            [
                history,
                target_panel,
            ],
            ignore_index=True,
        )

    return (
        pd.DataFrame(quality_rows),
        pd.DataFrame(daily_rows),
        pd.DataFrame(pick_rows),
    )


def build_status(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    available_days = (
        daily["target_date"].nunique()
        if not daily.empty
        else 0
    )

    if available_days < MIN_REVIEW_DAYS:
        status = "ACCUMULATING_FORWARD_DATA"
    else:
        status = "READY_FOR_REVIEW"

    return pd.DataFrame(
        [
            {
                "status": status,
                "development_end": DEVELOPMENT_END.date(),
                "forward_start": FORWARD_START.date(),
                "available_forward_days": available_days,
                "min_review_days": MIN_REVIEW_DAYS,
                "candidate_model": MODEL_NAME,
                "window": WINDOW,
                "topn": TOPN,
                "automatic_promotion": False,
            }
        ]
    )


def build_overall(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    if daily.empty:
        return pd.DataFrame(
            [
                {
                    "model": MODEL_NAME,
                    "forward_days": 0,
                    "avg_diff": np.nan,
                    "median_daily_avg_diff": np.nan,
                    "win_rate": np.nan,
                    "plus1000_rate": np.nan,
                    "plus2000_rate": np.nan,
                    "positive_days": np.nan,
                    "total_diff": 0.0,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "forward_days": daily[
                    "target_date"
                ].nunique(),
                "avg_diff": daily[
                    "avg_diff"
                ].mean(),
                "median_daily_avg_diff": daily[
                    "avg_diff"
                ].median(),
                "win_rate": daily[
                    "win_rate"
                ].mean(),
                "plus1000_rate": daily[
                    "plus1000_rate"
                ].mean(),
                "plus2000_rate": daily[
                    "plus2000_rate"
                ].mean(),
                "positive_days": daily[
                    "positive_day"
                ].mean()
                * 100,
                "total_diff": daily[
                    "total_diff"
                ].sum(),
            }
        ]
    )


def main() -> None:

    header(
        "08 - Big March Takasaki Oyagi "
        "Frozen JUGGLER RECENT7 WIN Top3 Forward Test"
    )

    development, locked_path = load_development()

    forward_files = discover_forward_daily_files()

    print(
        f"locked development    : {locked_path}"
    )
    print(
        f"development end       : {DEVELOPMENT_END.date()}"
    )
    print(
        f"forward start         : {FORWARD_START.date()}"
    )
    print(
        f"candidate model       : {MODEL_NAME}"
    )
    print(
        f"window                : {WINDOW}"
    )
    print(
        f"top N                 : {TOPN}"
    )
    print(
        f"min review days       : {MIN_REVIEW_DAYS}"
    )
    print(
        f"forward files found   : {len(forward_files)}"
    )

    if forward_files:
        print(
            "forward dates         : "
            + ", ".join(
                d.strftime("%Y-%m-%d")
                for d, _ in forward_files
            )
        )

    header(
        "FORWARD EVALUATION"
    )

    quality, daily, picks = evaluate_forward(
        development,
        forward_files,
    )

    overall = build_overall(
        daily
    )

    status = build_status(
        daily
    )

    header(
        "FORWARD OVERALL"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    header(
        "FORWARD STATUS"
    )

    print(
        status.to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_path = (
        OUTPUT_DIR
        / "08_forward_data_quality.csv"
    )

    daily_path = (
        OUTPUT_DIR
        / "08_forward_daily_results.csv"
    )

    picks_path = (
        OUTPUT_DIR
        / "08_forward_top3_picks.csv"
    )

    overall_path = (
        OUTPUT_DIR
        / "08_forward_overall.csv"
    )

    status_path = (
        OUTPUT_DIR
        / "08_forward_status.csv"
    )

    quality.to_csv(
        quality_path,
        index=False,
        encoding="utf-8-sig",
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

    overall.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    status.to_csv(
        status_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    for path in (
        quality_path,
        daily_path,
        picks_path,
        overall_path,
        status_path,
    ):
        print(
            path
        )

    print()
    print(
        "08 forward test complete."
    )
    print(
        "The development period through 2026-08-26 remains locked."
    )
    print(
        "No automatic promotion is performed."
    )
    print(
        "No Maruhan Maebashi files were modified."
    )


if __name__ == "__main__":
    main()
