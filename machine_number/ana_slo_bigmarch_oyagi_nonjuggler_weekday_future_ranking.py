from __future__ import annotations

from pathlib import Path
import argparse
import re

import pandas as pd


# ============================================================
# Big March Takasaki Oyagi
# Frozen NON_JUGGLER WEEKDAY_AVG Top1 Future Ranking
# ============================================================
#
# Frozen development period:
#   through 2026-08-26
#
# Forward period:
#   2026-08-27 onward
#
# Frozen rule:
#   - NON_JUGGLER only
#   - score = same-weekday historical average diff
#   - rank by weekday_avg desc
#   - tie break: machine_no asc
#   - rank 1 = PRIMARY
#   - ranks 2-10 = RESERVE
#
# Data source policy:
#   1) locked integrated development dataset through 2026-08-26
#   2) daily CSV files from 2026-08-27 onward
#
# IMPORTANT:
#   - Target-day actual data is never used.
#   - Frozen ranking rule is not changed here.
#   - No automatic model promotion is performed.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

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
    / "12_nonjuggler_weekday_future_ranking"
)

DEVELOPMENT_END = pd.Timestamp("2026-08-26")
FORWARD_START = pd.Timestamp("2026-08-27")

TOPN_PRIMARY = 1
TOPN_DISPLAY = 10

MODEL_NAME = "NON_JUGGLER_WEEKDAY_AVG_TOP1_FROZEN"

DAILY_FILE_RE = re.compile(
    r"^ana_slo_bigmarch_oyagi_(\d{8})\.csv$",
    re.IGNORECASE,
)


def header(title: str) -> None:
    print()
    print("=" * 122)
    print(title)
    print("=" * 122)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Frozen NON_JUGGLER WEEKDAY_AVG "
            "Top1 future ranking for Big March "
            "Takasaki Oyagi."
        )
    )

    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help=(
            "Optional target date YYYY-MM-DD. "
            "Default: latest data date + 1 day."
        ),
    )

    return parser.parse_args()


def normalize(
    df: pd.DataFrame,
    source: str,
) -> pd.DataFrame:

    x = df.copy()

    x.columns = [
        str(c).strip()
        for c in x.columns
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
            f"{source}: missing required columns: "
            f"{missing}"
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

    if x["G"].isna().any():
        raise RuntimeError(
            f"{source}: invalid or missing G exists."
        )

    if x["diff"].isna().any():
        raise RuntimeError(
            f"{source}: invalid or missing diff exists."
        )

    if (x["G"] < 0).any():
        raise RuntimeError(
            f"{source}: negative G exists."
        )

    duplicated = x.duplicated(
        subset=[
            "date",
            "machine_no",
        ],
        keep=False,
    )

    if duplicated.any():
        raise RuntimeError(
            f"{source}: duplicate date-machine rows exist."
        )

    x["weekday"] = (
        x["date"].dt.weekday
    )

    x["is_juggler"] = (
        x["machine_name"]
        .str.contains(
            "ジャグラー",
            na=False,
        )
    )

    return x


def find_locked_integrated_file() -> Path:

    files = sorted(
        QUALITY_DIR.glob(
            "01_bigmarch_oyagi_integrated_*_20260826.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Locked integrated development dataset "
            "ending 2026-08-26 was not found in:\n"
            f"{QUALITY_DIR}"
        )

    return files[-1]


def load_development() -> tuple[
    pd.DataFrame,
    Path,
]:

    path = find_locked_integrated_file()

    raw = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    x = normalize(
        raw,
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

        match = DAILY_FILE_RE.fullmatch(
            path.name
        )

        if not match:
            continue

        file_date = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="raise",
        ).normalize()

        if file_date >= FORWARD_START:
            found.append(
                (
                    file_date,
                    path,
                )
            )

    return sorted(
        found,
        key=lambda x: x[0],
    )


def load_forward_day(
    file_date: pd.Timestamp,
    path: Path,
) -> pd.DataFrame:

    raw = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    x = normalize(
        raw,
        source=path.name,
    )

    internal_dates = (
        x["date"]
        .drop_duplicates()
        .tolist()
    )

    if (
        len(internal_dates) != 1
        or internal_dates[0] != file_date
    ):
        raise RuntimeError(
            f"{path.name}: internal date does not "
            "match filename."
        )

    if len(x) < 200:
        raise RuntimeError(
            f"{path.name}: machine rows below 200 "
            f"({len(x)})."
        )

    machine_count = int(
        x["machine_no"].nunique()
    )

    if machine_count != len(x):
        raise RuntimeError(
            f"{path.name}: machine_no is not unique."
        )

    return x


def load_frozen_history() -> tuple[
    pd.DataFrame,
    Path,
    list,
]:

    development, locked_path = (
        load_development()
    )

    forward_files = (
        discover_forward_daily_files()
    )

    frames = [
        development
    ]

    for file_date, path in forward_files:

        forward_day = load_forward_day(
            file_date,
            path,
        )

        frames.append(
            forward_day
        )

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicated = history.duplicated(
        subset=[
            "date",
            "machine_no",
        ],
        keep=False,
    )

    if duplicated.any():
        examples = (
            history.loc[
                duplicated,
                [
                    "date",
                    "machine_no",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise RuntimeError(
            "Duplicate date-machine rows exist "
            "after frozen history assembly. "
            f"Examples: {examples}"
        )

    history = history.sort_values(
        [
            "date",
            "machine_no",
        ]
    ).reset_index(
        drop=True
    )

    return (
        history,
        locked_path,
        forward_files,
    )


def resolve_target_date(
    latest_date: pd.Timestamp,
    requested: str | None,
) -> pd.Timestamp:

    if requested is None:
        return (
            latest_date
            + pd.Timedelta(days=1)
        ).normalize()

    target_date = pd.to_datetime(
        requested,
        format="%Y-%m-%d",
        errors="raise",
    ).normalize()

    if target_date <= latest_date:
        raise RuntimeError(
            "--target-date must be later than "
            f"latest data date {latest_date.date()}."
        )

    return target_date


def build_future_ranking(
    history: pd.DataFrame,
    latest_date: pd.Timestamp,
    target_date: pd.Timestamp,
) -> pd.DataFrame:

    latest_panel = history[
        history["date"] == latest_date
    ].copy()

    if latest_panel.empty:
        raise RuntimeError(
            "Latest data panel is empty."
        )

    current_nonjuggler = latest_panel[
        ~latest_panel["is_juggler"]
    ][
        [
            "machine_no",
            "machine_name",
        ]
    ].copy()

    if current_nonjuggler.empty:
        raise RuntimeError(
            "No current NON_JUGGLER machines were "
            "found on the latest day."
        )

    # Safety:
    # only history strictly before target day.
    history_before_target = history[
        history["date"] < target_date
    ].copy()

    history_nonjuggler = (
        history_before_target[
            ~history_before_target[
                "is_juggler"
            ]
        ]
        .copy()
    )

    target_weekday = int(
        target_date.weekday()
    )

    weekday_history = history_nonjuggler[
        history_nonjuggler["weekday"]
        == target_weekday
    ].copy()

    weekday_stats = (
        weekday_history
        .groupby(
            "machine_no",
            as_index=False,
        )
        .agg(
            weekday_avg=(
                "diff",
                "mean",
            ),
            weekday_history_n=(
                "date",
                "nunique",
            ),
        )
    )

    ranking = (
        current_nonjuggler
        .merge(
            weekday_stats,
            on="machine_no",
            how="left",
        )
    )

    median_weekday_avg = (
        ranking["weekday_avg"]
        .median(
            skipna=True
        )
    )

    if pd.isna(
        median_weekday_avg
    ):
        median_weekday_avg = 0.0

    ranking["weekday_avg"] = (
        ranking["weekday_avg"]
        .fillna(
            float(
                median_weekday_avg
            )
        )
    )

    ranking[
        "weekday_history_n"
    ] = (
        ranking[
            "weekday_history_n"
        ]
        .fillna(0)
        .astype(int)
    )

    ranking["target_date"] = (
        target_date.date()
    )

    ranking[
        "latest_data_date"
    ] = (
        latest_date.date()
    )

    ranking["model"] = (
        MODEL_NAME
    )

    # Exact frozen baseline ranking:
    # weekday_avg desc, machine_no asc.
    #
    # weekday_history_n is diagnostic only.
    ranking = ranking.sort_values(
        [
            "weekday_avg",
            "machine_no",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    ranking["prediction_rank"] = (
        ranking.index + 1
    )

    ranking["tier"] = "RESERVE"

    ranking.loc[
        ranking["prediction_rank"]
        <= TOPN_PRIMARY,
        "tier",
    ] = "PRIMARY"

    return ranking


def main() -> None:

    args = parse_args()

    header(
        "12 - Big March Takasaki Oyagi "
        "Frozen NON_JUGGLER WEEKDAY_AVG "
        "Future Ranking"
    )

    (
        history,
        locked_path,
        forward_files,
    ) = load_frozen_history()

    latest_date = (
        history["date"].max()
    ).normalize()

    target_date = (
        resolve_target_date(
            latest_date,
            args.target_date,
        )
    )

    ranking = build_future_ranking(
        history,
        latest_date,
        target_date,
    )

    top10 = (
        ranking
        .head(TOPN_DISPLAY)
        .copy()
    )

    primary = top10[
        top10["prediction_rank"]
        <= TOPN_PRIMARY
    ].copy()

    reserve = top10[
        top10["prediction_rank"]
        > TOPN_PRIMARY
    ].copy()

    print(
        f"locked development     : "
        f"{locked_path}"
    )

    print(
        f"development end        : "
        f"{DEVELOPMENT_END.date()}"
    )

    print(
        f"forward start          : "
        f"{FORWARD_START.date()}"
    )

    print(
        f"forward daily files    : "
        f"{len(forward_files)}"
    )

    print(
        f"history rows           : "
        f"{len(history):,}"
    )

    print(
        f"latest data date       : "
        f"{latest_date.date()}"
    )

    print(
        f"target date            : "
        f"{target_date.date()}"
    )

    print(
        f"target weekday         : "
        f"{target_date.day_name()}"
    )

    print(
        f"model                  : "
        f"{MODEL_NAME}"
    )

    print(
        f"current NON_JUGGLER    : "
        f"{len(ranking)}"
    )

    print(
        f"primary TopN           : "
        f"{TOPN_PRIMARY}"
    )

    print(
        f"display TopN           : "
        f"{TOPN_DISPLAY}"
    )

    print(
        "automatic promotion   : False"
    )

    header(
        "PRIMARY TOP1"
    )

    print(
        primary[
            [
                "prediction_rank",
                "machine_no",
                "machine_name",
                "weekday_avg",
                "weekday_history_n",
            ]
        ].to_string(
            index=False
        )
    )

    header(
        "RESERVE 2-10"
    )

    print(
        reserve[
            [
                "prediction_rank",
                "machine_no",
                "machine_name",
                "weekday_avg",
                "weekday_history_n",
            ]
        ].to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    compact = target_date.strftime(
        "%Y%m%d"
    )

    all_path = (
        OUTPUT_DIR
        / (
            f"12_prediction_{compact}"
            "_all_nonjuggler.csv"
        )
    )

    top10_path = (
        OUTPUT_DIR
        / (
            f"12_prediction_{compact}"
            "_top10.csv"
        )
    )

    metadata_path = (
        OUTPUT_DIR
        / (
            f"12_prediction_{compact}"
            "_metadata.csv"
        )
    )

    ranking.to_csv(
        all_path,
        index=False,
        encoding="utf-8-sig",
    )

    top10.to_csv(
        top10_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = pd.DataFrame(
        [
            {
                "target_date":
                    target_date.date(),
                "latest_data_date":
                    latest_date.date(),
                "target_weekday":
                    target_date.day_name(),
                "model":
                    MODEL_NAME,
                "primary_topn":
                    TOPN_PRIMARY,
                "display_topn":
                    TOPN_DISPLAY,
                "current_nonjuggler_count":
                    len(ranking),
                "development_end":
                    DEVELOPMENT_END.date(),
                "forward_start":
                    FORWARD_START.date(),
                "locked_development_file":
                    locked_path.name,
                "forward_daily_files":
                    len(forward_files),
                "history_rows":
                    len(history),
                "automatic_promotion":
                    False,
            }
        ]
    )

    metadata.to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    header(
        "FILES SAVED"
    )

    print(all_path)
    print(top10_path)
    print(metadata_path)

    print()
    print(
        "Future ranking complete."
    )

    print(
        "The frozen NON_JUGGLER "
        "WEEKDAY_AVG ranking rule "
        "was not changed."
    )

    print(
        "Frozen development data through "
        "2026-08-26 plus forward daily data "
        "are used."
    )

    print(
        "Rank 1 is PRIMARY. "
        "Ranks 2-10 are RESERVE only."
    )

    print(
        "weekday_history_n is diagnostic "
        "only and is not used as a "
        "tie-break."
    )

    print(
        "No target-day actual data is used."
    )

    print(
        "No automatic model promotion "
        "is performed."
    )


if __name__ == "__main__":
    main()