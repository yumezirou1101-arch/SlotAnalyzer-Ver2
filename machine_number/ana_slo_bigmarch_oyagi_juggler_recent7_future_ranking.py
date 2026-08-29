from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


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
    / "09_juggler_recent7_future_ranking"
)

DEVELOPMENT_END = pd.Timestamp("2026-08-26")
FORWARD_START = pd.Timestamp("2026-08-27")

WINDOW = 7
TOPN_PRIMARY = 3
TOPN_DISPLAY = 10

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

    x["is_juggler"] = (
        x["machine_name"]
        .str.contains(
            "ジャグラー",
            na=False,
        )
    )

    return x


def discover_daily_files():
    found = []

    for path in DATA_DIR.glob(
        "ana_slo_bigmarch_oyagi_*.csv"
    ):
        m = DAILY_FILE_RE.fullmatch(path.name)

        if not m:
            continue

        date = pd.to_datetime(
            m.group(1),
            format="%Y%m%d",
            errors="raise",
        ).normalize()

        found.append(
            (
                date,
                path,
            )
        )

    if not found:
        raise RuntimeError(
            "No Big March Oyagi daily CSV files were found."
        )

    return sorted(
        found,
        key=lambda x: x[0],
    )


def load_history():
    daily_files = discover_daily_files()

    frames = []

    for file_date, path in daily_files:
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
                f"{path.name}: internal date does not match filename."
            )

        frames.append(x)

    history = pd.concat(
        frames,
        ignore_index=True,
    )

    history = history.sort_values(
        [
            "date",
            "machine_no",
        ]
    ).reset_index(drop=True)

    return history, daily_files


def build_future_ranking(
    history: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> pd.DataFrame:

    target_date = (
        latest_date
        + pd.Timedelta(days=1)
    ).normalize()

    latest_panel = history[
        history["date"] == latest_date
    ].copy()

    current_juggler = latest_panel[
        latest_panel["is_juggler"]
    ][
        [
            "machine_no",
            "machine_name",
        ]
    ].copy()

    if current_juggler.empty:
        raise RuntimeError(
            "No current Juggler machines were found on the latest day."
        )

    history_before_target = history[
        history["date"] < target_date
    ].copy()

    history_j = history_before_target[
        history_before_target["is_juggler"]
    ].copy()

    rows = []

    for _, current in current_juggler.iterrows():
        machine_no = int(
            current["machine_no"]
        )

        grp = history_j[
            history_j["machine_no"] == machine_no
        ].copy()

        recent = (
            grp.sort_values("date")
            .tail(WINDOW)
        )

        if recent.empty:
            recent7_win = -1.0
            recent7_n = 0
            recent7_avg_diff = 0.0
        else:
            recent7_win = float(
                recent["win"].mean()
            )
            recent7_n = int(
                recent["date"].nunique()
            )
            recent7_avg_diff = float(
                recent["diff"].mean()
            )

        rows.append(
            {
                "target_date": target_date.date(),
                "latest_data_date": latest_date.date(),
                "model": MODEL_NAME,
                "machine_no": machine_no,
                "machine_name": current["machine_name"],
                "recent7_win": recent7_win,
                "recent7_n": recent7_n,
                "recent7_avg_diff": recent7_avg_diff,
            }
        )

    ranking = pd.DataFrame(rows)

    ranking = ranking.sort_values(
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
    ).reset_index(drop=True)

    ranking["prediction_rank"] = (
        ranking.index + 1
    )

    ranking["tier"] = "RESERVE"
    ranking.loc[
        ranking["prediction_rank"] <= TOPN_PRIMARY,
        "tier",
    ] = "PRIMARY"

    return ranking


def main() -> None:
    header(
        "09 - Big March Takasaki Oyagi "
        "Frozen JUGGLER RECENT7 WIN Future Ranking"
    )

    history, daily_files = load_history()

    latest_date = daily_files[-1][0]
    target_date = (
        latest_date
        + pd.Timedelta(days=1)
    ).normalize()

    ranking = build_future_ranking(
        history,
        latest_date,
    )

    top10 = (
        ranking.head(TOPN_DISPLAY)
        .copy()
    )

    primary = top10[
        top10["prediction_rank"] <= TOPN_PRIMARY
    ].copy()

    reserve = top10[
        top10["prediction_rank"] > TOPN_PRIMARY
    ].copy()

    print(
        f"latest data date       : {latest_date.date()}"
    )
    print(
        f"target date            : {target_date.date()}"
    )
    print(
        f"model                  : {MODEL_NAME}"
    )
    print(
        f"window                 : {WINDOW}"
    )
    print(
        f"current Juggler count  : {len(ranking)}"
    )
    print(
        f"primary TopN           : {TOPN_PRIMARY}"
    )
    print(
        f"display TopN           : {TOPN_DISPLAY}"
    )

    header(
        "PRIMARY TOP3"
    )

    print(
        primary[
            [
                "prediction_rank",
                "machine_no",
                "machine_name",
                "recent7_win",
                "recent7_n",
                "recent7_avg_diff",
            ]
        ].to_string(
            index=False
        )
    )

    header(
        "RESERVE 4-10"
    )

    print(
        reserve[
            [
                "prediction_rank",
                "machine_no",
                "machine_name",
                "recent7_win",
                "recent7_n",
                "recent7_avg_diff",
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
        / f"09_prediction_{compact}_all_juggler.csv"
    )

    top10_path = (
        OUTPUT_DIR
        / f"09_prediction_{compact}_top10.csv"
    )

    metadata_path = (
        OUTPUT_DIR
        / f"09_prediction_{compact}_metadata.csv"
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
                "target_date": target_date.date(),
                "latest_data_date": latest_date.date(),
                "model": MODEL_NAME,
                "window": WINDOW,
                "primary_topn": TOPN_PRIMARY,
                "display_topn": TOPN_DISPLAY,
                "current_juggler_count": len(ranking),
                "development_end": DEVELOPMENT_END.date(),
                "forward_start": FORWARD_START.date(),
                "automatic_promotion": False,
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
    print("Future ranking complete.")
    print(
        "The frozen JUGGLER_RECENT7_WIN ranking rule was not changed."
    )
    print(
        "Ranks 1-3 are PRIMARY. Ranks 4-10 are RESERVE only."
    )
    print(
        "No target-day actual data is used."
    )


if __name__ == "__main__":
    main()
