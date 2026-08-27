from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 76 - Normal / A-Type / Juggler Live Prediction Evaluation
# ============================================================
#
# Purpose
# -------
# Compare actual performance of:
#   64 = Normal prediction
#   74 = A-type overall prediction
#   75 = Juggler-only prediction
#
# Evaluation bands:
#   TOP1 / TOP3 / TOP5 / TOP10
#
# Safety
# -------
# - Prediction files are read only.
# - Actual target-day data comes from ana_slo_YYYYMMDD.csv.
# - No model scores are recalculated.
# - No 64 / 74 / 75 files are modified.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

ANALYSIS_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)

NORMAL_DIR = (
    ANALYSIS_DIR
    / "64_Ver4_2_future_top10"
)

A_TYPE_DIR = (
    ANALYSIS_DIR
    / "74_Ver4_2_A_type_prediction"
)

JUGGLER_DIR = (
    ANALYSIS_DIR
    / "75_Ver4_2_Juggler_prediction"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "76_Normal_AType_Juggler_live_evaluation"
)

EXPECTED_STORE_MACHINES = 514

BANDS = {
    "TOP1": 1,
    "TOP3": 3,
    "TOP5": 5,
    "TOP10": 10,
}

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260825


def header(title: str) -> None:
    print()
    print("=" * 116)
    print(title)
    print("=" * 116)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None

    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CSV read failed: {path}\n"
        f"last_error={last_error}"
    )


def bootstrap_mean_ci(values: np.ndarray, seed_offset: int = 0):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(
        BOOTSTRAP_SEED + seed_offset
    )

    samples = rng.choice(
        x,
        size=(BOOTSTRAP_REPS, len(x)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def parse_actual_date(path: Path):
    m = re.fullmatch(
        r"ana_slo_(\d{8})\.csv",
        path.name,
        flags=re.IGNORECASE,
    )

    if not m:
        return None

    dt = pd.to_datetime(
        m.group(1),
        format="%Y%m%d",
        errors="coerce",
    )

    if pd.isna(dt):
        return None

    return pd.Timestamp(dt)


def discover_actual_files():
    result = {}

    for path in DATA_DIR.glob(
        "ana_slo_????????.csv"
    ):
        dt = parse_actual_date(path)

        if dt is not None:
            result[dt] = path

    return result


def canonicalize_actual(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:

    date_col = None
    for c in ("date", "日付"):
        if c in df.columns:
            date_col = c
            break

    no_col = None
    for c in ("machine_no", "台番号"):
        if c in df.columns:
            no_col = c
            break

    name_col = None
    for c in ("machine_name", "機種名"):
        if c in df.columns:
            name_col = c
            break

    diff_col = None
    for c in ("diff", "差枚"):
        if c in df.columns:
            diff_col = c
            break

    if not all([no_col, name_col, diff_col]):
        raise ValueError(
            "Actual required columns not found."
        )

    x = df.rename(
        columns={
            no_col: "machine_no",
            name_col: "machine_name",
            diff_col: "diff",
        }
    ).copy()

    if date_col is not None:
        x = x.rename(
            columns={
                date_col: "date"
            }
        )

        x["date"] = pd.to_datetime(
            x["date"],
            errors="coerce",
        )

        x = x[
            x["date"] == target_date
        ].copy()

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="coerce",
    )

    x["diff"] = (
        x["diff"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )

    x["diff"] = pd.to_numeric(
        x["diff"],
        errors="coerce",
    )

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    x = x.dropna(
        subset=[
            "machine_no",
            "machine_name",
            "diff",
        ]
    ).copy()

    x["machine_no"] = (
        x["machine_no"]
        .astype(int)
    )

    x = (
        x.sort_values("machine_no")
        .drop_duplicates(
            subset=["machine_no"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return x


def discover_prediction_files():
    found = []

    patterns = [
        (
            "NORMAL",
            NORMAL_DIR,
            re.compile(
                r"64_prediction_(\d{8})_top10\.csv",
                re.IGNORECASE,
            ),
        ),
        (
            "A_TYPE",
            A_TYPE_DIR,
            re.compile(
                r"74_A_type_prediction_(\d{8})_top10\.csv",
                re.IGNORECASE,
            ),
        ),
        (
            "JUGGLER",
            JUGGLER_DIR,
            re.compile(
                r"75_Juggler_prediction_(\d{8})_top10\.csv",
                re.IGNORECASE,
            ),
        ),
    ]

    for prediction_type, directory, rx in patterns:

        if not directory.exists():
            continue

        for path in directory.glob("*.csv"):
            m = rx.fullmatch(path.name)

            if not m:
                continue

            dt = pd.to_datetime(
                m.group(1),
                format="%Y%m%d",
                errors="coerce",
            )

            if pd.isna(dt):
                continue

            found.append(
                (
                    prediction_type,
                    pd.Timestamp(dt),
                    path,
                )
            )

    found.sort(
        key=lambda x: (
            x[1],
            x[0],
        )
    )

    return found


def prepare_prediction(
    prediction_type: str,
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = df.copy()

    if prediction_type == "NORMAL":
        rank_col = "prediction_rank"
        tier_col = "tier"

    elif prediction_type == "A_TYPE":
        rank_col = "a_type_rank"
        tier_col = "a_type_tier"

    elif prediction_type == "JUGGLER":
        rank_col = "juggler_rank"
        tier_col = "juggler_tier"

    else:
        raise ValueError(
            f"Unknown prediction type: {prediction_type}"
        )

    required = [
        "machine_no",
        "machine_name",
        "score",
        rank_col,
        tier_col,
        "target_date",
        "latest_data_date",
    ]

    missing = [
        c
        for c in required
        if c not in x.columns
    ]

    if missing:
        raise ValueError(
            f"{prediction_type} columns missing: {missing}"
        )

    x = x.rename(
        columns={
            rank_col: "local_rank",
            tier_col: "local_tier",
        }
    )

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="coerce",
    )

    x["local_rank"] = pd.to_numeric(
        x["local_rank"],
        errors="coerce",
    )

    x["score"] = pd.to_numeric(
        x["score"],
        errors="coerce",
    )

    x["target_date"] = pd.to_datetime(
        x["target_date"],
        errors="coerce",
    )

    x["latest_data_date"] = pd.to_datetime(
        x["latest_data_date"],
        errors="coerce",
    )

    x = x.dropna(
        subset=[
            "machine_no",
            "local_rank",
            "target_date",
            "latest_data_date",
        ]
    ).copy()

    x["machine_no"] = x["machine_no"].astype(int)
    x["local_rank"] = x["local_rank"].astype(int)

    x = (
        x.sort_values("local_rank")
        .reset_index(drop=True)
    )

    return x


def evaluate_one(
    prediction_type: str,
    target_date: pd.Timestamp,
    prediction_path: Path,
    actual_path: Path,
):
    pred_raw = read_csv_flexible(
        prediction_path
    )

    pred = prepare_prediction(
        prediction_type,
        pred_raw,
    )

    actual_raw = read_csv_flexible(
        actual_path
    )

    actual = canonicalize_actual(
        actual_raw,
        target_date,
    )

    if len(actual) != EXPECTED_STORE_MACHINES:
        raise RuntimeError(
            f"{target_date.date()}: actual rows={len(actual)} "
            f"expected={EXPECTED_STORE_MACHINES}"
        )

    merged = pred.merge(
        actual[
            [
                "machine_no",
                "machine_name",
                "diff",
            ]
        ].rename(
            columns={
                "machine_name":
                    "actual_machine_name",
                "diff":
                    "actual_diff",
            }
        ),
        on="machine_no",
        how="left",
        validate="one_to_one",
    )

    if merged["actual_diff"].isna().any():
        missing = merged.loc[
            merged["actual_diff"].isna(),
            "machine_no",
        ].tolist()

        raise RuntimeError(
            f"{prediction_type} {target_date.date()}: "
            f"actual missing for {missing}"
        )

    merged["prediction_type"] = (
        prediction_type
    )

    merged["actual_win"] = (
        merged["actual_diff"] > 0
    ).astype(int)

    merged["actual_plus1000"] = (
        merged["actual_diff"] >= 1000
    ).astype(int)

    merged["actual_plus2000"] = (
        merged["actual_diff"] >= 2000
    ).astype(int)

    store_avg = float(
        actual["diff"].mean()
    )

    daily_rows = []

    for band, cutoff in BANDS.items():
        selected = merged[
            merged["local_rank"] <= cutoff
        ].copy()

        d = selected[
            "actual_diff"
        ].astype(float)

        daily_rows.append(
            {
                "target_date":
                    target_date,
                "prediction_type":
                    prediction_type,
                "band":
                    band,
                "selected_n":
                    int(
                        len(selected)
                    ),
                "avg_diff":
                    float(
                        d.mean()
                    ),
                "median_diff":
                    float(
                        d.median()
                    ),
                "total_diff":
                    float(
                        d.sum()
                    ),
                "win_rate":
                    float(
                        (d > 0).mean()
                        * 100.0
                    ),
                "plus1000_rate":
                    float(
                        (d >= 1000).mean()
                        * 100.0
                    ),
                "plus2000_rate":
                    float(
                        (d >= 2000).mean()
                        * 100.0
                    ),
                "positive":
                    int(
                        d.sum() > 0
                    ),
                "store_avg_diff":
                    store_avg,
                "avg_diff_lift_vs_store":
                    float(
                        d.mean()
                        - store_avg
                    ),
                "excess_total_vs_store":
                    float(
                        (
                            d.mean()
                            - store_avg
                        )
                        * len(d)
                    ),
            }
        )

    return merged, pd.DataFrame(
        daily_rows
    )


def build_overall(
    daily: pd.DataFrame,
):
    rows = []

    for (
        prediction_type,
        band,
    ), g in daily.groupby(
        [
            "prediction_type",
            "band",
        ],
        sort=True,
    ):

        ci_low, ci_high = bootstrap_mean_ci(
            g["avg_diff"].to_numpy(),
            seed_offset=len(rows) + 1,
        )

        lift_low, lift_high = bootstrap_mean_ci(
            g[
                "avg_diff_lift_vs_store"
            ].to_numpy(),
            seed_offset=1000 + len(rows),
        )

        rows.append(
            {
                "prediction_type":
                    prediction_type,
                "band":
                    band,
                "evaluated_days":
                    int(
                        g[
                            "target_date"
                        ].nunique()
                    ),
                "selected_rows":
                    int(
                        g[
                            "selected_n"
                        ].sum()
                    ),
                "mean_daily_avg_diff":
                    float(
                        g[
                            "avg_diff"
                        ].mean()
                    ),
                "median_daily_avg_diff":
                    float(
                        g[
                            "avg_diff"
                        ].median()
                    ),
                "total_diff":
                    float(
                        g[
                            "total_diff"
                        ].sum()
                    ),
                "mean_win_rate":
                    float(
                        g[
                            "win_rate"
                        ].mean()
                    ),
                "mean_plus1000_rate":
                    float(
                        g[
                            "plus1000_rate"
                        ].mean()
                    ),
                "mean_plus2000_rate":
                    float(
                        g[
                            "plus2000_rate"
                        ].mean()
                    ),
                "positive_day_rate":
                    float(
                        g[
                            "positive"
                        ].mean()
                        * 100.0
                    ),
                "mean_store_avg_diff":
                    float(
                        g[
                            "store_avg_diff"
                        ].mean()
                    ),
                "mean_lift_vs_store":
                    float(
                        g[
                            "avg_diff_lift_vs_store"
                        ].mean()
                    ),
                "total_excess_vs_store":
                    float(
                        g[
                            "excess_total_vs_store"
                        ].sum()
                    ),
                "daily_avg_diff_ci95_low":
                    ci_low,
                "daily_avg_diff_ci95_high":
                    ci_high,
                "daily_lift_ci95_low":
                    lift_low,
                "daily_lift_ci95_high":
                    lift_high,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    header(
        "76 - Normal / A-Type / Juggler Live Prediction Evaluation"
    )

    predictions = discover_prediction_files()
    actual_map = discover_actual_files()

    print(
        f"prediction files      : {len(predictions)}"
    )
    print(
        f"actual daily files    : {len(actual_map)}"
    )

    status_rows = []
    detail_frames = []
    daily_frames = []

    for (
        prediction_type,
        target_date,
        prediction_path,
    ) in predictions:

        actual_path = actual_map.get(
            target_date
        )

        if actual_path is None:
            status_rows.append(
                {
                    "prediction_type":
                        prediction_type,
                    "target_date":
                        target_date,
                    "status":
                        "PENDING_ACTUAL_DATA",
                    "prediction_file":
                        str(
                            prediction_path
                        ),
                    "actual_file":
                        "",
                }
            )
            continue

        try:
            detail, daily = evaluate_one(
                prediction_type,
                target_date,
                prediction_path,
                actual_path,
            )

            detail_frames.append(
                detail
            )

            daily_frames.append(
                daily
            )

            status_rows.append(
                {
                    "prediction_type":
                        prediction_type,
                    "target_date":
                        target_date,
                    "status":
                        "EVALUATED",
                    "prediction_file":
                        str(
                            prediction_path
                        ),
                    "actual_file":
                        str(
                            actual_path
                        ),
                }
            )

        except Exception as exc:
            status_rows.append(
                {
                    "prediction_type":
                        prediction_type,
                    "target_date":
                        target_date,
                    "status":
                        f"ERROR:{exc}",
                    "prediction_file":
                        str(
                            prediction_path
                        ),
                    "actual_file":
                        str(
                            actual_path
                        ),
                }
            )

    status = pd.DataFrame(
        status_rows
    )

    detail = (
        pd.concat(
            detail_frames,
            ignore_index=True,
        )
        if detail_frames
        else pd.DataFrame()
    )

    daily = (
        pd.concat(
            daily_frames,
            ignore_index=True,
        )
        if daily_frames
        else pd.DataFrame()
    )

    overall = (
        build_overall(
            daily
        )
        if not daily.empty
        else pd.DataFrame()
    )

    header("STATUS")
    print(
        status.to_string(
            index=False
        )
    )

    header("DAILY RESULTS")

    if daily.empty:
        print(
            "No evaluated prediction yet."
        )
    else:
        print(
            daily[
                [
                    "target_date",
                    "prediction_type",
                    "band",
                    "selected_n",
                    "avg_diff",
                    "total_diff",
                    "win_rate",
                    "plus1000_rate",
                    "plus2000_rate",
                    "store_avg_diff",
                    "avg_diff_lift_vs_store",
                ]
            ].to_string(
                index=False
            )
        )

    header("OVERALL RESULTS")

    if overall.empty:
        print(
            "No overall result yet."
        )
    else:
        print(
            overall.to_string(
                index=False
            )
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        OUTPUT_DIR
        / "76_status.csv"
    )

    detail_path = (
        OUTPUT_DIR
        / "76_detail.csv"
    )

    daily_path = (
        OUTPUT_DIR
        / "76_daily.csv"
    )

    overall_path = (
        OUTPUT_DIR
        / "76_overall.csv"
    )

    status.to_csv(
        status_path,
        index=False,
        encoding="utf-8-sig",
    )

    detail.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    daily.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    overall.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    header("FILES SAVED")
    for path in (
        status_path,
        detail_path,
        daily_path,
        overall_path,
    ):
        print(path)

    print()
    print(
        "76 live comparison evaluation complete."
    )
    print(
        "Normal / A-type / Juggler predictions were not modified."
    )


if __name__ == "__main__":
    main()
