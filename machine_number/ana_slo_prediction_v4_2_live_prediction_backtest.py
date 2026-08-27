from __future__ import annotations

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


# ============================================================
# 69 - V4.2 Live Prediction Backtest
# ============================================================
#
# Purpose
# -------
# Evaluate ONLY the already-saved live prediction files created by 64.
# This script does NOT recalculate model scores and does NOT rewrite
# any prediction file.
#
# Prediction source:
#   64_prediction_YYYYMMDD_top10.csv
#
# Actual source:
#   ana_slo_YYYYMMDD.csv
#
# Main evaluation bands:
#   TOP1 / TOP3 / TOP5 / TOP10
#
# Safety principles:
# -------
# - target-day actual diff is read only from the matching actual CSV.
# - prediction rows are treated as frozen snapshots.
# - latest_data_date must be earlier than target_date.
# - no model re-ranking is performed here.
# - unavailable target dates are reported as PENDING, not guessed.
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

PREDICTION_DIR = (
    ANALYSIS_DIR
    / "64_Ver4_2_future_top10"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "69_Ver4_2_live_prediction_backtest"
)

EXPECTED_MACHINES_PER_DAY = 514

EVAL_BANDS = {
    "TOP1": 1,
    "TOP3": 3,
    "TOP5": 5,
    "TOP10": 10,
}

EXPECTED_MODEL = "CHAMPION_V4.2_C"
EXPECTED_WEIGHT_FINGERPRINT = "a1eaf45d71ded209"

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260825


# ============================================================
# GENERAL HELPERS
# ============================================================

def header(
    title: str,
) -> None:

    print()
    print("=" * 112)
    print(title)
    print("=" * 112)


def read_csv_flexible(
    path: Path,
) -> pd.DataFrame:

    last_error = None

    for enc in (
        "utf-8-sig",
        "utf-8",
        "cp932",
    ):

        try:
            return pd.read_csv(
                path,
                encoding=enc,
            )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"CSV read failed: {path}\n"
        f"last_error={last_error}"
    )


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for col in candidates:

        if col in df.columns:
            return col

    return None


def parse_target_date_from_prediction_filename(
    path: Path,
) -> pd.Timestamp | None:

    match = re.fullmatch(
        r"64_prediction_(\d{8})_top10\.csv",
        path.name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    date = pd.to_datetime(
        match.group(1),
        format="%Y%m%d",
        errors="coerce",
    )

    if pd.isna(date):
        return None

    return pd.Timestamp(date)


def parse_actual_date_from_filename(
    path: Path,
) -> pd.Timestamp | None:

    match = re.fullmatch(
        r"ana_slo_(\d{8})\.csv",
        path.name,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    date = pd.to_datetime(
        match.group(1),
        format="%Y%m%d",
        errors="coerce",
    )

    if pd.isna(date):
        return None

    return pd.Timestamp(date)


def safe_float(
    value,
) -> float:

    try:
        x = float(value)

        if np.isfinite(x):
            return x

    except Exception:
        pass

    return np.nan


def bootstrap_mean_ci(
    values: np.ndarray,
) -> tuple[float, float]:

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) < 2:
        return (
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    samples = rng.choice(
        x,
        size=(
            BOOTSTRAP_REPS,
            len(x),
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


# ============================================================
# ACTUAL DATA
# ============================================================

def canonicalize_actual(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:

    date_col = find_column(
        df,
        [
            "date",
            "日付",
        ],
    )

    no_col = find_column(
        df,
        [
            "machine_no",
            "台番号",
        ],
    )

    name_col = find_column(
        df,
        [
            "machine_name",
            "機種名",
        ],
    )

    diff_col = find_column(
        df,
        [
            "diff",
            "差枚",
        ],
    )

    if not all(
        [
            no_col,
            name_col,
            diff_col,
        ]
    ):
        raise ValueError(
            "Required actual columns not found: "
            f"machine_no={no_col}, "
            f"machine_name={name_col}, "
            f"diff={diff_col}"
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
                date_col: "date",
            }
        )

        x["date"] = pd.to_datetime(
            x["date"],
            errors="coerce",
        )

        x = x[
            x["date"] == target_date
        ].copy()

    else:

        x["date"] = target_date

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="coerce",
    )

    x["diff"] = (
        x["diff"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "+",
            "",
            regex=False,
        )
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
        x.sort_values(
            "machine_no"
        )
        .drop_duplicates(
            subset=[
                "machine_no",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
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

    return x


def discover_actual_files() -> dict[pd.Timestamp, Path]:

    actual_map: dict[
        pd.Timestamp,
        Path,
    ] = {}

    for path in DATA_DIR.glob(
        "ana_slo_????????.csv"
    ):

        date = parse_actual_date_from_filename(
            path
        )

        if date is None:
            continue

        actual_map[
            date
        ] = path

    return actual_map


# ============================================================
# PREDICTION DATA
# ============================================================

def discover_prediction_files() -> list[tuple[pd.Timestamp, Path]]:

    candidates = []

    for path in PREDICTION_DIR.glob(
        "64_prediction_????????_top10.csv"
    ):

        target_date = (
            parse_target_date_from_prediction_filename(
                path
            )
        )

        if target_date is None:
            continue

        candidates.append(
            (
                target_date,
                path,
            )
        )

    candidates.sort(
        key=lambda item:
            item[0]
    )

    return candidates


def load_prediction(
    path: Path,
    filename_target_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:

    raw = read_csv_flexible(
        path
    )

    required = [
        "machine_no",
        "machine_name",
        "score",
        "prediction_rank",
        "tier",
        "target_date",
        "latest_data_date",
    ]

    missing = [
        col
        for col in required
        if col not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"Prediction required columns missing: "
            f"{missing} in {path}"
        )

    x = raw.copy()

    x["machine_no"] = pd.to_numeric(
        x["machine_no"],
        errors="coerce",
    )

    x["score"] = pd.to_numeric(
        x["score"],
        errors="coerce",
    )

    x["prediction_rank"] = pd.to_numeric(
        x["prediction_rank"],
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

    x["machine_name"] = (
        x["machine_name"]
        .astype(str)
        .str.strip()
    )

    x["tier"] = (
        x["tier"]
        .astype(str)
        .str.strip()
    )

    x = x.dropna(
        subset=[
            "machine_no",
            "prediction_rank",
            "target_date",
            "latest_data_date",
        ]
    ).copy()

    x["machine_no"] = (
        x["machine_no"]
        .astype(int)
    )

    x["prediction_rank"] = (
        x["prediction_rank"]
        .astype(int)
    )

    x = (
        x.sort_values(
            "prediction_rank"
        )
        .reset_index(
            drop=True
        )
    )

    target_dates = sorted(
        pd.Timestamp(d)
        for d in x[
            "target_date"
        ].dropna().unique()
    )

    latest_dates = sorted(
        pd.Timestamp(d)
        for d in x[
            "latest_data_date"
        ].dropna().unique()
    )

    checks = {
        "rows": int(
            len(x)
        ),
        "unique_machines": int(
            x["machine_no"].nunique()
        ),
        "unique_ranks": int(
            x["prediction_rank"].nunique()
        ),
        "duplicate_machine_n": int(
            x.duplicated(
                subset=[
                    "machine_no",
                ]
            ).sum()
        ),
        "duplicate_rank_n": int(
            x.duplicated(
                subset=[
                    "prediction_rank",
                ]
            ).sum()
        ),
        "target_date_count": int(
            len(target_dates)
        ),
        "latest_data_date_count": int(
            len(latest_dates)
        ),
        "filename_target_matches": False,
        "latest_before_target": False,
        "rank_1_to_10_complete": False,
        "prediction_sha256": sha256_file(
            path
        ),
    }

    if len(target_dates) == 1:

        checks[
            "filename_target_matches"
        ] = (
            target_dates[0]
            == filename_target_date
        )

    if (
        len(target_dates) == 1
        and len(latest_dates) == 1
    ):

        checks[
            "latest_before_target"
        ] = (
            latest_dates[0]
            < target_dates[0]
        )

    checks[
        "rank_1_to_10_complete"
    ] = (
        set(
            x["prediction_rank"].tolist()
        )
        == set(
            range(
                1,
                11,
            )
        )
    )

    checks[
        "prediction_basic_ok"
    ] = bool(
        checks["rows"] == 10
        and checks["unique_machines"] == 10
        and checks["unique_ranks"] == 10
        and checks["duplicate_machine_n"] == 0
        and checks["duplicate_rank_n"] == 0
        and checks["target_date_count"] == 1
        and checks["latest_data_date_count"] == 1
        and checks["filename_target_matches"]
        and checks["latest_before_target"]
        and checks["rank_1_to_10_complete"]
    )

    return (
        x,
        checks,
    )


def load_metadata_for_target(
    target_date: pd.Timestamp,
) -> dict:

    path = (
        PREDICTION_DIR
        / (
            "64_prediction_"
            f"{target_date.strftime('%Y%m%d')}"
            "_metadata.csv"
        )
    )

    if not path.exists():
        return {
            "metadata_exists": False,
            "metadata_path": str(
                path
            ),
            "metadata_model": "",
            "metadata_weight_fingerprint": "",
            "metadata_target_date": pd.NaT,
            "metadata_latest_data_date": pd.NaT,
            "metadata_model_ok": False,
            "metadata_fingerprint_ok": False,
            "metadata_dates_ok": False,
        }

    meta = read_csv_flexible(
        path
    )

    if meta.empty:
        return {
            "metadata_exists": True,
            "metadata_path": str(
                path
            ),
            "metadata_model": "",
            "metadata_weight_fingerprint": "",
            "metadata_target_date": pd.NaT,
            "metadata_latest_data_date": pd.NaT,
            "metadata_model_ok": False,
            "metadata_fingerprint_ok": False,
            "metadata_dates_ok": False,
        }

    row = meta.iloc[0]

    meta_target = pd.to_datetime(
        row.get(
            "target_date",
            pd.NaT,
        ),
        errors="coerce",
    )

    meta_latest = pd.to_datetime(
        row.get(
            "latest_data_date",
            pd.NaT,
        ),
        errors="coerce",
    )

    model = str(
        row.get(
            "model",
            "",
        )
    ).strip()

    fingerprint = str(
        row.get(
            "weight_fingerprint",
            "",
        )
    ).strip()

    return {
        "metadata_exists": True,
        "metadata_path": str(
            path
        ),
        "metadata_model": model,
        "metadata_weight_fingerprint":
            fingerprint,
        "metadata_target_date":
            meta_target,
        "metadata_latest_data_date":
            meta_latest,
        "metadata_model_ok":
            model == EXPECTED_MODEL,
        "metadata_fingerprint_ok":
            fingerprint
            == EXPECTED_WEIGHT_FINGERPRINT,
        "metadata_dates_ok": bool(
            pd.notna(meta_target)
            and pd.notna(meta_latest)
            and pd.Timestamp(meta_target)
            == target_date
            and pd.Timestamp(meta_latest)
            < target_date
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def actual_quality_row(
    target_date: pd.Timestamp,
    actual_path: Path | None,
    actual: pd.DataFrame | None,
) -> dict:

    if (
        actual_path is None
        or actual is None
    ):

        return {
            "target_date": target_date,
            "actual_exists": False,
            "actual_path": "",
            "actual_rows": 0,
            "actual_unique_machines": 0,
            "actual_duplicates": 0,
            "actual_missing_diff": 0,
            "actual_machine_count_ok": False,
            "actual_basic_ok": False,
            "actual_eligible": False,
        }

    duplicates = int(
        actual.duplicated(
            subset=[
                "machine_no",
            ]
        ).sum()
    )

    missing_diff = int(
        actual["diff"].isna().sum()
    )

    machine_count_ok = bool(
        actual["machine_no"].nunique()
        == EXPECTED_MACHINES_PER_DAY
    )

    basic_ok = bool(
        duplicates == 0
        and missing_diff == 0
    )

    return {
        "target_date": target_date,
        "actual_exists": True,
        "actual_path": str(
            actual_path
        ),
        "actual_rows": int(
            len(actual)
        ),
        "actual_unique_machines": int(
            actual["machine_no"].nunique()
        ),
        "actual_duplicates": duplicates,
        "actual_missing_diff": missing_diff,
        "actual_machine_count_ok":
            machine_count_ok,
        "actual_basic_ok":
            basic_ok,
        "actual_eligible": bool(
            machine_count_ok
            and basic_ok
        ),
    }


def evaluate_band(
    merged: pd.DataFrame,
    actual: pd.DataFrame,
    target_date: pd.Timestamp,
    band_name: str,
    cutoff: int,
) -> dict:

    selected = (
        merged[
            merged["prediction_rank"]
            <= cutoff
        ]
        .sort_values(
            "prediction_rank"
        )
        .copy()
    )

    diffs = pd.to_numeric(
        selected[
            "actual_diff"
        ],
        errors="coerce",
    ).dropna()

    store_diffs = pd.to_numeric(
        actual["diff"],
        errors="coerce",
    ).dropna()

    if len(diffs) != cutoff:
        raise RuntimeError(
            f"{target_date.date()} {band_name}: "
            f"expected {cutoff} actual diffs, "
            f"got {len(diffs)}"
        )

    store_avg_diff = float(
        store_diffs.mean()
    )

    total_diff = float(
        diffs.sum()
    )

    avg_diff = float(
        diffs.mean()
    )

    expected_total_at_store_avg = float(
        store_avg_diff
        * cutoff
    )

    excess_total_vs_store = float(
        total_diff
        - expected_total_at_store_avg
    )

    return {
        "target_date": target_date,
        "band": band_name,
        "selected_n": int(
            cutoff
        ),
        "avg_diff": avg_diff,
        "median_diff": float(
            diffs.median()
        ),
        "win_rate": float(
            (
                diffs > 0
            ).mean()
            * 100.0
        ),
        "plus1000_rate": float(
            (
                diffs >= 1000
            ).mean()
            * 100.0
        ),
        "plus2000_rate": float(
            (
                diffs >= 2000
            ).mean()
            * 100.0
        ),
        "positive": int(
            total_diff > 0
        ),
        "total_diff": total_diff,
        "store_avg_diff":
            store_avg_diff,
        "store_win_rate": float(
            (
                store_diffs > 0
            ).mean()
            * 100.0
        ),
        "store_plus1000_rate": float(
            (
                store_diffs >= 1000
            ).mean()
            * 100.0
        ),
        "store_plus2000_rate": float(
            (
                store_diffs >= 2000
            ).mean()
            * 100.0
        ),
        "avg_diff_lift_vs_store": float(
            avg_diff
            - store_avg_diff
        ),
        "expected_total_at_store_avg":
            expected_total_at_store_avg,
        "excess_total_vs_store":
            excess_total_vs_store,
        "selected_machine_nos": tuple(
            int(x)
            for x in selected[
                "machine_no"
            ].tolist()
        ),
    }


def build_overall_summary(
    daily_df: pd.DataFrame,
) -> pd.DataFrame:

    if daily_df.empty:
        return pd.DataFrame()

    rows = []

    for band, group in daily_df.groupby(
        "band",
        sort=False,
    ):

        avg_values = (
            group[
                "avg_diff"
            ]
            .astype(float)
            .to_numpy()
        )

        lift_values = (
            group[
                "avg_diff_lift_vs_store"
            ]
            .astype(float)
            .to_numpy()
        )

        avg_ci_low, avg_ci_high = (
            bootstrap_mean_ci(
                avg_values
            )
        )

        lift_ci_low, lift_ci_high = (
            bootstrap_mean_ci(
                lift_values
            )
        )

        rows.append(
            {
                "band": band,
                "evaluated_days": int(
                    group[
                        "target_date"
                    ].nunique()
                ),
                "selected_rows": int(
                    group[
                        "selected_n"
                    ].sum()
                ),
                "mean_daily_avg_diff": float(
                    group[
                        "avg_diff"
                    ].mean()
                ),
                "median_daily_avg_diff": float(
                    group[
                        "avg_diff"
                    ].median()
                ),
                "mean_win_rate": float(
                    group[
                        "win_rate"
                    ].mean()
                ),
                "mean_plus1000_rate": float(
                    group[
                        "plus1000_rate"
                    ].mean()
                ),
                "mean_plus2000_rate": float(
                    group[
                        "plus2000_rate"
                    ].mean()
                ),
                "positive_day_rate": float(
                    group[
                        "positive"
                    ].mean()
                    * 100.0
                ),
                "total_diff": float(
                    group[
                        "total_diff"
                    ].sum()
                ),
                "mean_store_avg_diff": float(
                    group[
                        "store_avg_diff"
                    ].mean()
                ),
                "mean_avg_diff_lift_vs_store":
                    float(
                        group[
                            "avg_diff_lift_vs_store"
                        ].mean()
                    ),
                "total_excess_vs_store": float(
                    group[
                        "excess_total_vs_store"
                    ].sum()
                ),
                "mean_daily_avg_diff_ci95_low":
                    avg_ci_low,
                "mean_daily_avg_diff_ci95_high":
                    avg_ci_high,
                "mean_lift_vs_store_ci95_low":
                    lift_ci_low,
                "mean_lift_vs_store_ci95_high":
                    lift_ci_high,
            }
        )

    order = {
        name: i
        for i, name in enumerate(
            EVAL_BANDS.keys()
        )
    }

    out = pd.DataFrame(
        rows
    )

    out[
        "_order"
    ] = out[
        "band"
    ].map(
        order
    )

    out = (
        out.sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    header(
        "69 - V4.2 Live Prediction Backtest"
    )

    if not PREDICTION_DIR.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: "
            f"{PREDICTION_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_files = (
        discover_prediction_files()
    )

    actual_map = (
        discover_actual_files()
    )

    print(
        f"prediction files      : "
        f"{len(prediction_files)}"
    )
    print(
        f"actual daily files    : "
        f"{len(actual_map)}"
    )
    print(
        f"prediction directory  : "
        f"{PREDICTION_DIR}"
    )
    print(
        f"output directory      : "
        f"{OUTPUT_DIR}"
    )

    status_rows = []
    quality_rows = []
    detail_frames = []
    daily_rows = []

    for (
        filename_target_date,
        prediction_path,
    ) in prediction_files:

        header(
            f"TARGET {filename_target_date.date()}"
        )

        prediction, pred_checks = (
            load_prediction(
                prediction_path,
                filename_target_date,
            )
        )

        metadata = (
            load_metadata_for_target(
                filename_target_date
            )
        )

        actual_path = actual_map.get(
            filename_target_date
        )

        actual = None

        if actual_path is not None:

            actual_raw = read_csv_flexible(
                actual_path
            )

            actual = canonicalize_actual(
                actual_raw,
                filename_target_date,
            )

        actual_quality = actual_quality_row(
            filename_target_date,
            actual_path,
            actual,
        )

        quality_rows.append(
            {
                **{
                    "target_date":
                        filename_target_date,
                    "prediction_path":
                        str(
                            prediction_path
                        ),
                },
                **pred_checks,
                **metadata,
                **actual_quality,
            }
        )

        prediction_ok = bool(
            pred_checks[
                "prediction_basic_ok"
            ]
        )

        metadata_ok = bool(
            metadata[
                "metadata_exists"
            ]
            and metadata[
                "metadata_model_ok"
            ]
            and metadata[
                "metadata_fingerprint_ok"
            ]
            and metadata[
                "metadata_dates_ok"
            ]
        )

        actual_ok = bool(
            actual_quality[
                "actual_eligible"
            ]
        )

        if not prediction_ok:

            status = (
                "SKIPPED_PREDICTION_QUALITY_FAIL"
            )

        elif not metadata_ok:

            status = (
                "SKIPPED_METADATA_CHECK_FAIL"
            )

        elif not actual_quality[
            "actual_exists"
        ]:

            status = (
                "PENDING_ACTUAL_DATA"
            )

        elif not actual_ok:

            status = (
                "SKIPPED_ACTUAL_QUALITY_FAIL"
            )

        else:

            status = "EVALUATED"

        status_rows.append(
            {
                "target_date":
                    filename_target_date,
                "status":
                    status,
                "prediction_path":
                    str(
                        prediction_path
                    ),
                "actual_path":
                    (
                        str(
                            actual_path
                        )
                        if actual_path
                        is not None
                        else ""
                    ),
                "prediction_sha256":
                    pred_checks[
                        "prediction_sha256"
                    ],
                "metadata_model":
                    metadata[
                        "metadata_model"
                    ],
                "metadata_weight_fingerprint":
                    metadata[
                        "metadata_weight_fingerprint"
                    ],
            }
        )

        print(
            f"prediction quality    : "
            f"{prediction_ok}"
        )
        print(
            f"metadata check        : "
            f"{metadata_ok}"
        )
        print(
            f"actual exists         : "
            f"{actual_quality['actual_exists']}"
        )
        print(
            f"actual quality        : "
            f"{actual_ok}"
        )
        print(
            f"status                : "
            f"{status}"
        )

        if status != "EVALUATED":
            continue

        assert actual is not None

        merged = prediction.merge(
            actual[
                [
                    "machine_no",
                    "machine_name",
                    "diff",
                    "win",
                    "plus1000",
                    "plus2000",
                ]
            ].rename(
                columns={
                    "machine_name":
                        "actual_machine_name",
                    "diff":
                        "actual_diff",
                    "win":
                        "actual_win",
                    "plus1000":
                        "actual_plus1000",
                    "plus2000":
                        "actual_plus2000",
                }
            ),
            on="machine_no",
            how="left",
            validate="one_to_one",
        )

        if merged[
            "actual_diff"
        ].isna().any():

            missing_nos = (
                merged.loc[
                    merged[
                        "actual_diff"
                    ].isna(),
                    "machine_no",
                ]
                .tolist()
            )

            raise RuntimeError(
                f"{filename_target_date.date()}: "
                f"prediction machine(s) missing "
                f"from actual data: {missing_nos}"
            )

        merged[
            "machine_name_match"
        ] = (
            merged[
                "machine_name"
            ].astype(str)
            == merged[
                "actual_machine_name"
            ].astype(str)
        )

        merged[
            "target_date"
        ] = filename_target_date

        merged[
            "prediction_file"
        ] = prediction_path.name

        merged[
            "prediction_sha256"
        ] = pred_checks[
            "prediction_sha256"
        ]

        detail_frames.append(
            merged
        )

        for (
            band_name,
            cutoff,
        ) in EVAL_BANDS.items():

            daily_rows.append(
                evaluate_band(
                    merged,
                    actual,
                    filename_target_date,
                    band_name,
                    cutoff,
                )
            )

    status_df = pd.DataFrame(
        status_rows
    )

    quality_df = pd.DataFrame(
        quality_rows
    )

    if detail_frames:

        detail_df = pd.concat(
            detail_frames,
            ignore_index=True,
        )

        detail_df = detail_df.sort_values(
            [
                "target_date",
                "prediction_rank",
            ]
        ).reset_index(
            drop=True
        )

    else:

        detail_df = pd.DataFrame()

    daily_df = pd.DataFrame(
        daily_rows
    )

    if not daily_df.empty:

        band_order = {
            name: i
            for i, name
            in enumerate(
                EVAL_BANDS.keys()
            )
        }

        daily_df[
            "_band_order"
        ] = daily_df[
            "band"
        ].map(
            band_order
        )

        daily_df = (
            daily_df.sort_values(
                [
                    "target_date",
                    "_band_order",
                ]
            )
            .drop(
                columns=[
                    "_band_order",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    overall_df = (
        build_overall_summary(
            daily_df
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    status_path = (
        OUTPUT_DIR
        / "69_live_prediction_status.csv"
    )

    quality_path = (
        OUTPUT_DIR
        / "69_live_prediction_data_quality.csv"
    )

    detail_path = (
        OUTPUT_DIR
        / "69_live_prediction_detail.csv"
    )

    daily_path = (
        OUTPUT_DIR
        / "69_live_prediction_daily.csv"
    )

    overall_path = (
        OUTPUT_DIR
        / "69_live_prediction_overall.csv"
    )

    status_df.to_csv(
        status_path,
        index=False,
        encoding="utf-8-sig",
    )

    quality_df.to_csv(
        quality_path,
        index=False,
        encoding="utf-8-sig",
    )

    detail_df.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    daily_df.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    overall_df.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    header(
        "STATUS"
    )

    if status_df.empty:

        print(
            "No 64 prediction files found."
        )

    else:

        print(
            status_df[
                [
                    "target_date",
                    "status",
                    "metadata_model",
                    "metadata_weight_fingerprint",
                ]
            ].to_string(
                index=False
            )
        )

    header(
        "DAILY RESULTS"
    )

    if daily_df.empty:

        print(
            "No target date could be evaluated yet."
        )

    else:

        display_cols = [
            "target_date",
            "band",
            "selected_n",
            "avg_diff",
            "win_rate",
            "plus1000_rate",
            "plus2000_rate",
            "total_diff",
            "store_avg_diff",
            "avg_diff_lift_vs_store",
            "excess_total_vs_store",
        ]

        print(
            daily_df[
                display_cols
            ].to_string(
                index=False
            )
        )

    header(
        "OVERALL RESULTS"
    )

    if overall_df.empty:

        print(
            "No overall result yet."
        )

    else:

        print(
            overall_df.to_string(
                index=False
            )
        )

    header(
        "FILES SAVED"
    )

    for path in (
        status_path,
        quality_path,
        detail_path,
        daily_path,
        overall_path,
    ):
        print(path)

    print()
    print(
        "69 live prediction backtest complete."
    )
    print(
        "64 prediction files were read only; "
        "they were not modified."
    )


if __name__ == "__main__":
    main()
