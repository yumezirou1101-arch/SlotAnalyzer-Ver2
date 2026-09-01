from __future__ import annotations

from pathlib import Path
import hashlib
import re
from zoneinfo import ZoneInfo

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

PREDICTION_CLASS_FORWARD_VALID = "FORWARD_VALID"
PREDICTION_CLASS_LEGACY = "LEGACY_UNVERIFIED"
PREDICTION_CLASS_GUARD_FAIL = "FORWARD_GUARD_FAIL"

JST = ZoneInfo("Asia/Tokyo")
EXPECTED_FORWARD_CUTOFF = (
    9,
    0,
)

NEW_GUARD_EVIDENCE_FIELDS = {
    "generated_at_jst",
    "forward_guard_version",
    "forward_valid",
    "forward_cutoff_jst",
    "target_actual_absent_at_generation",
    "target_source_absent_at_generation",
    "all514_sha256",
    "top10_sha256",
}

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


def parse_strict_bool(
    value,
) -> bool | None:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    text = str(
        value
    ).strip().lower()

    if text == "true":
        return True
    if text == "false":
        return False
    return None


def clean_text(
    value,
) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_forward_cutoff(
    value,
) -> tuple[int, int] | None:
    match = re.fullmatch(
        r"(\d{2}):(\d{2})\s+Asia/Tokyo",
        clean_text(value),
    )
    if match is None:
        return None

    hour = int(
        match.group(1)
    )
    minute = int(
        match.group(2)
    )
    if hour > 23 or minute > 59:
        return None
    return (
        hour,
        minute,
    )


def empty_metadata_result(
    path: Path,
    exists: bool,
) -> dict:
    return {
        "metadata_exists": exists,
        "metadata_path": str(path),
        "metadata_model": "",
        "metadata_weight_fingerprint": "",
        "metadata_target_date": pd.NaT,
        "metadata_latest_data_date": pd.NaT,
        "metadata_model_ok": False,
        "metadata_fingerprint_ok": False,
        "metadata_dates_ok": False,
        "prediction_class": PREDICTION_CLASS_LEGACY,
        "forward_guard_evidence_present": False,
        "forward_guard_ok": False,
        "forward_guard_fail_reasons": "",
        "metadata_generated_at_jst": pd.NaT,
        "metadata_forward_guard_version": "",
        "metadata_forward_valid": None,
        "metadata_top10_sha256": "",
        "metadata_all514_sha256": "",
        "current_all514_sha256": "",
        "top10_hash_ok": False,
        "all514_hash_ok": False,
    }


def load_metadata_for_target(
    target_date: pd.Timestamp,
    prediction_path: Path,
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
        return empty_metadata_result(
            path,
            False,
        )

    meta = read_csv_flexible(
        path
    )
    if meta.empty:
        return empty_metadata_result(
            path,
            True,
        )

    row = meta.iloc[0]
    meta_target = pd.to_datetime(
        row.get("target_date", pd.NaT),
        errors="coerce",
    )
    meta_latest = pd.to_datetime(
        row.get("latest_data_date", pd.NaT),
        errors="coerce",
    )
    model = clean_text(
        row.get("model", "")
    )
    fingerprint = clean_text(
        row.get("weight_fingerprint", "")
    )
    metadata_dates_ok = bool(
        pd.notna(meta_target)
        and pd.notna(meta_latest)
        and pd.Timestamp(meta_target).normalize()
        == target_date.normalize()
        and pd.Timestamp(meta_latest).normalize()
        < target_date.normalize()
    )

    evidence_present = any(
        field in meta.columns
        for field in NEW_GUARD_EVIDENCE_FIELDS
    )

    result = {
        "metadata_exists": True,
        "metadata_path": str(path),
        "metadata_model": model,
        "metadata_weight_fingerprint": fingerprint,
        "metadata_target_date": meta_target,
        "metadata_latest_data_date": meta_latest,
        "metadata_model_ok": model == EXPECTED_MODEL,
        "metadata_fingerprint_ok": (
            fingerprint == EXPECTED_WEIGHT_FINGERPRINT
        ),
        "metadata_dates_ok": metadata_dates_ok,
        "prediction_class": PREDICTION_CLASS_LEGACY,
        "forward_guard_evidence_present": evidence_present,
        "forward_guard_ok": False,
        "forward_guard_fail_reasons": "",
        "metadata_generated_at_jst": pd.NaT,
        "metadata_forward_guard_version": "",
        "metadata_forward_valid": None,
        "metadata_top10_sha256": "",
        "metadata_all514_sha256": "",
        "current_all514_sha256": "",
        "top10_hash_ok": False,
        "all514_hash_ok": False,
    }

    if not evidence_present:
        return result

    reasons = []
    guard_version = clean_text(
        row.get("forward_guard_version", "")
    )
    forward_valid = parse_strict_bool(
        row.get("forward_valid", None)
    )
    actual_absent = parse_strict_bool(
        row.get(
            "target_actual_absent_at_generation",
            None,
        )
    )
    source_absent = parse_strict_bool(
        row.get(
            "target_source_absent_at_generation",
            None,
        )
    )
    generated_at = pd.to_datetime(
        row.get("generated_at_jst", pd.NaT),
        errors="coerce",
    )
    cutoff = parse_forward_cutoff(
        row.get("forward_cutoff_jst", "")
    )
    top10_expected_hash = clean_text(
        row.get("top10_sha256", "")
    ).lower()
    all514_expected_hash = clean_text(
        row.get("all514_sha256", "")
    ).lower()
    all514_path = (
        PREDICTION_DIR
        / (
            "64_prediction_"
            f"{target_date.strftime('%Y%m%d')}"
            "_all514.csv"
        )
    )

    current_top10_hash = sha256_file(
        prediction_path
    )
    current_all514_hash = (
        sha256_file(all514_path)
        if all514_path.exists()
        else ""
    )
    top10_hash_ok = bool(
        top10_expected_hash
        and top10_expected_hash == current_top10_hash.lower()
    )
    all514_hash_ok = bool(
        all514_expected_hash
        and all514_expected_hash == current_all514_hash.lower()
    )

    if forward_valid is not True:
        reasons.append("forward_valid_not_true")
    if not guard_version:
        reasons.append("forward_guard_version_missing")
    if not result["metadata_model_ok"]:
        reasons.append("model_mismatch")
    if not result["metadata_fingerprint_ok"]:
        reasons.append("weight_fingerprint_mismatch")
    if (
        pd.isna(meta_target)
        or pd.Timestamp(meta_target).normalize()
        != target_date.normalize()
    ):
        reasons.append("metadata_target_date_mismatch")

    expected_latest = target_date.normalize() - pd.Timedelta(
        days=1
    )
    if (
        pd.isna(meta_latest)
        or pd.Timestamp(meta_latest).normalize()
        != expected_latest
    ):
        reasons.append("latest_data_date_not_target_minus_one")
    if actual_absent is not True:
        reasons.append("target_actual_absence_not_proven")
    if source_absent is not True:
        reasons.append("target_source_absence_not_proven")
    if cutoff is None:
        reasons.append("forward_cutoff_invalid")
    elif cutoff != EXPECTED_FORWARD_CUTOFF:
        reasons.append("forward_cutoff_not_0900_jst")

    generated_at_jst = pd.NaT
    if pd.isna(generated_at):
        reasons.append("generated_at_jst_invalid")
    elif getattr(generated_at, "tzinfo", None) is None:
        reasons.append("generated_at_jst_timezone_missing")
    else:
        generated_at_jst = pd.Timestamp(
            generated_at
        ).tz_convert(
            JST
        )
        generated_date = pd.Timestamp(
            generated_at_jst.date()
        )
        if generated_date > target_date.normalize():
            reasons.append("generated_after_target_date")
        elif (
            generated_date == target_date.normalize()
            and cutoff is not None
            and (
                generated_at_jst.hour,
                generated_at_jst.minute,
                generated_at_jst.second,
                generated_at_jst.microsecond,
            ) >= (
                cutoff[0],
                cutoff[1],
                0,
                0,
            )
        ):
            reasons.append("generated_at_or_after_forward_cutoff")

    if not top10_hash_ok:
        reasons.append("top10_sha256_mismatch")
    if not all514_hash_ok:
        reasons.append("all514_sha256_mismatch")

    result.update(
        {
            "prediction_class": (
                PREDICTION_CLASS_FORWARD_VALID
                if not reasons
                else PREDICTION_CLASS_GUARD_FAIL
            ),
            "forward_guard_ok": not reasons,
            "forward_guard_fail_reasons": ";".join(reasons),
            "metadata_generated_at_jst": generated_at_jst,
            "metadata_forward_guard_version": guard_version,
            "metadata_forward_valid": forward_valid,
            "metadata_top10_sha256": top10_expected_hash,
            "metadata_all514_sha256": all514_expected_hash,
            "current_all514_sha256": current_all514_hash,
            "top10_hash_ok": top10_hash_ok,
            "all514_hash_ok": all514_hash_ok,
        }
    )
    return result


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
# TOP10 VS OUTSIDE TOP10 VS STORE
# ============================================================

def build_top10_comparison_daily_row(
    merged: pd.DataFrame,
    actual: pd.DataFrame,
    target_date: pd.Timestamp,
) -> dict:

    top10 = (
        merged[
            merged["prediction_rank"] <= 10
        ]
        .sort_values("prediction_rank")
        .copy()
    )

    if len(top10) != 10:
        raise RuntimeError(
            f"{target_date.date()}: "
            f"expected 10 TOP10 rows, got {len(top10)}"
        )

    top10_machine_nos = set(
        pd.to_numeric(
            top10["machine_no"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    if len(top10_machine_nos) != 10:
        raise RuntimeError(
            f"{target_date.date()}: "
            "TOP10 machine numbers are not 10 unique values"
        )

    actual_work = actual.copy()

    actual_work["machine_no"] = pd.to_numeric(
        actual_work["machine_no"],
        errors="coerce",
    )

    actual_work["diff"] = pd.to_numeric(
        actual_work["diff"],
        errors="coerce",
    )

    actual_work = actual_work.dropna(
        subset=[
            "machine_no",
            "diff",
        ]
    ).copy()

    actual_work["machine_no"] = (
        actual_work["machine_no"].astype(int)
    )

    store = actual_work.copy()

    outside = actual_work[
        ~actual_work["machine_no"].isin(
            top10_machine_nos
        )
    ].copy()

    top10_actual = actual_work[
        actual_work["machine_no"].isin(
            top10_machine_nos
        )
    ].copy()

    if len(top10_actual) != 10:
        raise RuntimeError(
            f"{target_date.date()}: "
            f"expected 10 TOP10 actual rows, "
            f"got {len(top10_actual)}"
        )

    expected_outside_n = (
        len(store) - 10
    )

    if len(outside) != expected_outside_n:
        raise RuntimeError(
            f"{target_date.date()}: "
            f"expected {expected_outside_n} outside rows, "
            f"got {len(outside)}"
        )

    def metrics(
        frame: pd.DataFrame,
    ) -> dict:

        diffs = frame["diff"].astype(float)

        return {
            "n": int(len(frame)),
            "avg_diff": float(diffs.mean()),
            "median_diff": float(diffs.median()),
            "win_rate": float(
                (diffs > 0).mean() * 100.0
            ),
            "plus1000_rate": float(
                (diffs >= 1000).mean() * 100.0
            ),
            "plus2000_rate": float(
                (diffs >= 2000).mean() * 100.0
            ),
            "total_diff": float(diffs.sum()),
        }

    top10_m = metrics(top10_actual)
    outside_m = metrics(outside)
    store_m = metrics(store)

    return {
        "target_date": target_date,

        "top10_n": top10_m["n"],
        "top10_avg_diff": top10_m["avg_diff"],
        "top10_median_diff": top10_m["median_diff"],
        "top10_win_rate": top10_m["win_rate"],
        "top10_plus1000_rate":
            top10_m["plus1000_rate"],
        "top10_plus2000_rate":
            top10_m["plus2000_rate"],
        "top10_total_diff":
            top10_m["total_diff"],

        "outside_top10_n":
            outside_m["n"],
        "outside_top10_avg_diff":
            outside_m["avg_diff"],
        "outside_top10_median_diff":
            outside_m["median_diff"],
        "outside_top10_win_rate":
            outside_m["win_rate"],
        "outside_top10_plus1000_rate":
            outside_m["plus1000_rate"],
        "outside_top10_plus2000_rate":
            outside_m["plus2000_rate"],
        "outside_top10_total_diff":
            outside_m["total_diff"],

        "store_n": store_m["n"],
        "store_avg_diff":
            store_m["avg_diff"],
        "store_median_diff":
            store_m["median_diff"],
        "store_win_rate":
            store_m["win_rate"],
        "store_plus1000_rate":
            store_m["plus1000_rate"],
        "store_plus2000_rate":
            store_m["plus2000_rate"],
        "store_total_diff":
            store_m["total_diff"],

        "top10_avg_diff_lift_vs_outside":
            float(
                top10_m["avg_diff"]
                - outside_m["avg_diff"]
            ),

        "top10_avg_diff_lift_vs_store":
            float(
                top10_m["avg_diff"]
                - store_m["avg_diff"]
            ),

        "top10_win_rate_lift_vs_outside":
            float(
                top10_m["win_rate"]
                - outside_m["win_rate"]
            ),

        "top10_win_rate_lift_vs_store":
            float(
                top10_m["win_rate"]
                - store_m["win_rate"]
            ),
    }




def build_top10_comparison_overall(
    comparison_daily_df: pd.DataFrame,
) -> pd.DataFrame:

    if comparison_daily_df.empty:
        return pd.DataFrame()

    x = comparison_daily_df.copy()

    lift_vs_outside_values = (
        x[
            "top10_avg_diff_lift_vs_outside"
        ]
        .astype(float)
        .to_numpy()
    )

    lift_vs_store_values = (
        x[
            "top10_avg_diff_lift_vs_store"
        ]
        .astype(float)
        .to_numpy()
    )

    outside_ci_low, outside_ci_high = (
        bootstrap_mean_ci(
            lift_vs_outside_values
        )
    )

    store_ci_low, store_ci_high = (
        bootstrap_mean_ci(
            lift_vs_store_values
        )
    )

    return pd.DataFrame(
        [
            {
                "evaluated_days": int(
                    x["target_date"].nunique()
                ),

                "mean_top10_avg_diff": float(
                    x["top10_avg_diff"].mean()
                ),

                "mean_outside_top10_avg_diff": float(
                    x[
                        "outside_top10_avg_diff"
                    ].mean()
                ),

                "mean_store_avg_diff": float(
                    x["store_avg_diff"].mean()
                ),

                "mean_top10_win_rate": float(
                    x["top10_win_rate"].mean()
                ),

                "mean_outside_top10_win_rate": float(
                    x[
                        "outside_top10_win_rate"
                    ].mean()
                ),

                "mean_store_win_rate": float(
                    x["store_win_rate"].mean()
                ),

                "mean_top10_plus1000_rate": float(
                    x[
                        "top10_plus1000_rate"
                    ].mean()
                ),

                "mean_outside_top10_plus1000_rate": float(
                    x[
                        "outside_top10_plus1000_rate"
                    ].mean()
                ),

                "mean_store_plus1000_rate": float(
                    x[
                        "store_plus1000_rate"
                    ].mean()
                ),

                "mean_top10_plus2000_rate": float(
                    x[
                        "top10_plus2000_rate"
                    ].mean()
                ),

                "mean_outside_top10_plus2000_rate": float(
                    x[
                        "outside_top10_plus2000_rate"
                    ].mean()
                ),

                "mean_store_plus2000_rate": float(
                    x[
                        "store_plus2000_rate"
                    ].mean()
                ),

                "mean_top10_avg_diff_lift_vs_outside":
                    float(
                        x[
                            "top10_avg_diff_lift_vs_outside"
                        ].mean()
                    ),

                "mean_top10_avg_diff_lift_vs_store":
                    float(
                        x[
                            "top10_avg_diff_lift_vs_store"
                        ].mean()
                    ),

                "mean_top10_win_rate_lift_vs_outside":
                    float(
                        x[
                            "top10_win_rate_lift_vs_outside"
                        ].mean()
                    ),

                "mean_top10_win_rate_lift_vs_store":
                    float(
                        x[
                            "top10_win_rate_lift_vs_store"
                        ].mean()
                    ),

                "top10_total_diff": float(
                    x["top10_total_diff"].sum()
                ),

                "outside_top10_total_diff": float(
                    x[
                        "outside_top10_total_diff"
                    ].sum()
                ),

                "store_total_diff": float(
                    x["store_total_diff"].sum()
                ),

                "mean_lift_vs_outside_ci95_low":
                    outside_ci_low,

                "mean_lift_vs_outside_ci95_high":
                    outside_ci_high,

                "mean_lift_vs_store_ci95_low":
                    store_ci_low,

                "mean_lift_vs_store_ci95_high":
                    store_ci_high,
            }
        ]
    )


def filter_prediction_class(
    frame: pd.DataFrame,
    prediction_class: str,
) -> pd.DataFrame:
    if (
        frame.empty
        or "prediction_class" not in frame.columns
    ):
        return pd.DataFrame(
            columns=frame.columns
        )
    return (
        frame[
            frame["prediction_class"]
            == prediction_class
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def next_forward_checkpoint(
    evaluated_days: int,
) -> str:
    if evaluated_days < 10:
        return "10"
    if evaluated_days < 15:
        return "15"
    if evaluated_days < 21:
        return "21"
    return "FORMAL_REVIEW_READY"


def build_forward_summary(
    status_df: pd.DataFrame,
) -> pd.DataFrame:
    if status_df.empty:
        forward_days = 0
        legacy_days = 0
    else:
        forward_days = int(
            status_df.loc[
                status_df["status"]
                == "EVALUATED_FORWARD_VALID",
                "target_date",
            ].nunique()
        )
        legacy_days = int(
            status_df.loc[
                status_df["status"]
                == "EVALUATED_LEGACY_UNVERIFIED",
                "target_date",
            ].nunique()
        )

    return pd.DataFrame(
        [
            {
                "forward_valid_evaluated_days":
                    forward_days,
                "legacy_evaluated_days":
                    legacy_days,
                "next_checkpoint":
                    next_forward_checkpoint(
                        forward_days
                    ),
            }
        ]
    )


def build_forward_coverage(
    actual_map: dict[pd.Timestamp, Path],
    prediction_files: list[tuple[pd.Timestamp, Path]],
    status_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "date",
        "actual_exists",
        "prediction_exists",
        "prediction_class",
        "evaluation_status",
    ]
    if status_df.empty:
        return pd.DataFrame(
            columns=columns
        )

    evidence_rows = status_df[
        status_df["prediction_class"].isin(
            [
                PREDICTION_CLASS_FORWARD_VALID,
                PREDICTION_CLASS_GUARD_FAIL,
            ]
        )
    ]
    if evidence_rows.empty:
        return pd.DataFrame(
            columns=columns
        )

    guard_start = pd.Timestamp(
        evidence_rows["target_date"].min()
    ).normalize()
    prediction_map = {
        pd.Timestamp(date).normalize(): path
        for date, path in prediction_files
    }
    status_map = {
        pd.Timestamp(row["target_date"]).normalize(): row
        for _, row in status_df.iterrows()
    }
    dates = sorted(
        {
            pd.Timestamp(date).normalize()
            for date in actual_map
        }
        | set(
            prediction_map
        )
    )

    rows = []
    for date in dates:
        if date < guard_start:
            continue

        actual_exists = date in actual_map
        prediction_exists = date in prediction_map
        status_row = status_map.get(
            date
        )

        if prediction_exists and status_row is not None:
            prediction_class = status_row[
                "prediction_class"
            ]
            evaluation_status = status_row[
                "status"
            ]
        elif actual_exists:
            prediction_class = "MISSING"
            evaluation_status = (
                "MISSING_FROZEN_PREDICTION"
            )
        else:
            prediction_class = ""
            evaluation_status = "NO_DATA"

        rows.append(
            {
                "date": date,
                "actual_exists": actual_exists,
                "prediction_exists": prediction_exists,
                "prediction_class": prediction_class,
                "evaluation_status": evaluation_status,
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def resolve_evaluation_status(
    prediction_ok: bool,
    metadata_ok: bool,
    prediction_class: str,
    actual_exists: bool,
    actual_ok: bool,
) -> str:
    if not prediction_ok:
        return "SKIPPED_PREDICTION_QUALITY_FAIL"
    if prediction_class == PREDICTION_CLASS_GUARD_FAIL:
        return "SKIPPED_FORWARD_GUARD_FAIL"
    if not metadata_ok:
        return "SKIPPED_METADATA_CHECK_FAIL"
    if not actual_exists:
        return (
            "PENDING_FORWARD_VALID"
            if prediction_class
            == PREDICTION_CLASS_FORWARD_VALID
            else "PENDING_LEGACY_UNVERIFIED"
        )
    if not actual_ok:
        return "SKIPPED_ACTUAL_QUALITY_FAIL"
    return (
        "EVALUATED_FORWARD_VALID"
        if prediction_class
        == PREDICTION_CLASS_FORWARD_VALID
        else "EVALUATED_LEGACY_UNVERIFIED"
    )



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
    comparison_rows = []

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
                filename_target_date,
                prediction_path,
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

        prediction_class = metadata[
            "prediction_class"
        ]

        status = resolve_evaluation_status(
            prediction_ok,
            metadata_ok,
            prediction_class,
            actual_quality["actual_exists"],
            actual_ok,
        )

        status_rows.append(
            {
                "target_date":
                    filename_target_date,
                "status":
                    status,
                "prediction_class":
                    prediction_class,
                "forward_guard_fail_reasons":
                    metadata[
                        "forward_guard_fail_reasons"
                    ],
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
            f"prediction class      : "
            f"{prediction_class}"
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

        if status not in (
            "EVALUATED_FORWARD_VALID",
            "EVALUATED_LEGACY_UNVERIFIED",
        ):
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
        merged[
            "prediction_class"
        ] = prediction_class

        detail_frames.append(
            merged
        )

        for (
            band_name,
            cutoff,
        ) in EVAL_BANDS.items():

            daily_row = evaluate_band(
                    merged,
                    actual,
                    filename_target_date,
                    band_name,
                    cutoff,
                )
            daily_row[
                "prediction_class"
            ] = prediction_class
            daily_rows.append(
                daily_row
            )

        comparison_row = build_top10_comparison_daily_row(
                merged,
                actual,
                filename_target_date,
            )
        comparison_row[
            "prediction_class"
        ] = prediction_class
        comparison_rows.append(
            comparison_row
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

    comparison_daily_df = pd.DataFrame(
        comparison_rows
    )

    if not comparison_daily_df.empty:
        comparison_daily_df = (
            comparison_daily_df
            .sort_values(
                "target_date"
            )
            .reset_index(
                drop=True
            )
        )

    comparison_overall_df = (
        build_top10_comparison_overall(
            comparison_daily_df
        )
    )

    forward_detail_df = filter_prediction_class(
        detail_df,
        PREDICTION_CLASS_FORWARD_VALID,
    )
    legacy_detail_df = filter_prediction_class(
        detail_df,
        PREDICTION_CLASS_LEGACY,
    )
    forward_daily_df = filter_prediction_class(
        daily_df,
        PREDICTION_CLASS_FORWARD_VALID,
    )
    legacy_daily_df = filter_prediction_class(
        daily_df,
        PREDICTION_CLASS_LEGACY,
    )
    forward_overall_df = build_overall_summary(
        forward_daily_df
    )
    legacy_overall_df = build_overall_summary(
        legacy_daily_df
    )
    if forward_overall_df.empty and not legacy_overall_df.empty:
        forward_overall_df = legacy_overall_df.head(
            0
        ).copy()
    if legacy_overall_df.empty and not forward_overall_df.empty:
        legacy_overall_df = forward_overall_df.head(
            0
        ).copy()
    forward_comparison_daily_df = filter_prediction_class(
        comparison_daily_df,
        PREDICTION_CLASS_FORWARD_VALID,
    )
    legacy_comparison_daily_df = filter_prediction_class(
        comparison_daily_df,
        PREDICTION_CLASS_LEGACY,
    )
    forward_comparison_overall_df = (
        build_top10_comparison_overall(
            forward_comparison_daily_df
        )
    )
    legacy_comparison_overall_df = (
        build_top10_comparison_overall(
            legacy_comparison_daily_df
        )
    )
    if (
        forward_comparison_overall_df.empty
        and not legacy_comparison_overall_df.empty
    ):
        forward_comparison_overall_df = (
            legacy_comparison_overall_df.head(
                0
            ).copy()
        )
    if (
        legacy_comparison_overall_df.empty
        and not forward_comparison_overall_df.empty
    ):
        legacy_comparison_overall_df = (
            forward_comparison_overall_df.head(
                0
            ).copy()
        )
    coverage_df = build_forward_coverage(
        actual_map,
        prediction_files,
        status_df,
    )
    forward_summary_df = build_forward_summary(
        status_df
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

    comparison_daily_path = (
        OUTPUT_DIR
        / "69_top10_vs_outside_daily.csv"
    )

    comparison_overall_path = (
        OUTPUT_DIR
        / "69_top10_vs_outside_overall.csv"
    )

    forward_detail_path = (
        OUTPUT_DIR
        / "69_forward_valid_detail.csv"
    )
    forward_daily_path = (
        OUTPUT_DIR
        / "69_forward_valid_daily.csv"
    )
    forward_overall_path = (
        OUTPUT_DIR
        / "69_forward_valid_overall.csv"
    )
    legacy_detail_path = (
        OUTPUT_DIR
        / "69_legacy_unverified_detail.csv"
    )
    legacy_daily_path = (
        OUTPUT_DIR
        / "69_legacy_unverified_daily.csv"
    )
    legacy_overall_path = (
        OUTPUT_DIR
        / "69_legacy_unverified_overall.csv"
    )
    forward_comparison_daily_path = (
        OUTPUT_DIR
        / "69_forward_valid_top10_vs_outside_daily.csv"
    )
    forward_comparison_overall_path = (
        OUTPUT_DIR
        / "69_forward_valid_top10_vs_outside_overall.csv"
    )
    legacy_comparison_daily_path = (
        OUTPUT_DIR
        / "69_legacy_unverified_top10_vs_outside_daily.csv"
    )
    legacy_comparison_overall_path = (
        OUTPUT_DIR
        / "69_legacy_unverified_top10_vs_outside_overall.csv"
    )
    coverage_path = (
        OUTPUT_DIR
        / "69_forward_coverage.csv"
    )
    forward_summary_path = (
        OUTPUT_DIR
        / "69_forward_summary.csv"
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

    comparison_daily_df.to_csv(
        comparison_daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_overall_df.to_csv(
        comparison_overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    for frame, path in (
        (
            forward_detail_df,
            forward_detail_path,
        ),
        (
            forward_daily_df,
            forward_daily_path,
        ),
        (
            forward_overall_df,
            forward_overall_path,
        ),
        (
            legacy_detail_df,
            legacy_detail_path,
        ),
        (
            legacy_daily_df,
            legacy_daily_path,
        ),
        (
            legacy_overall_df,
            legacy_overall_path,
        ),
        (
            forward_comparison_daily_df,
            forward_comparison_daily_path,
        ),
        (
            forward_comparison_overall_df,
            forward_comparison_overall_path,
        ),
        (
            legacy_comparison_daily_df,
            legacy_comparison_daily_path,
        ),
        (
            legacy_comparison_overall_df,
            legacy_comparison_overall_path,
        ),
        (
            coverage_df,
            coverage_path,
        ),
        (
            forward_summary_df,
            forward_summary_path,
        ),
    ):
        frame.to_csv(
            path,
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
                    "prediction_class",
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
        "FORWARD VALID PROGRESS"
    )
    print(
        forward_summary_df.to_string(
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
        comparison_daily_path,
        comparison_overall_path,
        forward_detail_path,
        forward_daily_path,
        forward_overall_path,
        legacy_detail_path,
        legacy_daily_path,
        legacy_overall_path,
        forward_comparison_daily_path,
        forward_comparison_overall_path,
        legacy_comparison_daily_path,
        legacy_comparison_overall_path,
        coverage_path,
        forward_summary_path,
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
