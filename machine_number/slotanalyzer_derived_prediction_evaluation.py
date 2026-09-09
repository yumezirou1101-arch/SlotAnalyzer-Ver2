from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from slotanalyzer_derived_prediction_evidence import (
    FORMALIZATION_START_DATE,
    SCHEMA_VERSION,
    DerivedPredictionError,
    sha256_file,
    verify_derived_prediction,
)
from slotanalyzer_evaluation_quarantine import assess_evaluation_quarantine


PREDICTION_CLASS_FORWARD_VALID = "FORWARD_VALID"
PREDICTION_CLASS_LEGACY = "LEGACY_UNVERIFIED"
PREDICTION_CLASS_GUARD_FAIL = "FORWARD_GUARD_FAIL"
FORMAL_BANDS = {"TOP3": 3, "TOP5": 5, "TOP10": 10}
CATEGORIES = {
    "A_TYPE": {
        "directory": "74_Ver4_2_A_type_prediction",
        "pattern": re.compile(r"74_A_type_prediction_(\d{8})_top10\.csv", re.I),
        "metadata": "74_A_type_prediction_{ymd}_metadata.csv",
        "rank": "a_type_rank",
        "stage": "74_A_TYPE",
    },
    "JUGGLER": {
        "directory": "75_Ver4_2_Juggler_prediction",
        "pattern": re.compile(r"75_Juggler_prediction_(\d{8})_top10\.csv", re.I),
        "metadata": "75_Juggler_prediction_{ymd}_metadata.csv",
        "rank": "juggler_rank",
        "stage": "75_JUGGLER",
    },
}
STATUS_COLUMNS = [
    "target_date", "category", "status", "prediction_class", "reason",
    "prediction_path", "metadata_path", "actual_path", "prediction_sha256", "metadata_sha256", "actual_sha256",
    "evaluation_eligible", "quarantine_reason_code",
]
DETAIL_COLUMNS = [
    "target_date", "category", "rank", "machine_no", "machine_name", "score",
    "actual_diff", "actual_win", "actual_ge_1000", "actual_ge_2000",
    "prediction_class", "evaluation_status", "prediction_sha256", "metadata_sha256",
    "actual_sha256", "prediction_file", "actual_file",
]
DAILY_COLUMNS = [
    "target_date", "category", "band", "n", "wins", "win_rate", "avg_diff", "sum_diff",
    "ge_1000", "ge_1000_rate", "ge_2000", "ge_2000_rate", "prediction_class",
    "evaluation_status", "prediction_sha256", "metadata_sha256", "actual_sha256",
]
COVERAGE_COLUMNS = [
    "target_date", "category", "prediction_exists", "actual_exists", "prediction_quality_ok",
    "actual_quality_ok", "formal_evidence_ok", "prediction_class", "evaluation_status",
    "detail_rank_count", "rank1_10_complete", "formal_evaluation_complete",
    "prediction_sha256", "metadata_sha256", "actual_sha256", "actual_filename_date_ok", "actual_internal_date_ok",
    "evaluation_eligible", "quarantine_reason_code",
]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    error = None
    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            error = exc
    raise RuntimeError(f"CSV read failed: {path}; last_error={error}")


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def discover_predictions(analysis_dir: Path):
    found = []
    for category, spec in CATEGORIES.items():
        directory = analysis_dir / spec["directory"]
        if not directory.exists():
            continue
        for path in directory.glob("*.csv"):
            match = spec["pattern"].fullmatch(path.name)
            if not match:
                continue
            parsed = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
            if not pd.isna(parsed):
                found.append((category, pd.Timestamp(parsed).date(), path))
    return sorted(found, key=lambda item: (item[1], item[0]))


def discover_actuals(data_dir: Path) -> dict:
    result = {}
    pattern = re.compile(r"ana_slo_(\d{8})\.csv", re.I)
    for path in data_dir.glob("ana_slo_????????.csv"):
        match = pattern.fullmatch(path.name)
        if match:
            parsed = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
            if not pd.isna(parsed):
                result[pd.Timestamp(parsed).date()] = path
    return result


def validate_prediction_quality(frame: pd.DataFrame, category: str, target_date, latest_date) -> pd.DataFrame:
    rank_col = CATEGORIES[category]["rank"]
    required = {"machine_no", "machine_name", "score", rank_col, "target_date", "latest_data_date"}
    if not required.issubset(frame.columns):
        raise ValueError(f"prediction columns missing: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    result[rank_col] = pd.to_numeric(result[rank_col], errors="coerce")
    result["machine_no"] = pd.to_numeric(result["machine_no"], errors="coerce")
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    if len(result) != 10:
        raise ValueError("prediction Top10 must contain exactly 10 rows")
    if result[[rank_col, "machine_no", "score", "machine_name"]].isna().any().any():
        raise ValueError("prediction contains required-value nulls")
    if result["machine_name"].astype(str).str.strip().eq("").any():
        raise ValueError("prediction contains an empty machine_name")
    if result[rank_col].duplicated().any() or set(result[rank_col].astype(int)) != set(range(1, 11)):
        raise ValueError("prediction ranks are not exactly 1-10")
    if result["machine_no"].duplicated().any():
        raise ValueError("prediction machine_no is duplicated")
    targets = set(pd.to_datetime(result["target_date"], errors="coerce").dt.date)
    latest = set(pd.to_datetime(result["latest_data_date"], errors="coerce").dt.date)
    if targets != {target_date} or latest != {latest_date}:
        raise ValueError("prediction target/latest date mismatch")
    result[rank_col] = result[rank_col].astype(int)
    result["machine_no"] = result["machine_no"].astype(int)
    return result.sort_values(rank_col).reset_index(drop=True)


def validate_actual_quality(path: Path, target_date) -> pd.DataFrame:
    if path.name != f"ana_slo_{target_date:%Y%m%d}.csv":
        raise ValueError("actual filename date mismatch")
    frame = read_csv_flexible(path)
    aliases = {}
    for canonical, choices in {
        "date": ("date", "日付"), "machine_no": ("machine_no", "台番号"),
        "machine_name": ("machine_name", "機種名"), "diff": ("diff", "差枚"),
    }.items():
        column = next((name for name in choices if name in frame.columns), None)
        if column is None:
            raise ValueError(f"actual column missing: {canonical}")
        aliases[column] = canonical
    result = frame.rename(columns=aliases).copy()
    dates = pd.to_datetime(result["date"], errors="coerce")
    result["machine_no"] = pd.to_numeric(result["machine_no"], errors="coerce")
    result["diff"] = pd.to_numeric(
        result["diff"].astype(str).str.replace(",", "", regex=False).str.replace("+", "", regex=False),
        errors="coerce",
    )
    if len(result) != 514 or dates.isna().any() or set(dates.dt.date) != {target_date}:
        raise ValueError("actual row count or internal date is invalid")
    if result["machine_no"].isna().any() or result["machine_no"].duplicated().any() or result["machine_no"].nunique() != 514:
        raise ValueError("actual machine inventory is invalid")
    if result["diff"].isna().any():
        raise ValueError("actual diff is missing or non-numeric")
    if result["machine_name"].isna().any() or result["machine_name"].astype(str).str.strip().eq("").any():
        raise ValueError("actual machine_name is missing")
    result["machine_no"] = result["machine_no"].astype(int)
    return result[["machine_no", "machine_name", "diff"]]


def classify_prediction(analysis_dir: Path, category: str, target_date, prediction_path: Path):
    spec = CATEGORIES[category]
    directory = analysis_dir / spec["directory"]
    metadata_path = directory / spec["metadata"].format(ymd=target_date.strftime("%Y%m%d"))
    if target_date < FORMALIZATION_START_DATE:
        return PREDICTION_CLASS_LEGACY, None, "formalization_start_date_not_reached", metadata_path
    if not metadata_path.exists():
        return PREDICTION_CLASS_LEGACY, None, "formal_schema_missing", metadata_path
    try:
        metadata = read_csv_flexible(metadata_path)
    except Exception as exc:
        return PREDICTION_CLASS_GUARD_FAIL, None, f"metadata_read_error:{type(exc).__name__}", metadata_path
    schema = str(metadata.iloc[0].get("schema_version", "")).strip() if len(metadata) == 1 else ""
    if not schema:
        return PREDICTION_CLASS_LEGACY, None, "formal_schema_missing", metadata_path
    if schema != SCHEMA_VERSION:
        return PREDICTION_CLASS_GUARD_FAIL, None, "formal_schema_mismatch", metadata_path
    try:
        verification = verify_derived_prediction(
            directory,
            analysis_dir / "64_Ver4_2_future_top10",
            target_date,
            category,
        )
        status79 = analysis_dir / "79_one_click_prediction_pipeline" / f"79_pipeline_{target_date:%Y%m%d}_status.csv"
        rows79 = read_csv_flexible(status79)
        rows79 = rows79[rows79["stage"].astype(str) == spec["stage"]]
        if len(rows79) != 1:
            raise DerivedPredictionError("79 stage evidence missing or duplicated")
        row79 = rows79.iloc[0]
        if not _truthy(row79.get("pipeline_complete")):
            raise DerivedPredictionError("79 pipeline_complete is false")
        if str(row79.get("prediction_class", "")).strip() != PREDICTION_CLASS_FORWARD_VALID:
            raise DerivedPredictionError("79 prediction_class mismatch")
        if str(row79.get("metadata_sha256", "")).strip().lower() != verification.metadata_sha256:
            raise DerivedPredictionError("79 metadata SHA-256 mismatch")
        return PREDICTION_CLASS_FORWARD_VALID, verification, "", metadata_path
    except Exception as exc:
        return PREDICTION_CLASS_GUARD_FAIL, None, f"{type(exc).__name__}:{exc}", metadata_path


def evaluate_formal_predictions(project_root: Path, data_dir: Path, analysis_dir: Path, output_dir: Path):
    actuals = discover_actuals(data_dir)
    status_rows, detail_frames, daily_rows, coverage_rows = [], [], [], []
    for category, target_date, prediction_path in discover_predictions(analysis_dir):
        prediction_class, verification, reason, metadata_path = classify_prediction(
            analysis_dir, category, target_date, prediction_path
        )
        prediction_sha = sha256_file(prediction_path)
        metadata_sha = sha256_file(metadata_path) if metadata_path.exists() else ""
        source64_dir = analysis_dir / "64_Ver4_2_future_top10"
        ymd = target_date.strftime("%Y%m%d")
        quarantine = assess_evaluation_quarantine(
            project_root,
            "MARUHAN_MAEBASHI",
            target_date,
            category,
            prediction_path,
            metadata_path,
            source64_dir / f"64_prediction_{ymd}_all514.csv",
            source64_dir / f"64_prediction_{ymd}_metadata.csv",
        )
        latest_date = target_date - pd.Timedelta(days=1)
        latest_date = latest_date.date() if hasattr(latest_date, "date") else latest_date
        prediction_ok = False
        prediction = None
        prediction_error = ""
        try:
            prediction = validate_prediction_quality(
                read_csv_flexible(prediction_path), category, target_date, latest_date
            )
            prediction_ok = True
        except Exception as exc:
            prediction_error = f"{type(exc).__name__}:{exc}"

        actual_path = actuals.get(target_date)
        actual_exists = actual_path is not None
        actual_sha = sha256_file(actual_path) if actual_path is not None else ""
        actual_ok = False
        actual = None
        actual_error = ""
        if actual_exists:
            try:
                actual = validate_actual_quality(actual_path, target_date)
                if prediction is not None and not set(prediction["machine_no"]).issubset(set(actual["machine_no"])):
                    raise ValueError("Top10 machine is missing from actual")
                actual_ok = True
            except Exception as exc:
                actual_error = f"{type(exc).__name__}:{exc}"

        if not prediction_ok:
            evaluation_status = "SKIPPED_PREDICTION_QUALITY_FAIL"
        elif prediction_class == PREDICTION_CLASS_GUARD_FAIL:
            evaluation_status = (
                "SKIPPED_METADATA_CHECK_FAIL"
                if reason.startswith(("metadata_read_error", "formal_schema_mismatch"))
                else "SKIPPED_FORWARD_GUARD_FAIL"
            )
        elif not actual_exists:
            evaluation_status = (
                "PENDING_FORWARD_VALID" if prediction_class == PREDICTION_CLASS_FORWARD_VALID
                else "PENDING_LEGACY_UNVERIFIED"
            )
        elif not actual_ok:
            evaluation_status = "SKIPPED_ACTUAL_QUALITY_FAIL"
        else:
            evaluation_status = (
                "EVALUATED_FORWARD_VALID" if prediction_class == PREDICTION_CLASS_FORWARD_VALID
                else "EVALUATED_LEGACY_UNVERIFIED"
            )
        if quarantine.quarantined:
            prediction_class = PREDICTION_CLASS_FORWARD_VALID
            evaluation_status = quarantine.status

        status_rows.append({
            "target_date": target_date, "category": category,
            "status": evaluation_status, "prediction_class": prediction_class,
            "reason": quarantine.reason or reason or prediction_error or actual_error,
            "prediction_path": str(prediction_path),
            "metadata_path": str(metadata_path) if metadata_path.exists() else "",
            "actual_path": str(actual_path) if actual_path else "",
            "prediction_sha256": prediction_sha, "metadata_sha256": metadata_sha,
            "actual_sha256": actual_sha,
            "evaluation_eligible": quarantine.evaluation_eligible,
            "quarantine_reason_code": quarantine.reason_code,
        })

        formal_complete = evaluation_status == "EVALUATED_FORWARD_VALID"
        detail_rank_count = 0
        rank_complete = False
        if formal_complete:
            rank_col = CATEGORIES[category]["rank"]
            merged = prediction.merge(
                actual.rename(columns={"machine_name": "actual_machine_name", "diff": "actual_diff"}),
                on="machine_no", how="left", validate="one_to_one",
            )
            merged["actual_win"] = (merged["actual_diff"] > 0).astype(int)
            merged["actual_ge_1000"] = (merged["actual_diff"] >= 1000).astype(int)
            merged["actual_ge_2000"] = (merged["actual_diff"] >= 2000).astype(int)
            detail = pd.DataFrame({
                "target_date": target_date, "category": category,
                "rank": merged[rank_col].astype(int), "machine_no": merged["machine_no"].astype(int),
                "machine_name": merged["machine_name"], "score": merged["score"],
                "actual_diff": merged["actual_diff"], "actual_win": merged["actual_win"],
                "actual_ge_1000": merged["actual_ge_1000"], "actual_ge_2000": merged["actual_ge_2000"],
                "prediction_class": PREDICTION_CLASS_FORWARD_VALID,
                "evaluation_status": evaluation_status,
                "prediction_sha256": prediction_sha, "metadata_sha256": metadata_sha,
                "actual_sha256": actual_sha,
                "prediction_file": prediction_path.name, "actual_file": actual_path.name,
            })
            detail_frames.append(detail)
            detail_rank_count = len(detail)
            rank_complete = set(detail["rank"]) == set(range(1, 11))
            for band, cutoff in FORMAL_BANDS.items():
                selected = detail[detail["rank"] <= cutoff]
                daily_rows.append({
                    "target_date": target_date, "category": category, "band": band,
                    "n": len(selected), "wins": int(selected["actual_win"].sum()),
                    "win_rate": float(selected["actual_win"].mean() * 100),
                    "avg_diff": float(selected["actual_diff"].mean()),
                    "sum_diff": float(selected["actual_diff"].sum()),
                    "ge_1000": int(selected["actual_ge_1000"].sum()),
                    "ge_1000_rate": float(selected["actual_ge_1000"].mean() * 100),
                    "ge_2000": int(selected["actual_ge_2000"].sum()),
                    "ge_2000_rate": float(selected["actual_ge_2000"].mean() * 100),
                    "prediction_class": PREDICTION_CLASS_FORWARD_VALID,
                    "evaluation_status": evaluation_status,
                    "prediction_sha256": prediction_sha, "metadata_sha256": metadata_sha,
                    "actual_sha256": actual_sha,
                })

        coverage_rows.append({
            "target_date": target_date, "category": category,
            "prediction_exists": True, "actual_exists": actual_exists,
            "prediction_quality_ok": prediction_ok, "actual_quality_ok": actual_ok,
            "formal_evidence_ok": prediction_class == PREDICTION_CLASS_FORWARD_VALID,
            "prediction_class": prediction_class, "evaluation_status": evaluation_status,
            "detail_rank_count": detail_rank_count, "rank1_10_complete": rank_complete,
            "formal_evaluation_complete": formal_complete and detail_rank_count == 10 and rank_complete,
            "prediction_sha256": prediction_sha, "metadata_sha256": metadata_sha,
            "actual_sha256": actual_sha,
            "actual_filename_date_ok": actual_ok, "actual_internal_date_ok": actual_ok,
            "evaluation_eligible": quarantine.evaluation_eligible,
            "quarantine_reason_code": quarantine.reason_code,
        })

    status = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    detail = (
        pd.concat(detail_frames, ignore_index=True).reindex(columns=DETAIL_COLUMNS)
        if detail_frames else pd.DataFrame(columns=DETAIL_COLUMNS)
    )
    daily = pd.DataFrame(daily_rows, columns=DAILY_COLUMNS)
    coverage = pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, name in (
        (status, "76_formal_status.csv"), (detail, "76_formal_detail.csv"),
        (daily, "76_formal_daily.csv"), (coverage, "76_formal_coverage.csv"),
    ):
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")
    return status, detail, daily, coverage
