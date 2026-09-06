from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib

import pandas as pd

from ana_slo_prediction_v4_2_forward_guard import (
    validate_consecutive_latest_date,
    validate_forward_time,
    validate_target_actual_absent,
)


SCHEMA_VERSION = "ATYPE_JUGGLER_FORMAL_V1"
GUARD_VERSION = "DERIVED_FORWARD_GUARD_V1"
FORMALIZATION_START_DATE = date(2026, 9, 7)
EXPECTED_MODEL = "CHAMPION_V4.2_C"
EXPECTED_WEIGHT_FINGERPRINT = "a1eaf45d71ded209"
EXPECTED_WEIGHT_SUM = 1.0
FORWARD_CUTOFF_TEXT = "09:00 Asia/Tokyo"
JST = ZoneInfo("Asia/Tokyo")


class DerivedPredictionError(RuntimeError):
    pass


class AlreadyFrozenError(DerivedPredictionError):
    pass


class PartialFrozenOutputError(DerivedPredictionError):
    pass


class LegacyFrozenOutputError(DerivedPredictionError):
    pass


@dataclass(frozen=True)
class Source64Evidence:
    target_date: date
    latest_data_date: date
    generated_at_jst: str
    model: str
    weight_fingerprint: str
    weight_sum: float
    all514_path: Path
    top10_path: Path
    metadata_path: Path
    all514_sha256: str
    top10_sha256: str
    metadata_sha256: str


@dataclass(frozen=True)
class DerivedVerification:
    kind: str
    target_date: date
    paths: tuple[Path, ...]
    metadata_path: Path
    metadata_sha256: str
    prediction_class: str = "FORWARD_VALID"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_one_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if len(frame) != 1:
        raise DerivedPredictionError(f"metadata must contain exactly one row: {path}")
    return frame.iloc[0]


def _date(value, label: str) -> date:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise DerivedPredictionError(f"invalid {label}")
    return pd.Timestamp(parsed).date()


def _generated_before_cutoff(value, target: date, label: str) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed) or getattr(parsed, "tzinfo", None) is None:
        raise DerivedPredictionError(f"{label} must be timezone-aware")
    generated = pd.Timestamp(parsed).tz_convert(JST)
    if generated.date() > target:
        raise DerivedPredictionError(f"{label} is after target date")
    if generated.date() == target and generated.time().replace(tzinfo=None) >= time(9, 0):
        raise DerivedPredictionError(f"{label} is at/after 09:00 JST")
    return generated.isoformat()


def derived_paths(output_dir: Path, target_date: date, kind: str) -> tuple[Path, ...]:
    ymd = target_date.strftime("%Y%m%d")
    if kind == "A_TYPE":
        return (
            output_dir / f"74_A_type_prediction_{ymd}_all.csv",
            output_dir / f"74_A_type_prediction_{ymd}_top10.csv",
            output_dir / f"74_A_type_candidate_review_{ymd}.csv",
            output_dir / f"74_A_type_prediction_{ymd}_metadata.csv",
        )
    if kind == "JUGGLER":
        return (
            output_dir / f"75_Juggler_prediction_{ymd}_all.csv",
            output_dir / f"75_Juggler_prediction_{ymd}_top10.csv",
            output_dir / f"75_Juggler_prediction_{ymd}_metadata.csv",
        )
    raise ValueError(f"unknown derived prediction kind: {kind}")


def verify_source_64(source_dir: Path, target_date: date) -> Source64Evidence:
    ymd = target_date.strftime("%Y%m%d")
    all_path = source_dir / f"64_prediction_{ymd}_all514.csv"
    top10_path = source_dir / f"64_prediction_{ymd}_top10.csv"
    metadata_path = source_dir / f"64_prediction_{ymd}_metadata.csv"
    for path in (all_path, top10_path, metadata_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise DerivedPredictionError(f"64 formal triplet missing or empty: {path}")

    row = _read_one_row(metadata_path)
    required_evidence = (
        "generated_at_jst", "target_date", "latest_data_date", "model",
        "weight_fingerprint", "weight_sum", "forward_guard_version",
        "forward_valid", "forward_cutoff_jst",
        "target_actual_absent_at_generation", "target_source_absent_at_generation",
        "daily_csv_sha256", "source_html_sha256", "all514_sha256", "top10_sha256",
    )
    for key in required_evidence:
        value = row.get(key, "")
        if pd.isna(value) or str(value).strip() == "":
            raise DerivedPredictionError(f"64 metadata evidence missing: {key}")
    meta_target = _date(row.get("target_date"), "64 target_date")
    latest = _date(row.get("latest_data_date"), "64 latest_data_date")
    if meta_target != target_date:
        raise DerivedPredictionError("64 metadata target_date mismatch")
    try:
        validate_consecutive_latest_date(target_date, latest)
    except RuntimeError as exc:
        raise DerivedPredictionError(str(exc)) from exc
    if not _truthy(row.get("forward_valid")):
        raise DerivedPredictionError("64 metadata forward_valid is not true")
    if not str(row.get("forward_guard_version", "")).strip():
        raise DerivedPredictionError("64 forward_guard_version is missing")
    if str(row.get("model", "")).strip() != EXPECTED_MODEL:
        raise DerivedPredictionError("64 model mismatch")
    if str(row.get("weight_fingerprint", "")).strip() != EXPECTED_WEIGHT_FINGERPRINT:
        raise DerivedPredictionError("64 weight fingerprint mismatch")
    try:
        weight_sum = float(row.get("weight_sum"))
    except (TypeError, ValueError) as exc:
        raise DerivedPredictionError("64 weight_sum is invalid") from exc
    if abs(weight_sum - EXPECTED_WEIGHT_SUM) > 1e-12:
        raise DerivedPredictionError("64 weight_sum mismatch")
    generated = _generated_before_cutoff(row.get("generated_at_jst"), target_date, "64 generated_at_jst")
    if str(row.get("forward_cutoff_jst", "")).strip() != FORWARD_CUTOFF_TEXT:
        raise DerivedPredictionError("64 forward cutoff mismatch")
    if not _truthy(row.get("target_actual_absent_at_generation")):
        raise DerivedPredictionError("64 target actual absence is not proven")
    if not _truthy(row.get("target_source_absent_at_generation")):
        raise DerivedPredictionError("64 target source absence is not proven")

    all_hash = sha256_file(all_path)
    top10_hash = sha256_file(top10_path)
    if all_hash != str(row.get("all514_sha256", "")).strip().lower():
        raise DerivedPredictionError("64 all514 SHA-256 mismatch")
    if top10_hash != str(row.get("top10_sha256", "")).strip().lower():
        raise DerivedPredictionError("64 top10 SHA-256 mismatch")

    all_frame = pd.read_csv(all_path, encoding="utf-8-sig")
    top_frame = pd.read_csv(top10_path, encoding="utf-8-sig")
    required_columns = {
        "machine_no", "machine_name", "score", "prediction_rank", "tier",
        "target_date", "latest_data_date",
    }
    if not required_columns.issubset(all_frame.columns) or not required_columns.issubset(top_frame.columns):
        raise DerivedPredictionError("64 prediction schema is incomplete")
    all_numbers = pd.to_numeric(all_frame["machine_no"], errors="coerce")
    if len(all_frame) != 514 or all_numbers.nunique(dropna=True) != 514 or all_numbers.duplicated().any():
        raise DerivedPredictionError("64 all514 inventory is invalid")
    ranks = set(pd.to_numeric(top_frame["prediction_rank"], errors="coerce").dropna().astype(int))
    if len(top_frame) != 10 or ranks != set(range(1, 11)):
        raise DerivedPredictionError("64 top10 ranks are incomplete")
    for label, frame in (("all514", all_frame), ("top10", top_frame)):
        targets = {_date(value, f"64 {label} target_date") for value in frame["target_date"]}
        latest_dates = {_date(value, f"64 {label} latest_data_date") for value in frame["latest_data_date"]}
        if targets != {target_date} or latest_dates != {latest}:
            raise DerivedPredictionError(f"64 {label} date evidence mismatch")

    return Source64Evidence(
        target_date, latest, generated, EXPECTED_MODEL,
        EXPECTED_WEIGHT_FINGERPRINT, weight_sum,
        all_path, top10_path, metadata_path,
        all_hash, top10_hash, sha256_file(metadata_path),
    )


def verify_derived_prediction(
    output_dir: Path,
    source_64_dir: Path,
    target_date: date,
    kind: str,
) -> DerivedVerification:
    paths = derived_paths(output_dir, target_date, kind)
    existing = tuple(path for path in paths if path.exists())
    if not existing:
        raise DerivedPredictionError("derived frozen set does not exist")
    if len(existing) != len(paths):
        raise PartialFrozenOutputError("PARTIAL_FROZEN_OUTPUT: " + ", ".join(str(p) for p in existing))
    if any(path.stat().st_size <= 0 for path in paths):
        raise DerivedPredictionError("derived frozen set contains an empty file")

    metadata_path = paths[-1]
    row = _read_one_row(metadata_path)
    if str(row.get("schema_version", "")).strip() != SCHEMA_VERSION:
        raise LegacyFrozenOutputError("LEGACY_FROZEN_OUTPUT: formal schema is absent or unsupported")
    if str(row.get("derived_guard_version", "")).strip() != GUARD_VERSION:
        raise DerivedPredictionError("derived guard version mismatch")
    if str(row.get("prediction_class", "")).strip() != "FORWARD_VALID":
        raise DerivedPredictionError("derived prediction_class is not FORWARD_VALID")
    if not _truthy(row.get("forward_valid")):
        raise DerivedPredictionError("derived forward_valid is not true")
    meta_target = _date(row.get("target_date"), "derived target_date")
    latest = _date(row.get("latest_data_date"), "derived latest_data_date")
    start = _date(row.get("formalization_start_date"), "formalization_start_date")
    if meta_target != target_date or start != FORMALIZATION_START_DATE or target_date < start:
        raise DerivedPredictionError("derived formalization boundary mismatch")
    try:
        validate_consecutive_latest_date(target_date, latest)
    except RuntimeError as exc:
        raise DerivedPredictionError(str(exc)) from exc
    _generated_before_cutoff(row.get("generated_at_jst"), target_date, "derived generated_at_jst")
    if str(row.get("forward_cutoff_jst", "")).strip() != FORWARD_CUTOFF_TEXT:
        raise DerivedPredictionError("derived forward cutoff mismatch")
    if str(row.get("model", "")).strip() != EXPECTED_MODEL:
        raise DerivedPredictionError("derived model mismatch")
    if str(row.get("weight_fingerprint", "")).strip() != EXPECTED_WEIGHT_FINGERPRINT:
        raise DerivedPredictionError("derived fingerprint mismatch")
    if abs(float(row.get("weight_sum")) - EXPECTED_WEIGHT_SUM) > 1e-12:
        raise DerivedPredictionError("derived weight_sum mismatch")
    if not _truthy(row.get("target_actual_absent_at_generation")) or not _truthy(row.get("target_source_absent_at_generation")):
        raise DerivedPredictionError("derived target actual/source absence is not proven")
    if _truthy(row.get("derived_score_recalculated")):
        raise DerivedPredictionError("derived score was recalculated")

    source = verify_source_64(source_64_dir, target_date)
    lineage = {
        "source_64_all514_file": source.all514_path.name,
        "source_64_top10_file": source.top10_path.name,
        "source_64_metadata_file": source.metadata_path.name,
        "source_64_all514_sha256": source.all514_sha256,
        "source_64_top10_sha256": source.top10_sha256,
        "source_64_metadata_sha256": source.metadata_sha256,
    }
    for key, expected in lineage.items():
        if str(row.get(key, "")).strip().lower() != expected.lower():
            raise DerivedPredictionError(f"derived lineage mismatch: {key}")

    own_keys = (
        ("all_sha256", paths[0]),
        ("top10_sha256", paths[1]),
    )
    if kind == "A_TYPE":
        own_keys += (("candidate_review_sha256", paths[2]),)
    for key, path in own_keys:
        if str(row.get(key, "")).strip().lower() != sha256_file(path):
            raise DerivedPredictionError(f"derived artifact SHA-256 mismatch: {key}")
    own_files = (("all_file", paths[0]), ("top10_file", paths[1]))
    if kind == "A_TYPE":
        own_files += (("candidate_review_file", paths[2]),)
    for key, path in own_files:
        if str(row.get(key, "")).strip() != path.name:
            raise DerivedPredictionError(f"derived artifact filename mismatch: {key}")

    top10 = pd.read_csv(paths[1], encoding="utf-8-sig")
    rank_col = "a_type_rank" if kind == "A_TYPE" else "juggler_rank"
    required = {"machine_no", "machine_name", "score", rank_col, "target_date", "latest_data_date"}
    if not required.issubset(top10.columns) or len(top10) != 10:
        raise DerivedPredictionError("derived top10 schema/count mismatch")
    ranks = set(pd.to_numeric(top10[rank_col], errors="coerce").dropna().astype(int))
    targets = {_date(value, "top10 target_date") for value in top10["target_date"]}
    latest_dates = {_date(value, "top10 latest_data_date") for value in top10["latest_data_date"]}
    if ranks != set(range(1, 11)) or targets != {target_date} or latest_dates != {latest}:
        raise DerivedPredictionError("derived top10 rank/date mismatch")

    return DerivedVerification(kind, target_date, paths, metadata_path, sha256_file(metadata_path))


def inspect_frozen_set(
    output_dir: Path,
    source_64_dir: Path,
    target_date: date,
    kind: str,
) -> DerivedVerification | None:
    paths = derived_paths(output_dir, target_date, kind)
    count = sum(path.exists() for path in paths)
    if count == 0:
        return None
    if count != len(paths):
        raise PartialFrozenOutputError(f"PARTIAL_FROZEN_OUTPUT: {count}/{len(paths)} files exist")
    return verify_derived_prediction(output_dir, source_64_dir, target_date, kind)


def generation_preflight(
    project_root: Path,
    data_dir: Path,
    output_dir: Path,
    source_64_dir: Path,
    target_date: date,
    kind: str,
    current_jst: datetime | None = None,
) -> tuple[Source64Evidence, datetime]:
    existing = inspect_frozen_set(output_dir, source_64_dir, target_date, kind)
    if existing is not None:
        raise AlreadyFrozenError(f"ALREADY_FROZEN: verified {kind} formal set")
    if target_date < FORMALIZATION_START_DATE:
        raise DerivedPredictionError(
            f"FORMALIZATION_BOUNDARY_REJECTED: target={target_date}, start={FORMALIZATION_START_DATE}"
        )
    generated = validate_forward_time(target_date, current_jst)
    validate_target_actual_absent(project_root, data_dir, target_date)
    source = verify_source_64(source_64_dir, target_date)
    validate_consecutive_latest_date(target_date, source.latest_data_date)
    return source, generated


def build_metadata(
    source: Source64Evidence,
    generated_at_jst: datetime,
    own_hashes: dict[str, str],
    extra: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "formalization_start_date": FORMALIZATION_START_DATE.isoformat(),
        "prediction_class": "FORWARD_VALID",
        "derived_guard_version": GUARD_VERSION,
        "generated_at_jst": generated_at_jst.astimezone(JST).isoformat(),
        "forward_cutoff_jst": FORWARD_CUTOFF_TEXT,
        "forward_valid": True,
        "target_date": source.target_date.isoformat(),
        "latest_data_date": source.latest_data_date.isoformat(),
        "model": source.model,
        "weight_fingerprint": source.weight_fingerprint,
        "weight_sum": source.weight_sum,
        "target_actual_absent_at_generation": True,
        "target_source_absent_at_generation": True,
        "target_actual_absent": True,
        "target_source_absent": True,
        "source_64_all514_file": source.all514_path.name,
        "source_64_top10_file": source.top10_path.name,
        "source_64_metadata_file": source.metadata_path.name,
        "source_64_all514_sha256": source.all514_sha256,
        "source_64_top10_sha256": source.top10_sha256,
        "source_64_metadata_sha256": source.metadata_sha256,
        "derived_score_recalculated": False,
        **own_hashes,
        **extra,
    }


def exclusive_write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", mode="x")
