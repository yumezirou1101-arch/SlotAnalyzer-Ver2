from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath


REGISTRY_RELATIVE_PATH = Path("config/evaluation_quarantine.json")
SCHEMA_VERSION = 1
VALID_CATEGORIES = frozenset({"NORMAL", "A_TYPE", "JUGGLER"})
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
STATUS_SKIPPED = "SKIPPED_INVENTORY_GUARD_INCIDENT"
STATUS_MISMATCH = "MANUAL_REVIEW_QUARANTINE_SHA_MISMATCH"


class QuarantineConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class QuarantineDecision:
    status: str
    evaluation_eligible: bool
    quarantined: bool
    reason_code: str = ""
    reason: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_relative_path(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or ":" in text:
        raise QuarantineConfigError(f"Invalid repository-relative quarantine path: {text}")
    return path.as_posix()


def _validate_entry(entry: object) -> dict:
    if not isinstance(entry, dict):
        raise QuarantineConfigError("Quarantine entry must be an object.")
    required = {
        "status", "store", "target_date", "category", "prediction_path",
        "prediction_sha256", "metadata_path", "metadata_sha256",
        "source_64_all514_sha256", "source_64_metadata_sha256",
        "reason_code", "reason", "incident_date", "created_at_jst",
    }
    missing = sorted(required - set(entry))
    if missing:
        raise QuarantineConfigError(f"Quarantine entry fields are missing: {missing}")
    value = dict(entry)
    if value["status"] != "ACTIVE":
        raise QuarantineConfigError("Only ACTIVE quarantine entries are supported.")
    if value["category"] not in VALID_CATEGORIES:
        raise QuarantineConfigError(f"Unknown quarantine category: {value['category']}")
    if not str(value["store"]).strip() or not str(value["reason"]).strip():
        raise QuarantineConfigError("Quarantine store and reason must be non-empty.")
    if value["reason_code"] != "INVENTORY_GUARD_INCIDENT":
        raise QuarantineConfigError("Unsupported quarantine reason_code.")
    try:
        date.fromisoformat(str(value["target_date"]))
        date.fromisoformat(str(value["incident_date"]))
        created = datetime.fromisoformat(str(value["created_at_jst"]))
    except ValueError as exc:
        raise QuarantineConfigError("Quarantine date evidence is invalid.") from exc
    if created.utcoffset() is None:
        raise QuarantineConfigError("created_at_jst must include a UTC offset.")
    value["prediction_path"] = _valid_relative_path(value["prediction_path"])
    value["metadata_path"] = _valid_relative_path(value["metadata_path"])
    for key in (
        "prediction_sha256", "metadata_sha256", "source_64_all514_sha256",
        "source_64_metadata_sha256",
    ):
        sha = str(value[key]).strip().lower()
        if not SHA_PATTERN.fullmatch(sha):
            raise QuarantineConfigError(f"Invalid quarantine SHA-256: {key}")
        value[key] = sha
    return value


def load_registry(project_root: Path, registry_path: Path | None = None) -> list[dict]:
    project_root = Path(project_root).resolve()
    path = Path(registry_path) if registry_path is not None else project_root / REGISTRY_RELATIVE_PATH
    if not path.is_file():
        raise QuarantineConfigError(f"Required quarantine registry is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuarantineConfigError(f"Quarantine registry is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise QuarantineConfigError("Quarantine registry schema_version is invalid.")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise QuarantineConfigError("Quarantine registry entries must be a list.")
    validated = [_validate_entry(entry) for entry in entries]
    keys = [(x["store"], x["target_date"], x["category"]) for x in validated]
    if len(keys) != len(set(keys)):
        raise QuarantineConfigError("Duplicate quarantine store/date/category entry.")
    return validated


def assess_evaluation_quarantine(
    project_root: Path,
    store: str,
    target_date: date,
    category: str,
    prediction_path: Path,
    metadata_path: Path,
    source_64_all514_path: Path,
    source_64_metadata_path: Path,
    *,
    registry_path: Path | None = None,
) -> QuarantineDecision:
    root = Path(project_root).resolve()
    category = str(category).strip()
    if category not in VALID_CATEGORIES:
        raise QuarantineConfigError(f"Unknown evaluation category: {category}")
    entries = load_registry(root, registry_path)
    matches = [
        entry for entry in entries
        if entry["store"] == store
        and entry["target_date"] == date.fromisoformat(str(target_date)).isoformat()
        and entry["category"] == category
    ]
    if not matches:
        return QuarantineDecision("NOT_QUARANTINED", True, False)
    entry = matches[0]
    expected_prediction = (root / entry["prediction_path"]).resolve()
    expected_metadata = (root / entry["metadata_path"]).resolve()
    try:
        expected_prediction.relative_to(root)
        expected_metadata.relative_to(root)
    except ValueError as exc:
        raise QuarantineConfigError("Quarantine path resolves outside the project root.") from exc
    mismatches = []
    if Path(prediction_path).resolve() != expected_prediction:
        mismatches.append("prediction_path")
    if Path(metadata_path).resolve() != expected_metadata:
        mismatches.append("metadata_path")
    files = {
        "prediction_sha256": expected_prediction,
        "metadata_sha256": expected_metadata,
        "source_64_all514_sha256": Path(source_64_all514_path).resolve(),
        "source_64_metadata_sha256": Path(source_64_metadata_path).resolve(),
    }
    for key, path in files.items():
        if not path.is_file() or sha256_file(path) != entry[key]:
            mismatches.append(key)
    if mismatches:
        return QuarantineDecision(
            STATUS_MISMATCH, False, True, entry["reason_code"], ",".join(mismatches)
        )
    return QuarantineDecision(
        STATUS_SKIPPED, False, True, entry["reason_code"], entry["reason"]
    )
