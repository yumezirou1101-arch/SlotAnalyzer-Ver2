from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from slotanalyzer_inventory_guard import inspect_inventory_transition


POLICY_VERSION = "BIGMARCH_MONITOR_V1"
STORE = "BIGMARCH_TAKASAKI_OYAGI"
DAILY_RE = re.compile(r"^ana_slo_bigmarch_oyagi_(\d{8})\.csv$")


def evidence_path(data_dir: Path, operation_date: date) -> Path:
    # Legacy operation-date evidence path. Read-only compatibility is preserved.
    return Path(data_dir) / "inventory_monitor" / f"inventory_monitor_{operation_date:%Y%m%d}.json"


# SLOTANALYZER_INVENTORY_EVIDENCE_PAIR_STORAGE_V1
def pair_evidence_path(
    data_dir: Path,
    previous_date: date,
    current_date: date,
) -> Path:
    # Canonical evidence path for one inventory comparison pair.
    previous_date = date.fromisoformat(str(previous_date))
    current_date = date.fromisoformat(str(current_date))
    return (
        Path(data_dir)
        / "inventory_monitor"
        / (
            f"inventory_monitor_{previous_date:%Y%m%d}"
            f"_to_{current_date:%Y%m%d}.json"
        )
    )


def discover_daily_files(data_dir: Path) -> list[tuple[date, Path]]:
    found = []
    for path in Path(data_dir).glob("ana_slo_bigmarch_oyagi_????????.csv"):
        match = DAILY_RE.fullmatch(path.name)
        if match:
            found.append((datetime.strptime(match.group(1), "%Y%m%d").date(), path))
    return sorted(found)


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError("Inventory monitor evidence is not an object.")
    return value


def load_inventory_monitor_evidence(data_dir: Path, operation_date: date) -> dict:
    path = evidence_path(data_dir, operation_date)
    if not path.is_file():
        return {}
    try:
        value = _read_json(path)
        return {**value, "status": "ALREADY_OBSERVED", "evidence_path": str(path)}
    except Exception as exc:
        return {
            "schema_version": 1, "policy_version": POLICY_VERSION,
            "mode": "MONITOR_ONLY", "store": STORE,
            "operation_date": operation_date.isoformat(),
            "status": "ERROR", "comparison_status": "ERROR",
            "comparison_performed": False,
            "affects_formal": False, "affects_provisional": False,
            "affects_morning_status": False, "affects_sleep": False,
            "evidence_path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_inventory_pair_evidence(
    data_dir: Path,
    previous_date: date,
    current_date: date,
    *,
    operation_date: date | None = None,
) -> dict:
    # Read pair evidence, with legacy operation-date evidence as read-only fallback.
    data_dir = Path(data_dir).resolve()
    previous_date = date.fromisoformat(str(previous_date))
    current_date = date.fromisoformat(str(current_date))

    path = pair_evidence_path(data_dir, previous_date, current_date)
    if path.is_file():
        try:
            value = _read_json(path)
            return {**value, "status": "ALREADY_OBSERVED", "evidence_path": str(path)}
        except Exception as exc:
            return {
                "schema_version": 1,
                "policy_version": POLICY_VERSION,
                "mode": "MONITOR_ONLY",
                "store": STORE,
                "previous_date": previous_date.isoformat(),
                "current_date": current_date.isoformat(),
                "status": "ERROR",
                "comparison_status": "ERROR",
                "comparison_performed": False,
                "affects_formal": False,
                "affects_provisional": False,
                "affects_morning_status": False,
                "affects_sleep": False,
                "evidence_path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    if operation_date is None:
        return {}

    legacy = evidence_path(data_dir, date.fromisoformat(str(operation_date)))
    if not legacy.is_file():
        return {}

    try:
        value = _read_json(legacy)
        if _pair_key(value) != (
            previous_date.isoformat(),
            current_date.isoformat(),
        ):
            return {}
        return {**value, "status": "ALREADY_OBSERVED", "evidence_path": str(legacy)}
    except Exception as exc:
        return {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "mode": "MONITOR_ONLY",
            "store": STORE,
            "previous_date": previous_date.isoformat(),
            "current_date": current_date.isoformat(),
            "status": "ERROR",
            "comparison_status": "ERROR",
            "comparison_performed": False,
            "affects_formal": False,
            "affects_provisional": False,
            "affects_morning_status": False,
            "affects_sleep": False,
            "evidence_path": str(legacy),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _identity(value: dict) -> tuple:
    # Legacy identity helper retained for backward compatibility.
    return (
        value.get("operation_date"), value.get("previous_date"), value.get("current_date"),
        value.get("previous_daily_path"), value.get("current_daily_path"),
        value.get("previous_daily_sha256"), value.get("current_daily_sha256"),
    )


def _pair_key(value: dict) -> tuple:
    return (
        value.get("previous_date"),
        value.get("current_date"),
    )


def _source_identity(value: dict) -> tuple:
    # operation_date is observation metadata, not source-pair identity.
    return (
        value.get("previous_date"),
        value.get("current_date"),
        value.get("previous_daily_path"),
        value.get("current_daily_path"),
        value.get("previous_daily_sha256"),
        value.get("current_daily_sha256"),
    )


def observe_big_march_inventory(
    data_dir: Path,
    operation_date: date,
    *,
    generated_at_jst: datetime,
) -> dict:
    data_dir = Path(data_dir).resolve()
    operation_date = date.fromisoformat(str(operation_date))
    latest_allowed = operation_date - timedelta(days=1)
    daily_files = [
        item for item in discover_daily_files(data_dir) if item[0] <= latest_allowed
    ]
    if len(daily_files) < 2:
        result = {
            "schema_version": 1, "policy_version": POLICY_VERSION,
            "mode": "MONITOR_ONLY", "store": STORE,
            "operation_date": operation_date.isoformat(),
            "generated_at_jst": generated_at_jst.isoformat(),
            "status": "COMPARISON_UNAVAILABLE",
            "comparison_status": "COMPARISON_UNAVAILABLE",
            "comparison_performed": False,
            "affects_formal": False, "affects_provisional": False,
            "affects_morning_status": False, "affects_sleep": False,
            "error": "At least two daily CSV files are required.",
        }
    else:
        (previous_date, previous_path), (current_date, current_path) = daily_files[-2:]
        comparison = inspect_inventory_transition(
            STORE, previous_path, current_path, previous_date, current_date,
            policy_version=POLICY_VERSION, mode="MONITOR_ONLY",
        ).to_dict()
        status = {
            "COMPARED_NO_CHANGE": "NO_CHANGE",
            "COMPARED_CHANGE": "CHANGE_OBSERVED",
            "NON_CONSECUTIVE": "NON_CONSECUTIVE",
        }.get(comparison["comparison_status"], "ERROR")
        result = {
            **comparison,
            "operation_date": operation_date.isoformat(),
            "generated_at_jst": generated_at_jst.isoformat(),
            "status": status,
            "affects_formal": False, "affects_provisional": False,
            "affects_morning_status": False, "affects_sleep": False,
        }
    # Pair-addressed evidence allows multiple valid comparisons to coexist.
    # Fail closed only when the SAME pair changes its source identity.
    if result.get("comparison_performed"):
        previous_date = date.fromisoformat(str(result["previous_date"]))
        current_date = date.fromisoformat(str(result["current_date"]))
        path = pair_evidence_path(data_dir, previous_date, current_date)
        result["evidence_path"] = str(path)

        if path.exists():
            existing = _read_json(path)
            if _source_identity(existing) == _source_identity(result):
                return {**existing, "status": "ALREADY_OBSERVED", "evidence_path": str(path)}
            return {
                **result,
                "status": "SOURCE_CHANGED_AFTER_OBSERVATION",
                "error": (
                    "Existing evidence for the same comparison pair has "
                    "different source paths or SHA-256 values."
                ),
            }

        # Legacy evidence is never deleted, overwritten, or migrated implicitly.
        legacy_path = evidence_path(data_dir, operation_date)
        if legacy_path.exists():
            legacy = _read_json(legacy_path)
            if _pair_key(legacy) == _pair_key(result):
                if _source_identity(legacy) == _source_identity(result):
                    return {
                        **legacy,
                        "status": "ALREADY_OBSERVED",
                        "evidence_path": str(legacy_path),
                    }
                return {
                    **result,
                    "evidence_path": str(legacy_path),
                    "status": "SOURCE_CHANGED_AFTER_OBSERVATION",
                    "error": (
                        "Legacy evidence for the same comparison pair has "
                        "different source paths or SHA-256 values."
                    ),
                }

        _atomic_write_json(path, result)
        return result

    # If no comparison pair exists, keep the legacy operation-date behavior.
    path = evidence_path(data_dir, operation_date)
    result["evidence_path"] = str(path)
    if path.exists():
        existing = _read_json(path)
        if _identity(existing) == _identity(result):
            return {**existing, "status": "ALREADY_OBSERVED", "evidence_path": str(path)}
        return {
            **result,
            "status": "SOURCE_CHANGED_AFTER_OBSERVATION",
            "error": (
                "Existing monitor evidence has different source dates, "
                "paths, or SHA-256 values."
            ),
        }
    _atomic_write_json(path, result)
    return result


def observe_big_march_inventory_best_effort(
    data_dir: Path,
    operation_date: date,
    *,
    generated_at_jst: datetime,
) -> dict:
    try:
        return observe_big_march_inventory(
            data_dir, operation_date, generated_at_jst=generated_at_jst
        )
    except Exception as exc:
        return {
            "schema_version": 1, "policy_version": POLICY_VERSION,
            "mode": "MONITOR_ONLY", "store": STORE,
            "operation_date": operation_date.isoformat(),
            "generated_at_jst": generated_at_jst.isoformat(),
            "status": "ERROR", "comparison_status": "ERROR",
            "comparison_performed": False,
            "affects_formal": False, "affects_provisional": False,
            "affects_morning_status": False, "affects_sleep": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
