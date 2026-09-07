from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


STATE_FILENAME = "inventory_guard_state.json"
CONFIRMED_INVENTORY_DIRNAME = "confirmed_inventory"


@dataclass(frozen=True)
class InventoryGuardPolicy:
    store: str
    known_change_dates: frozenset[date] = frozenset()
    confirmed_inventory_prefix: str = "inventory"
    daily_inventory_pattern: str = "ana_slo_{ymd}.csv"


# Store policy is separate from reusable inventory comparison/state logic.
MARUHAN_MAEBASHI_POLICY = InventoryGuardPolicy(
    store="MARUHAN_MAEBASHI",
    known_change_dates=frozenset({date(2026, 9, 8)}),
    confirmed_inventory_prefix="maruhan_inventory",
)
YASUDA_MAEBASHI_POLICY = InventoryGuardPolicy(
    store="YASUDA_MAEBASHI",
    daily_inventory_pattern="ana_slo_{ymd}.csv",
)
KNOWN_INVENTORY_CHANGE_DATES = MARUHAN_MAEBASHI_POLICY.known_change_dates


class InventoryGuardBlockedError(RuntimeError):
    def __init__(self, result: "InventoryGuardResult"):
        self.result = result
        super().__init__(result.summary())


@dataclass(frozen=True)
class RenamedMachine:
    machine_no: int
    previous_machine_name: str
    current_machine_name: str


@dataclass(frozen=True)
class InventoryDiff:
    previous_date: str
    current_date: str
    previous_machine_count: int
    current_machine_count: int
    added_machine_numbers: list[int] = field(default_factory=list)
    removed_machine_numbers: list[int] = field(default_factory=list)
    renamed_machine_numbers: list[RenamedMachine] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added_machine_numbers
            or self.removed_machine_numbers
            or self.renamed_machine_numbers
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        value["has_changes"] = self.has_changes
        return value


@dataclass(frozen=True)
class InventoryComparisonEvidence:
    schema_version: int
    policy_version: str
    mode: str
    store: str
    previous_date: str
    current_date: str
    calendar_gap_days: int
    comparison_performed: bool
    comparison_status: str
    previous_machine_count: int | None
    current_machine_count: int | None
    added_machine_numbers: list[int] | None
    removed_machine_numbers: list[int] | None
    renamed_machine_numbers: list[RenamedMachine] | None
    added_count: int | None
    removed_count: int | None
    renamed_count: int | None
    changed_machine_count: int | None
    change_rate: float | None
    has_changes: bool | None
    previous_daily_path: str
    current_daily_path: str
    previous_daily_sha256: str
    current_daily_sha256: str
    machine_name_comparison: str = "EXACT_STRING"
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InventoryGuardResult:
    status: str
    blocked: bool
    reason: str
    target_date: str
    latest_data_date: str
    known_change_date: bool
    confirmed_target_inventory_path: str
    confirmed_target_inventory_exists: bool
    comparison: InventoryDiff | None = None
    persistent_state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["comparison"] = self.comparison.to_dict() if self.comparison else None
        return value

    def summary(self) -> str:
        parts = [f"INVENTORY_GUARD_BLOCKED: {self.reason}"]
        diff = self.comparison
        if diff is None and self.persistent_state:
            diff = _diff_from_state(self.persistent_state)
        if diff:
            parts.append(
                f"previous/current={diff.previous_machine_count}/{diff.current_machine_count}"
            )
            if diff.added_machine_numbers:
                parts.append(f"added={diff.added_machine_numbers}")
            if diff.removed_machine_numbers:
                parts.append(f"removed={diff.removed_machine_numbers}")
            if diff.renamed_machine_numbers:
                changes = [
                    f"{item.machine_no}:{item.previous_machine_name}->{item.current_machine_name}"
                    for item in diff.renamed_machine_numbers
                ]
                parts.append(f"renamed={changes}")
        return "; ".join(parts)


def daily_inventory_path(
    data_dir: Path,
    value: date,
    policy: InventoryGuardPolicy = MARUHAN_MAEBASHI_POLICY,
) -> Path:
    return Path(data_dir) / policy.daily_inventory_pattern.format(ymd=f"{value:%Y%m%d}")


def confirmed_inventory_path(
    data_dir: Path,
    value: date,
    policy: InventoryGuardPolicy = MARUHAN_MAEBASHI_POLICY,
) -> Path:
    return (
        Path(data_dir)
        / CONFIRMED_INVENTORY_DIRNAME
        / f"{policy.confirmed_inventory_prefix}_{value:%Y%m%d}.csv"
    )


def inventory_guard_state_path(data_dir: Path) -> Path:
    return Path(data_dir) / STATE_FILENAME


def _load_inventory(path: Path) -> dict[int, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Inventory file is missing or empty: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        no_column = "machine_no" if "machine_no" in fields else "台番号" if "台番号" in fields else None
        name_column = "machine_name" if "machine_name" in fields else "機種名" if "機種名" in fields else None
        if no_column is None or name_column is None:
            raise RuntimeError(f"Inventory columns are missing: {path}")
        inventory: dict[int, str] = {}
        for row in reader:
            raw_no = str(row.get(no_column, "")).strip()
            name = str(row.get(name_column, "")).strip()
            if not raw_no or not name:
                raise RuntimeError(f"Inventory contains an empty machine number or name: {path}")
            try:
                machine_no = int(float(raw_no.replace(",", "")))
            except ValueError as exc:
                raise RuntimeError(f"Inventory contains an invalid machine number: {path}") from exc
            if machine_no in inventory:
                raise RuntimeError(f"Inventory contains duplicate machine number {machine_no}: {path}")
            inventory[machine_no] = name
    if not inventory:
        raise RuntimeError(f"Inventory contains no machines: {path}")
    return inventory


def compare_inventory_files(
    previous_path: Path,
    current_path: Path,
    previous_date: date,
    current_date: date,
) -> InventoryDiff:
    previous = _load_inventory(previous_path)
    current = _load_inventory(current_path)
    previous_numbers = set(previous)
    current_numbers = set(current)
    common = sorted(previous_numbers & current_numbers)
    renamed = [
        RenamedMachine(machine_no, previous[machine_no], current[machine_no])
        for machine_no in common
        if previous[machine_no] != current[machine_no]
    ]
    return InventoryDiff(
        previous_date=previous_date.isoformat(),
        current_date=current_date.isoformat(),
        previous_machine_count=len(previous),
        current_machine_count=len(current),
        added_machine_numbers=sorted(current_numbers - previous_numbers),
        removed_machine_numbers=sorted(previous_numbers - current_numbers),
        renamed_machine_numbers=renamed,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_file_date(path: Path) -> date | None:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        date_column = "date" if "date" in fields else "日付" if "日付" in fields else None
        if date_column is None:
            return None
        values = {str(row.get(date_column, "")).strip() for row in reader}
    if len(values) != 1 or "" in values:
        raise RuntimeError(f"Inventory contains invalid dates: {path}")
    try:
        return date.fromisoformat(values.pop())
    except ValueError as exc:
        raise RuntimeError(f"Inventory contains an invalid date: {path}") from exc


def inspect_inventory_transition(
    store: str,
    previous_path: Path,
    current_path: Path,
    previous_date: date,
    current_date: date,
    *,
    policy_version: str = "INVENTORY_DIFF_V1",
    mode: str = "READ_ONLY",
) -> InventoryComparisonEvidence:
    """Read and compare two inventories without writing files or state."""
    previous_path = Path(previous_path)
    current_path = Path(current_path)
    previous_date = date.fromisoformat(str(previous_date))
    current_date = date.fromisoformat(str(current_date))
    gap = (current_date - previous_date).days
    base = {
        "schema_version": 1,
        "policy_version": policy_version,
        "mode": mode,
        "store": store,
        "previous_date": previous_date.isoformat(),
        "current_date": current_date.isoformat(),
        "calendar_gap_days": gap,
        "previous_daily_path": str(previous_path),
        "current_daily_path": str(current_path),
        "previous_daily_sha256": "",
        "current_daily_sha256": "",
    }
    empty = {
        "comparison_performed": False,
        "previous_machine_count": None,
        "current_machine_count": None,
        "added_machine_numbers": None,
        "removed_machine_numbers": None,
        "renamed_machine_numbers": None,
        "added_count": None,
        "removed_count": None,
        "renamed_count": None,
        "changed_machine_count": None,
        "change_rate": None,
        "has_changes": None,
    }
    if gap != 1:
        return InventoryComparisonEvidence(
            **base, **empty, comparison_status="NON_CONSECUTIVE"
        )
    if not previous_path.is_file() or previous_path.stat().st_size <= 0:
        return InventoryComparisonEvidence(
            **base, **empty, comparison_status="PREVIOUS_MISSING",
            error=f"Previous inventory is missing or empty: {previous_path}",
        )
    if not current_path.is_file() or current_path.stat().st_size <= 0:
        return InventoryComparisonEvidence(
            **base, **empty, comparison_status="CURRENT_MISSING",
            error=f"Current inventory is missing or empty: {current_path}",
        )
    try:
        previous_sha = _sha256_file(previous_path)
        previous_inventory = _load_inventory(previous_path)
        embedded = _inventory_file_date(previous_path)
        if embedded is not None and embedded != previous_date:
            raise RuntimeError("Previous inventory date does not match the requested date.")
    except Exception as exc:
        return InventoryComparisonEvidence(
            **base, **empty, comparison_status="PREVIOUS_INVALID",
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        current_sha = _sha256_file(current_path)
        current_inventory = _load_inventory(current_path)
        embedded = _inventory_file_date(current_path)
        if embedded is not None and embedded != current_date:
            raise RuntimeError("Current inventory date does not match the requested date.")
    except Exception as exc:
        return InventoryComparisonEvidence(
            **{**base, "previous_daily_sha256": previous_sha}, **empty,
            comparison_status="CURRENT_INVALID",
            error=f"{type(exc).__name__}: {exc}",
        )
    previous_numbers = set(previous_inventory)
    current_numbers = set(current_inventory)
    added = sorted(current_numbers - previous_numbers)
    removed = sorted(previous_numbers - current_numbers)
    renamed = [
        RenamedMachine(number, previous_inventory[number], current_inventory[number])
        for number in sorted(previous_numbers & current_numbers)
        if previous_inventory[number] != current_inventory[number]
    ]
    changed = len(added) + len(removed) + len(renamed)
    has_changes = changed > 0
    return InventoryComparisonEvidence(
        **{
            **base,
            "previous_daily_sha256": previous_sha,
            "current_daily_sha256": current_sha,
        },
        comparison_performed=True,
        comparison_status="COMPARED_CHANGE" if has_changes else "COMPARED_NO_CHANGE",
        previous_machine_count=len(previous_inventory),
        current_machine_count=len(current_inventory),
        added_machine_numbers=added,
        removed_machine_numbers=removed,
        renamed_machine_numbers=renamed,
        added_count=len(added),
        removed_count=len(removed),
        renamed_count=len(renamed),
        changed_machine_count=changed,
        change_rate=changed / max(len(previous_inventory), len(current_inventory)),
        has_changes=has_changes,
    )


def _read_persistent_state(path: Path, policy: InventoryGuardPolicy) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Inventory guard state is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("store") != policy.store:
        raise RuntimeError(f"Inventory guard state has an invalid store or structure: {path}")
    return value


def _atomic_write_state(path: Path, value: dict) -> None:
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


def _state_for_change(policy: InventoryGuardPolicy, diff: InventoryDiff) -> dict:
    return {
        "version": 1,
        "store": policy.store,
        "detected_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "change_date": diff.current_date,
        "previous_date": diff.previous_date,
        "previous_machine_count": diff.previous_machine_count,
        "current_machine_count": diff.current_machine_count,
        "added_machine_numbers": diff.added_machine_numbers,
        "removed_machine_numbers": diff.removed_machine_numbers,
        "renamed_machine_numbers": [asdict(item) for item in diff.renamed_machine_numbers],
        "status": "BLOCKED",
        "approved_at": "",
        "approved_reason": "",
    }


def _diff_from_state(state: dict) -> InventoryDiff | None:
    if not state:
        return None
    try:
        return InventoryDiff(
            previous_date=str(state["previous_date"]),
            current_date=str(state["change_date"]),
            previous_machine_count=int(state["previous_machine_count"]),
            current_machine_count=int(state["current_machine_count"]),
            added_machine_numbers=[int(value) for value in state.get("added_machine_numbers", [])],
            removed_machine_numbers=[int(value) for value in state.get("removed_machine_numbers", [])],
            renamed_machine_numbers=[RenamedMachine(**value) for value in state.get("renamed_machine_numbers", [])],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Inventory guard state has invalid change details.") from exc


def _same_change(state: dict, diff: InventoryDiff) -> bool:
    stored = _diff_from_state(state)
    return stored is not None and stored.to_dict() == diff.to_dict()


def _is_explicitly_approved(state: dict) -> bool:
    approved = (
        state.get("status") == "APPROVED"
        and bool(str(state.get("approved_at", "")).strip())
        and bool(str(state.get("approved_reason", "")).strip())
    )
    if approved:
        # APPROVED without the exact persisted change details is not a valid
        # release record and must never silently unblock a zero-diff day.
        _diff_from_state(state)
    return approved


def assess_inventory_guard(
    data_dir: Path,
    target_date: date,
    latest_data_date: date,
    policy: InventoryGuardPolicy = MARUHAN_MAEBASHI_POLICY,
) -> InventoryGuardResult:
    data_dir = Path(data_dir)
    target_date = date.fromisoformat(str(target_date))
    latest_data_date = date.fromisoformat(str(latest_data_date))
    known_change = target_date in policy.known_change_dates
    confirmed_path = confirmed_inventory_path(data_dir, target_date, policy)
    confirmed_exists = confirmed_path.is_file() and confirmed_path.stat().st_size > 0
    state_path = inventory_guard_state_path(data_dir)
    persistent_state = _read_persistent_state(state_path, policy)

    if known_change and not confirmed_exists:
        return InventoryGuardResult(
            status="MANUAL_REVIEW", blocked=True,
            reason="Known inventory change date has no confirmed target inventory.",
            target_date=target_date.isoformat(), latest_data_date=latest_data_date.isoformat(),
            known_change_date=True, confirmed_target_inventory_path=str(confirmed_path),
            confirmed_target_inventory_exists=False, persistent_state=persistent_state,
        )

    current_path = confirmed_path if known_change else daily_inventory_path(data_dir, latest_data_date, policy)
    current_date = target_date if known_change else latest_data_date
    previous_date = current_date - timedelta(days=1)
    previous_path = daily_inventory_path(data_dir, previous_date, policy)
    comparison = None
    if previous_path.is_file() and current_path.is_file():
        comparison = compare_inventory_files(previous_path, current_path, previous_date, current_date)
        if comparison.has_changes:
            if not (_is_explicitly_approved(persistent_state) and _same_change(persistent_state, comparison)):
                same_blocked_change = (
                    persistent_state.get("status") == "BLOCKED"
                    and _same_change(persistent_state, comparison)
                )
                if not same_blocked_change:
                    persistent_state = _state_for_change(policy, comparison)
                    _atomic_write_state(state_path, persistent_state)
                return InventoryGuardResult(
                    status="MANUAL_REVIEW", blocked=True,
                    reason="Inventory change detected; explicit safety approval is required.",
                    target_date=target_date.isoformat(), latest_data_date=latest_data_date.isoformat(),
                    known_change_date=known_change, confirmed_target_inventory_path=str(confirmed_path),
                    confirmed_target_inventory_exists=confirmed_exists, comparison=comparison,
                    persistent_state=persistent_state,
                )

    if persistent_state and not _is_explicitly_approved(persistent_state):
        reason = (
            "Inventory change remains blocked because explicit approval has not been recorded."
            if persistent_state.get("status") == "BLOCKED"
            else "Persistent inventory state is not validly approved; formal Forward remains stopped."
        )
        return InventoryGuardResult(
            status="MANUAL_REVIEW", blocked=True, reason=reason,
            target_date=target_date.isoformat(), latest_data_date=latest_data_date.isoformat(),
            known_change_date=known_change, confirmed_target_inventory_path=str(confirmed_path),
            confirmed_target_inventory_exists=confirmed_exists, comparison=comparison,
            persistent_state=persistent_state,
        )

    return InventoryGuardResult(
        status="PASS", blocked=False,
        reason="No unapproved inventory change requiring a formal Forward stop was detected.",
        target_date=target_date.isoformat(), latest_data_date=latest_data_date.isoformat(),
        known_change_date=known_change, confirmed_target_inventory_path=str(confirmed_path),
        confirmed_target_inventory_exists=confirmed_exists, comparison=comparison,
        persistent_state=persistent_state,
    )


def enforce_inventory_guard(
    data_dir: Path,
    target_date: date,
    latest_data_date: date,
    policy: InventoryGuardPolicy = MARUHAN_MAEBASHI_POLICY,
) -> InventoryGuardResult:
    result = assess_inventory_guard(data_dir, target_date, latest_data_date, policy)
    if result.blocked:
        raise InventoryGuardBlockedError(result)
    return result


def assess_yasuda_inventory_guard(
    data_dir: Path,
    target_date: date,
    latest_data_date: date,
) -> InventoryGuardResult:
    """Y1 guard: enforce continuity only for validated 320-machine dailies."""
    data_dir = Path(data_dir)
    target_date = date.fromisoformat(str(target_date))
    latest_data_date = date.fromisoformat(str(latest_data_date))
    available_previous = []
    for path in data_dir.glob("ana_slo_????????.csv"):
        match = re.fullmatch(r"ana_slo_(\d{8})\.csv", path.name)
        if match:
            candidate = datetime.strptime(match.group(1), "%Y%m%d").date()
            if candidate < latest_data_date:
                available_previous.append((candidate, path))
    if available_previous:
        previous_date, previous_path = max(available_previous)
    else:
        previous_date = latest_data_date - timedelta(days=1)
        previous_path = daily_inventory_path(
            data_dir, previous_date, YASUDA_MAEBASHI_POLICY
        )
    current_path = daily_inventory_path(data_dir, latest_data_date, YASUDA_MAEBASHI_POLICY)
    state_path = inventory_guard_state_path(data_dir)
    persistent_state = _read_persistent_state(state_path, YASUDA_MAEBASHI_POLICY)
    evidence = inspect_inventory_transition(
        YASUDA_MAEBASHI_POLICY.store,
        previous_path,
        current_path,
        previous_date,
        latest_data_date,
        policy_version="YASUDA_Y1_V1",
        mode="ENFORCE",
    )
    comparison = None
    if evidence.comparison_performed:
        comparison = InventoryDiff(
            previous_date=evidence.previous_date,
            current_date=evidence.current_date,
            previous_machine_count=int(evidence.previous_machine_count),
            current_machine_count=int(evidence.current_machine_count),
            added_machine_numbers=list(evidence.added_machine_numbers or []),
            removed_machine_numbers=list(evidence.removed_machine_numbers or []),
            renamed_machine_numbers=list(evidence.renamed_machine_numbers or []),
        )
    if (
        not evidence.comparison_performed
        or evidence.previous_machine_count != 320
        or evidence.current_machine_count != 320
    ):
        reason = (
            "Inventory comparison is non-consecutive."
            if evidence.comparison_status == "NON_CONSECUTIVE"
            else "Inventory comparison is unavailable or outside the validated 320-machine baseline."
        )
        return InventoryGuardResult(
            status="MANUAL_REVIEW", blocked=True,
            reason=f"{reason} comparison_status={evidence.comparison_status}",
            target_date=target_date.isoformat(),
            latest_data_date=latest_data_date.isoformat(),
            known_change_date=False,
            confirmed_target_inventory_path="",
            confirmed_target_inventory_exists=False,
            comparison=comparison,
            persistent_state=persistent_state,
        )
    if comparison is not None and comparison.has_changes:
        if not (_is_explicitly_approved(persistent_state) and _same_change(persistent_state, comparison)):
            if not (
                persistent_state.get("status") == "BLOCKED"
                and _same_change(persistent_state, comparison)
            ):
                persistent_state = _state_for_change(YASUDA_MAEBASHI_POLICY, comparison)
                _atomic_write_state(state_path, persistent_state)
            return InventoryGuardResult(
                status="MANUAL_REVIEW", blocked=True,
                reason="Inventory change detected; explicit safety approval is required.",
                target_date=target_date.isoformat(),
                latest_data_date=latest_data_date.isoformat(),
                known_change_date=False,
                confirmed_target_inventory_path="",
                confirmed_target_inventory_exists=False,
                comparison=comparison,
                persistent_state=persistent_state,
            )
    if persistent_state and not _is_explicitly_approved(persistent_state):
        return InventoryGuardResult(
            status="MANUAL_REVIEW", blocked=True,
            reason="Inventory change remains blocked because explicit approval has not been recorded.",
            target_date=target_date.isoformat(),
            latest_data_date=latest_data_date.isoformat(),
            known_change_date=False,
            confirmed_target_inventory_path="",
            confirmed_target_inventory_exists=False,
            comparison=comparison,
            persistent_state=persistent_state,
        )
    return InventoryGuardResult(
        status="PASS", blocked=False,
        reason="Validated consecutive 320-machine inventories have no unapproved change.",
        target_date=target_date.isoformat(),
        latest_data_date=latest_data_date.isoformat(),
        known_change_date=False,
        confirmed_target_inventory_path="",
        confirmed_target_inventory_exists=False,
        comparison=comparison,
        persistent_state=persistent_state,
    )
