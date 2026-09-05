from __future__ import annotations

import csv
import json
import os
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
