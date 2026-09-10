from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MACHINE_DIR = ROOT / "machine_number"

if str(MACHINE_DIR) not in sys.path:
    sys.path.insert(0, str(MACHINE_DIR))

from slotanalyzer_inventory_guard import (
    YASUDA_MAEBASHI_POLICY,
    _atomic_write_state,
    _read_persistent_state,
    _same_change,
    assess_yasuda_inventory_guard,
    compare_inventory_files,
    daily_inventory_path,
    inspect_inventory_transition,
    inventory_guard_state_path,
)


EXPECTED_MACHINE_COUNT = 320
APPROVAL_TOOL = "YASUDA_INVENTORY_GUARD_APPROVAL_V1"


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD."
        ) from exc


def approve_yasuda_inventory_incident(
    data_dir: Path,
    previous_date: date,
    change_date: date,
    reason: str,
    *,
    approve: bool,
) -> dict:
    data_dir = Path(data_dir)
    previous_date = date.fromisoformat(str(previous_date))
    change_date = date.fromisoformat(str(change_date))
    reason = str(reason).strip()

    if not approve:
        raise RuntimeError(
            "Approval was not requested. Re-run with --approve after reviewing the incident."
        )

    if not reason:
        raise RuntimeError("Approval reason must not be empty.")

    if (change_date - previous_date).days != 1:
        raise RuntimeError(
            "Only an exact consecutive-day incident can be approved."
        )

    state_path = inventory_guard_state_path(data_dir)

    if not state_path.is_file():
        raise RuntimeError(
            f"Inventory Guard state does not exist: {state_path}"
        )

    state = _read_persistent_state(
        state_path,
        YASUDA_MAEBASHI_POLICY,
    )

    if state.get("version") != 1:
        raise RuntimeError(
            f"Unsupported Inventory Guard state version: {state.get('version')!r}"
        )

    if state.get("store") != YASUDA_MAEBASHI_POLICY.store:
        raise RuntimeError(
            f"Wrong store in Inventory Guard state: {state.get('store')!r}"
        )

    incident_kind = str(state.get("incident_kind", "")).strip()

    if incident_kind not in ("", "ACTUAL_CHANGE"):
        raise RuntimeError(
            "Only an ACTUAL_CHANGE incident or a verified legacy blank-kind "
            "incident can be approved by this tool."
        )

    if state.get("previous_date") != previous_date.isoformat():
        raise RuntimeError(
            "Requested previous_date does not match the persisted incident. "
            f"requested={previous_date.isoformat()} "
            f"persisted={state.get('previous_date')!r}"
        )

    if state.get("change_date") != change_date.isoformat():
        raise RuntimeError(
            "Requested change_date does not match the persisted incident. "
            f"requested={change_date.isoformat()} "
            f"persisted={state.get('change_date')!r}"
        )

    previous_path = daily_inventory_path(
        data_dir,
        previous_date,
        YASUDA_MAEBASHI_POLICY,
    )
    current_path = daily_inventory_path(
        data_dir,
        change_date,
        YASUDA_MAEBASHI_POLICY,
    )

    evidence = inspect_inventory_transition(
        YASUDA_MAEBASHI_POLICY.store,
        previous_path,
        current_path,
        previous_date,
        change_date,
        policy_version="YASUDA_APPROVAL_V1",
        mode="APPROVAL_VERIFY",
    )

    if not evidence.comparison_performed:
        raise RuntimeError(
            "Inventory comparison could not be completed. "
            f"status={evidence.comparison_status} error={evidence.error}"
        )

    if evidence.comparison_status != "COMPARED_CHANGE":
        raise RuntimeError(
            "The requested incident no longer reproduces as an inventory change. "
            f"status={evidence.comparison_status}"
        )

    if evidence.previous_machine_count != EXPECTED_MACHINE_COUNT:
        raise RuntimeError(
            "Previous inventory is outside the validated Yasuda baseline. "
            f"count={evidence.previous_machine_count}"
        )

    if evidence.current_machine_count != EXPECTED_MACHINE_COUNT:
        raise RuntimeError(
            "Current inventory is outside the validated Yasuda baseline. "
            f"count={evidence.current_machine_count}"
        )

    diff = compare_inventory_files(
        previous_path,
        current_path,
        previous_date,
        change_date,
    )

    if not diff.has_changes:
        raise RuntimeError(
            "The requested incident has no actual inventory changes."
        )

    if not _same_change(state, diff):
        raise RuntimeError(
            "SAFETY STOP: persisted Inventory Guard incident does not exactly "
            "match the current CSV-derived inventory diff."
        )

    if state.get("status") == "APPROVED":
        approved_at = str(state.get("approved_at", "")).strip()
        approved_reason = str(state.get("approved_reason", "")).strip()

        if not approved_at or not approved_reason:
            raise RuntimeError(
                "State says APPROVED but approval evidence is incomplete."
            )

        return {
            "status": "ALREADY_APPROVED",
            "state_path": str(state_path),
            "previous_date": previous_date.isoformat(),
            "change_date": change_date.isoformat(),
            "previous_machine_count": diff.previous_machine_count,
            "current_machine_count": diff.current_machine_count,
            "added_count": len(diff.added_machine_numbers),
            "removed_count": len(diff.removed_machine_numbers),
            "renamed_count": len(diff.renamed_machine_numbers),
            "approved_at": approved_at,
            "approved_reason": approved_reason,
        }

    if state.get("status") != "BLOCKED":
        raise RuntimeError(
            f"Incident is not in BLOCKED state: {state.get('status')!r}"
        )

    approved_at = datetime.now().astimezone().isoformat()

    approved_state = dict(state)
    approved_state["incident_kind"] = "ACTUAL_CHANGE"
    approved_state["status"] = "APPROVED"
    approved_state["approved_at"] = approved_at
    approved_state["approved_reason"] = reason
    approved_state["approval_tool"] = APPROVAL_TOOL
    approved_state["approval_previous_daily_sha256"] = (
        evidence.previous_daily_sha256
    )
    approved_state["approval_current_daily_sha256"] = (
        evidence.current_daily_sha256
    )

    if incident_kind == "":
        approved_state["legacy_state_normalized"] = True

    _atomic_write_state(state_path, approved_state)

    verified_state = _read_persistent_state(
        state_path,
        YASUDA_MAEBASHI_POLICY,
    )

    if verified_state.get("incident_kind") != "ACTUAL_CHANGE":
        raise RuntimeError(
            "Approval write verification failed: incident_kind mismatch."
        )

    if verified_state.get("status") != "APPROVED":
        raise RuntimeError(
            "Approval write verification failed: status is not APPROVED."
        )

    if verified_state.get("approved_at") != approved_at:
        raise RuntimeError(
            "Approval write verification failed: approved_at mismatch."
        )

    if verified_state.get("approved_reason") != reason:
        raise RuntimeError(
            "Approval write verification failed: approved_reason mismatch."
        )

    if not _same_change(verified_state, diff):
        raise RuntimeError(
            "Approval write verification failed: incident diff changed."
        )

    guard_result = assess_yasuda_inventory_guard(
        data_dir,
        change_date + date.resolution,
        change_date,
    )

    if guard_result.blocked or guard_result.status != "PASS":
        raise RuntimeError(
            "Approval was written but Inventory Guard did not release. "
            f"status={guard_result.status} reason={guard_result.reason}"
        )

    return {
        "status": "APPROVED",
        "state_path": str(state_path),
        "previous_date": previous_date.isoformat(),
        "change_date": change_date.isoformat(),
        "previous_machine_count": diff.previous_machine_count,
        "current_machine_count": diff.current_machine_count,
        "added_count": len(diff.added_machine_numbers),
        "removed_count": len(diff.removed_machine_numbers),
        "renamed_count": len(diff.renamed_machine_numbers),
        "previous_daily_sha256": evidence.previous_daily_sha256,
        "current_daily_sha256": evidence.current_daily_sha256,
        "approved_at": approved_at,
        "approved_reason": reason,
        "guard_status_after_approval": guard_result.status,
        "legacy_state_normalized": incident_kind == "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely approve one exact Yasuda Maebashi Inventory Guard "
            "ACTUAL_CHANGE incident."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "yasuda_maebashi" / "machine_number",
    )
    parser.add_argument(
        "--previous-date",
        required=True,
        type=parse_iso_date,
        help="Exact previous inventory date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--change-date",
        required=True,
        type=parse_iso_date,
        help="Exact changed inventory date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Human-reviewed approval reason.",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Required explicit approval switch.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        result = approve_yasuda_inventory_incident(
            data_dir=args.data_dir,
            previous_date=args.previous_date,
            change_date=args.change_date,
            reason=args.reason,
            approve=args.approve,
        )
    except Exception as exc:
        print(f"APPROVAL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"status={result['status']}")
    print(f"state_path={result['state_path']}")
    print(
        "incident="
        f"{result['previous_date']} -> {result['change_date']}"
    )
    print(
        "machine_count="
        f"{result['previous_machine_count']} -> "
        f"{result['current_machine_count']}"
    )
    print(
        "diff="
        f"added:{result['added_count']} "
        f"removed:{result['removed_count']} "
        f"renamed:{result['renamed_count']}"
    )
    print(f"approved_at={result['approved_at']}")
    print(f"approved_reason={result['approved_reason']}")

    if "guard_status_after_approval" in result:
        print(
            "guard_status_after_approval="
            f"{result['guard_status_after_approval']}"
        )

    if result.get("legacy_state_normalized"):
        print("legacy_state_normalized=True")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())