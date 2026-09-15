from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"

if str(MACHINE) not in sys.path:
    sys.path.insert(0, str(MACHINE))

from slotanalyzer_inventory_guard import (
    MARUHAN_MAEBASHI_POLICY,
    assess_inventory_guard,
    inventory_guard_state_path,
)
from slotanalyzer_maruhan_inventory_approve import (
    approve_maruhan_inventory_incident,
)


PREVIOUS_DATE = date(2026, 9, 13)
CHANGE_DATE = date(2026, 9, 14)
TARGET_DATE = date(2026, 9, 15)
REASON = "Reviewed exact Maruhan inventory incident in test."


def machines(count: int = 514) -> list[tuple[int, str]]:
    return [
        (number, f"machine-{number}")
        for number in range(1, count + 1)
    ]


def write_inventory(
    path: Path,
    rows: list[tuple[int, str]],
    day: date,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "日付",
                "台番号",
                "機種名",
            ]
        )

        for machine_no, machine_name in rows:
            writer.writerow(
                [
                    day.isoformat(),
                    machine_no,
                    machine_name,
                ]
            )


def prepare_blocked_incident(
    data: Path,
    *,
    legacy_blank_kind: bool = False,
) -> Path:
    previous = machines()
    current = machines()

    # Keep machine count unchanged while creating a real inventory rename.
    current[99] = (
        100,
        "replacement-machine",
    )

    write_inventory(
        data / "ana_slo_20260913.csv",
        previous,
        PREVIOUS_DATE,
    )
    write_inventory(
        data / "ana_slo_20260914.csv",
        current,
        CHANGE_DATE,
    )

    result = assess_inventory_guard(
        data,
        TARGET_DATE,
        CHANGE_DATE,
        MARUHAN_MAEBASHI_POLICY,
    )

    if not result.blocked:
        raise AssertionError(
            "Test fixture did not create a blocked Maruhan incident."
        )

    state_path = inventory_guard_state_path(data)

    if legacy_blank_kind:
        state = json.loads(
            state_path.read_text(
                encoding="utf-8",
            )
        )
        state.pop(
            "incident_kind",
            None,
        )
        state_path.write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return state_path


class MaruhanInventoryApprovalTests(
    unittest.TestCase,
):
    def test_requires_explicit_approve_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            state_path = prepare_blocked_incident(data)
            before = state_path.read_bytes()

            with self.assertRaisesRegex(
                RuntimeError,
                "Approval was not requested",
            ):
                approve_maruhan_inventory_incident(
                    data,
                    PREVIOUS_DATE,
                    CHANGE_DATE,
                    REASON,
                    approve=False,
                )

            after = state_path.read_bytes()

        self.assertEqual(
            before,
            after,
        )

    def test_rejects_empty_reason_without_modifying_state(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            state_path = prepare_blocked_incident(data)
            before = state_path.read_bytes()

            with self.assertRaisesRegex(
                RuntimeError,
                "Approval reason must not be empty",
            ):
                approve_maruhan_inventory_incident(
                    data,
                    PREVIOUS_DATE,
                    CHANGE_DATE,
                    "   ",
                    approve=True,
                )

            after = state_path.read_bytes()

        self.assertEqual(
            before,
            after,
        )

    def test_rejects_wrong_incident_dates_without_modifying_state(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            state_path = prepare_blocked_incident(
                data,
                legacy_blank_kind=True,
            )
            before = state_path.read_bytes()

            with self.assertRaisesRegex(
                RuntimeError,
                "Requested previous_date does not match",
            ):
                approve_maruhan_inventory_incident(
                    data,
                    date(2026, 9, 14),
                    date(2026, 9, 15),
                    REASON,
                    approve=True,
                )

            after = state_path.read_bytes()

        self.assertEqual(
            before,
            after,
        )

    def test_rejects_csv_diff_mismatch_without_approving(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            state_path = prepare_blocked_incident(data)

            altered = machines()
            altered[99] = (
                100,
                "different-replacement-machine",
            )

            write_inventory(
                data / "ana_slo_20260914.csv",
                altered,
                CHANGE_DATE,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "persisted Inventory Guard incident does not exactly match",
            ):
                approve_maruhan_inventory_incident(
                    data,
                    PREVIOUS_DATE,
                    CHANGE_DATE,
                    REASON,
                    approve=True,
                )

            state = json.loads(
                state_path.read_text(
                    encoding="utf-8",
                )
            )

        self.assertEqual(
            state["status"],
            "BLOCKED",
        )
        self.assertEqual(
            state["approved_at"],
            "",
        )
        self.assertEqual(
            state["approved_reason"],
            "",
        )

    def test_rejects_non_514_current_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            state_path = prepare_blocked_incident(data)

            current = machines(513)
            current[99] = (
                100,
                "replacement-machine",
            )

            write_inventory(
                data / "ana_slo_20260914.csv",
                current,
                CHANGE_DATE,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Current inventory is outside the validated Maruhan baseline",
            ):
                approve_maruhan_inventory_incident(
                    data,
                    PREVIOUS_DATE,
                    CHANGE_DATE,
                    REASON,
                    approve=True,
                )

            state = json.loads(
                state_path.read_text(
                    encoding="utf-8",
                )
            )

        self.assertEqual(
            state["status"],
            "BLOCKED",
        )

    def test_exact_incident_is_approved_and_guard_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            state_path = prepare_blocked_incident(data)

            result = approve_maruhan_inventory_incident(
                data,
                PREVIOUS_DATE,
                CHANGE_DATE,
                REASON,
                approve=True,
            )

            state = json.loads(
                state_path.read_text(
                    encoding="utf-8",
                )
            )

        self.assertEqual(
            result["status"],
            "APPROVED",
        )
        self.assertEqual(
            result["previous_machine_count"],
            514,
        )
        self.assertEqual(
            result["current_machine_count"],
            514,
        )
        self.assertEqual(
            result["added_count"],
            0,
        )
        self.assertEqual(
            result["removed_count"],
            0,
        )
        self.assertEqual(
            result["renamed_count"],
            1,
        )
        self.assertEqual(
            result["guard_status_after_approval"],
            "PASS",
        )
        self.assertFalse(
            result["legacy_state_normalized"],
        )

        self.assertEqual(
            state["incident_kind"],
            "ACTUAL_CHANGE",
        )
        self.assertEqual(
            state["status"],
            "APPROVED",
        )
        self.assertEqual(
            state["approved_reason"],
            REASON,
        )
        self.assertTrue(
            state["approved_at"],
        )
        self.assertEqual(
            state["approval_tool"],
            "MARUHAN_INVENTORY_GUARD_APPROVAL_V1",
        )
        self.assertEqual(
            len(
                state[
                    "approval_previous_daily_sha256"
                ]
            ),
            64,
        )
        self.assertEqual(
            len(
                state[
                    "approval_current_daily_sha256"
                ]
            ),
            64,
        )

    def test_legacy_blank_kind_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            state_path = prepare_blocked_incident(
                data,
                legacy_blank_kind=True,
            )

            result = approve_maruhan_inventory_incident(
                data,
                PREVIOUS_DATE,
                CHANGE_DATE,
                REASON,
                approve=True,
            )

            state = json.loads(
                state_path.read_text(
                    encoding="utf-8",
                )
            )

        self.assertEqual(
            result["status"],
            "APPROVED",
        )
        self.assertEqual(
            result["guard_status_after_approval"],
            "PASS",
        )
        self.assertTrue(
            result["legacy_state_normalized"],
        )
        self.assertEqual(
            state["incident_kind"],
            "ACTUAL_CHANGE",
        )
        self.assertTrue(
            state["legacy_state_normalized"],
        )

    def test_second_exact_approval_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            state_path = prepare_blocked_incident(data)

            first = approve_maruhan_inventory_incident(
                data,
                PREVIOUS_DATE,
                CHANGE_DATE,
                REASON,
                approve=True,
            )

            first_bytes = state_path.read_bytes()

            second = approve_maruhan_inventory_incident(
                data,
                PREVIOUS_DATE,
                CHANGE_DATE,
                "Different reason must not overwrite existing approval.",
                approve=True,
            )

            second_bytes = state_path.read_bytes()

        self.assertEqual(
            first["status"],
            "APPROVED",
        )
        self.assertEqual(
            second["status"],
            "ALREADY_APPROVED",
        )
        self.assertEqual(
            first_bytes,
            second_bytes,
        )
        self.assertEqual(
            second["approved_reason"],
            REASON,
        )
        self.assertEqual(
            second["approved_at"],
            first["approved_at"],
        )


if __name__ == "__main__":
    unittest.main()
