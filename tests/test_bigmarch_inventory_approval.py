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


import slotanalyzer_bigmarch_inventory_approve as approval
import slotanalyzer_bigmarch_inventory_guard as guard

from slotanalyzer_inventory_guard import (
    inventory_guard_state_path,
)


def machines(
    count: int = 276,
) -> list[tuple[int, str]]:
    return [
        (number, f"machine-{number}")
        for number in range(1, count + 1)
    ]


def write_daily(
    path: Path,
    day: str,
    rows: list[tuple[int, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "date",
                "machine_no",
                "machine_name",
            ]
        )

        for machine_no, machine_name in rows:
            writer.writerow(
                [
                    day,
                    machine_no,
                    machine_name,
                ]
            )


class BigMarchInventoryApprovalTests(
    unittest.TestCase
):
    def _create_blocked_change(
        self,
        data: Path,
    ) -> None:
        previous = machines()
        changed = machines()

        changed[0] = (
            1,
            "replacement-machine",
        )

        write_daily(
            data
            / "ana_slo_bigmarch_oyagi_20260901.csv",
            "2026-09-01",
            previous,
        )

        write_daily(
            data
            / "ana_slo_bigmarch_oyagi_20260902.csv",
            "2026-09-02",
            changed,
        )

        result = (
            guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )
        )

        self.assertTrue(
            result.blocked
        )

        self.assertEqual(
            result.status,
            "BLOCKED_ACTUAL_CHANGE",
        )

    def test_exact_change_can_be_approved(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            self._create_blocked_change(
                data
            )

            result = (
                approval.approve_big_march_inventory_incident(
                    data,
                    date(2026, 9, 1),
                    date(2026, 9, 2),
                    "Reviewed exact Big March inventory change.",
                    approve=True,
                )
            )

            saved = json.loads(
                inventory_guard_state_path(
                    data
                ).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            result["status"],
            "APPROVED",
        )

        self.assertEqual(
            result[
                "guard_status_after_approval"
            ],
            "PASS_FORMAL",
        )

        self.assertEqual(
            saved["status"],
            "APPROVED",
        )

        self.assertEqual(
            saved["incident_kind"],
            "ACTUAL_CHANGE",
        )

        self.assertTrue(
            saved["approved_at"]
        )

        self.assertEqual(
            saved["approved_reason"],
            "Reviewed exact Big March inventory change.",
        )

        self.assertEqual(
            saved["approval_tool"],
            "BIGMARCH_INVENTORY_GUARD_APPROVAL_V1",
        )

        self.assertEqual(
            len(
                saved[
                    "approval_previous_daily_sha256"
                ]
            ),
            64,
        )

        self.assertEqual(
            len(
                saved[
                    "approval_current_daily_sha256"
                ]
            ),
            64,
        )

    def test_approval_requires_explicit_switch(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            self._create_blocked_change(
                data
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Approval was not requested",
            ):
                approval.approve_big_march_inventory_incident(
                    data,
                    date(2026, 9, 1),
                    date(2026, 9, 2),
                    "Reviewed change.",
                    approve=False,
                )

    def test_wrong_dates_are_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            self._create_blocked_change(
                data
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Only an exact consecutive-day",
            ):
                approval.approve_big_march_inventory_incident(
                    data,
                    date(2026, 8, 31),
                    date(2026, 9, 2),
                    "Reviewed change.",
                    approve=True,
                )

    def test_non_consecutive_dates_are_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            self._create_blocked_change(
                data
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Only an exact consecutive-day",
            ):
                approval.approve_big_march_inventory_incident(
                    data,
                    date(2026, 9, 1),
                    date(2026, 9, 3),
                    "Reviewed change.",
                    approve=True,
                )

    def test_current_csv_drift_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            self._create_blocked_change(
                data
            )

            drifted = machines()

            drifted[0] = (
                1,
                "replacement-machine",
            )

            drifted[1] = (
                2,
                "different-later-change",
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                drifted,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "does not exactly match",
            ):
                approval.approve_big_march_inventory_incident(
                    data,
                    date(2026, 9, 1),
                    date(2026, 9, 2),
                    "Reviewed change.",
                    approve=True,
                )


if __name__ == "__main__":
    unittest.main()