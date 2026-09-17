from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"

if str(MACHINE) not in sys.path:
    sys.path.insert(0, str(MACHINE))


import slotanalyzer_bigmarch_inventory_guard as guard

from slotanalyzer_inventory_guard import (
    InventoryGuardPolicy,
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


class BigMarchInventoryGuardTests(
    unittest.TestCase
):
    def test_fresh_no_change_allows_formal(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                stable,
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

        self.assertFalse(result.blocked)
        self.assertTrue(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "PASS_FORMAL",
        )

        self.assertEqual(
            result.source_delay_days,
            0,
        )

        self.assertEqual(
            result.comparison_status,
            "COMPARED_NO_CHANGE",
        )

    def test_one_day_delay_with_safe_inventory_allows_only_provisional(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                stable,
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 4),
            )

        self.assertFalse(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertTrue(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "PASS_PROVISIONAL_ONLY",
        )

        self.assertEqual(
            result.source_delay_days,
            1,
        )

        self.assertEqual(
            result.comparison_status,
            "COMPARED_NO_CHANGE",
        )

    def test_two_day_delay_blocks_both(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                stable,
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 5),
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_SOURCE_DELAY",
        )

        self.assertEqual(
            result.source_delay_days,
            2,
        )

    def test_non_consecutive_inventory_blocks_both(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260903.csv",
                "2026-09-03",
                stable,
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 4),
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_NON_CONSECUTIVE",
        )

        self.assertEqual(
            result.comparison_status,
            "NON_CONSECUTIVE",
        )

    def test_one_day_delay_without_confirmed_normal_transition_blocks(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                machines(),
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 4),
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_COMPARISON_UNAVAILABLE",
        )

        self.assertEqual(
            result.comparison_status,
            "PREVIOUS_MISSING",
        )

    def test_added_machine_creates_persistent_block(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            previous = machines()
            current = machines()

            current.append(
                (
                    277,
                    "new-machine",
                )
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
                current,
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

            saved = json.loads(
                inventory_guard_state_path(
                    data
                ).read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result.blocked)

        self.assertEqual(
            result.status,
            "BLOCKED_ACTUAL_CHANGE",
        )

        self.assertEqual(
            saved["added_machine_numbers"],
            [277],
        )

        self.assertEqual(
            saved["removed_machine_numbers"],
            [],
        )

    def test_removed_machine_creates_persistent_block(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            previous = machines()
            current = machines()[:-1]

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
                current,
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

            saved = json.loads(
                inventory_guard_state_path(
                    data
                ).read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result.blocked)

        self.assertEqual(
            result.status,
            "BLOCKED_ACTUAL_CHANGE",
        )

        self.assertEqual(
            saved["added_machine_numbers"],
            [],
        )

        self.assertEqual(
            saved["removed_machine_numbers"],
            [276],
        )

    def test_renamed_machine_creates_persistent_block(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

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

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

            state_path = (
                inventory_guard_state_path(
                    data
                )
            )

            saved = json.loads(
                state_path.read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_ACTUAL_CHANGE",
        )

        self.assertEqual(
            saved["incident_kind"],
            "ACTUAL_CHANGE",
        )

        self.assertEqual(
            saved["status"],
            "BLOCKED",
        )

        self.assertEqual(
            saved["previous_date"],
            "2026-09-01",
        )

        self.assertEqual(
            saved["change_date"],
            "2026-09-02",
        )

        self.assertEqual(
            saved[
                "renamed_machine_numbers"
            ][0]["machine_no"],
            1,
        )

        self.assertEqual(
            len(
                saved[
                    "previous_daily_sha256"
                ]
            ),
            64,
        )

        self.assertEqual(
            len(
                saved[
                    "current_daily_sha256"
                ]
            ),
            64,
        )

    def test_change_then_next_day_no_change_remains_blocked(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

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

            first = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260903.csv",
                "2026-09-03",
                changed,
            )

            second = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 4),
            )

            saved = json.loads(
                inventory_guard_state_path(
                    data
                ).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            first.status,
            "BLOCKED_ACTUAL_CHANGE",
        )

        self.assertTrue(second.blocked)
        self.assertFalse(second.formal_allowed)
        self.assertFalse(second.provisional_allowed)

        self.assertEqual(
            second.status,
            "BLOCKED_PERSISTENT_INCIDENT",
        )

        self.assertEqual(
            second.comparison_status,
            "COMPARED_NO_CHANGE",
        )

        self.assertEqual(
            saved["incident_kind"],
            "ACTUAL_CHANGE",
        )

        self.assertEqual(
            saved["change_date"],
            "2026-09-02",
        )

        self.assertEqual(
            saved["status"],
            "BLOCKED",
        )

    def test_persistent_block_is_reloaded_on_later_assessment(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

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

            guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260903.csv",
                "2026-09-03",
                changed,
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 4),
            )

        self.assertTrue(result.blocked)

        self.assertEqual(
            result.status,
            "BLOCKED_PERSISTENT_INCIDENT",
        )

        self.assertEqual(
            result.persistent_state[
                "incident_kind"
            ],
            "ACTUAL_CHANGE",
        )

        self.assertEqual(
            result.persistent_state[
                "change_date"
            ],
            "2026-09-02",
        )

    def test_exact_preapproved_change_releases_formal(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

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

            first = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

            self.assertTrue(first.blocked)

            state_path = (
                inventory_guard_state_path(
                    data
                )
            )

            saved = json.loads(
                state_path.read_text(
                    encoding="utf-8"
                )
            )

            saved["status"] = "APPROVED"
            saved["approved_at"] = (
                "2026-09-03T09:00:00+09:00"
            )
            saved["approved_reason"] = (
                "Test fixture exact reviewed change."
            )

            state_path.write_text(
                json.dumps(
                    saved,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

        self.assertFalse(result.blocked)
        self.assertTrue(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "PASS_FORMAL",
        )

    def test_known_change_unconfirmed_blocks_both(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                stable,
            )

            test_policy = InventoryGuardPolicy(
                store="BIGMARCH_TAKASAKI_OYAGI",
                known_change_dates=frozenset(
                    {
                        date(2026, 9, 3),
                    }
                ),
                confirmed_inventory_prefix=(
                    "bigmarch_oyagi_inventory"
                ),
                daily_inventory_pattern=(
                    "ana_slo_bigmarch_oyagi_{ymd}.csv"
                ),
            )

            with mock.patch.object(
                guard,
                "BIGMARCH_TAKASAKI_OYAGI_POLICY",
                test_policy,
            ):
                result = (
                    guard.assess_big_march_inventory_guard(
                        data,
                        date(2026, 9, 4),
                    )
                )

            saved = json.loads(
                inventory_guard_state_path(
                    data
                ).read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_KNOWN_CHANGE_UNCONFIRMED",
        )

        self.assertTrue(
            result.known_change_date
        )

        self.assertEqual(
            saved["incident_kind"],
            "KNOWN_CHANGE_UNCONFIRMED",
        )

        self.assertEqual(
            saved["change_date"],
            "2026-09-03",
        )

    def test_source_changed_after_observation_blocks_both(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

            stable = machines()

            previous_path = (
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv"
            )

            current_path = (
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv"
            )

            write_daily(
                previous_path,
                "2026-09-01",
                stable,
            )

            write_daily(
                current_path,
                "2026-09-02",
                stable,
            )

            first = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

            self.assertFalse(first.blocked)

            write_daily(
                current_path,
                "2026-09-02",
                list(reversed(stable)),
            )

            second = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

        self.assertTrue(second.blocked)
        self.assertFalse(second.formal_allowed)
        self.assertFalse(second.provisional_allowed)

        self.assertEqual(
            second.status,
            "BLOCKED_SOURCE_CHANGED",
        )

        self.assertEqual(
            second.monitor_status,
            "SOURCE_CHANGED_AFTER_OBSERVATION",
        )

    def test_monitor_internal_error_fails_closed(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                stable,
            )

            with mock.patch.object(
                guard,
                "observe_big_march_inventory",
                side_effect=RuntimeError(
                    "simulated guard dependency failure"
                ),
            ):
                result = (
                    guard.assess_big_march_inventory_guard(
                        data,
                        date(2026, 9, 3),
                    )
                )

        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_GUARD_ERROR",
        )

        self.assertEqual(
            result.monitor_status,
            "ERROR",
        )

        self.assertIn(
            "simulated guard dependency failure",
            result.reason,
        )

    def test_formal_enforce_raises_when_only_provisional_is_safe(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                stable,
            )

            with self.assertRaises(
                guard.BigMarchInventoryGuardBlockedError
            ):
                guard.enforce_big_march_formal_inventory_guard(
                    data,
                    date(2026, 9, 4),
                )

    def test_provisional_enforce_raises_when_delay_is_two_days(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260902.csv",
                "2026-09-02",
                stable,
            )

            with self.assertRaises(
                guard.BigMarchInventoryGuardBlockedError
            ):
                guard.enforce_big_march_provisional_inventory_guard(
                    data,
                    date(2026, 9, 5),
                )


    def test_actual_change_still_writes_monitor_evidence(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)

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

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 3),
            )

            evidence_path = (
                data
                / "inventory_monitor"
                / "inventory_monitor_20260903.json"
            )

            self.assertTrue(
                evidence_path.is_file()
            )

            evidence = json.loads(
                evidence_path.read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result.blocked)

        self.assertEqual(
            result.status,
            "BLOCKED_ACTUAL_CHANGE",
        )

        self.assertEqual(
            result.monitor_status,
            "CHANGE_OBSERVED",
        )

        self.assertEqual(
            evidence["status"],
            "CHANGE_OBSERVED",
        )

    def test_non_consecutive_still_writes_monitor_evidence(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260901.csv",
                "2026-09-01",
                stable,
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260903.csv",
                "2026-09-03",
                stable,
            )

            result = guard.assess_big_march_inventory_guard(
                data,
                date(2026, 9, 4),
            )

            evidence_path = (
                data
                / "inventory_monitor"
                / "inventory_monitor_20260904.json"
            )

            self.assertTrue(
                evidence_path.is_file()
            )

            evidence = json.loads(
                evidence_path.read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result.blocked)

        self.assertEqual(
            result.status,
            "BLOCKED_NON_CONSECUTIVE",
        )

        self.assertEqual(
            result.monitor_status,
            "NON_CONSECUTIVE",
        )

        self.assertEqual(
            evidence["status"],
            "NON_CONSECUTIVE",
        )


if __name__ == "__main__":
    unittest.main()
