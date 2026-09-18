from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MACHINE_DIR = ROOT / "machine_number"

if str(MACHINE_DIR) not in sys.path:
    sys.path.insert(0, str(MACHINE_DIR))

import slotanalyzer_bigmarch_inventory_guard as guard

from slotanalyzer_inventory_guard import (
    InventoryGuardPolicy,
)
from tests.test_bigmarch_inventory_guard import (
    machines,
    write_daily,
)


class BigMarchStaleReferenceGuardTests(
    unittest.TestCase
):
    def test_two_day_delay_safe_inventory_allows_reference_only(
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

            result = (
                guard.assess_big_march_inventory_guard(
                    data,
                    date(2026, 9, 5),
                )
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)
        self.assertTrue(result.reference_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_SOURCE_DELAY",
        )

        self.assertEqual(
            result.source_delay_days,
            2,
        )

        self.assertEqual(
            result.comparison_status,
            "COMPARED_NO_CHANGE",
        )

    def test_three_day_delay_safe_inventory_allows_reference_only(
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

            result = (
                guard.assess_big_march_inventory_guard(
                    data,
                    date(2026, 9, 6),
                )
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)
        self.assertTrue(result.reference_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_SOURCE_DELAY",
        )

        self.assertEqual(
            result.source_delay_days,
            3,
        )

    def test_fresh_inventory_does_not_enable_reference(
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

            result = (
                guard.assess_big_march_inventory_guard(
                    data,
                    date(2026, 9, 3),
                )
            )

        self.assertTrue(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)
        self.assertFalse(result.reference_allowed)

    def test_one_day_delay_does_not_enable_reference(
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

            result = (
                guard.assess_big_march_inventory_guard(
                    data,
                    date(2026, 9, 4),
                )
            )

        self.assertFalse(result.formal_allowed)
        self.assertTrue(result.provisional_allowed)
        self.assertFalse(result.reference_allowed)

    def test_non_consecutive_latest_inventory_blocks_reference(
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

            result = (
                guard.assess_big_march_inventory_guard(
                    data,
                    date(2026, 9, 6),
                )
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.reference_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_NON_CONSECUTIVE",
        )

        self.assertEqual(
            result.comparison_status,
            "NON_CONSECUTIVE",
        )

    def test_actual_change_blocks_stale_reference(
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

            result = (
                guard.assess_big_march_inventory_guard(
                    data,
                    date(2026, 9, 5),
                )
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.reference_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_ACTUAL_CHANGE",
        )

    def test_persistent_incident_blocks_stale_reference(
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

            first = (
                guard.assess_big_march_inventory_guard(
                    data,
                    date(2026, 9, 3),
                )
            )

            self.assertEqual(
                first.status,
                "BLOCKED_ACTUAL_CHANGE",
            )

            write_daily(
                data
                / "ana_slo_bigmarch_oyagi_20260903.csv",
                "2026-09-03",
                changed,
            )

            result = (
                guard.assess_big_march_inventory_guard(
                    data,
                    date(2026, 9, 6),
                )
            )

        self.assertTrue(result.blocked)
        self.assertFalse(result.reference_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_PERSISTENT_INCIDENT",
        )

    def test_known_change_inside_stale_gap_blocks_reference(
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
                        date(2026, 9, 5),
                    )
                )

        self.assertTrue(result.blocked)
        self.assertFalse(result.reference_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_KNOWN_CHANGE_UNCONFIRMED",
        )

        self.assertTrue(
            result.known_change_date
        )

    def test_monitor_error_blocks_reference(
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
                    "simulated monitor failure"
                ),
            ):
                result = (
                    guard.assess_big_march_inventory_guard(
                        data,
                        date(2026, 9, 5),
                    )
                )

        self.assertTrue(result.blocked)
        self.assertFalse(result.reference_allowed)

        self.assertEqual(
            result.status,
            "BLOCKED_GUARD_ERROR",
        )

    def test_stale_reference_enforcer_accepts_safe_two_day_delay(
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

            result = (
                guard
                .enforce_big_march_stale_reference_inventory_guard(
                    data,
                    date(2026, 9, 5),
                )
            )

        self.assertTrue(result.reference_allowed)
        self.assertTrue(result.blocked)
        self.assertFalse(result.formal_allowed)
        self.assertFalse(result.provisional_allowed)

    def test_stale_reference_enforcer_rejects_one_day_delay(
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
                guard.enforce_big_march_stale_reference_inventory_guard(
                    data,
                    date(2026, 9, 4),
                )


if __name__ == "__main__":
    unittest.main()