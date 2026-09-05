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
    InventoryGuardPolicy,
    InventoryGuardBlockedError,
    assess_inventory_guard,
    compare_inventory_files,
    enforce_inventory_guard,
    inventory_guard_state_path,
)


def write_inventory(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["日付", "台番号", "機種名"])
        for machine_no, machine_name in rows:
            writer.writerow(["2026-09-01", machine_no, machine_name])


def machines(count: int = 514) -> list[tuple[int, str]]:
    return [(number, f"machine-{number}") for number in range(1, count + 1)]


class InventoryGuardTests(unittest.TestCase):
    def test_normal_514_inventory_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_inventory(data / "ana_slo_20260901.csv", machines())
            write_inventory(data / "ana_slo_20260902.csv", machines())
            result = assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
        self.assertFalse(result.blocked)
        self.assertEqual(result.status, "PASS")
        self.assertIsNotNone(result.comparison)
        self.assertFalse(result.comparison.has_changes)

    def test_known_change_date_without_confirmed_inventory_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            result = assess_inventory_guard(Path(directory), date(2026, 9, 8), date(2026, 9, 7))
        self.assertTrue(result.blocked)
        self.assertEqual(result.status, "MANUAL_REVIEW")
        self.assertTrue(result.known_change_date)

    def test_single_rename_with_514_machines_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines()
            current = machines()
            current[500 - 1] = (500, "new-machine")
            write_inventory(data / "ana_slo_20260901.csv", previous)
            write_inventory(data / "ana_slo_20260902.csv", current)
            result = assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
        self.assertTrue(result.blocked)
        self.assertEqual([item.machine_no for item in result.comparison.renamed_machine_numbers], [500])

    def test_one_removed_and_one_added_with_514_machines_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines()
            current = previous[1:] + [(900, "new-machine")]
            write_inventory(data / "ana_slo_20260901.csv", previous)
            write_inventory(data / "ana_slo_20260902.csv", current)
            result = assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
        self.assertTrue(result.blocked)
        self.assertEqual(result.comparison.removed_machine_numbers, [1])
        self.assertEqual(result.comparison.added_machine_numbers, [900])

    def test_513_and_515_are_explicit_and_blocked_when_changed(self):
        for count in (513, 515):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                data = Path(directory)
                write_inventory(data / "ana_slo_20260901.csv", machines(514))
                write_inventory(data / "ana_slo_20260902.csv", machines(count))
                result = assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
                self.assertTrue(result.blocked)
                self.assertEqual(result.comparison.current_machine_count, count)

    def test_enforcement_has_no_allow_gap_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InventoryGuardBlockedError):
                enforce_inventory_guard(Path(directory), date(2026, 9, 8), date(2026, 9, 7))

    def test_detected_rename_creates_persistent_block_state(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines()
            current = machines()
            current[499] = (500, "replacement-machine")
            write_inventory(data / "ana_slo_20260907.csv", previous)
            write_inventory(data / "ana_slo_20260908.csv", current)
            result = assess_inventory_guard(data, date(2026, 9, 9), date(2026, 9, 8))
            state_path = inventory_guard_state_path(data)
            saved = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertTrue(result.blocked)
        self.assertEqual(saved["status"], "BLOCKED")
        self.assertEqual(saved["store"], "MARUHAN_MAEBASHI")
        self.assertEqual(saved["change_date"], "2026-09-08")
        self.assertEqual(saved["renamed_machine_numbers"][0]["machine_no"], 500)
        self.assertEqual(saved["approved_at"], "")
        self.assertEqual(saved["approved_reason"], "")

    def test_next_day_zero_diff_remains_blocked_until_explicit_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines()
            changed = machines()
            changed[499] = (500, "replacement-machine")
            write_inventory(data / "ana_slo_20260907.csv", previous)
            write_inventory(data / "ana_slo_20260908.csv", changed)
            assess_inventory_guard(data, date(2026, 9, 9), date(2026, 9, 8))
            write_inventory(data / "ana_slo_20260909.csv", changed)
            result = assess_inventory_guard(data, date(2026, 9, 10), date(2026, 9, 9))
            with self.assertRaises(InventoryGuardBlockedError):
                enforce_inventory_guard(data, date(2026, 9, 10), date(2026, 9, 9))
        self.assertTrue(result.blocked)
        self.assertEqual(result.status, "MANUAL_REVIEW")
        self.assertFalse(result.comparison.has_changes)
        self.assertEqual(result.persistent_state["status"], "BLOCKED")
        self.assertIn("explicit approval", result.reason)

    def test_diff_and_persistent_state_are_store_agnostic(self):
        policy = InventoryGuardPolicy(
            store="TEST_STORE",
            known_change_dates=frozenset(),
            confirmed_inventory_prefix="test_inventory",
            daily_inventory_pattern="custom_store_{ymd}.csv",
        )
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_inventory(data / "custom_store_20260901.csv", [(1, "old")])
            write_inventory(data / "custom_store_20260902.csv", [(1, "new")])
            result = assess_inventory_guard(
                data, date(2026, 9, 3), date(2026, 9, 2), policy
            )
        self.assertTrue(result.blocked)
        self.assertEqual(result.persistent_state["store"], "TEST_STORE")

    def test_only_complete_explicit_approval_releases_persistent_block(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines()
            changed = machines()
            changed[499] = (500, "replacement-machine")
            write_inventory(data / "ana_slo_20260907.csv", previous)
            write_inventory(data / "ana_slo_20260908.csv", changed)
            assess_inventory_guard(data, date(2026, 9, 9), date(2026, 9, 8))
            state_path = inventory_guard_state_path(data)
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            saved["status"] = "APPROVED"
            saved["approved_at"] = "2026-09-09T12:00:00+09:00"
            saved["approved_reason"] = "Reviewed externally; test fixture approval."
            state_path.write_text(
                json.dumps(saved, ensure_ascii=False), encoding="utf-8"
            )
            write_inventory(data / "ana_slo_20260909.csv", changed)
            result = assess_inventory_guard(data, date(2026, 9, 10), date(2026, 9, 9))
        self.assertFalse(result.blocked)
        self.assertEqual(result.status, "PASS")


if __name__ == "__main__":
    unittest.main()
