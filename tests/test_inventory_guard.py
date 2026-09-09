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
    confirmed_inventory_path,
    enforce_inventory_guard,
    inspect_inventory_transition,
    inventory_guard_state_path,
    assess_yasuda_inventory_guard,
)


TEST_KNOWN_CHANGE_POLICY = InventoryGuardPolicy(
    store="TEST_MARUHAN_KNOWN_CHANGE",
    known_change_dates=frozenset({date(2026, 9, 8)}),
    confirmed_inventory_prefix="test_maruhan_inventory",
)


def write_inventory(
    path: Path, rows: list[tuple[int, str]], day: str = "2026-09-01"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["日付", "台番号", "機種名"])
        for machine_no, machine_name in rows:
            writer.writerow([day, machine_no, machine_name])


def machines(count: int = 514) -> list[tuple[int, str]]:
    return [(number, f"machine-{number}") for number in range(1, count + 1)]


class InventoryGuardTests(unittest.TestCase):
    @staticmethod
    def approve_saved_state(data: Path) -> dict:
        state_path = inventory_guard_state_path(data)
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        saved["status"] = "APPROVED"
        saved["approved_at"] = "2026-09-09T12:00:00+09:00"
        saved["approved_reason"] = "Reviewed externally; test fixture approval."
        state_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        return saved

    def test_pure_comparison_reports_change_counts_sha_and_exact_names(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = data / "previous.csv"
            current = data / "current.csv"
            previous.write_text(
                "date,machine_no,machine_name\n2026-09-01,1,A\n2026-09-01,2,B\n",
                encoding="utf-8-sig",
            )
            current.write_text(
                "date,machine_no,machine_name\n2026-09-02,2,B2\n2026-09-02,3,C\n",
                encoding="utf-8-sig",
            )
            result = inspect_inventory_transition(
                "TEST", previous, current, date(2026, 9, 1), date(2026, 9, 2)
            )
            self.assertFalse((data / "inventory_guard_state.json").exists())
        self.assertEqual(result.comparison_status, "COMPARED_CHANGE")
        self.assertEqual(result.added_machine_numbers, [3])
        self.assertEqual(result.removed_machine_numbers, [1])
        self.assertEqual(result.renamed_count, 1)
        self.assertEqual(result.changed_machine_count, 3)
        self.assertEqual(result.change_rate, 1.5)
        self.assertEqual(result.machine_name_comparison, "EXACT_STRING")
        self.assertEqual(len(result.previous_daily_sha256), 64)
        self.assertEqual(len(result.current_daily_sha256), 64)

    def test_pure_comparison_non_consecutive_is_not_no_change(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            result = inspect_inventory_transition(
                "TEST", data / "missing1.csv", data / "missing2.csv",
                date(2026, 9, 1), date(2026, 9, 3),
            )
        self.assertEqual(result.comparison_status, "NON_CONSECUTIVE")
        self.assertFalse(result.comparison_performed)
        self.assertIsNone(result.has_changes)
        self.assertIsNone(result.added_count)

    def test_pure_comparison_rejects_duplicate_and_missing_values(self):
        cases = {
            "duplicate": "date,machine_no,machine_name\n2026-09-01,1,A\n2026-09-01,1,B\n",
            "missing_no": "date,machine_no,machine_name\n2026-09-01,,A\n",
            "missing_name": "date,machine_no,machine_name\n2026-09-01,1,\n",
        }
        for label, content in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                data = Path(directory)
                previous = data / "previous.csv"
                current = data / "current.csv"
                previous.write_text(content, encoding="utf-8-sig")
                current.write_text(
                    "date,machine_no,machine_name\n2026-09-02,1,A\n",
                    encoding="utf-8-sig",
                )
                result = inspect_inventory_transition(
                    "TEST", previous, current, date(2026, 9, 1), date(2026, 9, 2)
                )
                self.assertEqual(result.comparison_status, "PREVIOUS_INVALID")

    def test_yasuda_320_no_change_passes_and_rename_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines(320)
            current = machines(320)
            write_inventory(data / "ana_slo_20260901.csv", previous)
            write_inventory(data / "ana_slo_20260902.csv", current, "2026-09-02")
            passed = assess_yasuda_inventory_guard(
                data, date(2026, 9, 3), date(2026, 9, 2)
            )
            current[99] = (100, "replacement")
            write_inventory(data / "ana_slo_20260902.csv", current, "2026-09-02")
            blocked = assess_yasuda_inventory_guard(
                data, date(2026, 9, 3), date(2026, 9, 2)
            )
        self.assertFalse(passed.blocked)
        self.assertTrue(blocked.blocked)
        self.assertEqual(blocked.comparison.renamed_machine_numbers[0].machine_no, 100)

    def test_yasuda_missing_comparison_never_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            result = assess_yasuda_inventory_guard(
                data, date(2026, 9, 3), date(2026, 9, 2)
            )
        self.assertTrue(result.blocked)
        self.assertIn("PREVIOUS_MISSING", result.reason)

    def test_yasuda_non_consecutive_never_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_inventory(data / "ana_slo_20260831.csv", machines(320), "2026-08-31")
            write_inventory(data / "ana_slo_20260902.csv", machines(320), "2026-09-02")
            result = assess_yasuda_inventory_guard(
                data, date(2026, 9, 3), date(2026, 9, 2)
            )
        self.assertTrue(result.blocked)
        self.assertIn("NON_CONSECUTIVE", result.reason)

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
            data = Path(directory)
            result = assess_inventory_guard(
                data,
                date(2026, 9, 8),
                date(2026, 9, 7),
                TEST_KNOWN_CHANGE_POLICY,
            )
            saved = json.loads(inventory_guard_state_path(data).read_text(encoding="utf-8"))
        self.assertTrue(result.blocked)
        self.assertEqual(result.status, "MANUAL_REVIEW")
        self.assertTrue(result.known_change_date)
        self.assertEqual(saved["incident_kind"], "KNOWN_CHANGE_UNCONFIRMED")

    def test_known_change_saves_actual_change_before_confirmed_gate_and_stays_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines()
            changed = machines()
            for index in range(121):
                machine_no, _ = changed[index]
                changed[index] = (machine_no, f"replacement-{machine_no}")
            write_inventory(data / "ana_slo_20260906.csv", previous, "2026-09-06")
            write_inventory(data / "ana_slo_20260907.csv", changed, "2026-09-07")
            first = assess_inventory_guard(
                data, date(2026, 9, 8), date(2026, 9, 7), TEST_KNOWN_CHANGE_POLICY
            )
            saved_first = inventory_guard_state_path(data).read_text(encoding="utf-8")
            write_inventory(data / "ana_slo_20260908.csv", changed, "2026-09-08")
            second = assess_inventory_guard(
                data, date(2026, 9, 9), date(2026, 9, 8), TEST_KNOWN_CHANGE_POLICY
            )
            saved_second = inventory_guard_state_path(data).read_text(encoding="utf-8")
        self.assertTrue(first.blocked)
        self.assertEqual(first.comparison.current_date, "2026-09-07")
        self.assertEqual(first.persistent_state["incident_kind"], "ACTUAL_CHANGE")
        self.assertEqual(len(first.persistent_state["renamed_machine_numbers"]), 121)
        self.assertTrue(second.blocked)
        self.assertFalse(second.comparison.has_changes)
        self.assertEqual(saved_first, saved_second)

    def test_same_change_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines()
            changed = machines()
            changed[0] = (1, "replacement")
            write_inventory(data / "ana_slo_20260901.csv", previous)
            write_inventory(data / "ana_slo_20260902.csv", changed, "2026-09-02")
            assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
            state_path = inventory_guard_state_path(data)
            first_content = state_path.read_bytes()
            first_mtime = state_path.stat().st_mtime_ns
            assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
            second_content = state_path.read_bytes()
            second_mtime = state_path.stat().st_mtime_ns
        self.assertEqual(first_content, second_content)
        self.assertEqual(first_mtime, second_mtime)

    def test_exact_approved_change_releases_but_different_change_reblocks(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            original = machines()
            first_change = machines()
            first_change[0] = (1, "replacement-one")
            write_inventory(data / "ana_slo_20260901.csv", original)
            write_inventory(data / "ana_slo_20260902.csv", first_change, "2026-09-02")
            assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
            self.approve_saved_state(data)
            approved = assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
            second_change = list(first_change)
            second_change[1] = (2, "replacement-two")
            write_inventory(data / "ana_slo_20260903.csv", second_change, "2026-09-03")
            reblocked = assess_inventory_guard(data, date(2026, 9, 4), date(2026, 9, 3))
        self.assertFalse(approved.blocked)
        self.assertTrue(reblocked.blocked)
        self.assertEqual(reblocked.persistent_state["change_date"], "2026-09-03")
        self.assertEqual(reblocked.persistent_state["status"], "BLOCKED")

    def test_unconfirmed_known_change_without_actual_comparison_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            first = assess_inventory_guard(
                data, date(2026, 9, 8), date(2026, 9, 7), TEST_KNOWN_CHANGE_POLICY
            )
            write_inventory(data / "ana_slo_20260907.csv", machines(), "2026-09-07")
            write_inventory(data / "ana_slo_20260908.csv", machines(), "2026-09-08")
            second = assess_inventory_guard(
                data, date(2026, 9, 9), date(2026, 9, 8), TEST_KNOWN_CHANGE_POLICY
            )
        self.assertTrue(first.blocked)
        self.assertEqual(first.persistent_state["incident_kind"], "KNOWN_CHANGE_UNCONFIRMED")
        self.assertTrue(second.blocked)
        self.assertFalse(second.comparison.has_changes)

    def test_planned_incident_is_enriched_when_matching_actual_change_arrives(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            assess_inventory_guard(
                data, date(2026, 9, 8), date(2026, 9, 7), TEST_KNOWN_CHANGE_POLICY
            )
            previous = machines()
            changed = machines()
            changed[0] = (1, "actual-change")
            write_inventory(data / "ana_slo_20260907.csv", previous, "2026-09-07")
            write_inventory(data / "ana_slo_20260908.csv", changed, "2026-09-08")
            result = assess_inventory_guard(
                data, date(2026, 9, 9), date(2026, 9, 8), TEST_KNOWN_CHANGE_POLICY
            )
        self.assertTrue(result.blocked)
        self.assertEqual(result.persistent_state["incident_kind"], "ACTUAL_CHANGE")
        self.assertEqual(result.persistent_state["change_date"], "2026-09-08")
        self.assertEqual(result.persistent_state["renamed_machine_numbers"][0]["machine_no"], 1)

    def test_confirmed_known_change_keeps_actual_evidence_first(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = machines()
            changed = machines()
            changed[0] = (1, "actual-change")
            write_inventory(data / "ana_slo_20260906.csv", previous, "2026-09-06")
            write_inventory(data / "ana_slo_20260907.csv", changed, "2026-09-07")
            write_inventory(
                confirmed_inventory_path(data, date(2026, 9, 8), TEST_KNOWN_CHANGE_POLICY),
                changed,
                "2026-09-08",
            )
            result = assess_inventory_guard(
                data, date(2026, 9, 8), date(2026, 9, 7), TEST_KNOWN_CHANGE_POLICY
            )
        self.assertTrue(result.blocked)
        self.assertEqual(result.persistent_state["change_date"], "2026-09-07")

    def test_confirmed_target_change_is_checked_after_no_actual_change(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            stable = machines()
            confirmed = machines()
            confirmed[0] = (1, "confirmed-change")
            write_inventory(data / "ana_slo_20260906.csv", stable, "2026-09-06")
            write_inventory(data / "ana_slo_20260907.csv", stable, "2026-09-07")
            write_inventory(
                confirmed_inventory_path(data, date(2026, 9, 8), TEST_KNOWN_CHANGE_POLICY),
                confirmed,
                "2026-09-08",
            )
            result = assess_inventory_guard(
                data, date(2026, 9, 8), date(2026, 9, 7), TEST_KNOWN_CHANGE_POLICY
            )
        self.assertTrue(result.blocked)
        self.assertEqual(result.comparison.current_date, "2026-09-08")
        self.assertEqual(result.persistent_state["change_date"], "2026-09-08")

    def test_different_change_does_not_overwrite_unapproved_incident(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            original = machines()
            first = machines()
            first[0] = (1, "first-change")
            second = list(first)
            second[1] = (2, "second-change")
            write_inventory(data / "ana_slo_20260901.csv", original)
            write_inventory(data / "ana_slo_20260902.csv", first, "2026-09-02")
            assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))
            state_path = inventory_guard_state_path(data)
            saved = state_path.read_bytes()
            write_inventory(data / "ana_slo_20260903.csv", second, "2026-09-03")
            result = assess_inventory_guard(data, date(2026, 9, 4), date(2026, 9, 3))
            saved_after = state_path.read_bytes()
        self.assertTrue(result.blocked)
        self.assertIn("different unapproved", result.reason)
        self.assertEqual(saved, saved_after)

    def test_corrupt_persistent_state_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_inventory(data / "ana_slo_20260901.csv", machines())
            write_inventory(data / "ana_slo_20260902.csv", machines(), "2026-09-02")
            inventory_guard_state_path(data).write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "state is unreadable"):
                assess_inventory_guard(data, date(2026, 9, 3), date(2026, 9, 2))

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
                enforce_inventory_guard(
                    Path(directory),
                    date(2026, 9, 8),
                    date(2026, 9, 7),
                    TEST_KNOWN_CHANGE_POLICY,
                )

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
