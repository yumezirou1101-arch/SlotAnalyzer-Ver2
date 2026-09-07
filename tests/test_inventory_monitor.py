from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"
if str(MACHINE) not in sys.path:
    sys.path.insert(0, str(MACHINE))

import slotanalyzer_inventory_monitor as monitor


NOW = datetime(2026, 9, 7, 8, 0, tzinfo=timezone.utc)


def write_daily(path: Path, day: str, count: int, rename: bool = False) -> None:
    rows = ["date,machine_no,machine_name"]
    for number in range(1, count + 1):
        name = "replacement" if rename and number == 1 else f"machine-{number}"
        rows.append(f"{day},{number},{name}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8-sig")


class InventoryMonitorTests(unittest.TestCase):
    def test_change_is_observed_without_guard_state(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_daily(data / "ana_slo_bigmarch_oyagi_20260901.csv", "2026-09-01", 276)
            write_daily(data / "ana_slo_bigmarch_oyagi_20260902.csv", "2026-09-02", 266, True)
            result = monitor.observe_big_march_inventory(
                data, date(2026, 9, 3), generated_at_jst=NOW
            )
            saved = json.loads(Path(result["evidence_path"]).read_text(encoding="utf-8"))
            self.assertFalse((data / "inventory_guard_state.json").exists())
        self.assertEqual(result["status"], "CHANGE_OBSERVED")
        self.assertEqual(result["removed_count"], 10)
        self.assertEqual(result["renamed_count"], 1)
        self.assertEqual(len(result["previous_daily_sha256"]), 64)
        self.assertFalse(saved["affects_formal"])
        self.assertFalse(saved["affects_provisional"])
        self.assertFalse(saved["affects_morning_status"])
        self.assertFalse(saved["affects_sleep"])

    def test_same_operation_is_idempotent_and_sha_change_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            previous = data / "ana_slo_bigmarch_oyagi_20260901.csv"
            current = data / "ana_slo_bigmarch_oyagi_20260902.csv"
            write_daily(previous, "2026-09-01", 276)
            write_daily(current, "2026-09-02", 276)
            first = monitor.observe_big_march_inventory(
                data, date(2026, 9, 3), generated_at_jst=NOW
            )
            path = Path(first["evidence_path"])
            before_text = path.read_text(encoding="utf-8")
            before_mtime = path.stat().st_mtime_ns
            second = monitor.observe_big_march_inventory(
                data, date(2026, 9, 3), generated_at_jst=NOW
            )
            self.assertEqual(second["status"], "ALREADY_OBSERVED")
            self.assertEqual(path.stat().st_mtime_ns, before_mtime)
            write_daily(current, "2026-09-02", 276, True)
            conflict = monitor.observe_big_march_inventory(
                data, date(2026, 9, 3), generated_at_jst=NOW
            )
            self.assertEqual(conflict["status"], "SOURCE_CHANGED_AFTER_OBSERVATION")
            self.assertEqual(path.read_text(encoding="utf-8"), before_text)

    def test_non_consecutive_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_daily(data / "ana_slo_bigmarch_oyagi_20260901.csv", "2026-09-01", 276)
            write_daily(data / "ana_slo_bigmarch_oyagi_20260903.csv", "2026-09-03", 276)
            result = monitor.observe_big_march_inventory(
                data, date(2026, 9, 4), generated_at_jst=NOW
            )
        self.assertEqual(result["status"], "NON_CONSECUTIVE")
        self.assertFalse(result["comparison_performed"])
        self.assertIsNone(result["has_changes"])

    def test_write_error_is_best_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_daily(data / "ana_slo_bigmarch_oyagi_20260901.csv", "2026-09-01", 276)
            write_daily(data / "ana_slo_bigmarch_oyagi_20260902.csv", "2026-09-02", 276)
            with mock.patch.object(monitor, "_atomic_write_json", side_effect=OSError("denied")):
                result = monitor.observe_big_march_inventory_best_effort(
                    data, date(2026, 9, 3), generated_at_jst=NOW
                )
        self.assertEqual(result["status"], "ERROR")
        self.assertFalse(result["affects_morning_status"])
        self.assertIn("denied", result["error"])


if __name__ == "__main__":
    unittest.main()
