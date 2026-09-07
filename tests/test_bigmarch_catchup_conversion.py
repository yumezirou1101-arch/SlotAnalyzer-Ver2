from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MACHINE) not in sys.path:
    sys.path.insert(0, str(MACHINE))

import run_slotanalyzer_morning_automation as automation
import slotanalyzer_morning_automation_support as support


def load_converter():
    path = MACHINE / "ana_slo_bigmarch_oyagi_batch_html_to_daily_csv.py"
    spec = importlib.util.spec_from_file_location("bigmarch_catchup_converter_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


converter = load_converter()
JST = support.JST


def write_source(root: Path, day: date, rows: int = 200, store: str = "ビッグマーチ高崎おおやぎ店") -> Path:
    body = "".join(
        f"<tr><td>機種{number}</td><td>{number}</td><td>1000</td><td>{number}</td></tr>"
        for number in range(1, rows + 1)
    )
    html = (
        f"<html><head><title>{store} {day:%Y/%m/%d}</title></head><body>"
        f"<div>{store} {day:%Y/%m/%d}</div>"
        "<table><thead><tr><th>機種名</th><th>台番号</th><th>G数</th><th>差枚</th></tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>"
    )
    path = root / f"ana_slo_bigmarch_oyagi_{day:%Y%m%d}_source.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_daily(root: Path, day: date, rows: int = 200, duplicate: bool = False) -> Path:
    data = root / "data/bigmarch_takasaki_oyagi/machine_number"
    data.mkdir(parents=True, exist_ok=True)
    numbers = list(range(1, rows + 1))
    if duplicate and rows > 1:
        numbers[-1] = numbers[-2]
    frame = pd.DataFrame({
        "date": [day.isoformat()] * rows,
        "machine_name": [f"機種{number}" for number in range(1, rows + 1)],
        "machine_no": numbers,
        "G": [1000] * rows,
        "diff": list(range(rows)),
    })
    path = data / f"ana_slo_bigmarch_oyagi_{day:%Y%m%d}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


class BigMarchCatchupConversionTests(unittest.TestCase):
    def test_global_batch_mode_keeps_partial_failure_behavior(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_source(root, date(2026, 9, 4), rows=199)
            write_source(root, date(2026, 9, 5), rows=200)
            output = root / "data/bigmarch_takasaki_oyagi/machine_number"
            with mock.patch.object(converter, "PROJECT_ROOT", root), \
                 mock.patch.object(converter, "OUTPUT_DIR", output), \
                 mock.patch.object(converter, "SUMMARY_DIR", root / "logs"), \
                 mock.patch.object(sys, "argv", ["converter", "--min-machines", "200"]):
                converter.main()
            self.assertFalse((output / "ana_slo_bigmarch_oyagi_20260904.csv").exists())
            self.assertTrue((output / "ana_slo_bigmarch_oyagi_20260905.csv").exists())

    def test_targeted_converter_atomic_success_and_existing_is_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_source(root, date(2026, 9, 5))
            output = root / "data/bigmarch_takasaki_oyagi/machine_number"
            summary = root / "data/bigmarch_takasaki_oyagi/batch_logs"
            argv = ["converter", "--only-date", "2026-09-05", "--min-machines", "200"]
            with mock.patch.object(converter, "PROJECT_ROOT", root), \
                 mock.patch.object(converter, "OUTPUT_DIR", output), \
                 mock.patch.object(converter, "SUMMARY_DIR", summary), \
                 mock.patch.object(sys, "argv", argv):
                converter.main()
                daily = output / "ana_slo_bigmarch_oyagi_20260905.csv"
                before = (support.sha256_file(daily), daily.stat().st_mtime_ns)
                converter.main()
                after = (support.sha256_file(daily), daily.stat().st_mtime_ns)
            self.assertEqual(before, after)
            self.assertTrue(support.verify_big_march_daily_for_date(root, date(2026, 9, 5)).ok)
            self.assertFalse(list(output.glob("*.tmp")))

    def test_targeted_converter_missing_or_invalid_source_fails(self):
        for mode in ("missing", "invalid"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                if mode == "invalid":
                    write_source(root, date(2026, 9, 5), rows=199)
                with mock.patch.object(converter, "PROJECT_ROOT", root), \
                     mock.patch.object(converter, "OUTPUT_DIR", root / "data"), \
                     mock.patch.object(converter, "SUMMARY_DIR", root / "logs"), \
                     mock.patch.object(sys, "argv", ["converter", "--only-date", "2026-09-05"]):
                    with self.assertRaises((FileNotFoundError, RuntimeError)):
                        converter.main()

    def test_assessment_contiguous_only_and_no_date_jump(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_daily(root, date(2026, 9, 3))
            write_source(root, date(2026, 9, 5))
            result = support.assess_big_march_catchup(
                root, date(2026, 9, 7), date(2026, 9, 6)
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.candidate_dates, [])

    def test_assessment_returns_all_contiguous_sources_oldest_first(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_daily(root, date(2026, 9, 3))
            for day in (date(2026, 9, 4), date(2026, 9, 5), date(2026, 9, 6)):
                write_source(root, day)
            result = support.assess_big_march_catchup(
                root, date(2026, 9, 8), date(2026, 9, 7)
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.candidate_dates, ["2026-09-04", "2026-09-05", "2026-09-06"])

    def test_valid_prefix_converts_before_later_invalid_source_stops(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_daily(root, date(2026, 9, 3))
            write_source(root, date(2026, 9, 4))
            write_source(root, date(2026, 9, 5), rows=199)
            now = datetime(2026, 9, 7, 8, 5, tzinfo=JST)
            state = automation.create_state(date(2026, 9, 7), "run", now)

            def convert(command, *args, **kwargs):
                day = date.fromisoformat(command[command.index("--only-date") + 1])
                write_daily(root, day)
                return support.ProcessResult(0, now.isoformat(), now.isoformat(), 0.1, "log")

            with mock.patch.object(automation, "PROJECT_ROOT", root), \
                 mock.patch.object(automation, "RUNS_DIR", root / "logs"), \
                 mock.patch.object(automation, "run_logged_subprocess", side_effect=convert) as run, \
                 mock.patch.object(automation, "_record_history"):
                ok = automation._run_big_march_catchup(
                    state, root / "state.json", date(2026, 9, 7),
                    date(2026, 9, 6), now, lambda: now,
                )
            self.assertFalse(ok)
            self.assertEqual(run.call_count, 1)
            self.assertTrue(support.verify_big_march_daily_for_date(root, date(2026, 9, 4)).ok)
            self.assertFalse((root / "data/bigmarch_takasaki_oyagi/machine_number/ana_slo_bigmarch_oyagi_20260905.csv").exists())
            self.assertEqual(state["stores"][support.STORE_BIGMARCH]["status"], "NEEDS_MANUAL_REVIEW")

    def test_20260907_deadline_catches_up_before_provisional(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_daily(root, date(2026, 9, 4))
            write_source(root, date(2026, 9, 5))
            now = datetime(2026, 9, 7, 9, 31, tzinfo=JST)
            state = automation.create_state(date(2026, 9, 7), "run", now)
            args = SimpleNamespace(max_fetch_attempts=20, retry_interval_sec=300, chrome_wait_sec=15)

            def run_converter(command, *unused_args, **unused_kwargs):
                self.assertNotIn("--overwrite", command)
                self.assertNotIn("--allow-gap", command)
                self.assertEqual(command[command.index("--only-date") + 1], "2026-09-05")
                write_daily(root, date(2026, 9, 5))
                return support.ProcessResult(0, now.isoformat(), now.isoformat(), 0.1, "catchup.log")

            def provisional(*unused_args, **unused_kwargs):
                latest = support.discover_big_march_daily_dates(root)[-1][0]
                self.assertEqual(latest, date(2026, 9, 5))

            with mock.patch.object(automation, "PROJECT_ROOT", root), \
                 mock.patch.object(automation, "RUNS_DIR", root / "logs"), \
                 mock.patch.object(automation, "run_logged_subprocess", side_effect=run_converter), \
                 mock.patch.object(automation, "_run_big_march_provisional", side_effect=provisional) as provisional_run, \
                 mock.patch.object(automation, "_record_history") as history:
                automation._process_store(
                    support.STORE_BIGMARCH, state, root / "state.json",
                    date(2026, 9, 7), date(2026, 9, 6), args, lambda: now,
                )
            item = state["stores"][support.STORE_BIGMARCH]
            provisional_run.assert_called_once()
            self.assertEqual(item["pipeline_attempt_count"], 0)
            self.assertEqual(item["catchup_attempt_count"], 1)
            self.assertEqual(item["catchup_converted_dates"], ["2026-09-05"])
            self.assertEqual(history.call_args.args[2], "CATCHUP_CONVERT")
            self.assertFalse(automation.should_sleep_on_success(True, state, 0))

    def test_converter_zero_without_valid_daily_is_manual_review(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                write_daily(root, date(2026, 9, 4))
                write_source(root, date(2026, 9, 5))
                now = datetime(2026, 9, 7, 8, 5, tzinfo=JST)
                state = automation.create_state(date(2026, 9, 7), "run", now)

                def fake_run(*args, **kwargs):
                    if invalid:
                        write_daily(root, date(2026, 9, 5), duplicate=True)
                    return support.ProcessResult(0, now.isoformat(), now.isoformat(), 0.1, "log")

                with mock.patch.object(automation, "PROJECT_ROOT", root), \
                     mock.patch.object(automation, "RUNS_DIR", root / "logs"), \
                     mock.patch.object(automation, "run_logged_subprocess", side_effect=fake_run), \
                     mock.patch.object(automation, "_record_history"):
                    ok = automation._run_big_march_catchup(
                        state, root / "state.json", date(2026, 9, 7),
                        date(2026, 9, 6), now, lambda: now,
                    )
                self.assertFalse(ok)
                self.assertEqual(state["stores"][support.STORE_BIGMARCH]["status"], "NEEDS_MANUAL_REVIEW")

    def test_expected_source_ready_before_deadline_uses_formal_pipeline_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_daily(root, date(2026, 9, 5))
            write_source(root, date(2026, 9, 6))
            now = datetime(2026, 9, 7, 8, 30, tzinfo=JST)
            state = automation.create_state(date(2026, 9, 7), "run", now)
            state["stores"][support.STORE_BIGMARCH]["catchup_converted_dates"] = ["2026-09-05"]
            args = SimpleNamespace(max_fetch_attempts=20, retry_interval_sec=300, chrome_wait_sec=15)
            process = support.ProcessResult(0, now.isoformat(), now.isoformat(), 0.1, "pipeline.log")
            complete = support.VerificationResult("COMPLETE", True, artifacts=["formal"])
            with mock.patch.object(automation, "PROJECT_ROOT", root), \
                 mock.patch.object(automation, "RUNS_DIR", root / "logs"), \
                 mock.patch.object(automation, "run_logged_subprocess", return_value=process) as run, \
                 mock.patch.object(automation, "verify_store_completion", return_value=complete), \
                 mock.patch.object(automation, "_record_history"):
                automation._process_store(
                    support.STORE_BIGMARCH, state, root / "state.json",
                    date(2026, 9, 7), date(2026, 9, 6), args, lambda: now,
                )
            item = state["stores"][support.STORE_BIGMARCH]
            self.assertEqual(run.call_count, 1)
            self.assertEqual(item["pipeline_attempt_count"], 1)
            self.assertEqual(item["status"], "SUCCESS")

    def test_invalid_existing_daily_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = write_daily(root, date(2026, 9, 5), duplicate=True)
            before = (support.sha256_file(path), path.stat().st_mtime_ns)
            now = datetime(2026, 9, 7, 8, 5, tzinfo=JST)
            state = automation.create_state(date(2026, 9, 7), "run", now)
            with mock.patch.object(automation, "PROJECT_ROOT", root), \
                 mock.patch.object(automation, "run_logged_subprocess") as run, \
                 mock.patch.object(automation, "_record_history"):
                ok = automation._run_big_march_catchup(
                    state, root / "state.json", date(2026, 9, 7),
                    date(2026, 9, 6), now, lambda: now,
                )
            self.assertFalse(ok)
            run.assert_not_called()
            self.assertEqual(before, (support.sha256_file(path), path.stat().st_mtime_ns))

    def test_crash_recovery_keeps_catchup_pre_pipeline_retryable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            now = datetime(2026, 9, 7, 8, 10, tzinfo=JST)
            state = automation.create_state(date(2026, 9, 7), "run", now)
            item = state["stores"][support.STORE_BIGMARCH]
            item.update(status="RUNNING", current_stage="CATCHUP_CONVERT")
            with mock.patch.object(automation, "verify_big_march_provisional_completion", return_value=support.VerificationResult("NONE", False)), \
                 mock.patch.object(automation, "verify_store_completion", return_value=support.VerificationResult("NONE", False)), \
                 mock.patch.object(automation, "_has_big_or_yasuda_completion_artifact", return_value=False):
                automation.reconcile_startup_state(state, root, date(2026, 9, 7), now)
            self.assertEqual(item["status"], "FAILED_RETRYABLE")
            self.assertEqual(item["error_category"], "RECOVERED_CATCHUP_INTERRUPTION")

    def test_catchup_command_is_isolated_and_has_no_bypass(self):
        command = support.build_big_march_catchup_command(
            ROOT, "python", date(2026, 9, 5)
        )
        self.assertIn("--only-date", command)
        self.assertNotIn("--overwrite", command)
        self.assertNotIn("--allow-gap", command)
        self.assertNotIn("one_click", " ".join(command).lower())


if __name__ == "__main__":
    unittest.main()
