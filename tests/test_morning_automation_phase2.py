from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MACHINE_DIR = PROJECT_ROOT / "machine_number"
for path in (PROJECT_ROOT, MACHINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_slotanalyzer_morning_automation as automation
from slotanalyzer_morning_automation_support import (
    JST,
    LockUnavailableError,
    StateCorruptError,
    WindowsFileLock,
    append_history_csv,
    atomic_write_json,
    build_fetch_command,
    build_pipeline_command,
    check_source_readiness,
    classify_fetch_result,
    classify_pipeline_result,
    determine_operation_dates,
    ensure_cdp,
    load_json_state,
    maruhan_pipeline_start_allowed,
    other_store_retry_allowed,
    run_logged_subprocess,
    verify_maruhan_completion,
    VerificationResult,
    STORE_BIGMARCH,
    STORE_MARUHAN,
    STORE_YASUDA,
)


def make_html(day: date, store: str, rows: int) -> str:
    body_rows = []
    for number in range(1, rows + 1):
        body_rows.append(
            f"<tr><td>Machine {number}</td><td>{number}</td><td>1000</td><td>{number}</td></tr>"
        )
    return (
        "<html><head><title>"
        f"{store} {day:%Y/%m/%d}"
        "</title></head><body>"
        f"<h1>{store} {day:%Y/%m/%d}</h1>"
        "<table><tr><th>機種名</th><th>台番号</th><th>G数</th><th>差枚</th></tr>"
        + "".join(body_rows)
        + "</table></body></html>"
    )


class Phase2SupportTests(unittest.TestCase):
    def test_operation_date_is_fixed_and_expected_is_previous_day(self):
        current = datetime(2026, 9, 2, 8, 0, tzinfo=JST)
        operation, expected = determine_operation_dates(current)
        self.assertEqual(operation, date(2026, 9, 2))
        self.assertEqual(expected, date(2026, 9, 1))
        later = datetime(2026, 9, 3, 0, 1, tzinfo=JST)
        self.assertEqual(operation, date(2026, 9, 2))
        self.assertNotEqual(operation, determine_operation_dates(later)[0])

    def test_automation_run_id_keeps_existing_morning_format(self):
        value = automation.new_automation_run_id(
            datetime(2026, 9, 2, 8, 0, tzinfo=JST)
        )
        self.assertRegex(
            value,
            r"^morning_\d{8}T\d{12}\+0900_[0-9a-f]{8}$",
        )

    def test_deadlines(self):
        operation = date(2026, 9, 2)
        self.assertTrue(
            maruhan_pipeline_start_allowed(datetime(2026, 9, 2, 8, 29, tzinfo=JST), operation)
        )
        self.assertFalse(
            maruhan_pipeline_start_allowed(datetime(2026, 9, 2, 8, 30, tzinfo=JST), operation)
        )
        self.assertFalse(
            maruhan_pipeline_start_allowed(datetime(2026, 9, 2, 8, 59, tzinfo=JST), operation)
        )
        self.assertFalse(
            maruhan_pipeline_start_allowed(datetime(2026, 9, 2, 9, 0, tzinfo=JST), operation)
        )
        self.assertTrue(
            other_store_retry_allowed(datetime(2026, 9, 2, 9, 29, tzinfo=JST), operation)
        )
        self.assertFalse(
            other_store_retry_allowed(datetime(2026, 9, 2, 9, 30, tzinfo=JST), operation)
        )

    def test_atomic_state_save_load_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = {"stores": {}, "operation_date": "2026-09-02"}
            atomic_write_json(path, state)
            self.assertEqual(load_json_state(path), state)
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaises(StateCorruptError):
                load_json_state(path)

    def test_history_append_has_one_header_and_two_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            fields = ["run_id", "status"]
            append_history_csv(path, {"run_id": "one", "status": "OK"}, fields)
            append_history_csv(path, {"run_id": "two", "status": "OK"}, fields)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines.count("run_id,status"), 1)
            self.assertEqual(len(lines), 3)

    def test_subprocess_log_records_command_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "safe.log"
            result = run_logged_subprocess(
                [sys.executable, "-c", "print('safe child')"],
                Path(directory),
                log_path,
                "UNIT_TEST",
            )
            self.assertEqual(result.returncode, 0)
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("stage=UNIT_TEST", text)
            self.assertIn("safe child", text)
            self.assertIn("returncode=0", text)

    def test_global_lock_rejects_second_process(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "global.lock"
            code = (
                "import sys;sys.path.insert(0,sys.argv[2]);"
                "from slotanalyzer_morning_automation_support import WindowsFileLock,LockUnavailableError;"
                "\ntry:\n"
                "  with WindowsFileLock(__import__('pathlib').Path(sys.argv[1])): print('ACQUIRED')\n"
                "except LockUnavailableError: print('REJECTED')\n"
            )
            with WindowsFileLock(lock_path):
                result = subprocess.run(
                    [sys.executable, "-c", code, str(lock_path), str(MACHINE_DIR)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("REJECTED", result.stdout)
            with WindowsFileLock(lock_path):
                pass

    def test_cdp_preflight_is_idempotent_when_probe_is_ready(self):
        expected = {"webSocketDebuggerUrl": "ws://127.0.0.1/test"}

        def fail_if_started(*args, **kwargs):
            self.fail("Chrome must not start when CDP is already ready.")

        result = ensure_cdp(
            PROJECT_ROOT,
            probe=lambda timeout: expected,
            popen=fail_if_started,
        )
        self.assertEqual(result, expected)

    def test_readiness_fixtures(self):
        expected = date(2026, 9, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                (STORE_MARUHAN, root / "ana_slo_20260901_source.html", "マルハンメガシティ前橋インター", 450),
                (STORE_BIGMARCH, root / "ana_slo_bigmarch_oyagi_20260901_source.html", "ビッグマーチ高崎おおやぎ店", 200),
                (STORE_YASUDA, root / "data/yasuda_maebashi/source_html/ana_slo_20260901_source.html", "やすだ前橋店", 300),
            ]
            for store, path, store_name, rows in cases:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(make_html(expected, store_name, rows), encoding="utf-8")
                result = check_source_readiness(store, root, expected)
                self.assertTrue(result.ready, result.error)
                self.assertEqual(result.details["records"], rows)

    def test_missing_expected_source_after_successful_fetch_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            readiness = check_source_readiness(
                STORE_MARUHAN, Path(directory), date(2026, 9, 1)
            )
            self.assertEqual(classify_fetch_result(0, readiness, True), "WAITING_FOR_DATA")
            self.assertEqual(classify_fetch_result(1, readiness, True), "FAILED_RETRYABLE")
            self.assertEqual(classify_fetch_result(1, readiness, False), "FAILED_FINAL")

    def test_pipeline_failure_is_not_retryable(self):
        verification = VerificationResult("NONE", False)
        self.assertEqual(classify_pipeline_result(1, verification), "FAILED_FINAL")

    def test_commands_have_no_forbidden_arguments_and_targets_are_correct(self):
        operation = date(2026, 9, 2)
        for store in (STORE_MARUHAN, STORE_BIGMARCH, STORE_YASUDA):
            fetch = build_fetch_command(store, PROJECT_ROOT, "python.exe")
            pipeline = build_pipeline_command(store, PROJECT_ROOT, "python.exe", operation)
            for command in (fetch, pipeline):
                self.assertNotIn("--allow-gap", command)
                self.assertNotIn("--overwrite", command)
        maruhan = build_pipeline_command(STORE_MARUHAN, PROJECT_ROOT, "python.exe", operation)
        yasuda = build_pipeline_command(STORE_YASUDA, PROJECT_ROOT, "python.exe", operation)
        bigmarch = build_pipeline_command(STORE_BIGMARCH, PROJECT_ROOT, "python.exe", operation)
        self.assertEqual(maruhan[maruhan.index("--target-date") + 1], "2026-09-02")
        self.assertEqual(yasuda[yasuda.index("--target-date") + 1], "2026-09-02")
        self.assertNotIn("--target-date", bigmarch)

    def test_terminal_state_skip_predicate(self):
        state = automation.create_state(
            date(2026, 9, 2),
            "automation_test",
            datetime(2026, 9, 2, 8, 0, tzinfo=JST),
        )
        for item in state["stores"].values():
            item["status"] = "SUCCESS"
        self.assertTrue(automation.all_terminal(state))

    def test_maruhan_partial_64_requires_manual_review(self):
        operation = date(2026, 9, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dir64 = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep/64_Ver4_2_future_top10"
            dir64.mkdir(parents=True)
            (dir64 / "64_prediction_20260902_all514.csv").write_text("a\n1\n", encoding="utf-8")
            result = verify_maruhan_completion(root, operation)
            self.assertEqual(result.status, "PARTIAL_64")

    def test_maruhan_downstream_without_64_requires_manual_review(self):
        operation = date(2026, 9, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dir77 = (
                root
                / "data/maruhan_maebashi/machine_number/analysis_31days_deep"
                / "77_live_integrated_prediction_report"
            )
            dir77.mkdir(parents=True)
            (dir77 / "77_integrated_prediction_20260902.csv").write_text(
                "a\n1\n", encoding="utf-8"
            )
            result = verify_maruhan_completion(root, operation)
            self.assertEqual(result.status, "INCONSISTENT_PIPELINE_ARTIFACTS")

    def test_maruhan_complete_64_without_77_requires_manual_review(self):
        operation = date(2026, 9, 2)
        expected = date(2026, 9, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data/maruhan_maebashi/machine_number"
            dir64 = data_dir / "analysis_31days_deep/64_Ver4_2_future_top10"
            dir64.mkdir(parents=True)
            daily = data_dir / "ana_slo_20260901.csv"
            source = root / "ana_slo_20260901_source.html"
            daily.parent.mkdir(parents=True, exist_ok=True)
            daily.write_text("date\n2026-09-01\n", encoding="utf-8")
            source.write_text("source", encoding="utf-8")
            all514 = dir64 / "64_prediction_20260902_all514.csv"
            top10 = dir64 / "64_prediction_20260902_top10.csv"
            metadata = dir64 / "64_prediction_20260902_metadata.csv"
            all514.write_text("rank\n1\n", encoding="utf-8")
            top10.write_text("rank\n1\n", encoding="utf-8")
            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()
            row = {
                "generated_at_jst": "2026-09-02T08:00:00+09:00",
                "target_date": operation.isoformat(),
                "latest_data_date": expected.isoformat(),
                "forward_guard_version": "test",
                "forward_valid": True,
                "forward_cutoff_jst": "09:00 Asia/Tokyo",
                "target_actual_absent_at_generation": True,
                "target_source_absent_at_generation": True,
                "daily_csv_sha256": digest(daily),
                "source_html_sha256": digest(source),
                "all514_sha256": digest(all514),
                "top10_sha256": digest(top10),
                "machines_ranked": 514,
                "target_actual_used": False,
                "model": "CHAMPION_V4.2_C",
                "weight_fingerprint": "test",
                "weight_sum": 1.0,
            }
            with metadata.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            result = verify_maruhan_completion(root, operation)
            self.assertEqual(result.status, "COMPLETE_64_INCOMPLETE_PIPELINE")


if __name__ == "__main__":
    unittest.main()
