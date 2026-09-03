from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MACHINE_DIR = PROJECT_ROOT / "machine_number"
for path in (PROJECT_ROOT, MACHINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sleep_windows_after_delay as helper
from slotanalyzer_morning_automation_support import JST


class SleepWindowsAfterDelayTests(unittest.TestCase):
    def _clock(self, count: int = 2):
        started = datetime(2026, 9, 3, 8, 1, 0, tzinfo=JST)
        values = iter(started + timedelta(seconds=index) for index in range(count))
        return lambda: next(values)

    def test_cli_dry_run_and_delay(self):
        args = helper.parse_args(
            [
                "--parent-automation-run-id",
                " test_run ",
                "--delay-sec",
                "3",
                "--dry-run",
            ]
        )

        self.assertEqual(args.parent_automation_run_id, "test_run")
        self.assertEqual(args.delay_sec, 3)
        self.assertTrue(args.dry_run)

    def test_dry_run_delays_logs_and_never_requests_sleep(self):
        delays = []
        rows = []

        def append(path, row, fields):
            rows.append(dict(row))

        def fail_if_called():
            raise AssertionError("dry-run must not request sleep")

        returncode = helper.run_helper(
            "test_run",
            delay_sec=7,
            dry_run=True,
            sleep_function=delays.append,
            sleep_request=fail_if_called,
            clock=self._clock(1),
            history_path=Path("unused.csv"),
            history_appender=append,
            helper_id="helper_test",
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(delays, [7])
        self.assertEqual(
            [row["status"] for row in rows], ["STARTED", "DRY_RUN_COMPLETE"]
        )

    def test_requesting_is_flushed_to_history_before_sleep_api(self):
        events = []

        def append(path, row, fields):
            events.append(row["status"])

        def request_sleep():
            events.append("API")

        returncode = helper.run_helper(
            "test_run",
            delay_sec=0,
            sleep_function=lambda seconds: None,
            sleep_request=request_sleep,
            clock=self._clock(),
            history_path=Path("unused.csv"),
            history_appender=append,
            helper_id="helper_test",
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(
            events, ["STARTED", "REQUESTING", "API", "RETURNED_AFTER_RESUME"]
        )

    def test_sleep_api_failure_logs_failed_and_returns_nonzero(self):
        rows = []

        def append(path, row, fields):
            rows.append(dict(row))

        def fail_sleep():
            raise OSError("mock API failure")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            returncode = helper.run_helper(
                "test_run",
                delay_sec=0,
                sleep_function=lambda seconds: None,
                sleep_request=fail_sleep,
                clock=self._clock(),
                history_path=Path("unused.csv"),
                history_appender=append,
                helper_id="helper_test",
            )

        self.assertEqual(returncode, 1)
        self.assertEqual(
            [row["status"] for row in rows],
            ["STARTED", "REQUESTING", "FAILED"],
        )
        self.assertIn("mock API failure", rows[-1]["error"])
        self.assertIn("ERROR: Windows sleep request failed", stderr.getvalue())

    def test_history_failure_does_not_prevent_sleep_request(self):
        requests = []

        def fail_append(path, row, fields):
            raise OSError("mock history failure")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            returncode = helper.run_helper(
                "test_run",
                delay_sec=0,
                sleep_function=lambda seconds: None,
                sleep_request=lambda: requests.append(True),
                clock=self._clock(),
                history_path=Path("unused.csv"),
                history_appender=fail_append,
                helper_id="helper_test",
            )

        self.assertEqual(returncode, 0)
        self.assertEqual(requests, [True])
        self.assertIn("WARNING: sleep helper history write failed", stderr.getvalue())

    def test_dry_run_history_has_one_header_and_two_flushed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "sleep_helper_history.csv"
            returncode = helper.run_helper(
                "test_run",
                delay_sec=0,
                dry_run=True,
                sleep_function=lambda seconds: None,
                clock=self._clock(1),
                history_path=history_path,
                helper_id="helper_test",
            )
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(returncode, 0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [row["status"] for row in rows], ["STARTED", "DRY_RUN_COMPLETE"]
        )

    def test_set_suspend_state_requests_normal_sleep(self):
        calls = []

        def set_suspend_state(hibernate, force_critical, disable_wake_event):
            calls.append((hibernate, force_critical, disable_wake_event))
            return True

        helper.request_windows_sleep(set_suspend_state)

        self.assertEqual(calls, [(False, False, False)])

    def test_cli_dry_run_flag_defaults_off(self):
        args = helper.parse_args(
            ["--parent-automation-run-id", "test_run", "--delay-sec", "0"]
        )
        self.assertFalse(args.dry_run)

    def test_main_passes_dry_run_without_calling_real_api(self):
        with patch.object(helper, "run_helper", return_value=0) as run:
            returncode = helper.main(
                [
                    "--parent-automation-run-id",
                    "test_run",
                    "--delay-sec",
                    "0",
                    "--dry-run",
                ]
            )

        self.assertEqual(returncode, 0)
        run.assert_called_once_with("test_run", delay_sec=0, dry_run=True)


if __name__ == "__main__":
    unittest.main()
