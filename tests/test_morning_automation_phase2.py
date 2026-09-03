from __future__ import annotations

import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import patch


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
    verify_big_march_completion,
    verify_maruhan_completion,
    verify_yasuda_completion,
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


def write_big_march_daily(root: Path, operation: date, rows: int = 200) -> Path:
    expected = operation - timedelta(days=1)
    path = (
        root
        / "data/bigmarch_takasaki_oyagi/machine_number"
        / f"ana_slo_bigmarch_oyagi_{expected:%Y%m%d}.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "machine_name", "machine_no", "G", "diff"],
        )
        writer.writeheader()
        for number in range(1, rows + 1):
            writer.writerow(
                {
                    "date": expected.isoformat(),
                    "machine_name": f"Machine {number}",
                    "machine_no": number,
                    "G": 1000,
                    "diff": number,
                }
            )
    return path


def write_big_march_target_artifacts(root: Path, operation: date) -> list[Path]:
    expected = operation - timedelta(days=1)
    ymd = operation.strftime("%Y%m%d")
    analysis = (
        root
        / "data/bigmarch_takasaki_oyagi/machine_number/analysis_31days_deep"
    )
    paths = [
        analysis / "08_juggler_recent7_top3_forward/08_forward_status.csv",
        analysis
        / "11_nonjuggler_weekday_top1_forward"
        / "11_nonjuggler_weekday_top1_forward_summary.csv",
        analysis
        / "09_juggler_recent7_future_ranking"
        / f"09_prediction_{ymd}_all_juggler.csv",
        analysis
        / "09_juggler_recent7_future_ranking"
        / f"09_prediction_{ymd}_top10.csv",
        analysis
        / "09_juggler_recent7_future_ranking"
        / f"09_prediction_{ymd}_metadata.csv",
        analysis
        / "12_nonjuggler_weekday_future_ranking"
        / f"12_prediction_{ymd}_all_nonjuggler.csv",
        analysis
        / "12_nonjuggler_weekday_future_ranking"
        / f"12_prediction_{ymd}_top10.csv",
        analysis
        / "12_nonjuggler_weekday_future_ranking"
        / f"12_prediction_{ymd}_metadata.csv",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.endswith("_metadata.csv"):
            path.write_text(
                "target_date,latest_data_date\n"
                f"{operation.isoformat()},{expected.isoformat()}\n",
                encoding="utf-8-sig",
            )
        else:
            path.write_text("status\nOK\n", encoding="utf-8-sig")
    return paths


def write_yasuda_daily(root: Path, operation: date) -> Path:
    expected = operation - timedelta(days=1)
    path = (
        root
        / "data/yasuda_maebashi/machine_number"
        / f"ana_slo_{expected:%Y%m%d}.csv"
    )
    columns = [
        "日付",
        "台番号",
        "機種名",
        "G数",
        "差枚",
        "BB",
        "RB",
        "合成確率",
        "BB確率",
        "RB確率",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for number in range(1, 321):
            writer.writerow(
                {
                    "日付": expected.isoformat(),
                    "台番号": number,
                    "機種名": f"Machine {number}",
                    "G数": 1000,
                    "差枚": number,
                    "BB": 1,
                    "RB": 1,
                    "合成確率": "1/100",
                    "BB確率": "1/200",
                    "RB確率": "1/200",
                }
            )
    return path


class Phase2SupportTests(unittest.TestCase):
    def _state_with_statuses(self, statuses: list[str]) -> dict:
        current = datetime(2026, 9, 3, 8, 0, tzinfo=JST)
        state = automation.create_state(date(2026, 9, 3), "test_run", current)
        for store, status in zip(automation.STORE_ORDER, statuses):
            state["stores"][store]["status"] = status
        return state

    def test_sleep_option_disabled_does_not_request_sleep(self):
        state = self._state_with_statuses(["SUCCESS", "SUCCESS", "SUCCESS"])
        calls = []

        result = automation.maybe_sleep_on_success(
            False, state, 0, lambda run_id: calls.append(run_id)
        )

        self.assertFalse(result)
        self.assertEqual(calls, [])

    def test_sleep_cli_option_defaults_off_and_can_be_enabled(self):
        with patch.object(sys, "argv", ["automation"]):
            self.assertFalse(automation.parse_args().sleep_on_success)
        with patch.object(sys, "argv", ["automation", "--sleep-on-success"]):
            self.assertTrue(automation.parse_args().sleep_on_success)

    def test_sleep_option_all_success_requests_sleep_once(self):
        state = self._state_with_statuses(["SUCCESS", "SUCCESS", "SUCCESS"])
        calls = []

        result = automation.maybe_sleep_on_success(
            True, state, 0, lambda run_id: calls.append(run_id) or 1234
        )

        self.assertTrue(result)
        self.assertEqual(calls, ["test_run"])

    def test_sleep_option_accepts_success_and_already_complete(self):
        state = self._state_with_statuses(
            ["SUCCESS", "ALREADY_COMPLETE", "SUCCESS"]
        )
        calls = []

        result = automation.maybe_sleep_on_success(
            True, state, 0, lambda run_id: calls.append(run_id) or 1234
        )

        self.assertTrue(result)
        self.assertEqual(calls, ["test_run"])

    def test_sleep_option_failed_final_does_not_request_sleep(self):
        state = self._state_with_statuses(["SUCCESS", "FAILED_FINAL", "SUCCESS"])
        calls = []

        result = automation.maybe_sleep_on_success(
            True, state, 1, lambda run_id: calls.append(run_id)
        )

        self.assertFalse(result)
        self.assertEqual(calls, [])

    def test_sleep_option_manual_review_does_not_request_sleep(self):
        state = self._state_with_statuses(
            ["SUCCESS", "NEEDS_MANUAL_REVIEW", "SUCCESS"]
        )
        calls = []

        result = automation.maybe_sleep_on_success(
            True, state, 1, lambda run_id: calls.append(run_id)
        )

        self.assertFalse(result)
        self.assertEqual(calls, [])

    def test_sleep_option_nonzero_wrapper_return_does_not_request_sleep(self):
        state = self._state_with_statuses(["SUCCESS", "SUCCESS", "SUCCESS"])
        calls = []

        result = automation.maybe_sleep_on_success(
            True, state, 1, lambda run_id: calls.append(run_id)
        )

        self.assertFalse(result)
        self.assertEqual(calls, [])

    def test_sleep_helper_launch_failure_preserves_success_and_returncode(self):
        state = self._state_with_statuses(["SUCCESS", "SUCCESS", "SUCCESS"])
        original_statuses = [
            state["stores"][store]["status"] for store in automation.STORE_ORDER
        ]

        def fail_launch(run_id):
            raise OSError(f"mock helper launch failure: {run_id}")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = automation.maybe_sleep_on_success(True, state, 0, fail_launch)

        self.assertFalse(result)
        self.assertIn(
            "WARNING: Windows sleep helper launch failed", stderr.getvalue()
        )
        self.assertEqual(
            [state["stores"][store]["status"] for store in automation.STORE_ORDER],
            original_statuses,
        )
        self.assertTrue(automation.should_sleep_on_success(True, state, 0))

    def test_sleep_option_missing_store_does_not_launch_helper(self):
        state = self._state_with_statuses(["SUCCESS", "SUCCESS", "SUCCESS"])
        del state["stores"][automation.STORE_YASUDA]
        calls = []

        result = automation.maybe_sleep_on_success(
            True, state, 0, lambda run_id: calls.append(run_id)
        )

        self.assertFalse(result)
        self.assertEqual(calls, [])

    def test_sleep_helper_popen_arguments_are_detached_and_not_waited(self):
        calls = []

        class Process:
            pid = 4321

            def wait(self):
                raise AssertionError("wrapper must not wait for the helper")

            def communicate(self):
                raise AssertionError("wrapper must not communicate with the helper")

        def fake_popen(command, **kwargs):
            calls.append((command, kwargs))
            return Process()

        pid = automation.launch_sleep_helper("test_run", 10, fake_popen)

        self.assertEqual(pid, 4321)
        self.assertEqual(len(calls), 1)
        command, kwargs = calls[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(automation.SLEEP_HELPER_PATH.resolve()))
        self.assertEqual(
            command[2:],
            ["--parent-automation-run-id", "test_run", "--delay-sec", "10"],
        )
        self.assertEqual(kwargs["cwd"], str(automation.PROJECT_ROOT))
        self.assertFalse(kwargs["shell"])
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(
            kwargs["creationflags"],
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    def test_sleep_helper_dry_run_is_explicitly_forwarded(self):
        calls = []

        class Process:
            pid = 4321

        def fake_popen(command, **kwargs):
            calls.append(command)
            return Process()

        automation.launch_sleep_helper(
            "detached_dry_run", 10, fake_popen, dry_run=True
        )

        self.assertEqual(calls[0][-1], "--dry-run")

    def test_finalization_order_is_save_summary_flush_then_helper_launch(self):
        state = self._state_with_statuses(["SUCCESS", "SUCCESS", "SUCCESS"])
        events = []

        def save_function(path, value, current):
            events.append("save")

        def summary_function(value):
            events.append("summary")

        def flush_function():
            events.append("flush")

        def helper_launcher(run_id):
            events.append("launch")
            return 4321

        returncode = automation.finalize_automation_run(
            Path("unused.json"),
            state,
            True,
            save_function=save_function,
            summary_function=summary_function,
            flush_function=flush_function,
            helper_launcher=helper_launcher,
            clock=lambda: datetime(2026, 9, 3, 8, 1, tzinfo=JST),
        )

        self.assertEqual(returncode, 0)
        self.assertEqual(events, ["save", "summary", "flush", "launch"])

    def test_finalization_keeps_zero_when_helper_launch_fails(self):
        state = self._state_with_statuses(["SUCCESS", "SUCCESS", "SUCCESS"])

        def fail_launch(run_id):
            raise OSError(f"mock helper launch failure: {run_id}")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            returncode = automation.finalize_automation_run(
                Path("unused.json"),
                state,
                True,
                save_function=lambda path, value, current: None,
                summary_function=lambda value: None,
                flush_function=lambda: None,
                helper_launcher=fail_launch,
                clock=lambda: datetime(2026, 9, 3, 8, 1, tzinfo=JST),
            )

        self.assertEqual(returncode, 0)
        self.assertIn(
            "WARNING: Windows sleep helper launch failed", stderr.getvalue()
        )
        self.assertEqual(
            [state["stores"][store]["status"] for store in automation.STORE_ORDER],
            ["SUCCESS", "SUCCESS", "SUCCESS"],
        )

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

    def test_big_march_daily_only_does_not_block_startup_recovery(self):
        operation = date(2026, 9, 2)
        current = datetime(2026, 9, 2, 8, 0, tzinfo=JST)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_big_march_daily(root, operation)
            verification = verify_big_march_completion(root, operation)
            self.assertEqual(verification.status, "PARTIAL")

            state = automation.create_state(operation, "automation_test", current)
            automation.reconcile_startup_state(state, root, operation, current)

            item = state["stores"][STORE_BIGMARCH]
            self.assertEqual(item["status"], "PENDING")
            self.assertNotEqual(
                item["error_category"], "PREEXISTING_COMPLETION_ARTIFACT"
            )

    def test_big_march_partial_target_artifact_requires_manual_review(self):
        operation = date(2026, 9, 2)
        current = datetime(2026, 9, 2, 8, 0, tzinfo=JST)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = (
                root
                / "data/bigmarch_takasaki_oyagi/machine_number"
                / "analysis_31days_deep/09_juggler_recent7_future_ranking"
                / "09_prediction_20260902_top10.csv"
            )
            target.parent.mkdir(parents=True)
            target.write_text("rank\n1\n", encoding="utf-8-sig")

            state = automation.create_state(operation, "automation_test", current)
            automation.reconcile_startup_state(state, root, operation, current)

            item = state["stores"][STORE_BIGMARCH]
            self.assertEqual(item["status"], "NEEDS_MANUAL_REVIEW")
            self.assertEqual(
                item["error_category"], "PREEXISTING_COMPLETION_ARTIFACT"
            )

    def test_big_march_complete_artifacts_without_state_stay_manual(self):
        operation = date(2026, 9, 2)
        current = datetime(2026, 9, 2, 8, 0, tzinfo=JST)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_big_march_daily(root, operation)
            write_big_march_target_artifacts(root, operation)
            self.assertTrue(verify_big_march_completion(root, operation).ok)

            state = automation.create_state(operation, "automation_test", current)
            automation.reconcile_startup_state(state, root, operation, current)

            item = state["stores"][STORE_BIGMARCH]
            self.assertEqual(item["status"], "NEEDS_MANUAL_REVIEW")
            self.assertEqual(
                item["error_category"], "PREEXISTING_COMPLETION_ARTIFACT"
            )

    def test_yasuda_daily_remains_complete_and_startup_stays_manual(self):
        operation = date(2026, 9, 2)
        current = datetime(2026, 9, 2, 8, 0, tzinfo=JST)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_yasuda_daily(root, operation)
            self.assertTrue(verify_yasuda_completion(root, operation).ok)

            state = automation.create_state(operation, "automation_test", current)
            automation.reconcile_startup_state(state, root, operation, current)

            item = state["stores"][STORE_YASUDA]
            self.assertEqual(item["status"], "NEEDS_MANUAL_REVIEW")
            self.assertEqual(
                item["error_category"], "PREEXISTING_COMPLETION_ARTIFACT"
            )

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
