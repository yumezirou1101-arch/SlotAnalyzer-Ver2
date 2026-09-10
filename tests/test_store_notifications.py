from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
from unittest import mock

from test_morning_notification import (
    FakeSMTP, JST, automation, notification, state_with, write_csv_rows, write_formal_69_fixture,
)
from slotanalyzer_notification_reference import SPECS, load_reference_result
from slotanalyzer_evaluation_quarantine import sha256_file


def reference_fixture(root, category="NORMAL", quarantined=True):
    target = date(2026, 9, 9)
    base = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep"
    folder, prefix, rank = SPECS[category]
    prediction = base / folder / f"{prefix}_20260909_top10.csv"
    metadata = base / folder / f"{prefix}_20260909_metadata.csv"
    write_csv_rows(prediction, [{"machine_no": i, "machine_name": f"M{i}", rank: i,
        "target_date": "2026-09-09", "score": 100-i} for i in range(1, 11)])
    write_csv_rows(metadata, [{"target_date": "2026-09-09",
        "generated_at_jst": "2026-09-09T08:01:00+09:00", "top10_sha256": sha256_file(prediction)}])
    source = base / "64_Ver4_2_future_top10"
    source.mkdir(exist_ok=True)
    all_path = source / "64_prediction_20260909_all514.csv"
    all_path.write_text("source evidence", encoding="utf-8")
    source_meta = source / "64_prediction_20260909_metadata.csv"
    if category != "NORMAL":
        source_meta.write_text("source metadata", encoding="utf-8")
    actual = base.parent / "ana_slo_20260909.csv"
    write_csv_rows(actual, [{"date": "2026-09-09", "machine_no": i,
        "machine_name": f"M{i}", "diff": 100 if i % 2 else -50} for i in range(1, 515)])
    entry = {"status": "ACTIVE", "store": "MARUHAN_MAEBASHI", "target_date": target.isoformat(),
        "category": category, "prediction_path": prediction.relative_to(root).as_posix(),
        "prediction_sha256": sha256_file(prediction), "metadata_path": metadata.relative_to(root).as_posix(),
        "metadata_sha256": sha256_file(metadata), "source_64_all514_sha256": sha256_file(all_path),
        "source_64_metadata_sha256": sha256_file(source_meta), "reason_code": "INVENTORY_GUARD_INCIDENT",
        "reason": "test incident", "incident_date": "2026-09-07", "created_at_jst": "2026-09-09T18:30:00+09:00"}
    registry = root / "config/evaluation_quarantine.json"
    registry.parent.mkdir(exist_ok=True)
    registry.write_text(json.dumps({"schema_version": 1, "entries": [entry] if quarantined else []}), encoding="utf-8")
    return prediction, metadata, actual


class StoreNotificationTests(unittest.TestCase):
    def test_main_sends_maruhan_before_bigmarch_wait_and_never_sleeps_early(self):
        state = state_with(["WAITING_FOR_DATA"] * 3)
        events = []
        passes = []
        def process(store, *args):
            events.append("process:" + store)
            status = {"maruhan": "SUCCESS", "yasuda": "NEEDS_MANUAL_REVIEW"}.get(store)
            if store == "bigmarch":
                passes.append(1)
                status = "WAITING_FOR_DATA" if len(passes) == 1 else "PROVISIONAL"
            state["stores"][store]["status"] = status
        with ExitStack() as stack:
            for name, value in {
                "parse_args": mock.Mock(return_value=mock.Mock(retry_interval_sec=300, sleep_on_success=True)),
                "now_jst": mock.Mock(return_value=datetime(2026, 9, 4, 8, 1, tzinfo=JST)),
                "WindowsFileLock": mock.MagicMock(), "load_json_state": mock.Mock(return_value=state),
                "reconcile_startup_state": mock.Mock(), "save_state": mock.Mock(),
                "print_summary": mock.Mock(), "_process_store": mock.Mock(side_effect=process),
                "launch_sleep_helper": mock.Mock(side_effect=AssertionError("early sleep")),
            }.items():
                stack.enter_context(mock.patch.object(automation, name, value))
            stack.enter_context(mock.patch.object(automation.time, "sleep", side_effect=lambda _: events.append("wait")))
            sender = stack.enter_context(mock.patch.object(notification, "send_notification_best_effort",
                side_effect=lambda state, root, store: events.append("mail:" + store)))
            self.assertEqual(automation.main(), 1)
        self.assertLess(events.index("mail:maruhan"), events.index("process:bigmarch"))
        self.assertLess(events.index("mail:yasuda"), events.index("wait"))
        self.assertLess(events.index("wait"), events.index("mail:bigmarch"))
        self.assertEqual(sender.call_count, 3)

    def test_single_store_subjects_content_and_maruhan_order(self):
        state = state_with(["SUCCESS", "PROVISIONAL", "NEEDS_MANUAL_REVIEW"])
        state["stores"]["yasuda"]["error_category"] = "INVENTORY_GUARD_BLOCKED"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_formal_69_fixture(root, date(2026, 9, 3))
            for store, expected in [("maruhan", "SUCCESS"), ("bigmarch", "PROVISIONAL"), ("yasuda", "MANUAL_REVIEW")]:
                message = notification.build_store_notification_message(state, root, store)
                self.assertIn(f"[{store.upper()}][{expected}]", message.subject)
                if store == "maruhan":
                    self.assertLess(message.plain.index("昨日の予測結果"), message.plain.index("Forward:"))
                    self.assertLess(message.plain.index("NORMAL Top10"), message.plain.index("A-TYPE Top10"))
                    self.assertLess(message.plain.index("A-TYPE Top10"), message.plain.index("JUGGLER Top10"))
                    self.assertNotIn("Big March", message.plain)
                elif store == "yasuda":
                    self.assertIn("NEEDS_MANUAL_REVIEW", message.plain)
                    self.assertIn("INVENTORY_GUARD_BLOCKED", message.plain)
                    self.assertNotIn("昨日の予測結果", message.plain)

    def test_history_compatibility_per_store_restart_and_old_aggregate(self):
        FakeSMTP.calls = []
        state = state_with(["SUCCESS"] * 3)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            history = root / "history.csv"
            send = lambda store: notification.send_notification_best_effort(state, root, history, store=store,
                credential_reader=lambda: notification.Credential("sender@example.com", "secret"),
                smtp_factory=FakeSMTP, recipient="reader@example.com")
            for _ in range(2):
                for store in automation.STORE_ORDER:
                    self.assertTrue(send(store))
            rows = notification._read_rows(history)
            self.assertEqual(len(rows), 3)
            self.assertEqual(set(rows[0]), set(notification.HISTORY_FIELDS))
            self.assertEqual({r["notification_type"] for r in rows},
                             {f"MORNING_RESULT:{s.upper()}" for s in automation.STORE_ORDER})
            rows[0]["notification_type"] = "MORNING_RESULT"
            write_csv_rows(history, [rows[0]])
            for store in automation.STORE_ORDER:
                self.assertTrue(send(store))
            self.assertEqual(len(notification._read_rows(history)), 1)

    def test_failure_isolated_no_repeated_attempt_or_state_mutation(self):
        state = state_with(["SUCCESS"] * 3)
        original = copy.deepcopy(state)
        sender = mock.Mock(side_effect=RuntimeError("offline"))
        attempts = set()
        for _ in range(2):
            notification.notify_terminal_stores_best_effort(state, Path("unused"), attempts, sender=sender)
        self.assertEqual(sender.call_count, 3)
        self.assertEqual(state, original)
        self.assertTrue(automation.should_sleep_on_success(True, state, 0))

    def test_build_failure_recorded_and_later_restart_can_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with mock.patch.object(notification, "build_notification_message", side_effect=ValueError("bad CSV")):
                self.assertFalse(notification.send_notification_best_effort(state_with(["SUCCESS"]*3), root, store="maruhan"))
            rows = notification._read_rows(root / "logs/morning_automation/notification_history.csv")
            self.assertEqual(rows[0]["status"], "FAILED")
            self.assertEqual(rows[0]["notification_type"], "MORNING_RESULT:MARUHAN")
            self.assertFalse(notification._already_sent(root / "logs/morning_automation/notification_history.csv",
                state_with(["SUCCESS"]*3), "MORNING_RESULT:MARUHAN"))

    def test_quarantined_reference_all_categories_read_only(self):
        for category in SPECS:
            with self.subTest(category=category), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                reference_fixture(root, category)
                before = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
                if category == "NORMAL":
                    result = notification._load_yesterday_normal_evaluation({}, root, date(2026, 9, 10))
                    rendered = notification._render_yesterday_plain(result)
                else:
                    result = notification._load_yesterday_derived_evaluation(root, date(2026, 9, 10), category)
                    rendered = notification._render_yesterday_derived_plain(result)
                self.assertEqual(result.status, "REFERENCE / QUARANTINED")
                self.assertFalse(result.formal)
                self.assertIn("+250枚", rendered)
                self.assertIn("正式Forward成績には加算しない", rendered)
                self.assertEqual(result.summary_rows[-1]["win_rate"], "50.0")
                self.assertEqual(before, {p: sha256_file(p) for p in root.rglob("*") if p.is_file()})

    def test_reference_unavailable_integrity_date_and_name_cases(self):
        for failure in ("sha", "actual_date", "duplicate", "late", "no_prediction", "name"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                prediction, metadata, actual = reference_fixture(root, quarantined=failure == "sha")
                if failure == "sha":
                    prediction.write_text(prediction.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")
                elif failure == "late":
                    metadata.write_text(metadata.read_text(encoding="utf-8-sig").replace("08:01", "18:01"), encoding="utf-8")
                elif failure == "no_prediction":
                    prediction.rename(prediction.with_name("64_prediction_20260908_top10.csv"))
                else:
                    rows = notification._read_rows(actual)
                    if failure == "actual_date": rows[0]["date"] = "2026-09-08"
                    if failure == "duplicate": rows[0]["machine_no"] = "2"
                    if failure == "name": rows[0]["machine_name"] = "replacement"
                    write_csv_rows(actual, rows)
                status, message, details, summary = load_reference_result(root, date(2026, 9, 9), "NORMAL", "FORWARD_GUARD_FAIL")
                if failure == "name":
                    self.assertEqual(status, "REFERENCE")
                    self.assertEqual(len(details), 9)
                    self.assertIn("未照合台: 1", message)
                else:
                    self.assertEqual(status, "REFERENCE_UNAVAILABLE")
                    self.assertFalse(details)

    def test_legacy_and_guard_fail_show_reference_without_promotion(self):
        for classification in ("LEGACY_UNVERIFIED", "FORWARD_GUARD_FAIL"):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                reference_fixture(root, quarantined=False)
                path = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep/69_Ver4_2_live_prediction_backtest/69_live_prediction_status.csv"
                write_csv_rows(path, [{"target_date": "2026-09-09", "status": "SKIPPED_" + classification,
                                       "prediction_class": classification}])
                before = path.read_bytes()
                result = notification._load_yesterday_normal_evaluation({}, root, date(2026, 9, 10))
                self.assertEqual(result.status, "REFERENCE")
                self.assertFalse(result.formal)
                self.assertEqual(path.read_bytes(), before)

    def test_quarantine_overrides_stale_formal_display_without_rewriting_69(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prediction, _, _ = reference_fixture(root)
            write_formal_69_fixture(root, date(2026, 9, 9), sha=sha256_file(prediction))
            base = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep/69_Ver4_2_live_prediction_backtest"
            before = {p: p.read_bytes() for p in base.iterdir()}
            result = notification._load_yesterday_normal_evaluation({}, root, date(2026, 9, 10))
            self.assertEqual(result.status, "REFERENCE / QUARANTINED")
            self.assertFalse(result.formal)
            self.assertEqual(before, {p: p.read_bytes() for p in base.iterdir()})

    def test_after_formal_cutoff_morning_prediction_is_reference_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, metadata, _ = reference_fixture(root, quarantined=False)
            metadata.write_text(metadata.read_text(encoding="utf-8-sig").replace("08:01", "09:15"), encoding="utf-8")
            status, message, details, _ = load_reference_result(root, date(2026, 9, 9), "NORMAL", "FORWARD_GUARD_FAIL")
            self.assertEqual(status, "REFERENCE")
            self.assertEqual(len(details), 10)
            self.assertIn("正式Forward成績には加算しない", message)

    def test_interruption_does_not_finalize_or_sleep(self):
        with ExitStack() as stack:
            state = state_with(["WAITING_FOR_DATA"]*3)
            for name, value in {
                "parse_args": mock.Mock(return_value=mock.Mock(retry_interval_sec=300, sleep_on_success=True)),
                "now_jst": mock.Mock(return_value=datetime(2026, 9, 4, 8, 1, tzinfo=JST)),
                "WindowsFileLock": mock.MagicMock(), "load_json_state": mock.Mock(return_value=state),
                "reconcile_startup_state": mock.Mock(), "save_state": mock.Mock(),
                "_process_store": mock.Mock(side_effect=KeyboardInterrupt),
            }.items():
                stack.enter_context(mock.patch.object(automation, name, value))
            finalize = stack.enter_context(mock.patch.object(automation, "finalize_automation_run"))
            with self.assertRaises(KeyboardInterrupt):
                automation.main()
            finalize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
