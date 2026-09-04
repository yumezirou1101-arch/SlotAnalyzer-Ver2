from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import date, datetime
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"
for value in (str(ROOT), str(MACHINE)):
    if value not in sys.path:
        sys.path.insert(0, value)

import run_slotanalyzer_morning_automation as automation
import slotanalyzer_morning_notification as notification
from slotanalyzer_morning_automation_support import JST


def state_with(statuses):
    return {
        "automation_run_id": "morning_test",
        "operation_date": "2026-09-04",
        "stores": {
            store: {
                "status": status,
                "latest_data_date": "2026-09-03",
                "error_category": "" if status == "SUCCESS" else "TEST_ERROR",
                "error": "" if status == "SUCCESS" else "test failure",
            }
            for store, status in zip(automation.STORE_ORDER, statuses)
        },
    }


class FakeSMTP:
    calls = []

    def __init__(self, host, port, **kwargs):
        self.calls.append(("connect", host, port, kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def login(self, username, password):
        self.calls.append(("login", username, password))

    def send_message(self, message):
        self.calls.append(("send", message))


class MorningNotificationTests(unittest.TestCase):
    def setUp(self):
        FakeSMTP.calls = []

    def test_overall_success_partial_failed_and_manual(self):
        self.assertEqual(notification.determine_overall_status(state_with(["SUCCESS"] * 3)), "SUCCESS")
        self.assertEqual(notification.determine_overall_status(state_with(["SUCCESS", "FAILED_FINAL", "FAILED_FINAL"])), "PARTIAL")
        self.assertEqual(notification.determine_overall_status(state_with(["FAILED_FINAL"] * 3)), "FAILED")
        self.assertEqual(notification.determine_overall_status(state_with(["NEEDS_MANUAL_REVIEW"] * 3)), "MANUAL_REVIEW")

    def test_credential_read_failure_is_safe(self):
        with mock.patch.object(notification.os, "name", "posix"):
            with self.assertRaises(notification.CredentialReadError) as caught:
                notification.read_windows_credential()
        self.assertNotIn("password", str(caught.exception).lower())

    def test_credential_read_success_contract(self):
        fake = notification.Credential("sender@example.com", "top-secret")
        self.assertEqual(fake.username, "sender@example.com")
        self.assertEqual(fake.password, "top-secret")

    def test_message_statuses_and_manual_warning(self):
        for statuses, expected in [
            (["SUCCESS"] * 3, "SUCCESS"),
            (["SUCCESS", "FAILED_FINAL", "FAILED_FINAL"], "PARTIAL"),
            (["FAILED_FINAL"] * 3, "FAILED"),
        ]:
            with tempfile.TemporaryDirectory() as directory:
                message = notification.build_notification_message(state_with(statuses), Path(directory))
            self.assertIn(f"[{expected}]", message.subject)
        with tempfile.TemporaryDirectory() as directory:
            message = notification.build_notification_message(state_with(["NEEDS_MANUAL_REVIEW"] * 3), Path(directory))
        self.assertIn("MANUAL_REVIEW", message.plain)

    def test_plain_ranking_mobile_format_and_safe_scores(self):
        rows = [
            {"prediction_rank": "1", "machine_no": "766", "machine_name": "東京喰種", "score": "75.536"},
            {"prediction_rank": "2", "machine_no": "910", "machine_name": "ヤバチバ", "score": "invalid"},
            {"prediction_rank": "3", "machine_no": "885", "machine_name": "攻殻機動隊"},
        ]
        lines = notification._rank_lines(rows, ("prediction_rank",))
        self.assertEqual(lines[0], "1. 766番台　東京喰種　75.54")
        self.assertEqual(lines[1], "2. 910番台　ヤバチバ")
        self.assertEqual(lines[2], "3. 885番台　攻殻機動隊")
        self.assertNotIn("台766", "\n".join(lines))
        self.assertNotIn("score=", "\n".join(lines))

    def test_html_ranking_is_mobile_table_and_escapes_values(self):
        block = notification.RankingBlock(
            "NORMAL Top10",
            [{
                "prediction_rank": "1", "machine_no": "766",
                "machine_name": "長い<機種>&名前", "score": "75.536",
            }],
            ("prediction_rank",),
        )
        rendered = notification._ranking_table_html(block)
        self.assertIn('<table width="100%"', rendered)
        self.assertIn("table-layout:fixed", rendered)
        self.assertIn("border-collapse:collapse", rendered)
        self.assertIn(">No.</th>", rendered)
        self.assertIn(">台</th>", rendered)
        self.assertNotIn(">順位</th>", rendered)
        self.assertNotIn(">台番号</th>", rendered)
        self.assertIn("機種", rendered)
        self.assertIn("Score", rendered)
        self.assertIn(">766</td>", rendered)
        self.assertNotIn("766番台", rendered)
        self.assertIn("75.54", rendered)
        self.assertIn("text-align:right", rendered)
        self.assertIn('width="10%"', rendered)
        self.assertIn('width="15%"', rendered)
        self.assertIn('width="57%"', rendered)
        self.assertIn('width="18%"', rendered)
        self.assertIn("padding:8px 6px", rendered)
        self.assertIn("border-left:1px solid #ddd", rendered)
        self.assertIn("text-align:center;white-space:nowrap", rendered)
        self.assertIn("overflow-wrap:anywhere", rendered)
        self.assertIn("長い&lt;機種&gt;&amp;名前", rendered)
        self.assertNotIn("overflow-x", rendered)
        self.assertNotIn("min-width", rendered)

    def test_html_uses_bare_four_digit_machine_number_but_plain_keeps_bandai(self):
        row = {
            "prediction_rank": "1", "machine_no": "1080",
            "machine_name": "マイジャグラーV", "score": "55.77",
        }
        plain = notification._rank_lines([row], ("prediction_rank",))
        rendered = notification._ranking_table_html(
            notification.RankingBlock("JUGGLER Top10", [row], ("prediction_rank",))
        )
        self.assertEqual(plain, ["1. 1080番台　マイジャグラーV　55.77"])
        self.assertIn(">1080</td>", rendered)
        self.assertNotIn("1080番台", rendered)

    def test_big_march_stale_does_not_show_old_ranking_as_today(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "data/bigmarch_takasaki_oyagi/machine_number/analysis_31days_deep/09_juggler_recent7_future_ranking/09_prediction_20260903_top10.csv"
            old.parent.mkdir(parents=True)
            old.write_text("machine_no,machine_name,target_date,latest_data_date\n1,OLD,2026-09-03,2026-09-02\n", encoding="utf-8")
            message = notification.build_notification_message(state_with(["SUCCESS"] * 3), root)
        self.assertIn("最新1日未反映", message.plain)
        self.assertIn("本日ランキング未生成", message.plain)
        self.assertNotIn("OLD", message.plain)
        self.assertNotIn("OLD", message.html)

    def test_yasuda_success_and_failure_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(notification, "verify_yasuda_completion") as verify:
                verify.return_value = mock.Mock(ok=True, status="COMPLETE")
                text = "\n".join(notification._yasuda_section(state_with(["SUCCESS"] * 3), root, date(2026, 9, 4))[0])
                self.assertIn("Freshness/quality: OK", text)
                verify.return_value = mock.Mock(ok=False, status="INVALID")
                text = "\n".join(notification._yasuda_section(state_with(["SUCCESS"] * 3), root, date(2026, 9, 4))[0])
                self.assertIn("FAILED: INVALID", text)

    def test_maruhan_formal_requires_existing_verifier(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(notification, "verify_maruhan_completion") as verify:
            verify.return_value = mock.Mock(ok=False, status="INVALID_64")
            lines, _ = notification._maruhan_section(state_with(["SUCCESS"] * 3), Path(directory), date(2026, 9, 4))
        self.assertIn("formal表示不可", "\n".join(lines))

    def test_maruhan_formal_valid_uses_verified_metadata_and_rankings(self):
        metadata = [{
            "forward_valid": "True", "target_date": "2026-09-04",
            "latest_data_date": "2026-09-03", "generated_at_jst": "2026-09-04T08:00:00+09:00",
            "model": "MODEL", "weight_fingerprint": "fingerprint",
        }]
        ranking = [{
            "machine_no": "101", "machine_name": "TEST", "score": "12.5",
            "prediction_rank": "1", "target_date": "2026-09-04", "latest_data_date": "2026-09-03",
        }]
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(notification, "verify_maruhan_completion") as verify, \
                mock.patch.object(notification, "_read_rows", side_effect=[metadata, ranking, ranking, ranking, ranking]):
            verify.return_value = mock.Mock(ok=True, status="COMPLETE")
            lines, warnings = notification._maruhan_section(
                state_with(["SUCCESS"] * 3), Path(directory), date(2026, 9, 4)
            )
        text = "\n".join(lines)
        self.assertIn("FORWARD_VALID (formal)", text)
        self.assertIn("1. 101番台　TEST　12.50", text)
        self.assertLess(text.index("NORMAL Top10"), text.index("generated_at_jst"))
        self.assertIn("詳細情報", text)
        self.assertEqual(warnings, [])

    def test_common_html_renderer_is_used_for_all_ranking_groups(self):
        row = {"rank": "1", "machine_no": "1", "machine_name": "TEST", "score": "1"}
        maruhan = notification.StoreSection(
            "MARUHAN", ["Forward: FORWARD_VALID"],
            [notification.RankingBlock(label, [row], ("rank",)) for label in
             ("NORMAL Top10", "A-TYPE Top10", "JUGGLER Top10", "統合", "代替候補・NEXT")],
            ["generated_at_jst: now", "model: test", "fingerprint: test"],
        )
        bigmarch = notification.StoreSection(
            "BIG MARCH", ["Forward検証段階"],
            [notification.RankingBlock(label, [row], ("rank",)) for label in
             ("JUGGLER", "NON_JUGGLER")], [],
        )
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(notification, "_maruhan_content", return_value=(maruhan, [])), \
                mock.patch.object(notification, "_bigmarch_content", return_value=(bigmarch, [])), \
                mock.patch.object(notification, "_yasuda_section", return_value=(["YASUDA", "status: SUCCESS"], [])):
            message = notification.build_notification_message(state_with(["SUCCESS"] * 3), Path(directory))
        self.assertEqual(message.html.count('<table width="100%"'), 7)
        self.assertLess(message.plain.index("Forward: FORWARD_VALID"), message.plain.index("NORMAL Top10"))
        self.assertLess(message.plain.index("NORMAL Top10"), message.plain.index("generated_at_jst"))
        self.assertLess(message.html.index("NORMAL Top10"), message.html.index("generated_at_jst"))

    def test_smtp_is_mocked_history_hash_mask_and_duplicate_sent(self):
        secret = "never-print-this-secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.csv"
            kwargs = dict(
                state=state_with(["SUCCESS"] * 3),
                project_root=root,
                history_path=history,
                credential_reader=lambda: notification.Credential("sender@example.com", secret),
                smtp_factory=FakeSMTP,
                recipient="galaxy@example.com",
                clock=lambda: datetime(2026, 9, 4, 8, 0, tzinfo=JST),
            )
            self.assertTrue(notification.send_notification_best_effort(**kwargs))
            self.assertTrue(notification.send_notification_best_effort(**kwargs))
            with history.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "SENT")
        self.assertEqual(rows[0]["masked_recipient"], "ga***@example.com")
        self.assertEqual(len(rows[0]["message_sha256"]), 64)
        self.assertEqual(len([call for call in FakeSMTP.calls if call[0] == "send"]), 1)
        sent_message = next(call[1] for call in FakeSMTP.calls if call[0] == "send")
        self.assertTrue(sent_message.is_multipart())
        self.assertEqual([part.get_content_type() for part in sent_message.iter_parts()], ["text/plain", "text/html"])
        self.assertNotIn(secret, str(rows))

    def test_notification_failure_redacts_secret_and_records_failed(self):
        secret = "never-print-this-secret"
        class FailingSMTP(FakeSMTP):
            def login(self, username, password):
                raise RuntimeError(f"authentication rejected {password}")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.csv"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = notification.send_notification_best_effort(
                    state_with(["FAILED_FINAL"] * 3), root, history,
                    credential_reader=lambda: notification.Credential("sender@example.com", secret),
                    smtp_factory=FailingSMTP, recipient="galaxy@example.com",
                )
            with history.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertFalse(result)
        self.assertEqual(rows[0]["status"], "FAILED")
        self.assertNotIn(secret, stderr.getvalue())
        self.assertNotIn(secret, str(rows))

    def test_notification_failure_keeps_return_code_and_sleep_eligibility(self):
        state = state_with(["SUCCESS"] * 3)
        events = []
        stderr = io.StringIO()
        def fail_notify(value, root):
            events.append("notify")
            raise RuntimeError("offline")
        with redirect_stderr(stderr):
            result = automation.finalize_automation_run(
                Path("unused"), state, True,
                save_function=lambda *args: events.append("save"),
                summary_function=lambda value: events.append("summary"),
                flush_function=lambda: events.append("flush"),
                notification_function=fail_notify,
                helper_launcher=lambda run_id: events.append("sleep") or 1,
                clock=lambda: datetime(2026, 9, 4, 8, 0, tzinfo=JST),
            )
        self.assertEqual(result, 0)
        self.assertTrue(automation.should_sleep_on_success(True, state, result))
        self.assertEqual(events, ["save", "summary", "flush", "notify", "flush", "sleep"])
        self.assertIn("WARNING", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
