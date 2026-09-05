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


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_formal_69_fixture(root: Path, target: date, *, sha: str = "frozen-sha") -> None:
    base = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep/69_Ver4_2_live_prediction_backtest"
    iso = target.isoformat()
    write_csv_rows(base / "69_live_prediction_status.csv", [{
        "target_date": iso,
        "status": "EVALUATED_FORWARD_VALID",
        "prediction_class": "FORWARD_VALID",
        "actual_path": str(root / f"data/maruhan_maebashi/machine_number/ana_slo_{target:%Y%m%d}.csv"),
        "prediction_sha256": sha,
    }])
    write_csv_rows(base / "69_forward_coverage.csv", [{
        "date": iso,
        "actual_exists": "True",
        "prediction_exists": "True",
        "prediction_class": "FORWARD_VALID",
        "evaluation_status": "EVALUATED_FORWARD_VALID",
    }])
    write_csv_rows(base / "69_forward_valid_detail.csv", [{
        "target_date": iso,
        "prediction_rank": rank,
        "machine_no": 700 + rank,
        "machine_name": f"NORMAL-{rank}",
        "score": 80 - rank / 10,
        "actual_diff": 2300 if rank == 1 else -1100,
        "actual_win": 1 if rank == 1 else 0,
        "prediction_sha256": sha,
        "prediction_class": "FORWARD_VALID",
    } for rank in range(1, 11)])
    write_csv_rows(base / "69_forward_valid_daily.csv", [{
        "target_date": iso,
        "band": band,
        "selected_n": selected,
        "avg_diff": average,
        "win_rate": rate,
        "plus1000_rate": rate,
        "plus2000_rate": rate,
        "prediction_class": "FORWARD_VALID",
    } for band, selected, average, rate in (
        ("TOP3", 3, 200, 33.3333),
        ("TOP5", 5, -340, 20),
        ("TOP10", 10, -680, 10),
    )])


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

    def test_yesterday_formal_normal_shows_top10_and_69_summaries(self):
        state = state_with(["SUCCESS"] * 3)
        state["operation_date"] = "2026-09-06"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_formal_69_fixture(root, date(2026, 9, 5))
            message = notification.build_notification_message(state, root)
        self.assertIn("【昨日の予測結果 2026-09-05】", message.plain)
        self.assertIn("EVALUATED_FORWARD_VALID", message.plain)
        self.assertEqual(message.plain.count("番台　NORMAL-"), 10)
        self.assertIn("+2,300枚　WIN", message.plain)
        self.assertIn("-1,100枚　LOSE", message.plain)
        self.assertIn("TOP3　平均 +200枚　勝率 33.3%", message.plain)
        self.assertIn("TOP5　平均 -340枚　勝率 20.0%", message.plain)
        self.assertIn("TOP10　平均 -680枚　勝率 10.0%", message.plain)
        self.assertIn(">実差枚</th>", message.html)
        self.assertNotIn("overflow-x", message.html)
        self.assertLess(message.plain.index("NORMAL Top10"), message.plain.index("【昨日の予測結果"))
        self.assertLess(message.plain.index("【昨日の予測結果"), message.plain.index("【詳細情報】"))

    def test_yesterday_does_not_fallback_to_older_69_result(self):
        state = state_with(["SUCCESS"] * 3)
        state["operation_date"] = "2026-09-06"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_formal_69_fixture(root, date(2026, 9, 4))
            message = notification.build_notification_message(state, root)
        self.assertIn("【昨日の予測結果 2026-09-05】", message.plain)
        self.assertIn("昨日の正式評価データがないため答え合わせ未評価", message.plain)
        self.assertNotIn("NORMAL-1", message.plain)

    def test_yesterday_legacy_and_forward_guard_fail_are_not_formal(self):
        cases = [
            ("EVALUATED_LEGACY_UNVERIFIED", "LEGACY_UNVERIFIED", "Legacy prediction"),
            ("SKIPPED_FORWARD_GUARD_FAIL", "FORWARD_GUARD_FAIL", "Forward Guard不成立"),
        ]
        for status, prediction_class, expected in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep/69_Ver4_2_live_prediction_backtest"
                write_csv_rows(base / "69_live_prediction_status.csv", [{
                    "target_date": "2026-09-05",
                    "status": status,
                    "prediction_class": prediction_class,
                }])
                result = notification._load_yesterday_normal_evaluation(
                    state_with(["SUCCESS"] * 3), root, date(2026, 9, 6)
                )
            self.assertFalse(result.formal)
            self.assertIn(expected, result.message)

    def test_yesterday_missing_frozen_prediction_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep/69_Ver4_2_live_prediction_backtest"
            write_csv_rows(base / "69_forward_coverage.csv", [{
                "date": "2026-09-05",
                "actual_exists": "True",
                "prediction_exists": "False",
                "prediction_class": "MISSING",
                "evaluation_status": "MISSING_FROZEN_PREDICTION",
            }])
            result = notification._load_yesterday_normal_evaluation(
                state_with(["SUCCESS"] * 3), root, date(2026, 9, 6)
            )
        self.assertFalse(result.formal)
        self.assertIn("MISSING_FROZEN_PREDICTION", result.message)

    def test_yesterday_pending_actual_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep/69_Ver4_2_live_prediction_backtest"
            write_csv_rows(base / "69_live_prediction_status.csv", [{
                "target_date": "2026-09-05",
                "status": "PENDING_FORWARD_VALID",
                "prediction_class": "FORWARD_VALID",
            }])
            result = notification._load_yesterday_normal_evaluation(
                state_with(["SUCCESS"] * 3), root, date(2026, 9, 6)
            )
        self.assertIn("昨日実績未取得", result.message)

    def test_yesterday_inventory_block_only_explains_missing_prediction(self):
        state = state_with(["NEEDS_MANUAL_REVIEW", "SUCCESS", "SUCCESS"])
        state["stores"][automation.STORE_MARUHAN]["inventory_guard"] = {"blocked": True}
        with tempfile.TemporaryDirectory() as directory:
            result = notification._load_yesterday_normal_evaluation(
                state, Path(directory), date(2026, 9, 6)
            )
        self.assertFalse(result.formal)
        self.assertEqual(result.message, "Inventory Guardにより正式予測なし")

    def test_yesterday_existing_formal_result_wins_over_current_inventory_block(self):
        state = state_with(["NEEDS_MANUAL_REVIEW", "SUCCESS", "SUCCESS"])
        state["stores"][automation.STORE_MARUHAN]["inventory_guard"] = {"blocked": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_formal_69_fixture(root, date(2026, 9, 5))
            result = notification._load_yesterday_normal_evaluation(
                state, root, date(2026, 9, 6)
            )
        self.assertTrue(result.formal)

    def test_yesterday_prediction_sha_mismatch_is_read_error_not_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_formal_69_fixture(root, date(2026, 9, 5))
            detail = root / "data/maruhan_maebashi/machine_number/analysis_31days_deep/69_Ver4_2_live_prediction_backtest/69_forward_valid_detail.csv"
            rows = notification._read_rows(detail)
            rows[0]["prediction_sha256"] = "different"
            write_csv_rows(detail, rows)
            result = notification._load_yesterday_normal_evaluation(
                state_with(["SUCCESS"] * 3), root, date(2026, 9, 6)
            )
        self.assertFalse(result.formal)
        self.assertEqual(result.status, "RESULT_READ_ERROR")

    def test_yesterday_read_error_does_not_change_status_or_today_content(self):
        state = state_with(["SUCCESS"] * 3)
        state["operation_date"] = "2026-09-06"
        original_statuses = [item["status"] for item in state["stores"].values()]
        original_reader = notification._read_rows

        def fail_69_only(path):
            if "69_Ver4_2_live_prediction_backtest" in str(path):
                raise PermissionError("fixture")
            return original_reader(path)

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(notification, "_read_rows", side_effect=fail_69_only):
            message = notification.build_notification_message(state, Path(directory))
        self.assertIn("昨日結果取得エラー", message.plain)
        self.assertIn("NORMAL Top10", message.plain)
        self.assertEqual(
            [item["status"] for item in state["stores"].values()], original_statuses
        )
        self.assertEqual(message.overall_status, "SUCCESS")

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

    def test_inventory_guard_warning_is_prominent_and_detailed(self):
        state = state_with(["NEEDS_MANUAL_REVIEW", "SUCCESS", "SUCCESS"])
        state["stores"][automation.STORE_MARUHAN]["inventory_guard"] = {
            "blocked": True,
            "reason": "Inventory change detected",
            "comparison": {
                "previous_machine_count": 514,
                "current_machine_count": 514,
                "added_machine_numbers": [900],
                "removed_machine_numbers": [1],
                "renamed_machine_numbers": [{
                    "machine_no": 700,
                    "previous_machine_name": "OLD",
                    "current_machine_name": "NEW",
                }],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            message = notification.build_notification_message(state, Path(directory))
        self.assertIn("⚠ Maruhan: 台構成変更を検知", message.plain)
        self.assertIn("⚠ Maruhan: 正式Forward停止", message.plain)
        self.assertIn("⚠ Maruhan: MANUAL_REVIEW", message.plain)
        self.assertIn("inventory previous/current: 514 / 514", message.plain)
        self.assertIn("renamed: 700 OLD -> NEW", message.plain)
        self.assertNotIn("FORWARD_VALID (formal)", message.plain)
        self.assertLess(message.plain.index("台構成変更を検知"), message.plain.index("【Maruhan"))
        self.assertIn("台構成変更を検知", message.html)

    def test_normal_success_gmail_has_no_inventory_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            message = notification.build_notification_message(
                state_with(["SUCCESS"] * 3), Path(directory)
            )
        self.assertNotIn("台構成変更を検知", message.plain)
        self.assertNotIn("正式Forward停止 (inventory guard)", message.html)

    def test_persistent_unapproved_block_warning_continues_on_zero_diff_day(self):
        state = state_with(["NEEDS_MANUAL_REVIEW", "SUCCESS", "SUCCESS"])
        state["operation_date"] = "2026-09-10"
        state["stores"][automation.STORE_MARUHAN]["inventory_guard"] = {
            "blocked": True,
            "reason": "Inventory change remains blocked because explicit approval has not been recorded.",
            "comparison": {
                "previous_machine_count": 514,
                "current_machine_count": 514,
                "added_machine_numbers": [],
                "removed_machine_numbers": [],
                "renamed_machine_numbers": [],
                "has_changes": False,
            },
            "persistent_state": {
                "status": "BLOCKED",
                "change_date": "2026-09-08",
                "detected_at": "2026-09-09T08:01:00+09:00",
                "previous_machine_count": 514,
                "current_machine_count": 514,
                "added_machine_numbers": [],
                "removed_machine_numbers": [],
                "renamed_machine_numbers": [{
                    "machine_no": 500,
                    "previous_machine_name": "OLD",
                    "current_machine_name": "NEW",
                }],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            message = notification.build_notification_message(state, Path(directory))
        self.assertIn("台構成変更による正式Forward停止中", message.plain)
        self.assertIn("未承認のため停止継続", message.plain)
        self.assertIn("renamed: 500 OLD -> NEW", message.plain)
        self.assertNotIn("FORWARD_VALID (formal)", message.plain)

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
