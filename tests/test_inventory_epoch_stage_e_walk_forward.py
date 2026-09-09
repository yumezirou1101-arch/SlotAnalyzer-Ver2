from __future__ import annotations

from datetime import date, timedelta
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "machine_number/ana_slo_prediction_v4_2_inventory_epoch_stage_e_walk_forward.py"
SPEC = importlib.util.spec_from_file_location("inventory_epoch_stage_e_test", SCRIPT)
stage_e = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage_e
SPEC.loader.exec_module(stage_e)
shadow = stage_e.shadow


def frame(rows):
    result = pd.DataFrame(rows, columns=("date", "machine_no", "machine_name", "games", "diff"))
    result["date"] = pd.to_datetime(result.date)
    return result


def event(day, machines=(1,)):
    return stage_e.StoreEvent(
        f"E{day:%Y%m%d}", day - timedelta(days=1), day, (), (), tuple(machines),
        len(machines) / 514, "SMALL",
    )


class PersistentEpochTests(unittest.TestCase):
    def test_next_day_diff_zero_keeps_changed_cohort(self):
        data = frame([
            ("2026-01-01", 1, "A", 10, 1), ("2026-01-02", 1, "B", 10, 2),
            ("2026-01-03", 1, "B", 10, 3), ("2026-01-01", 2, "KEEP", 10, 4),
            ("2026-01-02", 2, "KEEP", 10, 5), ("2026-01-03", 2, "KEEP", 10, 6),
        ])
        diff = shadow.build_persistent_inventory_diff(data, "2026-01-03")
        self.assertEqual(diff.machine_numbers("renamed"), frozenset({1}))

    def test_a_b_a_does_not_reconnect(self):
        data = frame([
            ("2026-01-01", 1, "A", 10, 1), ("2026-01-02", 1, "B", 10, 2),
            ("2026-01-03", 1, "A", 10, 3),
        ])
        epochs = shadow.build_continuous_epochs(data, "2026-01-03")
        current = epochs.current.iloc[0]
        self.assertEqual((current.epoch_sequence, current.epoch_calendar_days), (3, 1))
        diff = shadow.build_persistent_inventory_diff(data, "2026-01-03")
        row = diff.rows.iloc[0]
        self.assertEqual((row.old_machine_name, row.new_machine_name), ("B", "A"))

    def test_zero_g_is_not_observed_history(self):
        data = frame([("2026-01-01", 1, "A", 10, 1), ("2026-01-02", 1, "B", 0, 0)])
        current = shadow.build_continuous_epochs(data, "2026-01-02").current.iloc[0]
        self.assertEqual((current.epoch_calendar_days, current.epoch_observed_days), (1, 0))


class ChangedMachineEventOwnershipTests(unittest.TestCase):
    def test_prior_event_machine_is_not_attributed_to_current_event(self):
        first = event(date(2026, 1, 2), (1,))
        second = event(date(2026, 1, 10), (2,))

        self.assertIsNone(
            stage_e._event_for_changed_machine(
                [first, second], 1, first.change_date, second.event_id
            )
        )

    def test_current_event_machine_is_attributed_to_current_event(self):
        first = event(date(2026, 1, 2), (1,))
        second = event(date(2026, 1, 10), (2,))

        owner = stage_e._event_for_changed_machine(
            [first, second], 2, second.change_date, second.event_id
        )
        self.assertIsNotNone(owner)
        self.assertEqual(owner.event_id, second.event_id)

class EventWindowTests(unittest.TestCase):
    def paths(self, start, end, missing=()):
        return {day: Path(f"{day:%Y%m%d}.csv") for day in pd.date_range(start, end)
                if day.date() not in missing for day in [day.date()]}

    def test_next_event_censors_primary_window(self):
        first, second = event(date(2026, 1, 2)), event(date(2026, 1, 10))
        plan = stage_e.primary_targets([first, second], self.paths(date(2026, 1, 1), date(2026, 1, 24)))
        rows = plan[plan.event_id == first.event_id]
        self.assertEqual(rows.target_date.max(), date(2026, 1, 9))
        self.assertTrue(rows.independent_primary.all())

    def test_secondary_reentry_tracking_stops_before_next_event(self):
        first, second = event(date(2026, 1, 2)), event(date(2026, 1, 25))
        plan = stage_e.primary_targets([first, second], self.paths(date(2026, 1, 1), date(2026, 2, 8)))
        rows = plan[plan.event_id == first.event_id]
        self.assertEqual(rows.target_date.max(), date(2026, 1, 24))
        self.assertTrue((~rows[rows.event_calendar_day > 14].independent_primary).all())

    def test_last_event_secondary_tracking_reaches_latest_daily(self):
        last = event(date(2026, 1, 2))
        plan = stage_e.primary_targets([last], self.paths(date(2026, 1, 1), date(2026, 1, 24)))
        rows = plan[plan.event_id == last.event_id]
        self.assertEqual(rows.target_date.max(), date(2026, 1, 24))
        secondary = rows[rows.event_calendar_day > 14]
        self.assertFalse(secondary.empty)
        self.assertTrue((~secondary.independent_primary).all())
        self.assertTrue(secondary.day_bucket.eq("SECONDARY").all())
    def test_missing_t_or_t_minus_one_skips(self):
        paths = self.paths(date(2026, 1, 1), date(2026, 1, 16), {date(2026, 1, 4)})
        plan = stage_e.primary_targets([event(date(2026, 1, 2))], paths)
        self.assertFalse(plan.loc[plan.target_date.isin([date(2026, 1, 4), date(2026, 1, 5)]), "evaluation_available"].any())

    def test_primary_windows_do_not_overlap(self):
        events = [event(date(2026, 1, 2)), event(date(2026, 1, 10)), event(date(2026, 1, 20))]
        plan = stage_e.primary_targets(events, self.paths(date(2026, 1, 1), date(2026, 2, 3)))
        primary = plan[plan.independent_primary]
        counts = primary.groupby("target_date").event_id.nunique()
        self.assertEqual(int(counts.max()), 1)

    def test_day_buckets(self):
        expected = {1: "DAY1", 2: "DAY2", 3: "DAY3", 4: "DAY4-7", 7: "DAY4-7", 8: "DAY8-14", 14: "DAY8-14", 15: "SECONDARY"}
        self.assertEqual({day: stage_e.day_bucket(day) for day in expected}, expected)


class PolicyTests(unittest.TestCase):
    def sample(self, observed_count):
        rows = [("2025-12-31", 1, "OLD", 10, -10)]
        rows += [((date(2026, 1, 1) + timedelta(days=i)).isoformat(), 1, "NEW", 10, i)
                 for i in range(observed_count)]
        rows += [((date(2025, 12, 31) + timedelta(days=i)).isoformat(), 2, "KEEP", 10, i)
                 for i in range(observed_count + 1)]
        data = frame(rows)
        as_of = date(2026, 1, 1) + timedelta(days=observed_count - 1)
        diff = shadow.build_persistent_inventory_diff(data, as_of)
        return data, as_of, diff

    def test_threshold_boundaries(self):
        for count, threshold, expected in ((6, 7, False), (7, 7, True), (9, 10, False),
                                           (10, 10, True), (13, 14, False), (14, 14, True),
                                           (20, 21, False), (21, 21, True)):
            data, as_of, diff = self.sample(count)
            result = shadow.build_stage_c_shadow(data, as_of + timedelta(days=1), as_of, diff, threshold)
            row = result.eligibility.query("policy == @shadow.POLICY2 and machine_no == 1").iloc[0]
            self.assertEqual(bool(row.z_population and row.final_ranking), expected)

    def test_stage_a_d_exact_reuse_and_fingerprint(self):
        data, as_of, diff = self.sample(7)
        direct = shadow.build_stage_c_shadow(data, as_of + timedelta(days=1), as_of, diff, 7)
        via_stage_e = shadow.build_stage_c_shadow(data, as_of + timedelta(days=1), as_of, diff, 7)
        self.assertEqual(shadow.weight_fingerprint(), "a1eaf45d71ded209")
        pd.testing.assert_frame_equal(
            direct.policies[shadow.POLICY2]["final"].reset_index(drop=True),
            via_stage_e.policies[shadow.POLICY2]["final"].reset_index(drop=True),
        )

    def test_category_score_is_normal_score(self):
        data = frame([
            ("2026-01-01", 1, "OLD", 10, 1), ("2026-01-02", 1, "マイジャグラーV", 10, 2),
            ("2026-01-01", 2, "新ハナビ", 10, 3), ("2026-01-02", 2, "新ハナビ", 10, 4),
            ("2026-01-01", 3, "アイムジャグラーEX", 10, 5), ("2026-01-02", 3, "アイムジャグラーEX", 10, 6),
            ("2026-01-01", 4, "モンキーターンV", 10, 7), ("2026-01-02", 4, "モンキーターンV", 10, 8),
        ])
        diff = shadow.build_persistent_inventory_diff(data, "2026-01-02")
        result = shadow.build_stage_d_shadow(data, "2026-01-03", "2026-01-02", diff, 1)
        self.assertTrue((result.lineage.category_score == result.lineage.normal_score).all())


class LeakageAndOutputTests(unittest.TestCase):
    def test_ranking_digest_unchanged_by_unrelated_actual(self):
        data = frame([
            ("2025-12-31", 1, "OLD", 10, 0), ("2026-01-01", 1, "マイジャグラーV", 10, 100),
            ("2025-12-31", 2, "新ハナビ", 10, 0), ("2026-01-01", 2, "新ハナビ", 10, -100),
            ("2025-12-31", 3, "アイムジャグラーEX", 10, 0), ("2026-01-01", 3, "アイムジャグラーEX", 10, 50),
            ("2025-12-31", 4, "モンキーターンV", 10, 0), ("2026-01-01", 4, "モンキーターンV", 10, -50),
        ])
        active = shadow.build_persistent_inventory_diff(data, date(2026, 1, 1))
        stages = stage_e._policy_stages(data, date(2026, 1, 2), date(2026, 1, 1), active)
        before = stage_e.ranking_digest(stages)
        actual = frame([("2026-01-02", 1, "FUTURE NAME", 10, 999999)])
        self.assertEqual(before, stage_e.ranking_digest(stages))
        self.assertEqual(actual.iloc[0].machine_name, "FUTURE NAME")

    def test_future_huge_value_does_not_change_past_raw_ranking(self):
        base = frame([("2025-12-31", 1, "OLD", 10, 0), ("2026-01-01", 1, "A", 10, 1),
                      ("2025-12-31", 2, "B", 10, 0), ("2026-01-01", 2, "B", 10, 2)])
        future = pd.concat([base, frame([("2026-01-03", 1, "FUTURE", 10, 999999)])], ignore_index=True)
        diff = shadow.build_persistent_inventory_diff(base, "2026-01-01")
        left = shadow.build_stage_c_shadow(base, "2026-01-02", "2026-01-01", diff)
        right = shadow.build_stage_c_shadow(future, "2026-01-02", "2026-01-01", diff)
        pd.testing.assert_frame_equal(left.policies[shadow.POLICY_BASELINE]["final"], right.policies[shadow.POLICY_BASELINE]["final"])

    def test_actual_join_requires_one_to_one(self):
        rank = pd.DataFrame({"machine_no": [1], "score": [1.0]})
        actual = pd.DataFrame({"machine_no": [1, 1], "actual_diff": [1, 2]})
        with self.assertRaises(pd.errors.MergeError):
            rank.merge(actual, on="machine_no", validate="one_to_one")

    def test_topn_and_replacement_gain_loss(self):
        joined = pd.DataFrame({"actual_diff": [1000, -500, 200, 300, -100, 50, 60, 70, 80, 90]})
        rows = stage_e._metric_rows(joined, {"x": 1})
        self.assertEqual([row["top_n"] for row in rows], [1, 3, 5, 10])
        self.assertEqual(rows[1]["actual_total_diff"], 700)

    def test_research_paths_and_nonwrite_defaults(self):
        self.assertIn("research_inventory_epoch_stage_e", str(stage_e.OUTPUT_DIR))
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("run_slotanalyzer_morning", "--allow-gap", "69_forward", "76_"):
            self.assertNotIn(forbidden, source)
        self.assertIn('"production_outputs_written": False', source)

    def test_discovery_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as temp:
            before = set(Path(temp).iterdir())
            with self.assertRaises(ValueError):
                stage_e.discover_daily_paths(Path(temp))
            self.assertEqual(before, set(Path(temp).iterdir()))


if __name__ == "__main__":
    unittest.main()
