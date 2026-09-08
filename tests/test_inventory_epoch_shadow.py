from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "machine_number/ana_slo_prediction_v4_2_inventory_epoch_shadow.py"
SPEC = importlib.util.spec_from_file_location("inventory_epoch_shadow", SCRIPT)
shadow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shadow
SPEC.loader.exec_module(shadow)


def frame(rows):
    result = pd.DataFrame(
        rows, columns=["date", "machine_no", "machine_name", "games", "diff"]
    )
    result["date"] = pd.to_datetime(result["date"])
    return result


def write_daily(path, day, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["日付", "台番号", "機種名", "G数", "差枚"])
        for machine_no, name, games, difference in rows:
            writer.writerow([day, machine_no, name, games, difference])


class InventoryEpochShadowTests(unittest.TestCase):
    def epochs(self, rows, as_of="2026-01-10"):
        return shadow.build_continuous_epochs(frame(rows), date.fromisoformat(as_of))

    def test_unchanged_machine_stays_same_epoch(self):
        result = self.epochs([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-02", 1, "A", 1, 0)
        ])
        self.assertEqual(result.rows.epoch_sequence.tolist(), [1, 1])

    def test_a_to_b_starts_new_epoch(self):
        result = self.epochs([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-02", 1, "B", 1, 0)
        ])
        self.assertEqual(result.rows.epoch_sequence.tolist(), [1, 2])

    def test_a_b_a_is_three_epochs_without_reconnection(self):
        result = self.epochs([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-02", 1, "B", 1, 0),
            ("2026-01-03", 1, "A", 1, 0),
        ])
        self.assertEqual(result.rows.epoch_sequence.tolist(), [1, 2, 3])
        self.assertTrue(result.current.iloc[0].epoch_id.startswith("1:0003:"))
        self.assertEqual(result.current.iloc[0].epoch_calendar_days, 1)

    def test_zero_g_counts_only_calendar(self):
        row = self.epochs([("2026-01-01", 1, "A", 0, 0)]).current.iloc[0]
        self.assertEqual((row.epoch_calendar_days, row.epoch_observed_days), (1, 0))
        self.assertEqual((row.epoch_zero_g_days, row.epoch_total_games), (1, 0))

    def test_positive_g_counts_calendar_and_observed(self):
        row = self.epochs([("2026-01-01", 1, "A", 12, 0)]).current.iloc[0]
        self.assertEqual((row.epoch_calendar_days, row.epoch_observed_days), (1, 1))
        self.assertEqual(row.eligibility_history_n, 1)
        self.assertEqual(row.history_n_definition, "G_GT_0")

    def test_negative_g_rejected(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            self.epochs([("2026-01-01", 1, "A", -1, 0)])

    def test_duplicate_machine_rejected_by_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            write_daily(path, "2026-01-01", [(1, "A", 1, 0), (1, "B", 1, 0)])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                shadow.read_daily_csv(path, "2026-01-01", "2026-01-01")

    def test_empty_name_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            write_daily(path, "2026-01-01", [(1, "", 1, 0)])
            with self.assertRaisesRegex(ValueError, "empty"):
                shadow.read_daily_csv(path, "2026-01-01", "2026-01-01")

    def test_internal_date_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            write_daily(path, "2026-01-02", [(1, "A", 1, 0)])
            with self.assertRaisesRegex(ValueError, "mismatch"):
                shadow.read_daily_csv(path, "2026-01-01", "2026-01-02")

    def test_reader_accepts_thousands_separators_in_numeric_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            write_daily(path, "2026-01-01", [(1, "A", "5,307", "-1,200")])
            result = shadow.read_daily_csv(path, "2026-01-01", "2026-01-01")
        self.assertEqual((result.iloc[0].games, result.iloc[0]["diff"]), (5307, -1200))

    def test_future_file_rejected_before_open(self):
        with self.assertRaisesRegex(ValueError, "Future daily"):
            shadow.read_daily_csv(Path("must-not-open.csv"), "2026-01-02", "2026-01-01")

    def test_future_rows_excluded(self):
        result = self.epochs([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-02", 1, "B", 1, 0)
        ], "2026-01-01")
        self.assertEqual((len(result.rows), result.current.iloc[0].machine_name), (1, "A"))

    def test_future_a_cannot_change_past_b_epoch(self):
        result = self.epochs([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-02", 1, "B", 1, 0),
            ("2026-01-03", 1, "A", 1, 0),
        ], "2026-01-02")
        self.assertEqual((result.current.iloc[0].machine_name, result.current.iloc[0].epoch_sequence), ("B", 2))

    def test_gap_is_represented(self):
        result = self.epochs([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-03", 1, "A", 1, 0)
        ])
        self.assertFalse(result.current.iloc[0].calendar_continuous)
        self.assertEqual(result.current.iloc[0].epoch_missing_calendar_days, 1)
        self.assertTrue(result.rows.iloc[1].observation_continues_name)
        self.assertFalse(result.rows.iloc[1].calendar_contiguous_from_previous)

    def diff(self, previous, current):
        return shadow.compare_inventories(frame(previous), frame(current))

    def test_diff_unchanged(self):
        result = self.diff([("2026-01-01", 1, "A", 1, 0)], [("2026-01-02", 1, "A", 1, 0)])
        self.assertEqual(result.unchanged_count, 1)

    def test_diff_renamed(self):
        result = self.diff([("2026-01-01", 1, "A", 1, 0)], [("2026-01-02", 1, "B", 1, 0)])
        self.assertEqual(result.renamed_count, 1)
        self.assertEqual((result.rows.iloc[0].old_machine_name, result.rows.iloc[0].new_machine_name), ("A", "B"))

    def test_diff_added(self):
        result = self.diff([("2026-01-01", 1, "A", 1, 0)], [
            ("2026-01-02", 1, "A", 1, 0), ("2026-01-02", 2, "B", 1, 0)
        ])
        self.assertEqual(result.added_count, 1)

    def test_diff_removed(self):
        result = self.diff([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-01", 2, "B", 1, 0)
        ], [("2026-01-02", 1, "A", 1, 0)])
        self.assertEqual(result.removed_count, 1)

    def test_changed_and_unchanged_are_disjoint(self):
        result = self.diff([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-01", 2, "B", 1, 0)
        ], [
            ("2026-01-02", 1, "C", 1, 0), ("2026-01-02", 2, "B", 1, 0)
        ])
        self.assertTrue(result.machine_numbers("renamed").isdisjoint(result.machine_numbers("unchanged")))

    def test_current_epoch_has_no_prior_name_rows(self):
        result = self.epochs([
            ("2026-01-01", 1, "OLD", 1, 0), ("2026-01-02", 1, "NEW", 1, 0)
        ])
        current = result.rows[result.rows.epoch_id == result.current.iloc[0].epoch_id]
        self.assertEqual(current.machine_name.tolist(), ["NEW"])

    def test_last_observed_date(self):
        row = self.epochs([
            ("2026-01-01", 1, "A", 1, 0), ("2026-01-02", 1, "A", 0, 0),
            ("2026-01-03", 1, "A", 2, 0),
        ]).current.iloc[0]
        self.assertEqual(row.last_observed_date, date(2026, 1, 3))

    def test_no_observed_date_is_none(self):
        row = self.epochs([("2026-01-01", 1, "A", 0, 0)]).current.iloc[0]
        self.assertIsNone(row.last_observed_date)

    def test_module_has_no_production_writers_or_imports(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in (
            "to_csv",
            "Forward Guard",
            "Morning Automation",
            "formal evidence",
            "64_Ver4_2_future_top10",
        ):
            self.assertNotIn(forbidden, source)


class RawFeatureShadowTests(unittest.TestCase):
    def sample(self, future=False, no_type_prior=False):
        rows = [
            ("2026-01-01", 1, "OLD", 100, -1000), ("2026-01-02", 1, "OLD", 100, 2000),
            ("2026-01-03", 1, "NEW", 0, 0),
            ("2026-01-01", 2, "KEEP", 100, 100), ("2026-01-02", 2, "KEEP", 100, 200),
            ("2026-01-03", 2, "KEEP", 100, 300),
            ("2026-01-01", 3, "NEW" if not no_type_prior else "OTHER", 100, 900),
            ("2026-01-03", 3, "NEW" if not no_type_prior else "OTHER", 100, -300),
        ]
        if future:
            rows += [("2026-01-04", 1, "FUTURE", 999, 999999), ("2026-01-05", 2, "KEEP", 999, -999999)]
        data = frame(rows)
        previous = data[data.date == pd.Timestamp("2026-01-02")]
        current = data[data.date == pd.Timestamp("2026-01-03")]
        return shadow.build_raw_feature_shadow(data, "2026-01-04", "2026-01-03", shadow.compare_inventories(previous, current))

    def changed(self, result, policy):
        return result.lineage[(result.lineage.machine_no == 1) & (result.lineage.policy == policy)].iloc[0]

    def test_baseline_production_raw_definition(self):
        row = self.sample().baseline.set_index("machine_no").loc[1]
        self.assertEqual(row.avg31, 1000 / 3)
        self.assertEqual((row.last_diff, row.prev_change), (0, -2000))
        self.assertEqual((row.plus1000_rate, row.plus2000_rate), (1 / 3, 1 / 3))

    def test_baseline_recent7_is_row_tail_not_calendar_window(self):
        self.assertEqual(self.sample().baseline.set_index("machine_no").loc[1].recent7_avg, 1000 / 3)

    def test_unchanged_row_exact(self):
        r = self.sample()
        self.assertTrue(r.baseline.set_index("machine_no").loc[2, shadow.RAW_FEATURES].equals(r.row_based.set_index("machine_no").loc[2, shadow.RAW_FEATURES]))

    def test_unchanged_observed_exact(self):
        r = self.sample()
        self.assertTrue(r.baseline.set_index("machine_no").loc[2, shadow.RAW_FEATURES].equals(r.observed.set_index("machine_no").loc[2, shadow.RAW_FEATURES]))

    def test_row_excludes_pre_epoch(self):
        self.assertEqual(self.changed(self.sample(), shadow.ROW_BASED).pre_epoch_row_count, 0)

    def test_observed_excludes_pre_epoch(self):
        self.assertEqual(self.changed(self.sample(), shadow.OBSERVED_G_GT_0).pre_epoch_row_count, 0)

    def test_old_name_excluded_both(self):
        self.assertTrue((self.sample().lineage.old_machine_name_row_count == 0).all())

    def test_row_includes_zero_g(self):
        self.assertEqual(self.changed(self.sample(), shadow.ROW_BASED).history_n, 1)

    def test_observed_excludes_zero_g(self):
        self.assertEqual(self.changed(self.sample(), shadow.OBSERVED_G_GT_0).history_n, 0)

    def test_observed_zero_history_explicit_fallback(self):
        self.assertEqual(self.changed(self.sample(), shadow.OBSERVED_G_GT_0).fallback_kind, "TYPE_PRIOR")

    def test_type_prior_lineage_separate(self):
        row = self.changed(self.sample(), shadow.OBSERVED_G_GT_0)
        self.assertEqual((row.history_n, row.fallback_source_row_count), (0, 3))

    def test_store_prior_lineage_separate(self):
        hist, target, _ = shadow._prepare_history(frame([
            ("2026-01-01", 1, "OTHER", 1, 100),
        ]), "2026-01-02", "2026-01-01")
        _, kind, source_n = shadow._prior_values(hist, target, "ABSENT")
        self.assertEqual(kind, "STORE_PRIOR")
        self.assertEqual(source_n, 1)

    def test_future_actual_cannot_affect_seven(self):
        a, b = self.sample(), self.sample(future=True)
        self.assertTrue(a.observed.set_index("machine_no").loc[1, shadow.EPOCH_FEATURES].equals(b.observed.set_index("machine_no").loc[1, shadow.EPOCH_FEATURES]))

    def test_future_actual_cannot_affect_type(self):
        self.assertEqual(self.sample().baseline.set_index("machine_no").loc[1].type_avg, self.sample(future=True).baseline.set_index("machine_no").loc[1].type_avg)

    def test_future_actual_cannot_affect_neighbor(self):
        self.assertEqual(self.sample().baseline.set_index("machine_no").loc[1].neighbor_avg, self.sample(future=True).baseline.set_index("machine_no").loc[1].neighbor_avg)

    def test_future_name_cannot_affect_lineage(self):
        self.assertEqual(self.changed(self.sample(future=True), shadow.ROW_BASED).current_machine_name, "NEW")

    def test_type_avg_unchanged_in_shadow(self):
        r = self.sample()
        self.assertEqual(r.baseline.set_index("machine_no").loc[1].type_avg, r.observed.set_index("machine_no").loc[1].type_avg)

    def test_neighbor_avg_unchanged_in_shadow(self):
        r = self.sample()
        self.assertEqual(r.baseline.set_index("machine_no").loc[1].neighbor_avg, r.row_based.set_index("machine_no").loc[1].neighbor_avg)

    def test_neighbor_diagnostic_zero_g(self):
        self.assertGreaterEqual(self.sample().neighbor_diagnostics.iloc[0].zero_g_neighbor_count, 0)

    def test_neighbor_diagnostic_changed_neighbor(self):
        self.assertIn("neighbor_inventory_changed", self.sample().neighbor_diagnostics.columns)

    def test_all_raw_values_finite(self):
        r = self.sample()
        for panel in (r.baseline, r.row_based, r.observed):
            self.assertFalse(panel[list(shadow.RAW_FEATURES)].isna().any().any())

    def test_target_actual_never_loaded(self):
        self.assertFalse(self.sample(future=True).target_actual_loaded)

    def test_a_b_a_uses_latest_a_only(self):
        data = frame([("2026-01-01", 1, "A", 1, 100), ("2026-01-02", 1, "B", 1, 200), ("2026-01-03", 1, "A", 1, 300)])
        diff = shadow.compare_inventories(data[data.date == pd.Timestamp("2026-01-02")], data[data.date == pd.Timestamp("2026-01-03")])
        result = shadow.build_raw_feature_shadow(data, "2026-01-04", "2026-01-03", diff)
        self.assertEqual(result.row_based.set_index("machine_no").loc[1].avg31, 300)

    def test_history_max_never_reaches_target(self):
        row = self.changed(self.sample(future=True), shadow.ROW_BASED)
        self.assertLess(row.history_max_date, date(2026, 1, 4))


class StageCShadowTests(unittest.TestCase):
    def result(self, future=False):
        return RawFeatureShadowTests().sample(future=future), self._stage(future)

    def _stage(self, future=False):
        data_rows = [("2026-01-01", 1, "OLD", 100, -1000), ("2026-01-02", 1, "OLD", 100, 2000),
            ("2026-01-03", 1, "NEW", 0, 0), ("2026-01-01", 2, "KEEP", 100, 100),
            ("2026-01-02", 2, "KEEP", 100, 200), ("2026-01-03", 2, "KEEP", 100, 300),
            ("2026-01-01", 3, "NEW", 100, 900), ("2026-01-03", 3, "NEW", 100, -300)]
        if future:
            data_rows += [("2026-01-04", 1, "FUTURE", 99, 999999), ("2026-01-05", 2, "KEEP", 99, -999999)]
        data = frame(data_rows)
        diff = shadow.compare_inventories(data[data.date == pd.Timestamp("2026-01-02")], data[data.date == pd.Timestamp("2026-01-03")])
        return shadow.build_stage_c_shadow(data, "2026-01-04", "2026-01-03", diff)

    def test_fingerprint(self):
        self.assertEqual(shadow.weight_fingerprint(), "a1eaf45d71ded209")

    def test_baseline_zscore_formula(self):
        r = self._stage(); a = r.component_audit
        x = a[(a.policy == shadow.POLICY_BASELINE) & (a.feature == "avg31")].set_index("machine_no")
        raw = r.raw.baseline.set_index("machine_no").avg31
        self.assertAlmostEqual(x.loc[1].z, (raw.loc[1] - raw.mean()) / raw.std(ddof=0))

    def test_baseline_score_matches_production_pure_function(self):
        spec = importlib.util.spec_from_file_location(
            "production_v42_for_test",
            ROOT / "machine_number/ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py",
        )
        production = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(production)
        result = self._stage()
        expected = production.rank_score(result.raw.baseline, production.V42_C_WEIGHTS)
        actual = result.policies[shadow.POLICY_BASELINE]["scored"]
        self.assertEqual(expected.machine_no.tolist(), actual.machine_no.tolist())
        self.assertTrue(expected.score.reset_index(drop=True).equals(actual.score.reset_index(drop=True)))

    def test_component_formula(self):
        row = self._stage().component_audit.iloc[0]
        self.assertEqual(row.component, min(100, max(0, 50 + row.z * 12.5)))

    def test_weighted_component_formula(self):
        row = self._stage().component_audit.iloc[0]
        self.assertAlmostEqual(row.weighted_component, row.component * shadow.V42_C_WEIGHTS[row.feature])

    def test_score_is_component_sum(self):
        r = self._stage(); audit = r.component_audit
        expected = audit[(audit.policy == shadow.POLICY_BASELINE) & (audit.machine_no == 1)].weighted_component.sum()
        actual = r.policies[shadow.POLICY_BASELINE]["scored"].set_index("machine_no").loc[1].score
        self.assertAlmostEqual(actual, expected)

    def test_policy1_population_includes_changed(self):
        row = self._stage().eligibility.query("policy == @shadow.POLICY1 and machine_no == 1").iloc[0]
        self.assertTrue(row.z_population)

    def test_policy1_final_excludes_changed(self):
        row = self._stage().eligibility.query("policy == @shadow.POLICY1 and machine_no == 1").iloc[0]
        self.assertFalse(row.final_ranking)

    def test_policy2_population_excludes_changed(self):
        row = self._stage().eligibility.query("policy == @shadow.POLICY2 and machine_no == 1").iloc[0]
        self.assertFalse(row.z_population)

    def test_policy2_final_excludes_changed(self):
        row = self._stage().eligibility.query("policy == @shadow.POLICY2 and machine_no == 1").iloc[0]
        self.assertFalse(row.final_ranking)

    def test_policy3_unchanged_score_exact(self):
        r = self._stage(); row = r.score_ripple.query("policy == @shadow.POLICY3").iloc[0]
        self.assertEqual(row.max_abs_delta, 0)

    def test_score_and_ranks_separate(self):
        columns = r = self._stage().policies[shadow.POLICY1]["scored"].columns
        self.assertTrue({"score", "full_population_score_order", "final_candidate_rank"}.issubset(columns))

    def test_top10_comparison(self):
        self.assertEqual(len(self._stage().top10_comparisons), 5)

    def test_row_policy_sensitivity_exists(self):
        self.assertIn(shadow.ROW_POLICY1, self._stage().policies)

    def test_future_population_stats_unchanged(self):
        a, b = self._stage(), self._stage(True)
        self.assertTrue(a.z_audit.equals(b.z_audit))

    def test_future_score_unchanged(self):
        a, b = self._stage(), self._stage(True)
        self.assertTrue(a.policies[shadow.POLICY1]["scored"][["machine_no", "score"]].equals(b.policies[shadow.POLICY1]["scored"][["machine_no", "score"]]))

    def test_future_top10_unchanged(self):
        a, b = self._stage(), self._stage(True)
        self.assertEqual(a.policies[shadow.POLICY1]["top10"].machine_no.tolist(), b.policies[shadow.POLICY1]["top10"].machine_no.tolist())

    def test_std_zero_z_is_zero(self):
        panel = pd.DataFrame({"machine_no": [1, 2], **{f: [1.0, 1.0] for f in shadow.RAW_FEATURES}})
        stats, _ = shadow._population_stats(panel, {1, 2}, "X")
        scored, _, audit = shadow._score_panel(panel, stats, "X", {1, 2}, {1, 2})
        self.assertTrue((audit.z == 0).all()); self.assertTrue((scored.score == 50).all())

    def test_sort_is_score_descending(self):
        scores = self._stage().policies[shadow.POLICY_BASELINE]["scored"].score.tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_tie_sorting_matches_production(self):
        spec = importlib.util.spec_from_file_location(
            "production_v42_tie_test",
            ROOT / "machine_number/ana_slo_prediction_v4_2_machine_number_position_ablation_oos.py",
        )
        production = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(production)
        panel = pd.DataFrame({"machine_no": [3, 1, 2], **{f: [1.0, 1.0, 1.0] for f in shadow.RAW_FEATURES}})
        stats, _ = shadow._population_stats(panel, {1, 2, 3}, "TIE")
        actual, _, _ = shadow._score_panel(panel, stats, "TIE", {1, 2, 3}, {1, 2, 3})
        expected = production.rank_score(panel, production.V42_C_WEIGHTS)
        self.assertEqual(actual.machine_no.tolist(), expected.machine_no.tolist())

    def test_history_13_excluded_boundary(self):
        self.assertFalse(self._boundary(13).eligibility.query("policy == @shadow.POLICY1 and machine_no == 1").iloc[0].eligible)

    def test_history_14_eligible_boundary(self):
        self.assertTrue(self._boundary(14).eligibility.query("policy == @shadow.POLICY1 and machine_no == 1").iloc[0].eligible)

    def _boundary(self, count):
        rows = [(f"2026-01-{day:02d}", 1, "OLD" if day == 1 else "NEW", 1, day) for day in range(1, count + 2)]
        rows += [(f"2026-01-{day:02d}", 2, "KEEP", 1, day) for day in range(1, count + 2)]
        data = frame(rows); prev = data[data.date == pd.Timestamp("2026-01-01")]; current = data[data.date == pd.Timestamp("2026-01-02")]
        return shadow.build_stage_c_shadow(data, pd.Timestamp("2026-01-01") + pd.Timedelta(days=count + 2), pd.Timestamp("2026-01-01") + pd.Timedelta(days=count), shadow.compare_inventories(prev, current))

    def test_population_counts(self):
        r = self._stage(); z = r.z_audit.groupby("policy").population_n.first()
        self.assertEqual((z[shadow.POLICY1], z[shadow.POLICY2]), (3, 2))

    def test_candidate_counts(self):
        r = self._stage()
        self.assertEqual((len(r.policies[shadow.POLICY_BASELINE]["final"]), len(r.policies[shadow.POLICY1]["final"])), (3, 2))

    def test_metadata_is_nonformal(self):
        meta = self._stage().metadata
        self.assertEqual((meta["formal"], meta["forward_valid"], meta["shadow_only"], meta["target_actual_loaded"]), (False, False, True, False))


class StageDShadowTests(unittest.TestCase):
    def stage(self, future=False):
        rows = [("2026-01-01", 1, "OLD", 100, -1000), ("2026-01-02", 1, "OLD", 100, 2000),
            ("2026-01-03", 1, "マイジャグラーV", 100, 500),
            ("2026-01-01", 2, "新ハナビ", 100, 100), ("2026-01-02", 2, "新ハナビ", 100, 200),
            ("2026-01-03", 2, "新ハナビ", 100, 300),
            ("2026-01-01", 3, "モンキーターンV", 100, 900), ("2026-01-03", 3, "モンキーターンV", 100, -300),
            ("2026-01-01", 4, "アイムジャグラーEX", 100, -50), ("2026-01-03", 4, "アイムジャグラーEX", 100, 50)]
        if future:
            rows += [("2026-01-04", 2, "FUTUREジャグラー", 99, 999999), ("2026-01-05", 3, "新ハナビ", 99, -999999)]
        data = frame(rows); prev = data[data.date == pd.Timestamp("2026-01-02")]; current = data[data.date == pd.Timestamp("2026-01-03")]
        return shadow.build_stage_d_shadow(data, "2026-01-04", "2026-01-03", shadow.compare_inventories(prev, current))

    def test_a_type_production_classification_match(self):
        classify, _ = shadow._load_classifiers()
        r = self.stage().membership.query("category == 'A-TYPE' and policy == @shadow.POLICY_BASELINE").set_index("machine_no")
        for no, name in ((1, "マイジャグラーV"), (2, "新ハナビ"), (3, "モンキーターンV")):
            self.assertEqual(r.loc[no].classified_membership, classify(name)["is_a_type"])

    def test_juggler_production_classification_match(self):
        _, classify = shadow._load_classifiers()
        r = self.stage().membership.query("category == 'JUGGLER' and policy == @shadow.POLICY_BASELINE").set_index("machine_no")
        for no, name in ((1, "マイジャグラーV"), (2, "新ハナビ"), (4, "アイムジャグラーEX")):
            self.assertEqual(r.loc[no].classified_membership, classify(name))

    def test_a_type_score_equals_normal(self):
        self._assert_lineage("A-TYPE")

    def test_juggler_score_equals_normal(self):
        self._assert_lineage("JUGGLER")

    def _assert_lineage(self, category):
        rows = self.stage().lineage.query("category == @category")
        self.assertTrue((rows.category_score == rows.normal_score).all())

    def test_baseline_a_type_candidates(self):
        self.assertEqual(set(self.stage().categories["A-TYPE"][shadow.POLICY_BASELINE]["candidates"].machine_no), {1, 2, 4})

    def test_policy1_a_type_excludes_changed(self):
        self.assertNotIn(1, set(self.stage().categories["A-TYPE"][shadow.POLICY1]["candidates"].machine_no))

    def test_policy2_a_type_excludes_changed(self):
        self.assertNotIn(1, set(self.stage().categories["A-TYPE"][shadow.POLICY2]["candidates"].machine_no))

    def test_baseline_juggler_candidates(self):
        self.assertEqual(set(self.stage().categories["JUGGLER"][shadow.POLICY_BASELINE]["candidates"].machine_no), {1, 4})

    def test_policy1_juggler_excludes_changed(self):
        self.assertNotIn(1, set(self.stage().categories["JUGGLER"][shadow.POLICY1]["candidates"].machine_no))

    def test_policy2_juggler_excludes_changed(self):
        self.assertNotIn(1, set(self.stage().categories["JUGGLER"][shadow.POLICY2]["candidates"].machine_no))

    def test_a_type_rank_descending(self):
        x = self.stage().categories["A-TYPE"][shadow.POLICY_BASELINE]["candidates"]
        self.assertEqual(x.score.tolist(), sorted(x.score, reverse=True))

    def test_juggler_rank_descending(self):
        x = self.stage().categories["JUGGLER"][shadow.POLICY_BASELINE]["candidates"]
        self.assertEqual(x.score.tolist(), sorted(x.score, reverse=True))

    def test_membership_independent_from_eligibility(self):
        x = self.stage().membership.query("category == 'JUGGLER' and policy == @shadow.POLICY1 and machine_no == 1").iloc[0]
        self.assertTrue(x.classified_membership); self.assertFalse(x.ranking_eligible); self.assertFalse(x.final_category_candidate)

    def test_category_top10_comparisons(self):
        self.assertEqual(len(self.stage().comparisons), 6)

    def test_category_ripple_summary(self):
        self.assertEqual(len(self.stage().ripple), 4)

    def test_future_does_not_change_a_type(self):
        self._future_equal("A-TYPE")

    def test_future_does_not_change_juggler(self):
        self._future_equal("JUGGLER")

    def _future_equal(self, category):
        a, b = self.stage(), self.stage(True)
        for policy in (shadow.POLICY_BASELINE, shadow.POLICY1, shadow.POLICY2):
            left, right = a.categories[category][policy]["candidates"], b.categories[category][policy]["candidates"]
            self.assertTrue(left[["machine_no", "score", "category_rank"]].equals(right[["machine_no", "score", "category_rank"]]))

    def test_all_policy_category_lineage_exact(self):
        self.assertTrue((self.stage().lineage.category_score == self.stage().lineage.normal_score).all())

    def test_changed_classified_excluded_state(self):
        x = self.stage().membership.query("category == 'A-TYPE' and policy == @shadow.POLICY2 and machine_no == 1").iloc[0]
        self.assertEqual((x.changed, x.classified_membership, x.ranking_eligible), (True, True, False))

    def test_summary_separates_classified_and_candidate(self):
        x = self.stage().summaries.query("category == 'A-TYPE' and policy == @shadow.POLICY1").iloc[0]
        self.assertGreater(x.classified_count, x.candidate_count)

    def test_no_formal_metadata(self):
        r = self.stage()
        self.assertEqual((r.metadata["formal"], r.metadata["forward_valid"], r.metadata["shadow_only"]), (False, False, True))

    def test_target_actual_not_loaded(self):
        self.assertFalse(self.stage(True).metadata["target_actual_loaded"])

    def test_no_writer_or_production_execution_import(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("to_csv(", "to_json(", "subprocess", "run_pipeline"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
