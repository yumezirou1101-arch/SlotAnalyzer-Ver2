from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

from tests import test_derived_prediction_formal as phase2a


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"
if str(MACHINE) not in sys.path:
    sys.path.insert(0, str(MACHINE))

import slotanalyzer_derived_prediction_evaluation as evaluation
import slotanalyzer_morning_notification as notification
import slotanalyzer_morning_automation_support as support


TARGET = phase2a.TARGET
OPERATION_DATE = date(2026, 9, 8)


class DerivedPredictionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.analysis = self.root / "analysis"
        self.source = phase2a.write_source64(self.root)
        self.data = self.root / "data"
        self.data.mkdir()
        self.output = self.analysis / "76_Normal_AType_Juggler_live_evaluation"

    def tearDown(self):
        self.temp.cleanup()

    def _generate(self, module, kind: str) -> Path:
        directory = self.analysis / evaluation.CATEGORIES[kind]["directory"]
        module.PROJECT_ROOT = self.root
        module.ANALYSIS_DIR = self.analysis
        module.SOURCE_64_DIR = self.source
        module.OUTPUT_DIR = directory
        real = phase2a.evidence.generation_preflight
        with mock.patch.object(
            module, "generation_preflight",
            side_effect=lambda project, data, out, src, target, derived_kind: real(
                project, data, out, src, target, derived_kind, phase2a.NOW
            ),
        ), mock.patch.object(sys, "argv", ["script", "--target-date", TARGET.isoformat()]):
            module.main()
        return directory

    def _write_status79(self, directories: dict[str, Path], sha_override: dict[str, str] | None = None):
        rows = []
        for kind, directory in directories.items():
            verification = phase2a.evidence.verify_derived_prediction(directory, self.source, TARGET, kind)
            rows.append({
                "stage": evaluation.CATEGORIES[kind]["stage"],
                "pipeline_complete": True,
                "prediction_class": "FORWARD_VALID",
                "metadata_sha256": (sha_override or {}).get(kind, verification.metadata_sha256),
            })
        path = self.analysis / "79_one_click_prediction_pipeline" / f"79_pipeline_{TARGET:%Y%m%d}_status.csv"
        path.parent.mkdir()
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")

    def _write_actual(self, count: int = 514, duplicate: bool = False, missing_diff: bool = False):
        rows = []
        for number in range(1, count + 1):
            rows.append({
                "date": TARGET.isoformat(), "machine_no": number,
                "machine_name": "actual machine", "diff": number * 100 - 1000,
            })
        if duplicate and len(rows) >= 2:
            rows[-1]["machine_no"] = rows[-2]["machine_no"]
        if missing_diff:
            rows[0]["diff"] = None
        path = self.data / f"ana_slo_{TARGET:%Y%m%d}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def _formal_fixture(self, actual: bool = True):
        directories = {
            "A_TYPE": self._generate(phase2a.atype, "A_TYPE"),
            "JUGGLER": self._generate(phase2a.juggler, "JUGGLER"),
        }
        self._write_status79(directories)
        if actual:
            self._write_actual()
        return directories

    def _evaluate(self):
        return evaluation.evaluate_formal_predictions(
            self.root, self.data, self.analysis, self.output
        )

    def test_atype_and_juggler_formal_evaluation_outputs_and_summaries(self):
        self._formal_fixture()
        status, detail, daily, coverage = self._evaluate()
        self.assertEqual(set(status["status"]), {"EVALUATED_FORWARD_VALID"})
        self.assertEqual(set(status["prediction_class"]), {"FORWARD_VALID"})
        self.assertEqual(len(detail), 20)
        self.assertEqual(set(daily["band"]), {"TOP3", "TOP5", "TOP10"})
        self.assertEqual(len(daily), 6)
        self.assertTrue(coverage["formal_evaluation_complete"].all())
        for category in ("A_TYPE", "JUGGLER"):
            subset = detail[detail["category"] == category]
            self.assertEqual(set(subset["rank"]), set(range(1, 11)))
            status_row = status[status["category"] == category].iloc[0]
            self.assertEqual(set(subset["prediction_sha256"]), {status_row.prediction_sha256})
            self.assertEqual(set(subset["metadata_sha256"]), {status_row.metadata_sha256})
        for filename in (
            "76_formal_status.csv", "76_formal_detail.csv",
            "76_formal_daily.csv", "76_formal_coverage.csv",
        ):
            self.assertTrue((self.output / filename).is_file())

    def test_formal_prediction_without_actual_is_pending(self):
        self._formal_fixture(actual=False)
        status, detail, daily, coverage = self._evaluate()
        self.assertEqual(set(status["status"]), {"PENDING_FORWARD_VALID"})
        self.assertTrue(detail.empty)
        self.assertTrue(daily.empty)
        self.assertFalse(coverage["formal_evaluation_complete"].any())

    def test_legacy_and_pre_boundary_predictions_never_become_formal(self):
        legacy_date = date(2026, 9, 6)
        directory = self.analysis / evaluation.CATEGORIES["A_TYPE"]["directory"]
        directory.mkdir(parents=True)
        rows = [{
            "machine_no": number, "machine_name": "legacy", "score": 100 - number,
            "a_type_rank": number, "target_date": legacy_date.isoformat(),
            "latest_data_date": "2026-09-05",
        } for number in range(1, 11)]
        pd.DataFrame(rows).to_csv(
            directory / "74_A_type_prediction_20260906_top10.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame([{
            "date": legacy_date.isoformat(), "machine_no": number,
            "machine_name": "legacy", "diff": number,
        } for number in range(1, 515)]).to_csv(
            self.data / "ana_slo_20260906.csv", index=False, encoding="utf-8-sig"
        )
        status, detail, _, _ = self._evaluate()
        self.assertEqual(status.iloc[0].prediction_class, "LEGACY_UNVERIFIED")
        self.assertEqual(status.iloc[0].status, "EVALUATED_LEGACY_UNVERIFIED")
        self.assertTrue(detail.empty)

    def test_metadata_and_prediction_sha_fail_closed(self):
        for failure in ("metadata79", "prediction"):
            with self.subTest(failure=failure):
                self.tearDown(); self.setUp()
                directories = {
                    "A_TYPE": self._generate(phase2a.atype, "A_TYPE"),
                    "JUGGLER": self._generate(phase2a.juggler, "JUGGLER"),
                }
                self._write_status79(directories, {"A_TYPE": "0" * 64} if failure == "metadata79" else None)
                if failure == "prediction":
                    top = phase2a.evidence.derived_paths(directories["A_TYPE"], TARGET, "A_TYPE")[1]
                    frame = pd.read_csv(top, encoding="utf-8-sig")
                    frame.loc[0, "score"] = int(frame.loc[0, "score"]) + 1
                    frame.to_csv(top, index=False, encoding="utf-8-sig")
                self._write_actual()
                status, _, _, coverage = self._evaluate()
                row = status[status["category"] == "A_TYPE"].iloc[0]
                self.assertEqual(row.prediction_class, "FORWARD_GUARD_FAIL")
                self.assertEqual(row.status, "SKIPPED_FORWARD_GUARD_FAIL")
                self.assertFalse(bool(coverage[coverage["category"] == "A_TYPE"].iloc[0].formal_evaluation_complete))

    def test_unsupported_formal_schema_is_metadata_check_fail(self):
        directories = self._formal_fixture()
        metadata_path = phase2a.evidence.derived_paths(
            directories["A_TYPE"], TARGET, "A_TYPE"
        )[-1]
        metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        metadata.loc[0, "schema_version"] = "UNSUPPORTED"
        metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
        status, _, _, _ = self._evaluate()
        row = status[status["category"] == "A_TYPE"].iloc[0]
        self.assertEqual(row.prediction_class, "FORWARD_GUARD_FAIL")
        self.assertEqual(row.status, "SKIPPED_METADATA_CHECK_FAIL")

    def test_prediction_rank_and_machine_quality_failures(self):
        for failure in ("missing_rank", "duplicate_rank", "duplicate_machine"):
            with self.subTest(failure=failure):
                self.tearDown(); self.setUp()
                directories = self._formal_fixture(actual=True)
                top = phase2a.evidence.derived_paths(directories["JUGGLER"], TARGET, "JUGGLER")[1]
                frame = pd.read_csv(top, encoding="utf-8-sig")
                if failure == "missing_rank":
                    frame = frame.iloc[:-1]
                elif failure == "duplicate_rank":
                    frame.loc[9, "juggler_rank"] = 9
                else:
                    frame.loc[9, "machine_no"] = frame.loc[8, "machine_no"]
                frame.to_csv(top, index=False, encoding="utf-8-sig")
                status, _, _, _ = self._evaluate()
                row = status[status["category"] == "JUGGLER"].iloc[0]
                self.assertEqual(row.status, "SKIPPED_PREDICTION_QUALITY_FAIL")

    def test_actual_quality_failures(self):
        for failure in ("short", "duplicate", "missing_diff", "top10_missing"):
            with self.subTest(failure=failure):
                self.tearDown(); self.setUp()
                self._formal_fixture(actual=False)
                if failure == "short":
                    self._write_actual(count=513)
                elif failure == "duplicate":
                    self._write_actual(duplicate=True)
                elif failure == "missing_diff":
                    self._write_actual(missing_diff=True)
                else:
                    path = self._write_actual()
                    frame = pd.read_csv(path, encoding="utf-8-sig")
                    frame.loc[0, "machine_no"] = 9999
                    frame.to_csv(path, index=False, encoding="utf-8-sig")
                status, _, _, _ = self._evaluate()
                if failure == "top10_missing":
                    self.assertIn("SKIPPED_ACTUAL_QUALITY_FAIL", set(status["status"]))
                else:
                    self.assertEqual(set(status["status"]), {"SKIPPED_ACTUAL_QUALITY_FAIL"})

    def test_gmail_loaders_render_only_exact_formal_d_minus_one(self):
        directories = self._formal_fixture()
        self._evaluate()
        production_analysis = self.root / "data/maruhan_maebashi/machine_number/analysis_31days_deep"
        production_analysis.parent.mkdir(parents=True)
        # The loader uses production-relative paths; point its strict dependencies at that tree.
        import shutil
        shutil.copytree(self.analysis, production_analysis, dirs_exist_ok=True)
        production_data = self.root / "data/maruhan_maebashi/machine_number"
        shutil.copy2(self.data / f"ana_slo_{TARGET:%Y%m%d}.csv", production_data)
        status_path = production_analysis / "76_Normal_AType_Juggler_live_evaluation/76_formal_status.csv"
        status = pd.read_csv(status_path, encoding="utf-8-sig")
        for index, row in status.iterrows():
            category = row["category"]
            spec = evaluation.CATEGORIES[category]
            status.loc[index, "prediction_path"] = str(
                production_analysis / spec["directory"] /
                ("74_A_type_prediction_20260907_top10.csv" if category == "A_TYPE" else "75_Juggler_prediction_20260907_top10.csv")
            )
            status.loc[index, "metadata_path"] = str(
                production_analysis / spec["directory"] / spec["metadata"].format(ymd="20260907")
            )
            status.loc[index, "actual_path"] = str(production_data / "ana_slo_20260907.csv")
        status.to_csv(status_path, index=False, encoding="utf-8-sig")
        result = notification._load_yesterday_derived_evaluation(
            self.root, OPERATION_DATE, "A_TYPE"
        )
        self.assertTrue(result.formal, result.message)
        self.assertEqual(len(result.detail_rows), 10)
        self.assertEqual(len(result.summary_rows), 3)
        plain = notification._render_yesterday_derived_plain(result)
        html = notification._render_yesterday_derived_html(result)
        self.assertIn("前日 A-TYPE 正式結果", plain)
        self.assertIn("LOSE", plain)
        self.assertIn("Score", html)
        self.assertIn("TOP10", html)
        coverage_path = production_analysis / "76_Normal_AType_Juggler_live_evaluation/76_formal_coverage.csv"
        coverage = pd.read_csv(coverage_path, encoding="utf-8-sig")
        coverage.loc[coverage["category"] == "A_TYPE", "formal_evaluation_complete"] = False
        coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
        incomplete = notification._load_yesterday_derived_evaluation(
            self.root, OPERATION_DATE, "A_TYPE"
        )
        self.assertFalse(incomplete.formal)
        wrong_day = notification._load_yesterday_derived_evaluation(
            self.root, date(2026, 9, 9), "A_TYPE"
        )
        self.assertFalse(wrong_day.formal)

    def test_morning_completion_requires_both_strict_derived_sets(self):
        operation = TARGET
        analysis = self.root / "data/maruhan_maebashi/machine_number/analysis_31days_deep"
        dir64 = analysis / "64_Ver4_2_future_top10"
        dir77 = analysis / "77_live_integrated_prediction_report"
        dir79 = analysis / "79_one_click_prediction_pipeline"
        for path in (
            dir64 / "64_prediction_20260907_all514.csv",
            dir64 / "64_prediction_20260907_top10.csv",
            dir64 / "64_prediction_20260907_metadata.csv",
            dir77 / "77_integrated_prediction_20260907.csv",
            dir77 / "77_integrated_prediction_20260907_summary.csv",
            dir79 / "79_pipeline_20260907_status.csv",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x", encoding="utf-8")
        metadata = pd.DataFrame([{
            "generated_at_jst": "2026-09-07T08:00:00+09:00",
            "target_date": "2026-09-07", "latest_data_date": "2026-09-06",
            "forward_guard_version": "1", "forward_valid": True,
            "forward_cutoff_jst": "09:00 Asia/Tokyo",
            "target_actual_absent_at_generation": True,
            "target_source_absent_at_generation": True,
            "daily_csv_sha256": "h", "source_html_sha256": "h",
            "all514_sha256": "h", "top10_sha256": "h", "machines_ranked": 514,
            "target_actual_used": False, "model": "CHAMPION_V4.2_C",
            "weight_fingerprint": "a1eaf45d71ded209", "weight_sum": 1.0,
        }])
        status79 = pd.DataFrame({
            "stage": ["64_NORMAL", "74_A_TYPE", "75_JUGGLER", "77_INTEGRATED"],
            "pipeline_complete": [True, True, True, True],
        })

        def read_csv(path, *args, **kwargs):
            return status79 if "79_pipeline" in str(path) else metadata

        verified = mock.Mock(paths=(Path("derived"),))
        with mock.patch.object(support.pd, "read_csv", side_effect=read_csv), \
             mock.patch.object(support, "sha256_file", return_value="h"), \
             mock.patch.object(support, "_nonempty", return_value=True), \
             mock.patch.object(support, "verify_derived_prediction", return_value=verified) as strict:
            result = support.verify_maruhan_completion(self.root, operation)
        self.assertTrue(result.ok)
        self.assertEqual([call.args[3] for call in strict.call_args_list], ["A_TYPE", "JUGGLER"])

        with mock.patch.object(support.pd, "read_csv", side_effect=read_csv), \
             mock.patch.object(support, "sha256_file", return_value="h"), \
             mock.patch.object(support, "_nonempty", return_value=True), \
             mock.patch.object(support, "verify_derived_prediction", side_effect=RuntimeError("SHA mismatch")):
            result = support.verify_maruhan_completion(self.root, operation)
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "INVALID_DERIVED_PREDICTION")


if __name__ == "__main__":
    unittest.main()
