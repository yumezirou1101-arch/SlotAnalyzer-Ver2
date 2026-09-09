from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"
if str(MACHINE) not in sys.path:
    sys.path.insert(0, str(MACHINE))

import ana_slo_prediction_v4_2_live_prediction_backtest as normal69
import ana_slo_prediction_v4_2_normal_atype_juggler_live_evaluation as legacy76
from slotanalyzer_evaluation_quarantine import QuarantineDecision, STATUS_SKIPPED


SKIPPED = QuarantineDecision(STATUS_SKIPPED, False, True, "INVENTORY_GUARD_INCIDENT", "test")


class EvaluationQuarantineRoutingTests(unittest.TestCase):
    def test_69_quarantine_status_never_enters_aggregate_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_dir = root / "prediction"
            output = root / "output"
            prediction_dir.mkdir()
            prediction = prediction_dir / "64_prediction_20260909_top10.csv"
            prediction.write_text("fixture", encoding="utf-8")
            metadata_path = prediction_dir / "64_prediction_20260909_metadata.csv"
            metadata = {
                "metadata_exists": True, "metadata_model_ok": True,
                "metadata_fingerprint_ok": True, "metadata_dates_ok": True,
                "prediction_class": "FORWARD_VALID", "forward_guard_fail_reasons": "",
                "metadata_path": str(metadata_path), "metadata_model": "CHAMPION_V4.2_C",
                "metadata_weight_fingerprint": "a1eaf45d71ded209",
            }
            with mock.patch.multiple(
                normal69, PROJECT_ROOT=root, DATA_DIR=root / "data",
                PREDICTION_DIR=prediction_dir, OUTPUT_DIR=output,
            ), mock.patch.object(normal69, "discover_prediction_files", return_value=[(pd.Timestamp("2026-09-09"), prediction)]), \
                    mock.patch.object(normal69, "discover_actual_files", return_value={}), \
                    mock.patch.object(normal69, "load_prediction", return_value=(pd.DataFrame(), {"prediction_basic_ok": True, "prediction_sha256": "a" * 64})), \
                    mock.patch.object(normal69, "load_metadata_for_target", return_value=metadata), \
                    mock.patch.object(normal69, "assess_evaluation_quarantine", return_value=SKIPPED), \
                    redirect_stdout(StringIO()):
                normal69.main()
            status = pd.read_csv(output / "69_live_prediction_status.csv", encoding="utf-8-sig")
            summary = pd.read_csv(output / "69_forward_summary.csv", encoding="utf-8-sig")
        self.assertEqual(status.loc[0, "status"], STATUS_SKIPPED)
        self.assertEqual(status.loc[0, "prediction_class"], "FORWARD_VALID")
        self.assertFalse(bool(status.loc[0, "evaluation_eligible"]))
        self.assertEqual(int(summary.loc[0, "forward_valid_evaluated_days"]), 0)
        self.assertEqual(str(summary.loc[0, "next_checkpoint"]), "10")

    def test_legacy_76_excludes_all_three_categories_before_evaluate_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            target = pd.Timestamp("2026-09-09")
            predictions = [(category, target, root / f"{category}.csv") for category in ("NORMAL", "A_TYPE", "JUGGLER")]
            with mock.patch.multiple(legacy76, PROJECT_ROOT=root, OUTPUT_DIR=output), \
                    mock.patch.object(legacy76, "discover_prediction_files", return_value=predictions), \
                    mock.patch.object(legacy76, "discover_actual_files", return_value={target: root / "actual.csv"}), \
                    mock.patch.object(legacy76, "assess_evaluation_quarantine", return_value=SKIPPED), \
                    mock.patch.object(legacy76, "evaluate_one") as evaluate_one, \
                    mock.patch.object(legacy76, "evaluate_formal_predictions"), \
                    redirect_stdout(StringIO()):
                legacy76.main()
            status = pd.read_csv(output / "76_status.csv", encoding="utf-8-sig")
        evaluate_one.assert_not_called()
        self.assertEqual(set(status["prediction_type"]), {"NORMAL", "A_TYPE", "JUGGLER"})
        self.assertEqual(set(status["status"]), {STATUS_SKIPPED})


if __name__ == "__main__":
    unittest.main()
