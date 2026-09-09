from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"
if str(MACHINE) not in sys.path:
    sys.path.insert(0, str(MACHINE))

import slotanalyzer_evaluation_quarantine as quarantine


class EvaluationQuarantineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {name: self.root / f"artifacts/{name}.csv" for name in ("prediction", "metadata", "source_all", "source_metadata")}
        for name, path in self.paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        self.registry = self.root / "config/evaluation_quarantine.json"
        self.registry.parent.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def entry(self, **overrides):
        value = {
            "status": "ACTIVE", "store": "MARUHAN_MAEBASHI", "target_date": "2026-09-09", "category": "NORMAL",
            "prediction_path": "artifacts/prediction.csv", "prediction_sha256": quarantine.sha256_file(self.paths["prediction"]),
            "metadata_path": "artifacts/metadata.csv", "metadata_sha256": quarantine.sha256_file(self.paths["metadata"]),
            "source_64_all514_sha256": quarantine.sha256_file(self.paths["source_all"]),
            "source_64_metadata_sha256": quarantine.sha256_file(self.paths["source_metadata"]),
            "reason_code": "INVENTORY_GUARD_INCIDENT", "reason": "test incident",
            "incident_date": "2026-09-07", "created_at_jst": "2026-09-09T12:00:00+09:00",
        }
        value.update(overrides)
        return value

    def write_registry(self, payload=None):
        payload = payload or {"schema_version": 1, "entries": [self.entry()]}
        self.registry.write_text(json.dumps(payload), encoding="utf-8")

    def assess(self, **overrides):
        args = dict(project_root=self.root, store="MARUHAN_MAEBASHI", target_date=date(2026, 9, 9), category="NORMAL",
                    prediction_path=self.paths["prediction"], metadata_path=self.paths["metadata"],
                    source_64_all514_path=self.paths["source_all"], source_64_metadata_path=self.paths["source_metadata"])
        args.update(overrides)
        return quarantine.assess_evaluation_quarantine(**args)

    def test_exact_identity_and_no_wildcard_matching(self):
        self.write_registry()
        result = self.assess()
        self.assertEqual(result.status, quarantine.STATUS_SKIPPED)
        self.assertFalse(result.evaluation_eligible)
        self.assertFalse(self.assess(target_date=date(2026, 9, 8)).quarantined)
        self.assertFalse(self.assess(category="A_TYPE").quarantined)

    def test_path_and_all_sha_mismatches_fail_closed(self):
        self.write_registry()
        for overrides in ({"prediction_path": self.paths["metadata"]}, {"metadata_path": self.paths["prediction"]}):
            with self.subTest(overrides=overrides):
                self.assertEqual(self.assess(**overrides).status, quarantine.STATUS_MISMATCH)
        for name, path in self.paths.items():
            with self.subTest(name=name):
                original = path.read_text(encoding="utf-8")
                path.write_text(original + "changed", encoding="utf-8")
                self.assertEqual(self.assess().status, quarantine.STATUS_MISMATCH)
                path.write_text(original, encoding="utf-8")

    def test_registry_errors_fail_closed(self):
        with self.assertRaises(quarantine.QuarantineConfigError):
            self.assess()
        self.registry.write_text("{bad", encoding="utf-8")
        with self.assertRaises(quarantine.QuarantineConfigError):
            self.assess()
        cases = [
            {"schema_version": 2, "entries": []},
            {"schema_version": 1, "entries": [self.entry(), self.entry()]},
            {"schema_version": 1, "entries": [self.entry(category="UNKNOWN")]},
            {"schema_version": 1, "entries": [self.entry(prediction_sha256="bad")]},
            {"schema_version": 1, "entries": [self.entry(prediction_path="../outside.csv")]},
            {"schema_version": 1, "entries": [self.entry(prediction_path="C:/outside.csv")]},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.write_registry(payload)
                with self.assertRaises(quarantine.QuarantineConfigError):
                    self.assess()

if __name__ == "__main__":
    unittest.main()
