from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"
if str(MACHINE) not in sys.path:
    sys.path.insert(0, str(MACHINE))

import slotanalyzer_derived_prediction_evidence as evidence


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, MACHINE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


atype = load_module("atype_formal_test", "ana_slo_prediction_v4_2_A_type_separated.py")
juggler = load_module("juggler_formal_test", "ana_slo_prediction_v4_2_Juggler_separated.py")
pipeline79 = load_module("pipeline79_formal_test", "ana_slo_prediction_v4_2_one_click_live_pipeline.py")

TARGET = date(2026, 9, 7)
LATEST = date(2026, 9, 6)
NOW = datetime(2026, 9, 7, 8, 0, tzinfo=ZoneInfo("Asia/Tokyo"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source64(root: Path, **overrides) -> Path:
    source_dir = root / "analysis/64_Ver4_2_future_top10"
    source_dir.mkdir(parents=True)
    ymd = TARGET.strftime("%Y%m%d")
    rows = []
    for number in range(1, 515):
        name = "マイジャグラーV" if number % 2 else "新ハナビ"
        rows.append({
            "machine_no": number, "machine_name": name,
            "score": 1000 - number, "prediction_rank": number,
            "tier": "PRIMARY" if number <= 5 else "NEXT",
            "target_date": TARGET.isoformat(), "latest_data_date": LATEST.isoformat(),
        })
    all_path = source_dir / f"64_prediction_{ymd}_all514.csv"
    top_path = source_dir / f"64_prediction_{ymd}_top10.csv"
    meta_path = source_dir / f"64_prediction_{ymd}_metadata.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(all_path, index=False, encoding="utf-8-sig")
    frame.head(10).to_csv(top_path, index=False, encoding="utf-8-sig")
    metadata = {
        "generated_at_jst": NOW.isoformat(), "target_date": TARGET.isoformat(),
        "latest_data_date": LATEST.isoformat(), "model": evidence.EXPECTED_MODEL,
        "weight_fingerprint": evidence.EXPECTED_WEIGHT_FINGERPRINT, "weight_sum": 1.0,
        "forward_guard_version": "1.0", "forward_valid": True,
        "forward_cutoff_jst": evidence.FORWARD_CUTOFF_TEXT,
        "target_actual_absent_at_generation": True,
        "target_source_absent_at_generation": True,
        "daily_csv_sha256": "d" * 64, "source_html_sha256": "s" * 64,
        "all514_sha256": sha(all_path), "top10_sha256": sha(top_path),
    }
    metadata.update(overrides)
    pd.DataFrame([metadata]).to_csv(meta_path, index=False, encoding="utf-8-sig")
    return source_dir


def configure(module, root: Path, source_dir: Path, kind: str):
    data_dir = root / "data"
    output = root / ("out74" if kind == "A_TYPE" else "out75")
    module.PROJECT_ROOT = root
    module.ANALYSIS_DIR = data_dir / "analysis_31days_deep"
    module.SOURCE_64_DIR = source_dir
    module.OUTPUT_DIR = output
    real = evidence.generation_preflight
    return output, mock.patch.object(
        module,
        "generation_preflight",
        side_effect=lambda project, data, out, src, target, derived_kind: real(
            project, data, out, src, target, derived_kind, NOW
        ),
    )


class DerivedPredictionFormalTests(unittest.TestCase):
    def generate_into(self, module, kind: str, root: Path, source: Path):
        output, preflight_patch = configure(module, root, source, kind)
        with preflight_patch, mock.patch.object(sys, "argv", ["script", "--target-date", TARGET.isoformat()]):
            module.main()
        return output

    def generate(self, module, kind: str):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        source = write_source64(root)
        output, preflight_patch = configure(module, root, source, kind)
        with preflight_patch, mock.patch.object(sys, "argv", ["script", "--target-date", TARGET.isoformat()]):
            module.main()
        return temp, root, source, output

    def assert_formal_metadata(self, output: Path, kind: str):
        verification = evidence.verify_derived_prediction(output, output.parent / "analysis/64_Ver4_2_future_top10", TARGET, kind)
        row = pd.read_csv(verification.metadata_path, encoding="utf-8-sig").iloc[0]
        self.assertEqual(row.schema_version, evidence.SCHEMA_VERSION)
        self.assertEqual(row.formalization_start_date, TARGET.isoformat())
        self.assertEqual(row.prediction_class, "FORWARD_VALID")
        self.assertEqual(row.derived_guard_version, evidence.GUARD_VERSION)
        self.assertEqual(row.target_date, TARGET.isoformat())
        self.assertEqual(row.latest_data_date, LATEST.isoformat())
        self.assertEqual(row.model, evidence.EXPECTED_MODEL)
        self.assertEqual(row.weight_fingerprint, evidence.EXPECTED_WEIGHT_FINGERPRINT)
        self.assertEqual(float(row.weight_sum), 1.0)
        self.assertIn("+09:00", row.generated_at_jst)
        self.assertEqual(row.source_64_all514_sha256, sha(output.parent / "analysis/64_Ver4_2_future_top10" / f"64_prediction_{TARGET:%Y%m%d}_all514.csv"))
        return verification, row

    def test_normal_atype_formal_generation_and_lineage(self):
        temp, _, _, output = self.generate(atype, "A_TYPE")
        try:
            verification, row = self.assert_formal_metadata(output, "A_TYPE")
            self.assertEqual(row.all_sha256, sha(verification.paths[0]))
            self.assertEqual(row.top10_sha256, sha(verification.paths[1]))
            self.assertEqual(row.candidate_review_sha256, sha(verification.paths[2]))
        finally:
            temp.cleanup()

    def test_normal_juggler_formal_generation_and_lineage(self):
        temp, _, _, output = self.generate(juggler, "JUGGLER")
        try:
            verification, row = self.assert_formal_metadata(output, "JUGGLER")
            self.assertEqual(row.all_sha256, sha(verification.paths[0]))
            self.assertEqual(row.top10_sha256, sha(verification.paths[1]))
        finally:
            temp.cleanup()

    def test_complete_frozen_sets_are_verified_without_overwrite(self):
        for module, kind in ((atype, "A_TYPE"), (juggler, "JUGGLER")):
            with self.subTest(kind=kind):
                temp, root, source, output = self.generate(module, kind)
                try:
                    before = {p: (sha(p), p.stat().st_mtime_ns) for p in evidence.derived_paths(output, TARGET, kind)}
                    _, patcher = configure(module, root, source, kind)
                    with patcher, mock.patch.object(sys, "argv", ["script", "--target-date", TARGET.isoformat()]):
                        module.main()
                    after = {p: (sha(p), p.stat().st_mtime_ns) for p in evidence.derived_paths(output, TARGET, kind)}
                    self.assertEqual(before, after)
                finally:
                    temp.cleanup()

    def test_partial_and_legacy_sets_fail_closed(self):
        for kind in ("A_TYPE", "JUGGLER"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                root = Path(td); source = write_source64(root); output = root / "out"
                paths = evidence.derived_paths(output, TARGET, kind); output.mkdir()
                paths[0].write_text("partial", encoding="utf-8")
                with self.assertRaises(evidence.PartialFrozenOutputError):
                    evidence.inspect_frozen_set(output, source, TARGET, kind)
            with self.subTest(kind=f"{kind}_legacy"), tempfile.TemporaryDirectory() as td:
                root = Path(td); source = write_source64(root); output = root / "out"; output.mkdir()
                paths = evidence.derived_paths(output, TARGET, kind)
                for path in paths:
                    path.write_text("x\n", encoding="utf-8")
                pd.DataFrame([{"target_date": TARGET.isoformat()}]).to_csv(paths[-1], index=False, encoding="utf-8-sig")
                with self.assertRaises(evidence.LegacyFrozenOutputError):
                    evidence.inspect_frozen_set(output, source, TARGET, kind)

    def test_cutoff_past_actual_source_and_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = write_source64(root); output = root / "out"; data = root / "data"; data.mkdir()
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                evidence.generation_preflight(root, data, output, source, TARGET, "A_TYPE", NOW.replace(hour=9))
            with self.assertRaisesRegex(RuntimeError, "past"):
                evidence.generation_preflight(root, data, output, source, TARGET, "A_TYPE", NOW.replace(day=8))
            (data / f"ana_slo_{TARGET:%Y%m%d}.csv").write_text("actual", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "actual"):
                evidence.generation_preflight(root, data, output, source, TARGET, "A_TYPE", NOW)
            (data / f"ana_slo_{TARGET:%Y%m%d}.csv").unlink()
            (root / f"ana_slo_{TARGET:%Y%m%d}_source.html").write_text("source", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source"):
                evidence.generation_preflight(root, data, output, source, TARGET, "JUGGLER", NOW)
            with self.assertRaisesRegex(evidence.DerivedPredictionError, "FORMALIZATION_BOUNDARY"):
                evidence.generation_preflight(root, data, output, source, date(2026, 9, 6), "A_TYPE", datetime(2026, 9, 6, 8, tzinfo=NOW.tzinfo))

    def test_invalid_source64_evidence_fails_closed(self):
        cases = {
            "forward_valid": False,
            "model": "OTHER",
            "weight_fingerprint": "bad",
            "weight_sum": 0.9,
            "latest_data_date": "2026-09-05",
        }
        for key, value in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as td:
                source = write_source64(Path(td), **{key: value})
                with self.assertRaises(evidence.DerivedPredictionError):
                    evidence.verify_source_64(source, TARGET)

    def test_missing_metadata_and_source64_hash_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            source = write_source64(Path(td))
            (source / f"64_prediction_{TARGET:%Y%m%d}_metadata.csv").unlink()
            with self.assertRaises(evidence.DerivedPredictionError):
                evidence.verify_source_64(source, TARGET)
        for filename in (f"64_prediction_{TARGET:%Y%m%d}_all514.csv", f"64_prediction_{TARGET:%Y%m%d}_top10.csv"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as td:
                source = write_source64(Path(td)); (source / filename).write_bytes((source / filename).read_bytes() + b"x")
                with self.assertRaisesRegex(evidence.DerivedPredictionError, "SHA-256"):
                    evidence.verify_source_64(source, TARGET)

    def test_derived_sha_and_metadata_mismatch_fail_closed(self):
        temp, _, source, output = self.generate(atype, "A_TYPE")
        try:
            paths = evidence.derived_paths(output, TARGET, "A_TYPE")
            paths[1].write_bytes(paths[1].read_bytes() + b"x")
            with self.assertRaisesRegex(evidence.DerivedPredictionError, "SHA-256"):
                evidence.verify_derived_prediction(output, source, TARGET, "A_TYPE")
        finally:
            temp.cleanup()
        temp, _, source, output = self.generate(juggler, "JUGGLER")
        try:
            meta = evidence.derived_paths(output, TARGET, "JUGGLER")[-1]
            frame = pd.read_csv(meta, encoding="utf-8-sig"); frame.loc[0, "prediction_class"] = "LEGACY_UNVERIFIED"; frame.to_csv(meta, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(evidence.DerivedPredictionError, "prediction_class"):
                evidence.verify_derived_prediction(output, source, TARGET, "JUGGLER")
        finally:
            temp.cleanup()

    def test_pipeline79_reuses_only_strict_verified_set(self):
        temp, _, source, output = self.generate(atype, "A_TYPE")
        try:
            with mock.patch.object(pipeline79, "DIR_64", source), mock.patch.object(pipeline79, "run_stage") as run:
                row = pipeline79.run_or_reuse_derived("74 A-TYPE", "74_A_TYPE", Path("unused.py"), output, pd.Timestamp(TARGET), "A_TYPE")
            run.assert_not_called()
            self.assertEqual(row["status"], "ALREADY_FROZEN")
            self.assertEqual(row["prediction_class"], "FORWARD_VALID")
            self.assertEqual(row["metadata_sha256"], sha(evidence.derived_paths(output, TARGET, "A_TYPE")[-1]))
        finally:
            temp.cleanup()

    def test_pipeline79_strict_reuse_and_downstream_stop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = write_source64(root)
            out74 = self.generate_into(atype, "A_TYPE", root, source)
            out75 = self.generate_into(juggler, "JUGGLER", root, source)
            out77 = root / "out77"; log_dir = root / "log79"

            def run77(stage_name, script_path, args):
                self.assertEqual(stage_name, "77 INTEGRATED")
                out77.mkdir(parents=True, exist_ok=True)
                (out77 / f"77_integrated_prediction_{TARGET:%Y%m%d}.csv").write_text("x\n", encoding="utf-8")
                (out77 / f"77_integrated_prediction_{TARGET:%Y%m%d}_summary.csv").write_text("x\n", encoding="utf-8")
                return 0, 0.1

            patches = (
                mock.patch.object(pipeline79, "PROJECT_ROOT", root),
                mock.patch.object(pipeline79, "ANALYSIS_DIR", root / "analysis"),
                mock.patch.object(pipeline79, "DIR_64", source),
                mock.patch.object(pipeline79, "DIR_74", out74),
                mock.patch.object(pipeline79, "DIR_75", out75),
                mock.patch.object(pipeline79, "DIR_77", out77),
                mock.patch.object(pipeline79, "LOG_DIR", log_dir),
                mock.patch.object(pipeline79, "run_stage", side_effect=run77),
                mock.patch.object(sys, "argv", ["script", "--target-date", TARGET.isoformat()]),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
                pipeline79.main()
            status = pd.read_csv(log_dir / f"79_pipeline_{TARGET:%Y%m%d}_status.csv", encoding="utf-8-sig")
            mapped = status.set_index("stage")
            self.assertEqual(mapped.loc["64_NORMAL", "status"], "ALREADY_FROZEN")
            self.assertEqual(mapped.loc["74_A_TYPE", "status"], "ALREADY_FROZEN")
            self.assertEqual(mapped.loc["75_JUGGLER", "status"], "ALREADY_FROZEN")
            self.assertEqual(mapped.loc["74_A_TYPE", "prediction_class"], "FORWARD_VALID")
            self.assertTrue(str(mapped.loc["75_JUGGLER", "metadata_sha256"]))

            for kind, output, index in (("A_TYPE", out74, 1), ("JUGGLER", out75, 1)):
                with self.subTest(kind=f"{kind}_mismatch_stops"):
                    artifact = evidence.derived_paths(output, TARGET, kind)[index]
                    original = artifact.read_bytes(); artifact.write_bytes(original + b"x")
                    with mock.patch.object(pipeline79, "DIR_64", source), mock.patch.object(pipeline79, "run_stage") as run:
                        with self.assertRaises(evidence.DerivedPredictionError):
                            pipeline79.run_or_reuse_derived(kind, kind, Path("unused"), output, pd.Timestamp(TARGET), kind)
                    run.assert_not_called()
                    artifact.write_bytes(original)

    def test_no_allow_gap_interface_in_derived_generators(self):
        for path in (MACHINE / "ana_slo_prediction_v4_2_A_type_separated.py", MACHINE / "ana_slo_prediction_v4_2_Juggler_separated.py"):
            self.assertNotIn("--allow-gap", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
