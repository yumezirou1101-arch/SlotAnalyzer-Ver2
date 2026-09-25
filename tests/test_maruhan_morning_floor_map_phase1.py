from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"
for value in (str(ROOT), str(MACHINE)):
    if value not in sys.path:
        sys.path.insert(0, value)

import maruhan_morning_floor_map as floor_map


class MaruhanMorningFloorMapPhase1Tests(unittest.TestCase):
    def test_ranking_pairs_require_exact_top15_and_preserve_rank_machine_identity(self):
        rows = [
            {
                "prediction_rank": str(rank),
                "machine_no": str(700 + rank),
                "machine_name": f"TEST-{rank}",
                "score": str(80 - rank / 10),
            }
            for rank in range(1, 16)
        ]
        pairs = floor_map._ranking_pairs(rows, ("prediction_rank",))
        self.assertEqual(pairs[0], (1, 701))
        self.assertEqual(pairs[-1], (15, 715))
        self.assertEqual(len(pairs), 15)
        self.assertEqual(rows[0]["machine_name"], "TEST-1")
        self.assertEqual(rows[0]["score"], "79.9")

    def test_ranking_pairs_reject_duplicate_machine_or_incomplete_rank(self):
        rows = [
            {"prediction_rank": str(rank), "machine_no": str(700 + rank)}
            for rank in range(1, 16)
        ]
        rows[-1]["machine_no"] = rows[-2]["machine_no"]
        with self.assertRaises(floor_map.FloorMapValidationError):
            floor_map._ranking_pairs(rows, ("prediction_rank",))

        with self.assertRaises(floor_map.FloorMapValidationError):
            floor_map._ranking_pairs(rows[:14], ("prediction_rank",))

    def test_category_failure_is_isolated_and_successes_are_published(self):
        rankings = {
            category: (
                [
                    {
                        "prediction_rank": str(rank),
                        "machine_no": str(700 + rank),
                    }
                    for rank in range(1, 16)
                ],
                ("prediction_rank",),
            )
            for category in ("NORMAL", "A-TYPE", "JUGGLER")
        }

        base = {
            "seats": [],
            "font_path": Path("font.ttf"),
            "base_png": b"base",
            "base_rendering": {},
            "data_date": date(2026, 9, 24),
            "daily_path": Path("ana_slo_20260924.csv"),
            "layout_path": Path("layout.csv"),
            "source_sha256": "a" * 64,
            "layout_sha256": "b" * 64,
        }

        def overlay(_base_png, _seats, _pairs, category, _target, _font):
            if category == "A-TYPE":
                raise floor_map.FloorMapValidationError("intentional category failure")
            return (
                f"full-{category}".encode(),
                {
                    "width": 4030,
                    "height": 3610,
                    "marker_count": 15,
                    "marker_overlap_count": 0,
                    "marker_clipping_count": 0,
                    "markers": [],
                },
            )

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(floor_map, "_prepare_base", return_value=base), \
                mock.patch.object(floor_map, "_draw_category_overlay", side_effect=overlay), \
                mock.patch.object(
                    floor_map,
                    "_mail_variant",
                    side_effect=lambda payload, width: (b"mail-" + payload, width, 1433),
                ):
            output = Path(directory)
            results = floor_map.generate_floor_map_bundle(
                Path(directory),
                date(2026, 9, 25),
                rankings,
                output_dir=output,
            )

            self.assertEqual(results["NORMAL"].status, "OK")
            self.assertEqual(results["A-TYPE"].status, "UNAVAILABLE")
            self.assertEqual(results["JUGGLER"].status, "OK")
            self.assertIn("intentional category failure", results["A-TYPE"].error)
            self.assertTrue(results["NORMAL"].artifact.full_path.is_file())
            self.assertTrue(results["JUGGLER"].artifact.email_path.is_file())

            manifest = json.loads(
                (output / "maruhan_top15_floor_20260925_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["categories"]["NORMAL"]["status"], "OK")
            self.assertEqual(manifest["categories"]["A-TYPE"]["status"], "UNAVAILABLE")
            self.assertEqual(manifest["categories"]["JUGGLER"]["status"], "OK")

    def test_base_failure_returns_all_categories_unavailable_without_raising(self):
        rankings = {
            "NORMAL": ([{"prediction_rank": str(i), "machine_no": str(700 + i)} for i in range(1, 16)], ("prediction_rank",)),
            "A-TYPE": ([{"a_type_rank": str(i), "machine_no": str(800 + i)} for i in range(1, 16)], ("a_type_rank", "prediction_rank")),
            "JUGGLER": ([{"juggler_rank": str(i), "machine_no": str(900 + i)} for i in range(1, 16)], ("juggler_rank", "prediction_rank")),
        }
        with mock.patch.object(floor_map, "_prepare_base", side_effect=PermissionError("fixture")):
            results = floor_map.generate_floor_map_bundle(
                ROOT, date(2026, 9, 25), rankings
            )
        self.assertEqual(set(results), {"NORMAL", "A-TYPE", "JUGGLER"})
        self.assertTrue(all(result.status == "UNAVAILABLE" for result in results.values()))
        self.assertTrue(all("PermissionError" in result.error for result in results.values()))


if __name__ == "__main__":
    unittest.main()
