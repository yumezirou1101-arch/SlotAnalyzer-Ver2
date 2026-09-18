from __future__ import annotations

import sys
import tempfile
import unittest

from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MACHINE_DIR = ROOT / "machine_number"

if str(MACHINE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MACHINE_DIR),
    )


import slotanalyzer_morning_automation_support as support


class BigMarchStaleReferenceSupportTests(
    unittest.TestCase
):
    def test_build_stale_reference_command(
        self,
    ):
        root = Path(
            r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
        )

        command = (
            support.build_big_march_stale_reference_command(
                root,
                "python",
                date(2026, 9, 18),
            )
        )

        self.assertEqual(
            command[0],
            "python",
        )

        self.assertTrue(
            command[1].endswith(
                "ana_slo_bigmarch_oyagi_"
                "stale_reference_future_ranking.py"
            )
        )

        self.assertIn(
            "--operation-date",
            command,
        )

        self.assertIn(
            "2026-09-18",
            command,
        )

        self.assertNotIn(
            "--allow-gap",
            command,
        )

        self.assertNotIn(
            "--overwrite",
            command,
        )

    def test_verify_stale_reference_completion(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            operation_date = date(
                2026,
                9,
                18,
            )

            compact = (
                operation_date.strftime(
                    "%Y%m%d"
                )
            )

            output_dir = (
                root
                / "data"
                / "bigmarch_takasaki_oyagi"
                / "machine_number"
                / "analysis_31days_deep"
                / "91_stale_reference_future_ranking"
                / compact
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            prefix = (
                f"91_stale_reference_{compact}"
            )

            common = {
                "target_date": [
                    operation_date.isoformat()
                ],
                "expected_data_date": [
                    "2026-09-17"
                ],
                "latest_data_date": [
                    "2026-09-15"
                ],
                "source_delay_days": [2],
                "source_delay_bucket": [
                    "LAG_2"
                ],
                "ranking_class": [
                    "STALE_REFERENCE"
                ],
                "formal": [False],
                "provisional": [False],
                "stale_reference": [True],
                "forward_valid": [False],
                "inventory_currentness": [
                    "UNCONFIRMED"
                ],
                "machine_mapping": [
                    "LAST_KNOWN"
                ],
            }

            ranking = pd.DataFrame(
                {
                    **common,
                    "rank": [1],
                    "machine_no": [101],
                    "machine_name": [
                        "dummy-machine"
                    ],
                    "score": [10.0],
                }
            )

            for suffix in (
                "juggler_all",
                "juggler_top10",
                "nonjuggler_all",
                "nonjuggler_top10",
            ):
                ranking.to_csv(
                    output_dir
                    / f"{prefix}_{suffix}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            metadata = pd.DataFrame(
                [
                    {
                        "ranking_class": (
                            "STALE_REFERENCE"
                        ),
                        "formal": False,
                        "provisional": False,
                        "stale_reference": True,
                        "forward_valid": False,
                        "operation_date": (
                            "2026-09-18"
                        ),
                        "target_date": (
                            "2026-09-18"
                        ),
                        "expected_data_date": (
                            "2026-09-17"
                        ),
                        "latest_data_date": (
                            "2026-09-15"
                        ),
                        "source_delay_days": 2,
                        "source_delay_bucket": (
                            "LAG_2"
                        ),
                        "target_to_latest_gap_days": 3,
                        "source_status": (
                            "EXPECTED_DATE_MISSING_STALE"
                        ),
                        "inventory_currentness": (
                            "UNCONFIRMED"
                        ),
                        "machine_mapping": (
                            "LAST_KNOWN"
                        ),
                        "model": (
                            "TEST_J|TEST_N"
                        ),
                        "automatic_promotion": False,
                        "eligible_for_formal_evaluation": False,
                    }
                ]
            )

            metadata.to_csv(
                output_dir
                / f"{prefix}_metadata.csv",
                index=False,
                encoding="utf-8-sig",
            )

            status = metadata.copy()
            status["status"] = (
                "STALE_REFERENCE"
            )

            status.to_csv(
                output_dir
                / f"{prefix}_status.csv",
                index=False,
                encoding="utf-8-sig",
            )

            result = (
                support.verify_big_march_stale_reference_completion(
                    root,
                    operation_date,
                )
            )

        self.assertTrue(result.ok)

        self.assertEqual(
            result.status,
            "STALE_REFERENCE",
        )

        self.assertEqual(
            result.details[
                "latest_data_date"
            ],
            "2026-09-15",
        )

        self.assertEqual(
            result.details[
                "source_delay_days"
            ],
            2,
        )

        self.assertEqual(
            result.details[
                "source_delay_bucket"
            ],
            "LAG_2",
        )

    def test_formal_flag_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            operation_date = date(
                2026,
                9,
                18,
            )

            compact = (
                operation_date.strftime(
                    "%Y%m%d"
                )
            )

            output_dir = (
                root
                / "data"
                / "bigmarch_takasaki_oyagi"
                / "machine_number"
                / "analysis_31days_deep"
                / "91_stale_reference_future_ranking"
                / compact
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            prefix = (
                f"91_stale_reference_{compact}"
            )

            frame = pd.DataFrame(
                {
                    "target_date": [
                        "2026-09-18"
                    ],
                    "expected_data_date": [
                        "2026-09-17"
                    ],
                    "latest_data_date": [
                        "2026-09-15"
                    ],
                    "source_delay_days": [2],
                    "source_delay_bucket": [
                        "LAG_2"
                    ],
                    "ranking_class": [
                        "STALE_REFERENCE"
                    ],
                    "formal": [True],
                    "provisional": [False],
                    "stale_reference": [True],
                    "forward_valid": [False],
                    "inventory_currentness": [
                        "UNCONFIRMED"
                    ],
                    "machine_mapping": [
                        "LAST_KNOWN"
                    ],
                }
            )

            for suffix in (
                "juggler_all",
                "juggler_top10",
                "nonjuggler_all",
                "nonjuggler_top10",
            ):
                frame.to_csv(
                    output_dir
                    / f"{prefix}_{suffix}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            metadata = pd.DataFrame(
                [
                    {
                        "target_date": (
                            "2026-09-18"
                        ),
                        "expected_data_date": (
                            "2026-09-17"
                        ),
                        "latest_data_date": (
                            "2026-09-15"
                        ),
                        "source_delay_days": 2,
                        "source_delay_bucket": (
                            "LAG_2"
                        ),
                        "ranking_class": (
                            "STALE_REFERENCE"
                        ),
                        "formal": True,
                        "provisional": False,
                        "stale_reference": True,
                        "forward_valid": False,
                        "inventory_currentness": (
                            "UNCONFIRMED"
                        ),
                        "machine_mapping": (
                            "LAST_KNOWN"
                        ),
                        "eligible_for_formal_evaluation": False,
                        "source_status": (
                            "EXPECTED_DATE_MISSING_STALE"
                        ),
                        "target_to_latest_gap_days": 3,
                    }
                ]
            )

            metadata.to_csv(
                output_dir
                / f"{prefix}_metadata.csv",
                index=False,
                encoding="utf-8-sig",
            )

            status = metadata.copy()
            status["status"] = (
                "STALE_REFERENCE"
            )

            status.to_csv(
                output_dir
                / f"{prefix}_status.csv",
                index=False,
                encoding="utf-8-sig",
            )

            result = (
                support.verify_big_march_stale_reference_completion(
                    root,
                    operation_date,
                )
            )

        self.assertFalse(result.ok)

        self.assertEqual(
            result.status,
            "INVALID_STALE_REFERENCE",
        )

    def test_old_operation_date_output_is_not_reused(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            old_date = date(
                2026,
                9,
                17,
            )

            old_compact = (
                old_date.strftime(
                    "%Y%m%d"
                )
            )

            old_dir = (
                root
                / "data"
                / "bigmarch_takasaki_oyagi"
                / "machine_number"
                / "analysis_31days_deep"
                / "91_stale_reference_future_ranking"
                / old_compact
            )

            old_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            old_prefix = (
                f"91_stale_reference_{old_compact}"
            )

            (
                old_dir
                / f"{old_prefix}_status.csv"
            ).write_text(
                "status\nSTALE_REFERENCE\n",
                encoding="utf-8-sig",
            )

            result = (
                support.verify_big_march_stale_reference_completion(
                    root,
                    date(2026, 9, 18),
                )
            )

        self.assertFalse(result.ok)

        self.assertEqual(
            result.status,
            "NONE",
        )


if __name__ == "__main__":
    unittest.main()