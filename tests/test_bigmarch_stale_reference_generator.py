from __future__ import annotations

import sys
import tempfile
import unittest

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MACHINE_DIR = ROOT / "machine_number"

if str(MACHINE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MACHINE_DIR),
    )


import ana_slo_bigmarch_oyagi_stale_reference_future_ranking as stale


def write_daily(
    data_dir: Path,
    daily_date: date,
    *,
    machine_count: int = 200,
) -> Path:
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(
        {
            "date": [
                daily_date.isoformat()
            ]
            * machine_count,
            "machine_name": [
                f"machine-{number}"
                for number
                in range(
                    1,
                    machine_count + 1,
                )
            ],
            "machine_no": list(
                range(
                    1,
                    machine_count + 1,
                )
            ),
            "G": [5000] * machine_count,
            "diff": [0] * machine_count,
        }
    )

    path = (
        data_dir
        / (
            "ana_slo_bigmarch_oyagi_"
            f"{daily_date:%Y%m%d}"
            ".csv"
        )
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


def make_dummy_module(
    latest_date: date,
    model_name: str,
):
    history = pd.DataFrame(
        {
            "date": [
                pd.Timestamp(
                    latest_date
                )
            ],
            "machine_no": [1],
            "machine_name": [
                "dummy-machine"
            ],
            "G": [5000],
            "diff": [100],
        }
    )

    def load_frozen_history():
        return (
            history.copy(),
            None,
            None,
        )

    def build_future_ranking(
        history_frame,
        latest,
        target,
    ):
        return pd.DataFrame(
            {
                "rank": [1, 2],
                "machine_no": [1, 2],
                "machine_name": [
                    "dummy-machine-1",
                    "dummy-machine-2",
                ],
                "score": [
                    10.0,
                    9.0,
                ],
            }
        )

    return SimpleNamespace(
        MODEL_NAME=model_name,
        load_frozen_history=(
            load_frozen_history
        ),
        build_future_ranking=(
            build_future_ranking
        ),
    )


def allowed_guard_decision():
    return SimpleNamespace(
        reference_allowed=True,
    )


class BigMarchStaleReferenceGeneratorTests(
    unittest.TestCase
):
    def test_lag_two_is_eligible(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            data_dir = (
                root
                / stale.DATA_REL
            )

            write_daily(
                data_dir,
                date(2026, 9, 2),
            )

            result = (
                stale.assess_eligibility(
                    root,
                    date(2026, 9, 5),
                )
            )

        self.assertEqual(
            result.expected_data_date,
            date(2026, 9, 4),
        )

        self.assertEqual(
            result.latest_data_date,
            date(2026, 9, 2),
        )

        self.assertEqual(
            result.source_delay_days,
            2,
        )

    def test_lag_one_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            data_dir = (
                root
                / stale.DATA_REL
            )

            write_daily(
                data_dir,
                date(2026, 9, 3),
            )

            with self.assertRaisesRegex(
                stale.StaleReferenceBlockedError,
                "SOURCE_DELAY_NOT_STALE",
            ):
                stale.assess_eligibility(
                    root,
                    date(2026, 9, 5),
                )

    def test_expected_daily_present_is_rejected(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            data_dir = (
                root
                / stale.DATA_REL
            )

            write_daily(
                data_dir,
                date(2026, 9, 2),
            )

            write_daily(
                data_dir,
                date(2026, 9, 4),
            )

            with self.assertRaisesRegex(
                stale.StaleReferenceBlockedError,
                "EXPECTED_DAILY_PRESENT",
            ):
                stale.assess_eligibility(
                    root,
                    date(2026, 9, 5),
                )

    def test_guard_block_stops_before_ranking_modules(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            data_dir = (
                root
                / stale.DATA_REL
            )

            write_daily(
                data_dir,
                date(2026, 9, 2),
            )

            with (
                mock.patch.object(
                    stale,
                    "enforce_big_march_stale_reference_inventory_guard",
                    side_effect=RuntimeError(
                        "guard blocked"
                    ),
                ),
                mock.patch.object(
                    stale,
                    "load_ranking_modules",
                ) as load_modules,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "guard blocked",
                ):
                    stale.generate(
                        root,
                        date(2026, 9, 5),
                    )

            load_modules.assert_not_called()

    def test_lag_two_generates_isolated_stale_reference(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            data_dir = (
                root
                / stale.DATA_REL
            )

            latest_date = date(
                2026,
                9,
                2,
            )

            write_daily(
                data_dir,
                latest_date,
            )

            juggler = make_dummy_module(
                latest_date,
                "TEST_JUGGLER",
            )

            nonjuggler = make_dummy_module(
                latest_date,
                "TEST_NONJUGGLER",
            )

            with (
                mock.patch.object(
                    stale,
                    "enforce_big_march_stale_reference_inventory_guard",
                    return_value=(
                        allowed_guard_decision()
                    ),
                ),
                mock.patch.object(
                    stale,
                    "load_ranking_modules",
                    return_value=(
                        juggler,
                        nonjuggler,
                    ),
                ),
            ):
                result = stale.generate(
                    root,
                    date(2026, 9, 5),
                )

            self.assertEqual(
                result["status"],
                "STALE_REFERENCE",
            )

            paths = result["paths"]

            metadata = pd.read_csv(
                paths["metadata"],
                encoding="utf-8-sig",
            )

            self.assertEqual(
                len(metadata),
                1,
            )

            row = metadata.iloc[0]

            self.assertEqual(
                row["ranking_class"],
                "STALE_REFERENCE",
            )

            self.assertFalse(
                stale._bool_value(
                    row["formal"]
                )
            )

            self.assertFalse(
                stale._bool_value(
                    row["provisional"]
                )
            )

            self.assertTrue(
                stale._bool_value(
                    row[
                        "stale_reference"
                    ]
                )
            )

            self.assertFalse(
                stale._bool_value(
                    row["forward_valid"]
                )
            )

            self.assertFalse(
                stale._bool_value(
                    row[
                        "eligible_for_formal_evaluation"
                    ]
                )
            )

            self.assertEqual(
                int(
                    row[
                        "source_delay_days"
                    ]
                ),
                2,
            )

            self.assertEqual(
                row[
                    "source_delay_bucket"
                ],
                "LAG_2",
            )

            self.assertEqual(
                row[
                    "inventory_currentness"
                ],
                "UNCONFIRMED",
            )

            self.assertEqual(
                row[
                    "machine_mapping"
                ],
                "LAST_KNOWN",
            )

            self.assertTrue(
                paths[
                    "juggler_top10"
                ].is_file()
            )

            self.assertTrue(
                paths[
                    "nonjuggler_top10"
                ].is_file()
            )

    def test_lag_three_plus_uses_lag_three_plus_bucket(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            data_dir = (
                root
                / stale.DATA_REL
            )

            latest_date = date(
                2026,
                9,
                1,
            )

            write_daily(
                data_dir,
                latest_date,
            )

            juggler = make_dummy_module(
                latest_date,
                "TEST_JUGGLER",
            )

            nonjuggler = make_dummy_module(
                latest_date,
                "TEST_NONJUGGLER",
            )

            with (
                mock.patch.object(
                    stale,
                    "enforce_big_march_stale_reference_inventory_guard",
                    return_value=(
                        allowed_guard_decision()
                    ),
                ),
                mock.patch.object(
                    stale,
                    "load_ranking_modules",
                    return_value=(
                        juggler,
                        nonjuggler,
                    ),
                ),
            ):
                result = stale.generate(
                    root,
                    date(2026, 9, 5),
                )

            metadata = pd.read_csv(
                result["paths"]["metadata"],
                encoding="utf-8-sig",
            )

            row = metadata.iloc[0]

            self.assertEqual(
                int(
                    row[
                        "source_delay_days"
                    ]
                ),
                3,
            )

            self.assertEqual(
                row[
                    "source_delay_bucket"
                ],
                "LAG_3_PLUS",
            )

    def test_existing_stale_reference_is_reused_not_overwritten(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            data_dir = (
                root
                / stale.DATA_REL
            )

            latest_date = date(
                2026,
                9,
                2,
            )

            write_daily(
                data_dir,
                latest_date,
            )

            juggler = make_dummy_module(
                latest_date,
                "TEST_JUGGLER",
            )

            nonjuggler = make_dummy_module(
                latest_date,
                "TEST_NONJUGGLER",
            )

            with (
                mock.patch.object(
                    stale,
                    "enforce_big_march_stale_reference_inventory_guard",
                    return_value=(
                        allowed_guard_decision()
                    ),
                ),
                mock.patch.object(
                    stale,
                    "load_ranking_modules",
                    return_value=(
                        juggler,
                        nonjuggler,
                    ),
                ),
            ):
                first = stale.generate(
                    root,
                    date(2026, 9, 5),
                )

                second = stale.generate(
                    root,
                    date(2026, 9, 5),
                )

            self.assertEqual(
                first["status"],
                "STALE_REFERENCE",
            )

            self.assertEqual(
                second["status"],
                "ALREADY_STALE_REFERENCE",
            )


if __name__ == "__main__":
    unittest.main()