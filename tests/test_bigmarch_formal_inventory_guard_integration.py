from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"

if str(MACHINE) not in sys.path:
    sys.path.insert(
        0,
        str(MACHINE),
    )


FORMAL_SCRIPT = (
    MACHINE
    / "ana_slo_bigmarch_oyagi_one_click_daily_update_v3.py"
)


def load_formal_module():
    module_name = (
        "test_target_"
        "ana_slo_bigmarch_oyagi_one_click_daily_update_v3"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        FORMAL_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load Big March Formal child module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


class BigMarchFormalInventoryGuardIntegrationTests(
    unittest.TestCase
):
    def setUp(
        self,
    ) -> None:
        self.formal = load_formal_module()

    @staticmethod
    def args(
        *,
        allow_gap: bool = False,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            fetch_days=1,
            min_machines=200,
            skip_fetch=True,
            chrome_wait_sec=15,
            allow_gap=allow_gap,
        )

    @staticmethod
    def freshness_result() -> dict:
        import pandas as pd

        return {
            "latest_path": Path(
                "dummy_latest.csv"
            ),
            "latest_data_date": pd.Timestamp(
                "2026-09-16"
            ),
            "target_date": pd.Timestamp(
                "2026-09-17"
            ),
            "rows": 276,
            "unique_machines": 276,
        }

    def test_guard_block_stops_before_forward_and_ranking(
        self,
    ):
        executed_labels: list[str] = []

        def fake_run_stage(
            label,
            script,
            args=None,
        ):
            executed_labels.append(
                label
            )

            return 0.01

        with (
            mock.patch.object(
                self.formal,
                "parse_args",
                return_value=self.args(
                    allow_gap=False
                ),
            ),
            mock.patch.object(
                self.formal,
                "check_file",
            ),
            mock.patch.object(
                self.formal,
                "ensure_cdp",
            ),
            mock.patch.object(
                self.formal,
                "compile_script",
            ),
            mock.patch.object(
                self.formal,
                "run_stage",
                side_effect=fake_run_stage,
            ),
            mock.patch.object(
                self.formal,
                "freshness_guard",
                return_value=(
                    self.freshness_result()
                ),
            ),
            mock.patch.object(
                self.formal,
                "enforce_big_march_formal_inventory_guard",
                side_effect=RuntimeError(
                    "simulated inventory guard block"
                ),
            ) as guard_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated inventory guard block",
            ):
                self.formal.main()

        guard_mock.assert_called_once_with(
            self.formal.DATA_DIR,
            date.today(),
        )

        self.assertEqual(
            executed_labels,
            [
                "BATCH HTML TO DAILY CSV",
            ],
        )

        self.assertNotIn(
            "JUGGLER FROZEN FORWARD",
            executed_labels,
        )

        self.assertNotIn(
            "NONJUGGLER FROZEN FORWARD",
            executed_labels,
        )

        self.assertNotIn(
            "JUGGLER FUTURE RANKING",
            executed_labels,
        )

        self.assertNotIn(
            "NONJUGGLER FUTURE RANKING",
            executed_labels,
        )

    def test_allow_gap_does_not_bypass_inventory_guard(
        self,
    ):
        executed_labels: list[str] = []

        def fake_run_stage(
            label,
            script,
            args=None,
        ):
            executed_labels.append(
                label
            )

            return 0.01

        with (
            mock.patch.object(
                self.formal,
                "parse_args",
                return_value=self.args(
                    allow_gap=True
                ),
            ),
            mock.patch.object(
                self.formal,
                "check_file",
            ),
            mock.patch.object(
                self.formal,
                "ensure_cdp",
            ),
            mock.patch.object(
                self.formal,
                "compile_script",
            ),
            mock.patch.object(
                self.formal,
                "run_stage",
                side_effect=fake_run_stage,
            ),
            mock.patch.object(
                self.formal,
                "freshness_guard",
                return_value=(
                    self.freshness_result()
                ),
            ) as freshness_mock,
            mock.patch.object(
                self.formal,
                "enforce_big_march_formal_inventory_guard",
                side_effect=RuntimeError(
                    "simulated guard block despite allow-gap"
                ),
            ) as guard_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated guard block despite allow-gap",
            ):
                self.formal.main()

        freshness_mock.assert_called_once_with(
            min_machines=200,
            allow_gap=True,
        )

        guard_mock.assert_called_once_with(
            self.formal.DATA_DIR,
            date.today(),
        )

        self.assertEqual(
            executed_labels,
            [
                "BATCH HTML TO DAILY CSV",
            ],
        )

        self.assertNotIn(
            "JUGGLER FROZEN FORWARD",
            executed_labels,
        )

        self.assertNotIn(
            "NONJUGGLER FROZEN FORWARD",
            executed_labels,
        )

        self.assertNotIn(
            "JUGGLER FUTURE RANKING",
            executed_labels,
        )

        self.assertNotIn(
            "NONJUGGLER FUTURE RANKING",
            executed_labels,
        )


if __name__ == "__main__":
    unittest.main()
