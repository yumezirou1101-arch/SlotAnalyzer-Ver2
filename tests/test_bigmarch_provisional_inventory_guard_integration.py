from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MACHINE = ROOT / "machine_number"

if str(MACHINE) not in sys.path:
    sys.path.insert(
        0,
        str(MACHINE),
    )


PROVISIONAL_SCRIPT = (
    MACHINE
    / "ana_slo_bigmarch_oyagi_provisional_future_ranking.py"
)


def load_provisional_module():
    module_name = (
        "test_target_"
        "ana_slo_bigmarch_oyagi_provisional_future_ranking"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        PROVISIONAL_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load Big March provisional child module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


class BigMarchProvisionalInventoryGuardIntegrationTests(
    unittest.TestCase
):
    def setUp(
        self,
    ) -> None:
        self.provisional = load_provisional_module()

    def eligibility(
        self,
        project_root: Path,
        operation_date: date,
    ):
        expected_data_date = (
            operation_date
            - timedelta(days=1)
        )

        latest_data_date = (
            expected_data_date
            - timedelta(days=1)
        )

        data_dir = (
            project_root
            / self.provisional.DATA_REL
        )

        return self.provisional.Eligibility(
            operation_date=operation_date,
            expected_data_date=expected_data_date,
            latest_data_date=latest_data_date,
            latest_daily_path=(
                data_dir
                / (
                    "ana_slo_bigmarch_oyagi_"
                    f"{latest_data_date:%Y%m%d}.csv"
                )
            ),
            expected_source_path=(
                project_root
                / (
                    "ana_slo_bigmarch_oyagi_"
                    f"{expected_data_date:%Y%m%d}_source.html"
                )
            ),
            expected_daily_path=(
                data_dir
                / (
                    "ana_slo_bigmarch_oyagi_"
                    f"{expected_data_date:%Y%m%d}.csv"
                )
            ),
        )

    def test_guard_block_stops_before_module_load_and_validation(
        self,
    ):
        project_root = Path(
            r"C:\SlotAnalyzer_Test"
        )

        operation_date = date(
            2026,
            9,
            17,
        )

        eligibility = self.eligibility(
            project_root,
            operation_date,
        )

        with (
            mock.patch.object(
                self.provisional,
                "assess_eligibility",
                return_value=eligibility,
            ) as eligibility_mock,
            mock.patch.object(
                self.provisional,
                "enforce_big_march_provisional_inventory_guard",
                side_effect=RuntimeError(
                    "simulated provisional inventory guard block"
                ),
            ) as guard_mock,
            mock.patch.object(
                self.provisional,
                "load_ranking_modules",
            ) as load_modules_mock,
            mock.patch.object(
                self.provisional,
                "validate_existing",
            ) as validate_existing_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated provisional inventory guard block",
            ):
                self.provisional.generate(
                    project_root,
                    operation_date,
                )

        eligibility_mock.assert_called_once_with(
            project_root,
            operation_date,
        )

        guard_mock.assert_called_once_with(
            project_root
            / self.provisional.DATA_REL,
            operation_date,
        )

        load_modules_mock.assert_not_called()
        validate_existing_mock.assert_not_called()

    def test_existing_provisional_cannot_bypass_guard(
        self,
    ):
        project_root = Path(
            r"C:\SlotAnalyzer_Test"
        )

        operation_date = date(
            2026,
            9,
            17,
        )

        eligibility = self.eligibility(
            project_root,
            operation_date,
        )

        with (
            mock.patch.object(
                self.provisional,
                "assess_eligibility",
                return_value=eligibility,
            ),
            mock.patch.object(
                self.provisional,
                "enforce_big_march_provisional_inventory_guard",
                side_effect=RuntimeError(
                    "simulated persistent inventory incident"
                ),
            ) as guard_mock,
            mock.patch.object(
                self.provisional,
                "load_ranking_modules",
            ) as load_modules_mock,
            mock.patch.object(
                self.provisional,
                "validate_existing",
                return_value=True,
            ) as validate_existing_mock,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated persistent inventory incident",
            ):
                self.provisional.generate(
                    project_root,
                    operation_date,
                )

        guard_mock.assert_called_once_with(
            project_root
            / self.provisional.DATA_REL,
            operation_date,
        )

        load_modules_mock.assert_not_called()
        validate_existing_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
