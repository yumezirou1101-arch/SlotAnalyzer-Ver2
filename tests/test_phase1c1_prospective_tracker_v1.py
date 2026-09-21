from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]

TRACKER_PATH = (
    ROOT
    / "machine_number"
    / "ana_slo_v42c_store_behavior_phase1c1_prospective_tracker_v1.py"
)


def load_tracker_module():
    module_name = (
        "phase1c1_prospective_tracker_v1"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        TRACKER_PATH,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load tracker module: {TRACKER_PATH}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    # Python 3.14:
    # dataclasses expects the module to exist in
    # sys.modules while the module body is executed.
    sys.modules[
        module_name
    ] = module

    try:
        spec.loader.exec_module(
            module
        )

    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    return module


tracker = load_tracker_module()


def write_daily_csv(
    path: Path,
    target_date: date,
    *,
    machine_count: int = 514,
    duplicate_machine_no: bool = False,
    blank_diff_index: int | None = None,
    strong_count: int = 100,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for index in range(
        machine_count
    ):
        machine_no = (
            500
            + index
        )

        if (
            duplicate_machine_no
            and index
            == machine_count - 1
        ):
            machine_no = 500

        diff: str | int

        if (
            blank_diff_index
            is not None
            and index
            == blank_diff_index
        ):
            diff = ""

        elif (
            index
            < strong_count
        ):
            diff = 2000

        else:
            diff = 0

        rows.append(
            {
                "日付":
                    target_date.isoformat(),
                "台番号":
                    machine_no,
                "機種名":
                    f"TEST-{machine_no}",
                "G数":
                    5000,
                "差枚":
                    diff,
                "BB":
                    0,
                "RB":
                    0,
                "合成確率":
                    "",
                "BB確率":
                    "",
                "RB確率":
                    "",
            }
        )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "日付",
                "台番号",
                "機種名",
                "G数",
                "差枚",
                "BB",
                "RB",
                "合成確率",
                "BB確率",
                "RB確率",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_morning_state(
    state_dir: Path,
    data_date: date,
    *,
    guard_status: str = "PASS",
    blocked: bool = False,
    guard_latest_data_date: date | None = None,
) -> None:
    operation_date = (
        data_date
        + timedelta(
            days=1
        )
    )

    if (
        guard_latest_data_date
        is None
    ):
        guard_latest_data_date = (
            data_date
        )

    path = (
        state_dir
        / (
            "morning_automation_state_"
            f"{operation_date:%Y%m%d}.json"
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    value = {
        "operation_date":
            operation_date.isoformat(),
        "stores": {
            "maruhan": {
                "inventory_guard": {
                    "status":
                        guard_status,
                    "blocked":
                        blocked,
                    "latest_data_date":
                        guard_latest_data_date.isoformat(),
                }
            }
        },
    }

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class Phase1C1ProspectiveTrackerTests(
    unittest.TestCase
):
    def test_pre_start_date_is_not_discovered_into_tracker(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            before = date(
                2026,
                9,
                21,
            )

            start = date(
                2026,
                9,
                22,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260921.csv",
                before,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260922.csv",
                start,
            )

            write_morning_state(
                state_dir,
                start,
            )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                result = (
                    tracker.build_tracker()
                )

            self.assertEqual(
                len(
                    result
                ),
                1,
            )

            self.assertEqual(
                result.iloc[
                    0
                ][
                    "target_date"
                ],
                "2026-09-22",
            )

            self.assertTrue(
                bool(
                    result.iloc[
                        0
                    ][
                        "in_primary_sample"
                    ]
                )
            )

    def test_rebuild_is_idempotent_and_has_no_duplicate_target_date(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            target = date(
                2026,
                9,
                24,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260924.csv",
                target,
            )

            write_morning_state(
                state_dir,
                target,
            )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                first = (
                    tracker.build_tracker()
                )

                second = (
                    tracker.build_tracker()
                )

            self.assertEqual(
                len(
                    first
                ),
                1,
            )

            self.assertEqual(
                len(
                    second
                ),
                1,
            )

            self.assertFalse(
                first[
                    "target_date"
                ].duplicated().any()
            )

            self.assertFalse(
                second[
                    "target_date"
                ].duplicated().any()
            )

            self.assertEqual(
                first.to_dict(
                    orient="records"
                ),
                second.to_dict(
                    orient="records"
                ),
            )

    def test_machine_count_failure_is_not_valid(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            target = date(
                2026,
                9,
                24,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260924.csv",
                target,
                machine_count=513,
            )

            write_morning_state(
                state_dir,
                target,
            )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                result = (
                    tracker.build_tracker()
                )

            row = result.iloc[
                0
            ]

            self.assertEqual(
                row[
                    "data_quality_status"
                ],
                "INVALID",
            )

            self.assertFalse(
                bool(
                    row[
                        "eligible_valid_day"
                    ]
                )
            )

            self.assertFalse(
                bool(
                    row[
                        "in_primary_sample"
                    ]
                )
            )

    def test_duplicate_machine_failure_is_not_valid(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            target = date(
                2026,
                9,
                24,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260924.csv",
                target,
                duplicate_machine_no=True,
            )

            write_morning_state(
                state_dir,
                target,
            )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                result = (
                    tracker.build_tracker()
                )

            row = result.iloc[
                0
            ]

            self.assertEqual(
                row[
                    "data_quality_status"
                ],
                "INVALID",
            )

            self.assertGreater(
                int(
                    row[
                        "duplicate_machine_n"
                    ]
                ),
                0,
            )

            self.assertFalse(
                bool(
                    row[
                        "in_primary_sample"
                    ]
                )
            )

    def test_blank_diff_failure_is_not_valid(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            target = date(
                2026,
                9,
                24,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260924.csv",
                target,
                blank_diff_index=0,
            )

            write_morning_state(
                state_dir,
                target,
            )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                result = (
                    tracker.build_tracker()
                )

            row = result.iloc[
                0
            ]

            self.assertEqual(
                row[
                    "data_quality_status"
                ],
                "INVALID",
            )

            self.assertEqual(
                int(
                    row[
                        "actual_diff_missing_n"
                    ]
                ),
                1,
            )

            self.assertFalse(
                bool(
                    row[
                        "in_primary_sample"
                    ]
                )
            )

    def test_inventory_guard_not_pass_is_not_valid(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            target = date(
                2026,
                9,
                24,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260924.csv",
                target,
            )

            write_morning_state(
                state_dir,
                target,
                guard_status="MANUAL_REVIEW",
                blocked=True,
            )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                result = (
                    tracker.build_tracker()
                )

            row = result.iloc[
                0
            ]

            self.assertEqual(
                row[
                    "data_quality_status"
                ],
                "INVALID",
            )

            self.assertEqual(
                row[
                    "inventory_guard_evidence_status"
                ],
                "INVENTORY_GUARD_NOT_PASS",
            )

            self.assertFalse(
                bool(
                    row[
                        "in_primary_sample"
                    ]
                )
            )

    def test_missing_inventory_guard_is_not_valid(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            target = date(
                2026,
                9,
                24,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260924.csv",
                target,
            )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                result = (
                    tracker.build_tracker()
                )

            row = result.iloc[
                0
            ]

            self.assertEqual(
                row[
                    "data_quality_status"
                ],
                "INVALID",
            )

            self.assertEqual(
                row[
                    "inventory_guard_evidence_status"
                ],
                "INVENTORY_GUARD_EVIDENCE_MISSING",
            )

            self.assertFalse(
                bool(
                    row[
                        "in_primary_sample"
                    ]
                )
            )

    def test_calendar_classification_matches_fixed_phase1c1_definition(
        self,
    ):
        cases = [
            (
                date(
                    2026,
                    9,
                    22,
                ),
                "WEEKEND_HOLIDAY",
                False,
                True,
            ),
            (
                date(
                    2026,
                    9,
                    23,
                ),
                "WEEKEND_HOLIDAY",
                False,
                True,
            ),
            (
                date(
                    2026,
                    9,
                    24,
                ),
                "WEEKDAY",
                False,
                False,
            ),
            (
                date(
                    2026,
                    9,
                    26,
                ),
                "WEEKEND_HOLIDAY",
                True,
                False,
            ),
            (
                date(
                    2026,
                    10,
                    12,
                ),
                "WEEKEND_HOLIDAY",
                False,
                True,
            ),
        ]

        for (
            target_date,
            expected_group,
            expected_weekend,
            expected_holiday,
        ) in cases:
            with self.subTest(
                target_date=target_date
            ):
                (
                    group,
                    is_weekend,
                    is_holiday,
                    _holiday_name,
                ) = (
                    tracker.classify_calendar_day(
                        target_date
                    )
                )

                self.assertEqual(
                    group,
                    expected_group,
                )

                self.assertEqual(
                    is_weekend,
                    expected_weekend,
                )

                self.assertEqual(
                    is_holiday,
                    expected_holiday,
                )

    def test_primary_threshold_includes_exactly_2000(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            target = date(
                2026,
                9,
                24,
            )

            write_daily_csv(
                data_dir
                / "ana_slo_20260924.csv",
                target,
                strong_count=123,
            )

            write_morning_state(
                state_dir,
                target,
            )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                result = (
                    tracker.build_tracker()
                )

            row = result.iloc[
                0
            ]

            self.assertEqual(
                int(
                    row[
                        "strong_n"
                    ]
                ),
                123,
            )

            self.assertAlmostEqual(
                float(
                    row[
                        "strong_rate"
                    ]
                ),
                123 / 514,
            )

    def test_41_valid_days_is_accumulating(
        self,
    ):
        rows = []

        current = (
            tracker.PROSPECTIVE_START_DATE
        )

        for index in range(
            41
        ):
            rows.append(
                {
                    "target_date":
                        current.isoformat(),
                    "calendar_group":
                        (
                            "WEEKEND_HOLIDAY"
                            if current.weekday()
                            >= 5
                            else "WEEKDAY"
                        ),
                    "in_primary_sample":
                        True,
                    "data_quality_status":
                        "VALID",
                    "eligible_valid_day":
                        True,
                    "sample_index":
                        index + 1,
                }
            )

            current += timedelta(
                days=1
            )

        frame = (
            tracker.pd.DataFrame(
                rows
            )
        )

        progress = (
            tracker.build_progress_summary(
                frame
            )
        )

        self.assertEqual(
            int(
                progress.iloc[
                    0
                ][
                    "valid_target_day_n"
                ]
            ),
            41,
        )

        self.assertEqual(
            progress.iloc[
                0
            ][
                "status"
            ],
            "ACCUMULATING",
        )

    def test_42_valid_days_is_review_ready(
        self,
    ):
        rows = []

        current = (
            tracker.PROSPECTIVE_START_DATE
        )

        for index in range(
            42
        ):
            rows.append(
                {
                    "target_date":
                        current.isoformat(),
                    "calendar_group":
                        (
                            "WEEKEND_HOLIDAY"
                            if current.weekday()
                            >= 5
                            else "WEEKDAY"
                        ),
                    "in_primary_sample":
                        True,
                    "data_quality_status":
                        "VALID",
                    "eligible_valid_day":
                        True,
                    "sample_index":
                        index + 1,
                }
            )

            current += timedelta(
                days=1
            )

        frame = (
            tracker.pd.DataFrame(
                rows
            )
        )

        progress = (
            tracker.build_progress_summary(
                frame
            )
        )

        self.assertEqual(
            int(
                progress.iloc[
                    0
                ][
                    "valid_target_day_n"
                ]
            ),
            42,
        )

        self.assertEqual(
            progress.iloc[
                0
            ][
                "status"
            ],
            (
                "PHASE1C1_"
                "PROSPECTIVE_REVIEW_READY"
            ),
        )

    def test_first_42_valid_days_only_are_primary_sample(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            data_dir = (
                root
                / "data"
            )

            state_dir = (
                root
                / "state"
            )

            current = (
                tracker.PROSPECTIVE_START_DATE
            )

            for _index in range(
                43
            ):
                write_daily_csv(
                    data_dir
                    / (
                        "ana_slo_"
                        f"{current:%Y%m%d}.csv"
                    ),
                    current,
                )

                write_morning_state(
                    state_dir,
                    current,
                )

                current += timedelta(
                    days=1
                )

            with (
                patch.object(
                    tracker,
                    "DATA_DIR",
                    data_dir,
                ),
                patch.object(
                    tracker,
                    "MORNING_STATE_DIR",
                    state_dir,
                ),
            ):
                result = (
                    tracker.build_tracker()
                )

            self.assertEqual(
                len(
                    result
                ),
                43,
            )

            self.assertEqual(
                int(
                    result[
                        "in_primary_sample"
                    ].astype(
                        bool
                    ).sum()
                ),
                42,
            )

            self.assertFalse(
                bool(
                    result.iloc[
                        42
                    ][
                        "in_primary_sample"
                    ]
                )
            )

    def test_metadata_declares_no_production_changes(
        self,
    ):
        metadata = (
            tracker.build_metadata()
        )

        values = {
            str(
                row[
                    "key"
                ]
            ): str(
                row[
                    "value"
                ]
            )
            for row in metadata.to_dict(
                orient="records"
            )
        }

        self.assertEqual(
            values[
                "morning_automation_modified"
            ],
            "False",
        )

        self.assertEqual(
            values[
                "production_ranking_modified"
            ],
            "False",
        )

        self.assertEqual(
            values[
                "champion_modified"
            ],
            "False",
        )

        self.assertEqual(
            values[
                "forward_guard_modified"
            ],
            "False",
        )

        self.assertEqual(
            values[
                "formal_forward_modified"
            ],
            "False",
        )

        self.assertEqual(
            values[
                "neighbor_avg_modified"
            ],
            "False",
        )


if __name__ == "__main__":
    unittest.main()