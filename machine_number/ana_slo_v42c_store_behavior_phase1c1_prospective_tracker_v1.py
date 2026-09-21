from __future__ import annotations

"""
STORE_BEHAVIOR_RESEARCH
Phase 1C-1 Prospective 42-day Tracker v1

OFFLINE RESEARCH SUPPORT ONLY

Purpose
-------
Track the precommitted Phase 1C-1 prospective confirmation sample
using only validated Maruhan Mega City Maebashi Inter daily CSV files
whose target_date is on or after 2026-09-22.

Rules
-----
- PROSPECTIVE_START_DATE = 2026-09-22 fixed
- TARGET_DAYS = 42 fixed
- STRONG_PRIMARY = actual_diff >= +2000 fixed
- WEEKDAY / WEEKEND_HOLIDAY definition fixed
- historical dates are never included
- daily CSVs are rescanned and tracker is rebuilt on every run
- append mode is never used
- duplicate target_date is prohibited
- only VALID days enter the 42-day primary sample
- exact machine count = 514
- unique machine count = 514
- duplicate machine count = 0
- actual_diff missing count = 0
- Inventory Guard must explicitly PASS
- bootstrap / permutation / LOO are not run here
- formal statistical evaluation belongs to 03_2 after 42 VALID days
- production systems are not modified
"""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

MACHINE_DIR = SCRIPT_PATH.parent

PROJECT_ROOT = MACHINE_DIR.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "maruhan_maebashi"
    / "machine_number"
)

ANALYSIS_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "offline_store_behavior_phase1c1_prospective_tracker_v1"
)

MORNING_STATE_DIR = (
    PROJECT_ROOT
    / "logs"
    / "morning_automation"
    / "state"
)


# ============================================================
# Fixed prospective research specification
# ============================================================

RESEARCH_NAME = "STORE_BEHAVIOR_RESEARCH"

RESEARCH_PHASE = (
    "PHASE1C1_WEEKDAY_VS_WEEKEND_HOLIDAY_PROSPECTIVE"
)

RESEARCH_SCOPE = "OFFLINE_RESEARCH_SUPPORT_ONLY"

STORE = "MARUHAN_MEGA_CITY_MAEBASHI_INTER"

CHAMPION_MODEL = "CHAMPION_V4.2_C"

CHAMPION_FINGERPRINT = (
    "a1eaf45d71ded209"
)

PROSPECTIVE_START_DATE = date(
    2026,
    9,
    22,
)

TARGET_DAYS = 42

EXPECTED_MACHINES_PER_DAY = 514

PRIMARY_THRESHOLD = 2000

DAILY_FILE_PATTERN = re.compile(
    r"^ana_slo_(\d{8})\.csv$"
)


# ============================================================
# Fixed Japanese holiday calendar
# ============================================================

HOLIDAY_SOURCE_NAME = (
    "Cabinet Office, Government of Japan "
    "National Holidays 2026"
)

HOLIDAY_SOURCE_URL = (
    "https://www8.cao.go.jp/chosei/shukujitsu/gaiyou.html"
)

HOLIDAY_SOURCE_RETRIEVED = "2026-09-21"

FIXED_HOLIDAYS: dict[date, str] = {
    date(
        2026,
        9,
        22,
    ): (
        "Holiday under Article 3 Paragraph 3 / "
        "休日（祝日法第3条第3項）"
    ),
    date(
        2026,
        9,
        23,
    ): (
        "Autumnal Equinox Day / 秋分の日"
    ),
    date(
        2026,
        10,
        12,
    ): (
        "Sports Day / スポーツの日"
    ),
    date(
        2026,
        11,
        3,
    ): (
        "Culture Day / 文化の日"
    ),
    date(
        2026,
        11,
        23,
    ): (
        "Labor Thanksgiving Day / 勤労感謝の日"
    ),
}

HOLIDAY_CALENDAR_SUPPORTED_THROUGH = date(
    2026,
    12,
    31,
)


# ============================================================
# Fixed input schema
# ============================================================

COL_DATE = "日付"

COL_MACHINE_NO = "台番号"

COL_MACHINE_NAME = "機種名"

COL_GAMES = "G数"

COL_DIFF = "差枚"

REQUIRED_COLUMNS = {
    COL_DATE,
    COL_MACHINE_NO,
    COL_MACHINE_NAME,
    COL_GAMES,
    COL_DIFF,
}


# ============================================================
# Outputs
# ============================================================

TRACKER_CSV = (
    OUTPUT_DIR
    / "01_prospective_daily_tracker.csv"
)

PROGRESS_CSV = (
    OUTPUT_DIR
    / "02_progress_summary.csv"
)

METADATA_CSV = (
    OUTPUT_DIR
    / "03_metadata.csv"
)

MANIFEST_JSON = (
    OUTPUT_DIR
    / "04_output_manifest.json"
)


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class InventoryGuardEvidence:
    status: str
    blocked: bool | None
    state_path: str
    evidence_status: str
    reason: str


# ============================================================
# Generic helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def atomic_write_text(
    path: Path,
    text: str,
    encoding: str = "utf-8",
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            handle.write(
                text
            )

            temporary_path = Path(
                handle.name
            )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def write_dataframe(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    atomic_write_text(
        path,
        frame.to_csv(
            index=False,
            lineterminator="\n",
        ),
        encoding="utf-8-sig",
    )


def write_json(
    value: dict,
    path: Path,
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def bool_text(
    value: bool | None,
) -> str:
    if value is None:
        return ""

    return (
        "True"
        if value
        else "False"
    )


# ============================================================
# Calendar
# ============================================================

def classify_calendar_day(
    target_date: date,
) -> tuple[
    str,
    bool,
    bool,
    str,
]:
    if (
        target_date
        > HOLIDAY_CALENDAR_SUPPORTED_THROUGH
    ):
        raise RuntimeError(
            "Prospective calendar date exceeds "
            "fixed holiday support. "
            f"target_date={target_date}, "
            "supported_through="
            f"{HOLIDAY_CALENDAR_SUPPORTED_THROUGH}"
        )

    is_weekend = (
        target_date.weekday()
        >= 5
    )

    is_holiday = (
        target_date
        in FIXED_HOLIDAYS
    )

    holiday_name = (
        FIXED_HOLIDAYS.get(
            target_date,
            "",
        )
    )

    calendar_group = (
        "WEEKEND_HOLIDAY"
        if (
            is_weekend
            or is_holiday
        )
        else "WEEKDAY"
    )

    return (
        calendar_group,
        is_weekend,
        is_holiday,
        holiday_name,
    )


# ============================================================
# Daily-file discovery
# ============================================================

def discover_daily_files() -> list[
    tuple[
        date,
        Path,
    ]
]:
    rows: list[
        tuple[
            date,
            Path,
        ]
    ] = []

    for path in DATA_DIR.glob(
        "ana_slo_????????.csv"
    ):
        match = (
            DAILY_FILE_PATTERN.fullmatch(
                path.name
            )
        )

        if match is None:
            continue

        target_date = (
            pd.to_datetime(
                match.group(
                    1
                ),
                format="%Y%m%d",
                errors="raise",
            ).date()
        )

        rows.append(
            (
                target_date,
                path,
            )
        )

    rows.sort(
        key=lambda item: (
            item[0],
            item[1].name,
        )
    )

    return rows


# ============================================================
# Inventory Guard evidence
# ============================================================

def morning_state_path_for_data_date(
    data_date: date,
) -> Path:
    operation_date = (
        data_date
        + timedelta(
            days=1
        )
    )

    return (
        MORNING_STATE_DIR
        / (
            "morning_automation_state_"
            f"{operation_date:%Y%m%d}.json"
        )
    )


def load_inventory_guard_evidence(
    data_date: date,
) -> InventoryGuardEvidence:
    state_path = (
        morning_state_path_for_data_date(
            data_date
        )
    )

    if not state_path.is_file():
        return InventoryGuardEvidence(
            status="",
            blocked=None,
            state_path=str(
                state_path
            ),
            evidence_status=(
                "INVENTORY_GUARD_EVIDENCE_MISSING"
            ),
            reason=(
                "Morning Automation state was not found."
            ),
        )

    try:
        with state_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            state = json.load(
                handle
            )

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        return InventoryGuardEvidence(
            status="",
            blocked=None,
            state_path=str(
                state_path
            ),
            evidence_status=(
                "INVENTORY_GUARD_EVIDENCE_INVALID"
            ),
            reason=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

    try:
        maruhan_state = (
            state[
                "stores"
            ][
                "maruhan"
            ]
        )

    except (
        KeyError,
        TypeError,
    ):
        return InventoryGuardEvidence(
            status="",
            blocked=None,
            state_path=str(
                state_path
            ),
            evidence_status=(
                "INVENTORY_GUARD_EVIDENCE_INVALID"
            ),
            reason=(
                "stores.maruhan was not found."
            ),
        )

    inventory_guard = (
        maruhan_state.get(
            "inventory_guard"
        )
    )

    if not isinstance(
        inventory_guard,
        dict,
    ):
        return InventoryGuardEvidence(
            status="",
            blocked=None,
            state_path=str(
                state_path
            ),
            evidence_status=(
                "INVENTORY_GUARD_EVIDENCE_MISSING"
            ),
            reason=(
                "Maruhan inventory_guard was not found."
            ),
        )

    guard_status = str(
        inventory_guard.get(
            "status",
            "",
        )
    ).strip()

    blocked_raw = (
        inventory_guard.get(
            "blocked"
        )
    )

    blocked = (
        blocked_raw
        if isinstance(
            blocked_raw,
            bool,
        )
        else None
    )

    guard_latest_date = str(
        inventory_guard.get(
            "latest_data_date",
            "",
        )
    ).strip()

    if (
        guard_latest_date
        and guard_latest_date
        != data_date.isoformat()
    ):
        return InventoryGuardEvidence(
            status=guard_status,
            blocked=blocked,
            state_path=str(
                state_path
            ),
            evidence_status=(
                "INVENTORY_GUARD_DATE_MISMATCH"
            ),
            reason=(
                "Inventory Guard latest_data_date "
                "does not match candidate data date. "
                f"guard={guard_latest_date}, "
                f"candidate={data_date.isoformat()}"
            ),
        )

    if (
        guard_status == "PASS"
        and blocked is False
    ):
        return InventoryGuardEvidence(
            status=guard_status,
            blocked=blocked,
            state_path=str(
                state_path
            ),
            evidence_status="PASS",
            reason="",
        )

    return InventoryGuardEvidence(
        status=guard_status,
        blocked=blocked,
        state_path=str(
            state_path
        ),
        evidence_status=(
            "INVENTORY_GUARD_NOT_PASS"
        ),
        reason=(
            "Inventory Guard is not an explicit "
            "PASS with blocked=False."
        ),
    )


# ============================================================
# Daily CSV validation
# ============================================================

def read_daily_csv(
    path: Path,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )


def inspect_daily_file(
    filename_date: date,
    path: Path,
) -> dict:
    base = {
        "target_date":
            filename_date.isoformat(),
        "source_csv":
            str(
                path
            ),
        "source_csv_name":
            path.name,
        "source_csv_sha256":
            "",
        "calendar_group":
            "",
        "weekday_number":
            filename_date.weekday(),
        "weekday_name":
            filename_date.strftime(
                "%A"
            ),
        "is_weekend":
            False,
        "is_japanese_holiday":
            False,
        "holiday_name":
            "",
        "machine_n":
            0,
        "unique_machine_n":
            0,
        "duplicate_machine_n":
            0,
        "invalid_machine_no_n":
            0,
        "actual_diff_missing_n":
            0,
        "strong_n":
            None,
        "strong_rate":
            None,
        "inventory_guard_status":
            "",
        "inventory_guard_blocked":
            "",
        "inventory_guard_evidence_status":
            "",
        "inventory_guard_state_path":
            "",
        "inventory_guard_reason":
            "",
        "data_quality_status":
            "INVALID",
        "data_quality_reason":
            "",
        "eligible_valid_day":
            False,
        "in_primary_sample":
            False,

        # IMPORTANT:
        # nullable numeric value.
        # Do not initialize this as "" because pandas can infer
        # a string dtype and then reject integer sample indexes.
        "sample_index":
            None,
    }

    if (
        filename_date
        < PROSPECTIVE_START_DATE
    ):
        base[
            "data_quality_status"
        ] = "SKIPPED_BEFORE_START"

        base[
            "data_quality_reason"
        ] = (
            "target_date is before "
            "PROSPECTIVE_START_DATE"
        )

        return base

    try:
        (
            calendar_group,
            is_weekend,
            is_holiday,
            holiday_name,
        ) = classify_calendar_day(
            filename_date
        )

    except Exception as exc:
        base[
            "data_quality_status"
        ] = "INVALID_CALENDAR"

        base[
            "data_quality_reason"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        return base

    base[
        "calendar_group"
    ] = calendar_group

    base[
        "is_weekend"
    ] = is_weekend

    base[
        "is_japanese_holiday"
    ] = is_holiday

    base[
        "holiday_name"
    ] = holiday_name

    try:
        base[
            "source_csv_sha256"
        ] = sha256_file(
            path
        )

        raw = read_daily_csv(
            path
        )

    except Exception as exc:
        base[
            "data_quality_status"
        ] = "INVALID_READ_ERROR"

        base[
            "data_quality_reason"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        return base

    missing_columns = sorted(
        REQUIRED_COLUMNS
        - set(
            raw.columns
        )
    )

    if missing_columns:
        base[
            "data_quality_status"
        ] = "INVALID_SCHEMA"

        base[
            "data_quality_reason"
        ] = (
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

        return base

    machine_n = len(
        raw
    )

    date_values = pd.to_datetime(
        raw[
            COL_DATE
        ],
        errors="coerce",
    )

    valid_dates = (
        date_values
        .dropna()
        .dt.date
    )

    unique_dates = set(
        valid_dates.tolist()
    )

    invalid_date_n = int(
        date_values.isna().sum()
    )

    machine_no = pd.to_numeric(
        raw[
            COL_MACHINE_NO
        ],
        errors="coerce",
    )

    invalid_machine_no_n = int(
        machine_no.isna().sum()
    )

    unique_machine_n = int(
        machine_no.nunique(
            dropna=True
        )
    )

    duplicate_machine_n = int(
        machine_no.duplicated(
            keep=False
        ).sum()
    )

    actual_diff = pd.to_numeric(
        raw[
            COL_DIFF
        ]
        .astype(
            str
        )
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "+",
            "",
            regex=False,
        )
        .str.strip(),
        errors="coerce",
    )

    actual_diff_missing_n = int(
        actual_diff.isna().sum()
    )

    base[
        "machine_n"
    ] = machine_n

    base[
        "unique_machine_n"
    ] = unique_machine_n

    base[
        "duplicate_machine_n"
    ] = duplicate_machine_n

    base[
        "invalid_machine_no_n"
    ] = invalid_machine_no_n

    base[
        "actual_diff_missing_n"
    ] = actual_diff_missing_n

    failures: list[str] = []

    if (
        machine_n
        != EXPECTED_MACHINES_PER_DAY
    ):
        failures.append(
            "machine_n="
            f"{machine_n} "
            "expected="
            f"{EXPECTED_MACHINES_PER_DAY}"
        )

    if (
        unique_machine_n
        != EXPECTED_MACHINES_PER_DAY
    ):
        failures.append(
            "unique_machine_n="
            f"{unique_machine_n} "
            "expected="
            f"{EXPECTED_MACHINES_PER_DAY}"
        )

    if (
        duplicate_machine_n
        != 0
    ):
        failures.append(
            "duplicate_machine_n="
            f"{duplicate_machine_n}"
        )

    if (
        invalid_machine_no_n
        != 0
    ):
        failures.append(
            "invalid_machine_no_n="
            f"{invalid_machine_no_n}"
        )

    if (
        actual_diff_missing_n
        != 0
    ):
        failures.append(
            "actual_diff_missing_n="
            f"{actual_diff_missing_n}"
        )

    if (
        invalid_date_n
        != 0
    ):
        failures.append(
            "invalid_date_n="
            f"{invalid_date_n}"
        )

    if (
        unique_dates
        != {
            filename_date
        }
    ):
        failures.append(
            "CSV date does not match filename "
            "target_date. "
            f"csv_dates={sorted(unique_dates)}, "
            f"filename_date={filename_date}"
        )

    inventory_guard = (
        load_inventory_guard_evidence(
            filename_date
        )
    )

    base[
        "inventory_guard_status"
    ] = inventory_guard.status

    base[
        "inventory_guard_blocked"
    ] = bool_text(
        inventory_guard.blocked
    )

    base[
        "inventory_guard_evidence_status"
    ] = inventory_guard.evidence_status

    base[
        "inventory_guard_state_path"
    ] = inventory_guard.state_path

    base[
        "inventory_guard_reason"
    ] = inventory_guard.reason

    if (
        inventory_guard.evidence_status
        != "PASS"
    ):
        failures.append(
            "inventory_guard="
            f"{inventory_guard.evidence_status}"
        )

    if failures:
        base[
            "data_quality_status"
        ] = "INVALID"

        base[
            "data_quality_reason"
        ] = " | ".join(
            failures
        )

        return base

    strong_n = int(
        (
            actual_diff
            >= PRIMARY_THRESHOLD
        ).sum()
    )

    strong_rate = (
        strong_n
        / machine_n
    )

    base[
        "strong_n"
    ] = strong_n

    base[
        "strong_rate"
    ] = strong_rate

    base[
        "data_quality_status"
    ] = "VALID"

    base[
        "data_quality_reason"
    ] = ""

    base[
        "eligible_valid_day"
    ] = True

    return base


# ============================================================
# Tracker
# ============================================================

def tracker_columns() -> list[str]:
    return [
        "target_date",
        "calendar_group",
        "weekday_number",
        "weekday_name",
        "is_weekend",
        "is_japanese_holiday",
        "holiday_name",
        "machine_n",
        "unique_machine_n",
        "duplicate_machine_n",
        "invalid_machine_no_n",
        "actual_diff_missing_n",
        "strong_n",
        "strong_rate",
        "inventory_guard_status",
        "inventory_guard_blocked",
        "inventory_guard_evidence_status",
        "inventory_guard_state_path",
        "inventory_guard_reason",
        "data_quality_status",
        "data_quality_reason",
        "eligible_valid_day",
        "in_primary_sample",
        "sample_index",
        "source_csv_name",
        "source_csv",
        "source_csv_sha256",
    ]


def build_tracker() -> pd.DataFrame:
    rows: list[dict] = []

    for (
        filename_date,
        path,
    ) in discover_daily_files():
        if (
            filename_date
            < PROSPECTIVE_START_DATE
        ):
            continue

        rows.append(
            inspect_daily_file(
                filename_date,
                path,
            )
        )

    columns = tracker_columns()

    if not rows:
        empty = pd.DataFrame(
            columns=columns
        )

        empty[
            "sample_index"
        ] = pd.Series(
            dtype="Int64"
        )

        return empty

    tracker = pd.DataFrame(
        rows
    )

    tracker = tracker.sort_values(
        [
            "target_date",
            "source_csv_name",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    if tracker[
        "target_date"
    ].duplicated().any():
        duplicates = tracker.loc[
            tracker[
                "target_date"
            ].duplicated(
                keep=False
            ),
            [
                "target_date",
                "source_csv_name",
            ],
        ]

        raise RuntimeError(
            "Duplicate prospective target_date "
            "detected:\n"
            + duplicates.to_string(
                index=False
            )
        )

    # Explicit nullable integer dtype avoids pandas string
    # inference when the initial values are missing.
    tracker[
        "sample_index"
    ] = pd.Series(
        [
            pd.NA
        ]
        * len(
            tracker
        ),
        dtype="Int64",
        index=tracker.index,
    )

    tracker[
        "in_primary_sample"
    ] = tracker[
        "in_primary_sample"
    ].astype(
        bool
    )

    tracker[
        "eligible_valid_day"
    ] = tracker[
        "eligible_valid_day"
    ].astype(
        bool
    )

    sample_index = 0

    for row_index in tracker.index:
        if not bool(
            tracker.at[
                row_index,
                "eligible_valid_day",
            ]
        ):
            continue

        if (
            sample_index
            >= TARGET_DAYS
        ):
            continue

        sample_index += 1

        tracker.at[
            row_index,
            "in_primary_sample",
        ] = True

        tracker.at[
            row_index,
            "sample_index",
        ] = sample_index

    return tracker[
        columns
    ].copy()


# ============================================================
# Progress
# ============================================================

def build_progress_summary(
    tracker: pd.DataFrame,
) -> pd.DataFrame:
    if tracker.empty:
        sample = tracker.copy()

    else:
        sample = tracker.loc[
            tracker[
                "in_primary_sample"
            ].astype(
                bool
            )
        ].copy()

    valid_day_n = len(
        sample
    )

    weekday_day_n = int(
        sample[
            "calendar_group"
        ].eq(
            "WEEKDAY"
        ).sum()
        if not sample.empty
        else 0
    )

    weekend_holiday_day_n = int(
        sample[
            "calendar_group"
        ].eq(
            "WEEKEND_HOLIDAY"
        ).sum()
        if not sample.empty
        else 0
    )

    remaining_day_n = max(
        TARGET_DAYS
        - valid_day_n,
        0,
    )

    status = (
        "PHASE1C1_PROSPECTIVE_REVIEW_READY"
        if valid_day_n
        >= TARGET_DAYS
        else "ACCUMULATING"
    )

    latest_sample_date = ""

    if not sample.empty:
        latest_sample_date = str(
            sample[
                "target_date"
            ].max()
        )

    return pd.DataFrame(
        [
            {
                "research_name":
                    RESEARCH_NAME,
                "research_phase":
                    RESEARCH_PHASE,
                "research_scope":
                    RESEARCH_SCOPE,
                "store":
                    STORE,
                "prospective_start_date":
                    PROSPECTIVE_START_DATE.isoformat(),
                "target_days":
                    TARGET_DAYS,
                "valid_target_day_n":
                    valid_day_n,
                "weekday_day_n":
                    weekday_day_n,
                "weekend_holiday_day_n":
                    weekend_holiday_day_n,
                "remaining_day_n":
                    remaining_day_n,
                "latest_sample_date":
                    latest_sample_date,
                "status":
                    status,
                "primary_threshold":
                    PRIMARY_THRESHOLD,
                "comparison":
                    "WEEKEND_HOLIDAY - WEEKDAY",
                "formal_statistics_run":
                    False,
            }
        ]
    )


# ============================================================
# Metadata
# ============================================================

def build_metadata() -> pd.DataFrame:
    holiday_text = "|".join(
        (
            f"{holiday_date.isoformat()}="
            f"{holiday_name}"
        )
        for (
            holiday_date,
            holiday_name,
        ) in sorted(
            FIXED_HOLIDAYS.items()
        )
    )

    rows = [
        (
            "research_name",
            RESEARCH_NAME,
        ),
        (
            "research_phase",
            RESEARCH_PHASE,
        ),
        (
            "research_scope",
            RESEARCH_SCOPE,
        ),
        (
            "store",
            STORE,
        ),
        (
            "champion_model",
            CHAMPION_MODEL,
        ),
        (
            "champion_fingerprint",
            CHAMPION_FINGERPRINT,
        ),
        (
            "prospective_start_date",
            PROSPECTIVE_START_DATE.isoformat(),
        ),
        (
            "target_days",
            TARGET_DAYS,
        ),
        (
            "expected_machines_per_day",
            EXPECTED_MACHINES_PER_DAY,
        ),
        (
            "primary_endpoint",
            "actual_diff >= +2000",
        ),
        (
            "primary_threshold",
            PRIMARY_THRESHOLD,
        ),
        (
            "weekday_definition",
            (
                "Monday-Friday and not "
                "Japanese public holiday"
            ),
        ),
        (
            "weekend_holiday_definition",
            (
                "Saturday or Sunday or "
                "Japanese public holiday"
            ),
        ),
        (
            "comparison",
            "WEEKEND_HOLIDAY - WEEKDAY",
        ),
        (
            "holiday_source",
            HOLIDAY_SOURCE_NAME,
        ),
        (
            "holiday_source_url",
            HOLIDAY_SOURCE_URL,
        ),
        (
            "holiday_source_retrieved",
            HOLIDAY_SOURCE_RETRIEVED,
        ),
        (
            "holiday_calendar_supported_through",
            HOLIDAY_CALENDAR_SUPPORTED_THROUGH.isoformat(),
        ),
        (
            "fixed_holidays_from_prospective_start",
            holiday_text,
        ),
        (
            "tracker_rebuild_method",
            "FULL_REBUILD_FROM_DAILY_CSV_NO_APPEND",
        ),
        (
            "historical_data_allowed",
            False,
        ),
        (
            "bootstrap_in_tracker",
            False,
        ),
        (
            "permutation_in_tracker",
            False,
        ),
        (
            "leave_one_day_out_in_tracker",
            False,
        ),
        (
            "formal_review_owner",
            "03_2_AFTER_42_VALID_DAYS",
        ),
        (
            "morning_automation_modified",
            False,
        ),
        (
            "production_ranking_modified",
            False,
        ),
        (
            "champion_modified",
            False,
        ),
        (
            "forward_guard_modified",
            False,
        ),
        (
            "formal_forward_modified",
            False,
        ),
        (
            "neighbor_avg_modified",
            False,
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "key",
            "value",
        ],
    )


# ============================================================
# Manifest
# ============================================================

def build_manifest(
    tracker: pd.DataFrame,
    progress: pd.DataFrame,
) -> dict:
    sample = tracker.loc[
        tracker[
            "in_primary_sample"
        ].astype(
            bool
        )
    ].copy()

    invalid_n = int(
        tracker[
            "data_quality_status"
        ].ne(
            "VALID"
        ).sum()
        if not tracker.empty
        else 0
    )

    return {
        "schema_version":
            1,
        "research_name":
            RESEARCH_NAME,
        "research_phase":
            RESEARCH_PHASE,
        "research_scope":
            RESEARCH_SCOPE,
        "store":
            STORE,
        "champion_model":
            CHAMPION_MODEL,
        "champion_fingerprint":
            CHAMPION_FINGERPRINT,
        "prospective_start_date":
            PROSPECTIVE_START_DATE.isoformat(),
        "target_days":
            TARGET_DAYS,
        "primary_threshold":
            PRIMARY_THRESHOLD,
        "comparison":
            "WEEKEND_HOLIDAY - WEEKDAY",
        "candidate_day_n":
            int(
                len(
                    tracker
                )
            ),
        "valid_primary_sample_day_n":
            int(
                len(
                    sample
                )
            ),
        "invalid_or_review_day_n":
            invalid_n,
        "status":
            str(
                progress.iloc[
                    0
                ][
                    "status"
                ]
            ),
        "formal_statistics_run":
            False,
        "outputs": {
            "tracker_csv":
                str(
                    TRACKER_CSV
                ),
            "progress_csv":
                str(
                    PROGRESS_CSV
                ),
            "metadata_csv":
                str(
                    METADATA_CSV
                ),
            "manifest_json":
                str(
                    MANIFEST_JSON
                ),
        },
        "production_changes": {
            "morning_automation":
                False,
            "production_ranking":
                False,
            "champion":
                False,
            "weights":
                False,
            "forward_guard":
                False,
            "formal_forward":
                False,
            "neighbor_avg":
                False,
        },
    }


# ============================================================
# Invariants
# ============================================================

def enforce_tracker_invariants(
    tracker: pd.DataFrame,
    progress: pd.DataFrame,
) -> None:
    if tracker.empty:
        sample = tracker.copy()

    else:
        target_dates = pd.to_datetime(
            tracker[
                "target_date"
            ],
            errors="raise",
        ).dt.date

        if any(
            value
            < PROSPECTIVE_START_DATE
            for value in target_dates
        ):
            raise RuntimeError(
                "Pre-prospective date entered tracker."
            )

        if tracker[
            "target_date"
        ].duplicated().any():
            raise RuntimeError(
                "Duplicate target_date entered tracker."
            )

        sample = tracker.loc[
            tracker[
                "in_primary_sample"
            ].astype(
                bool
            )
        ].copy()

    if (
        len(
            sample
        )
        > TARGET_DAYS
    ):
        raise RuntimeError(
            "Prospective primary sample exceeds "
            f"{TARGET_DAYS} days."
        )

    if not sample.empty:
        if not sample[
            "data_quality_status"
        ].eq(
            "VALID"
        ).all():
            raise RuntimeError(
                "Invalid day entered primary sample."
            )

        if not sample[
            "eligible_valid_day"
        ].astype(
            bool
        ).all():
            raise RuntimeError(
                "Ineligible day entered primary sample."
            )

        expected_indices = list(
            range(
                1,
                len(
                    sample
                )
                + 1,
            )
        )

        actual_indices = [
            int(
                value
            )
            for value in sample[
                "sample_index"
            ].tolist()
        ]

        if (
            actual_indices
            != expected_indices
        ):
            raise RuntimeError(
                "sample_index is not sequential."
            )

    progress_row = (
        progress.iloc[
            0
        ]
    )

    if (
        int(
            progress_row[
                "valid_target_day_n"
            ]
        )
        != len(
            sample
        )
    ):
        raise RuntimeError(
            "Progress count does not match tracker."
        )

    expected_status = (
        "PHASE1C1_PROSPECTIVE_REVIEW_READY"
        if len(
            sample
        )
        >= TARGET_DAYS
        else "ACCUMULATING"
    )

    if (
        progress_row[
            "status"
        ]
        != expected_status
    ):
        raise RuntimeError(
            "Progress status mismatch."
        )


# ============================================================
# Console
# ============================================================

def print_summary(
    tracker: pd.DataFrame,
    progress: pd.DataFrame,
) -> None:
    row = progress.iloc[
        0
    ]

    print(
        "=" * 78
    )

    print(
        "STORE_BEHAVIOR_RESEARCH "
        "Phase1C-1 Prospective 42-day Tracker"
    )

    print(
        "=" * 78
    )

    print(
        f"scope             : {RESEARCH_SCOPE}"
    )

    print(
        f"store             : {STORE}"
    )

    print(
        "start date        : "
        f"{PROSPECTIVE_START_DATE}"
    )

    print(
        "PRIMARY           : "
        "actual_diff >= +2000"
    )

    print(
        f"target days       : {TARGET_DAYS}"
    )

    print(
        f"candidate days    : {len(tracker)}"
    )

    print(
        "valid sample      : "
        f"{int(row['valid_target_day_n'])}"
        f" / {TARGET_DAYS}"
    )

    print(
        "WEEKDAY           : "
        f"{int(row['weekday_day_n'])}"
    )

    print(
        "WEEKEND_HOLIDAY   : "
        f"{int(row['weekend_holiday_day_n'])}"
    )

    print(
        "remaining         : "
        f"{int(row['remaining_day_n'])}"
    )

    print(
        f"status            : {row['status']}"
    )

    print()

    print(
        "formal statistics : NOT RUN"
    )

    print(
        "production change : NONE"
    )

    print(
        "Morning Automation: UNCHANGED"
    )

    print(
        f"Champion          : {CHAMPION_MODEL}"
    )

    print(
        f"fingerprint       : {CHAMPION_FINGERPRINT}"
    )

    print()

    print(
        "outputs:"
    )

    print(
        f"  {TRACKER_CSV}"
    )

    print(
        f"  {PROGRESS_CSV}"
    )

    print(
        f"  {METADATA_CSV}"
    )

    print(
        f"  {MANIFEST_JSON}"
    )

    print(
        "=" * 78
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    if (
        PROSPECTIVE_START_DATE
        != date(
            2026,
            9,
            22,
        )
    ):
        raise RuntimeError(
            "PROSPECTIVE_START_DATE must remain "
            "2026-09-22."
        )

    if (
        TARGET_DAYS
        != 42
    ):
        raise RuntimeError(
            "TARGET_DAYS must remain 42."
        )

    if (
        PRIMARY_THRESHOLD
        != 2000
    ):
        raise RuntimeError(
            "PRIMARY_THRESHOLD must remain +2000."
        )

    if (
        EXPECTED_MACHINES_PER_DAY
        != 514
    ):
        raise RuntimeError(
            "EXPECTED_MACHINES_PER_DAY must remain 514."
        )

    tracker = build_tracker()

    progress = (
        build_progress_summary(
            tracker
        )
    )

    metadata = (
        build_metadata()
    )

    enforce_tracker_invariants(
        tracker,
        progress,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_dataframe(
        tracker,
        TRACKER_CSV,
    )

    write_dataframe(
        progress,
        PROGRESS_CSV,
    )

    write_dataframe(
        metadata,
        METADATA_CSV,
    )

    manifest = (
        build_manifest(
            tracker,
            progress,
        )
    )

    write_json(
        manifest,
        MANIFEST_JSON,
    )

    print_summary(
        tracker,
        progress,
    )


if __name__ == "__main__":
    main()