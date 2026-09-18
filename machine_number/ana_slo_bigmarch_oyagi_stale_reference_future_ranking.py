from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import tempfile

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from slotanalyzer_bigmarch_inventory_guard import (
    enforce_big_march_stale_reference_inventory_guard,
)


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

DATA_REL = Path(
    "data/bigmarch_takasaki_oyagi/machine_number"
)

OUTPUT_REL = (
    DATA_REL
    / "analysis_31days_deep"
    / "91_stale_reference_future_ranking"
)

DAILY_RE = re.compile(
    r"^ana_slo_bigmarch_oyagi_(\d{8})\.csv$",
    re.I,
)

REQUIRED_DAILY_COLUMNS = {
    "date",
    "machine_name",
    "machine_no",
    "G",
    "diff",
}

MIN_MACHINES = 200

JST = ZoneInfo("Asia/Tokyo")


class StaleReferenceBlockedError(
    RuntimeError
):
    pass


class StaleReferenceManualReviewError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class Eligibility:
    operation_date: date
    expected_data_date: date
    latest_data_date: date
    source_delay_days: int
    latest_daily_path: Path
    expected_source_path: Path
    expected_daily_path: Path


def _load_module(
    name: str,
    path: Path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Unable to load module: {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(module)

    return module


def load_ranking_modules(
    project_root: Path,
):
    machine_dir = (
        project_root
        / "machine_number"
    )

    juggler = _load_module(
        "bigmarch_stale_reference_juggler",
        machine_dir
        / (
            "ana_slo_bigmarch_oyagi_"
            "juggler_recent7_future_ranking.py"
        ),
    )

    nonjuggler = _load_module(
        "bigmarch_stale_reference_nonjuggler",
        machine_dir
        / (
            "ana_slo_bigmarch_oyagi_"
            "nonjuggler_weekday_future_ranking.py"
        ),
    )

    return (
        juggler,
        nonjuggler,
    )


def discover_daily_files(
    data_dir: Path,
) -> list[tuple[date, Path]]:
    found: list[
        tuple[date, Path]
    ] = []

    for path in data_dir.glob(
        "ana_slo_bigmarch_oyagi_*.csv"
    ):
        match = DAILY_RE.fullmatch(
            path.name
        )

        if match is None:
            continue

        file_date = datetime.strptime(
            match.group(1),
            "%Y%m%d",
        ).date()

        found.append(
            (
                file_date,
                path,
            )
        )

    return sorted(
        found,
        key=lambda item: item[0],
    )


def validate_latest_daily(
    path: Path,
    expected_date: date,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    missing = sorted(
        REQUIRED_DAILY_COLUMNS
        - set(frame.columns)
    )

    if missing:
        raise StaleReferenceBlockedError(
            "LATEST_DAILY_SCHEMA_INVALID: "
            f"missing={missing}"
        )

    if len(frame) < MIN_MACHINES:
        raise StaleReferenceBlockedError(
            "LATEST_DAILY_TOO_SMALL: "
            f"rows={len(frame)}"
        )

    dates = (
        pd.to_datetime(
            frame["date"],
            errors="raise",
        )
        .dt.date
        .unique()
        .tolist()
    )

    machine_no = pd.to_numeric(
        frame["machine_no"],
        errors="coerce",
    )

    machine_name = (
        frame["machine_name"]
        .astype("string")
        .str.strip()
    )

    games = pd.to_numeric(
        frame["G"],
        errors="coerce",
    )

    differences = pd.to_numeric(
        frame["diff"],
        errors="coerce",
    )

    if dates != [expected_date]:
        raise StaleReferenceBlockedError(
            "LATEST_DAILY_INTERNAL_DATE_MISMATCH"
        )

    if (
        machine_no.isna().any()
        or machine_no.duplicated().any()
        or (
            machine_no.nunique()
            != len(frame)
        )
    ):
        raise StaleReferenceBlockedError(
            "LATEST_DAILY_MACHINE_NO_INVALID"
        )

    if (
        machine_name.isna().any()
        or machine_name.isin(
            [
                "",
                "nan",
                "None",
            ]
        ).any()
    ):
        raise StaleReferenceBlockedError(
            "LATEST_DAILY_MACHINE_NAME_INVALID"
        )

    if (
        games.isna().any()
        or differences.isna().any()
        or (games < 0).any()
    ):
        raise StaleReferenceBlockedError(
            "LATEST_DAILY_VALUES_INVALID"
        )

    return frame


def assess_eligibility(
    project_root: Path,
    operation_date: date,
) -> Eligibility:
    operation_date = (
        date.fromisoformat(
            str(operation_date)
        )
    )

    expected_data_date = (
        operation_date
        - timedelta(days=1)
    )

    data_dir = (
        project_root
        / DATA_REL
    )

    expected_source_path = (
        project_root
        / (
            "ana_slo_bigmarch_oyagi_"
            f"{expected_data_date:%Y%m%d}"
            "_source.html"
        )
    )

    expected_daily_path = (
        data_dir
        / (
            "ana_slo_bigmarch_oyagi_"
            f"{expected_data_date:%Y%m%d}"
            ".csv"
        )
    )

    if expected_source_path.exists():
        raise StaleReferenceBlockedError(
            "EXPECTED_SOURCE_PRESENT"
        )

    if expected_daily_path.exists():
        raise StaleReferenceBlockedError(
            "EXPECTED_DAILY_PRESENT"
        )

    daily_files = (
        discover_daily_files(
            data_dir
        )
    )

    if not daily_files:
        raise StaleReferenceBlockedError(
            "NO_DAILY_FILES"
        )

    latest_data_date, latest_path = (
        daily_files[-1]
    )

    if latest_data_date >= expected_data_date:
        raise StaleReferenceBlockedError(
            "LATEST_DATE_NOT_BEFORE_EXPECTED"
        )

    source_delay_days = (
        expected_data_date
        - latest_data_date
    ).days

    if source_delay_days < 2:
        raise StaleReferenceBlockedError(
            "SOURCE_DELAY_NOT_STALE: "
            f"delay={source_delay_days}"
        )

    validate_latest_daily(
        latest_path,
        latest_data_date,
    )

    return Eligibility(
        operation_date=operation_date,
        expected_data_date=(
            expected_data_date
        ),
        latest_data_date=(
            latest_data_date
        ),
        source_delay_days=(
            source_delay_days
        ),
        latest_daily_path=latest_path,
        expected_source_path=(
            expected_source_path
        ),
        expected_daily_path=(
            expected_daily_path
        ),
    )


def output_paths(
    project_root: Path,
    operation_date: date,
) -> dict[str, Path]:
    compact = (
        operation_date.strftime(
            "%Y%m%d"
        )
    )

    directory = (
        project_root
        / OUTPUT_REL
        / compact
    )

    prefix = (
        f"91_stale_reference_{compact}"
    )

    return {
        "directory": directory,
        "juggler_all": (
            directory
            / f"{prefix}_juggler_all.csv"
        ),
        "juggler_top10": (
            directory
            / f"{prefix}_juggler_top10.csv"
        ),
        "nonjuggler_all": (
            directory
            / f"{prefix}_nonjuggler_all.csv"
        ),
        "nonjuggler_top10": (
            directory
            / f"{prefix}_nonjuggler_top10.csv"
        ),
        "metadata": (
            directory
            / f"{prefix}_metadata.csv"
        ),
        "status": (
            directory
            / f"{prefix}_status.csv"
        ),
    }


def _source_delay_bucket(
    source_delay_days: int,
) -> str:
    if source_delay_days == 2:
        return "LAG_2"

    return "LAG_3_PLUS"


def _expected_metadata(
    eligibility: Eligibility,
    models: str,
) -> dict:
    return {
        "ranking_class": (
            "STALE_REFERENCE"
        ),
        "formal": False,
        "provisional": False,
        "stale_reference": True,
        "forward_valid": False,
        "operation_date": (
            eligibility
            .operation_date
            .isoformat()
        ),
        "target_date": (
            eligibility
            .operation_date
            .isoformat()
        ),
        "expected_data_date": (
            eligibility
            .expected_data_date
            .isoformat()
        ),
        "latest_data_date": (
            eligibility
            .latest_data_date
            .isoformat()
        ),
        "source_delay_days": (
            eligibility
            .source_delay_days
        ),
        "source_delay_bucket": (
            _source_delay_bucket(
                eligibility
                .source_delay_days
            )
        ),
        "target_to_latest_gap_days": (
            eligibility
            .source_delay_days
            + 1
        ),
        "source_status": (
            "EXPECTED_DATE_MISSING_STALE"
        ),
        "inventory_currentness": (
            "UNCONFIRMED"
        ),
        "machine_mapping": (
            "LAST_KNOWN"
        ),
        "model": models,
        "automatic_promotion": False,
        "eligible_for_formal_evaluation": (
            False
        ),
    }


def _bool_value(
    value,
) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
        }
    )


def validate_existing(
    paths: dict[str, Path],
    expected: dict,
) -> bool:
    artifacts = [
        value
        for key, value
        in paths.items()
        if key != "directory"
    ]

    if not any(
        path.exists()
        for path in artifacts
    ):
        if paths["directory"].exists():
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_DIRECTORY_"
                    "EXISTS_WITHOUT_ARTIFACTS"
                )
            )

        return False

    if not all(
        path.is_file()
        and path.stat().st_size > 0
        for path in artifacts
    ):
        raise (
            StaleReferenceManualReviewError(
                "STALE_REFERENCE_ARTIFACTS_PARTIAL"
            )
        )

    metadata = pd.read_csv(
        paths["metadata"],
        encoding="utf-8-sig",
    )

    if len(metadata) != 1:
        raise (
            StaleReferenceManualReviewError(
                "STALE_REFERENCE_METADATA_INVALID"
            )
        )

    row = metadata.iloc[0]

    for key, value in expected.items():
        actual = row.get(key)

        if isinstance(value, bool):
            matches = (
                _bool_value(actual)
                == value
            )
        else:
            matches = (
                str(actual)
                == str(value)
            )

        if not matches:
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_METADATA_"
                    f"MISMATCH: {key}"
                )
            )

    required = {
        "target_date",
        "expected_data_date",
        "latest_data_date",
        "source_delay_days",
        "source_delay_bucket",
        "ranking_class",
        "formal",
        "provisional",
        "stale_reference",
        "forward_valid",
        "inventory_currentness",
        "machine_mapping",
    }

    for key in (
        "juggler_all",
        "juggler_top10",
        "nonjuggler_all",
        "nonjuggler_top10",
    ):
        frame = pd.read_csv(
            paths[key],
            encoding="utf-8-sig",
        )

        if (
            frame.empty
            or not required.issubset(
                frame.columns
            )
        ):
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_RANKING_"
                    f"INVALID: {key}"
                )
            )

        if not (
            frame["target_date"]
            .astype(str)
            == expected["target_date"]
        ).all():
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_TARGET_"
                    f"MISMATCH: {key}"
                )
            )

        if not (
            frame["expected_data_date"]
            .astype(str)
            == expected[
                "expected_data_date"
            ]
        ).all():
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_EXPECTED_DATE_"
                    f"MISMATCH: {key}"
                )
            )

        if not (
            frame["latest_data_date"]
            .astype(str)
            == expected[
                "latest_data_date"
            ]
        ).all():
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_LATEST_DATE_"
                    f"MISMATCH: {key}"
                )
            )

        if not (
            frame["ranking_class"]
            .astype(str)
            == "STALE_REFERENCE"
        ).all():
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_CLASS_"
                    f"MISMATCH: {key}"
                )
            )

        if (
            frame["formal"]
            .map(_bool_value)
            .any()
        ):
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_FORMAL_FLAG_"
                    f"MISMATCH: {key}"
                )
            )

        if (
            frame["provisional"]
            .map(_bool_value)
            .any()
        ):
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_PROVISIONAL_FLAG_"
                    f"MISMATCH: {key}"
                )
            )

        if not (
            frame["stale_reference"]
            .map(_bool_value)
            .all()
        ):
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_FLAG_"
                    f"MISMATCH: {key}"
                )
            )

        if (
            frame["forward_valid"]
            .map(_bool_value)
            .any()
        ):
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_FORWARD_FLAG_"
                    f"MISMATCH: {key}"
                )
            )

        if not (
            frame["inventory_currentness"]
            .astype(str)
            == "UNCONFIRMED"
        ).all():
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_INVENTORY_"
                    f"CURRENTNESS_MISMATCH: {key}"
                )
            )

        if not (
            frame["machine_mapping"]
            .astype(str)
            == "LAST_KNOWN"
        ).all():
            raise (
                StaleReferenceManualReviewError(
                    "STALE_REFERENCE_MACHINE_MAPPING_"
                    f"MISMATCH: {key}"
                )
            )

    status = pd.read_csv(
        paths["status"],
        encoding="utf-8-sig",
    )

    if (
        len(status) != 1
        or status.iloc[0].get("status")
        != "STALE_REFERENCE"
    ):
        raise (
            StaleReferenceManualReviewError(
                "STALE_REFERENCE_STATUS_INVALID"
            )
        )

    return True


def _annotate(
    ranking: pd.DataFrame,
    eligibility: Eligibility,
) -> pd.DataFrame:
    result = ranking.copy()

    result["target_date"] = (
        eligibility
        .operation_date
        .isoformat()
    )

    result["expected_data_date"] = (
        eligibility
        .expected_data_date
        .isoformat()
    )

    result["latest_data_date"] = (
        eligibility
        .latest_data_date
        .isoformat()
    )

    result["source_delay_days"] = (
        eligibility
        .source_delay_days
    )

    result["source_delay_bucket"] = (
        _source_delay_bucket(
            eligibility
            .source_delay_days
        )
    )

    result["ranking_class"] = (
        "STALE_REFERENCE"
    )

    result["formal"] = False
    result["provisional"] = False
    result["stale_reference"] = True
    result["forward_valid"] = False

    result["inventory_currentness"] = (
        "UNCONFIRMED"
    )

    result["machine_mapping"] = (
        "LAST_KNOWN"
    )

    return result


def generate(
    project_root: Path,
    operation_date: date,
) -> dict:
    project_root = Path(
        project_root
    )

    eligibility = (
        assess_eligibility(
            project_root,
            operation_date,
        )
    )

    data_dir = (
        project_root
        / DATA_REL
    )

    guard_decision = (
        enforce_big_march_stale_reference_inventory_guard(
            data_dir,
            eligibility.operation_date,
        )
    )

    if not guard_decision.reference_allowed:
        raise StaleReferenceBlockedError(
            "INVENTORY_GUARD_REFERENCE_NOT_ALLOWED"
        )

    juggler_module, nonjuggler_module = (
        load_ranking_modules(
            project_root
        )
    )

    models = (
        f"{juggler_module.MODEL_NAME}"
        "|"
        f"{nonjuggler_module.MODEL_NAME}"
    )

    metadata_base = (
        _expected_metadata(
            eligibility,
            models,
        )
    )

    paths = output_paths(
        project_root,
        eligibility.operation_date,
    )

    if validate_existing(
        paths,
        metadata_base,
    ):
        return {
            "status": (
                "ALREADY_STALE_REFERENCE"
            ),
            "paths": paths,
            "metadata": (
                metadata_base
            ),
        }

    (
        juggler_history,
        _,
        _,
    ) = (
        juggler_module
        .load_frozen_history()
    )

    (
        nonjuggler_history,
        _,
        _,
    ) = (
        nonjuggler_module
        .load_frozen_history()
    )

    for label, history in (
        (
            "JUGGLER",
            juggler_history,
        ),
        (
            "NON_JUGGLER",
            nonjuggler_history,
        ),
    ):
        latest_history_date = (
            pd.Timestamp(
                history["date"].max()
            )
            .date()
        )

        if (
            latest_history_date
            != eligibility.latest_data_date
        ):
            raise (
                StaleReferenceBlockedError(
                    f"{label}_HISTORY_LATEST_"
                    "MISMATCH: "
                    f"{latest_history_date}"
                )
            )

    target = pd.Timestamp(
        eligibility.operation_date
    )

    latest = pd.Timestamp(
        eligibility.latest_data_date
    )

    juggler_ranking = _annotate(
        juggler_module
        .build_future_ranking(
            juggler_history,
            latest,
            target,
        ),
        eligibility,
    )

    nonjuggler_ranking = _annotate(
        nonjuggler_module
        .build_future_ranking(
            nonjuggler_history,
            latest,
            target,
        ),
        eligibility,
    )

    frames = {
        "juggler_all": (
            juggler_ranking
        ),
        "juggler_top10": (
            juggler_ranking
            .head(10)
            .copy()
        ),
        "nonjuggler_all": (
            nonjuggler_ranking
        ),
        "nonjuggler_top10": (
            nonjuggler_ranking
            .head(10)
            .copy()
        ),
    }

    parent = (
        paths["directory"]
        .parent
    )

    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=(
                "."
                f"{eligibility.operation_date:%Y%m%d}"
                "_"
            ),
            dir=parent,
        )
    )

    try:
        for key, frame in (
            frames.items()
        ):
            frame.to_csv(
                temporary
                / paths[key].name,
                index=False,
                encoding="utf-8-sig",
            )

        metadata = {
            **metadata_base,
            "generated_at_jst": (
                datetime.now(
                    JST
                ).isoformat()
            ),
        }

        pd.DataFrame(
            [
                metadata,
            ]
        ).to_csv(
            temporary
            / paths["metadata"].name,
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(
            [
                {
                    **metadata,
                    "status": (
                        "STALE_REFERENCE"
                    ),
                },
            ]
        ).to_csv(
            temporary
            / paths["status"].name,
            index=False,
            encoding="utf-8-sig",
        )

        os.replace(
            temporary,
            paths["directory"],
        )

    except Exception:
        if temporary.exists():
            shutil.rmtree(
                temporary
            )

        raise

    return {
        "status": "STALE_REFERENCE",
        "paths": paths,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate isolated Big March "
            "STALE_REFERENCE rankings."
        )
    )

    parser.add_argument(
        "--operation-date",
        required=True,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = generate(
            args.project_root,
            date.fromisoformat(
                args.operation_date
            ),
        )

    except (
        StaleReferenceManualReviewError
    ) as exc:
        print(
            json.dumps(
                {
                    "status": (
                        "MANUAL_REVIEW"
                    ),
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )

        return 2

    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                },
                ensure_ascii=False,
            )
        )

        return 1

    print(
        json.dumps(
            {
                "status": (
                    result["status"]
                ),
                "target_date": (
                    args.operation_date
                ),
            },
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )