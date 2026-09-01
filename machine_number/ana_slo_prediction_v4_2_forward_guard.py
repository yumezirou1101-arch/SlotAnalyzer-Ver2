from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib

import pandas as pd


JST = ZoneInfo("Asia/Tokyo")
FORWARD_CUTOFF_HOUR_JST = 9
FORWARD_CUTOFF_JST = time(
    hour=FORWARD_CUTOFF_HOUR_JST,
    minute=0,
)
FORWARD_GUARD_VERSION = "1.0"


def normalize_date(value) -> pd.Timestamp:
    return pd.Timestamp(
        pd.to_datetime(
            value,
            errors="raise",
        )
    ).normalize()


def now_jst() -> datetime:
    return datetime.now(
        JST
    )


def validate_forward_time(
    target_date,
    current_jst: datetime | None = None,
) -> datetime:
    target = normalize_date(
        target_date
    )
    current = (
        current_jst
        if current_jst is not None
        else now_jst()
    )

    if current.tzinfo is None:
        current = current.replace(
            tzinfo=JST
        )
    else:
        current = current.astimezone(
            JST
        )

    today = pd.Timestamp(
        current.date()
    )

    if target < today:
        raise RuntimeError(
            "FORWARD_GUARD_REJECTED: target date is in the past. "
            f"target={target.date()}, today_jst={today.date()}"
        )

    if (
        target == today
        and current.time().replace(
            tzinfo=None
        ) >= FORWARD_CUTOFF_JST
    ):
        raise RuntimeError(
            "FORWARD_GUARD_REJECTED: same-day formal prediction "
            "deadline has passed. "
            f"target={target.date()}, now_jst={current.isoformat()}, "
            f"cutoff={FORWARD_CUTOFF_HOUR_JST:02d}:00 JST"
        )

    return current


def validate_consecutive_latest_date(
    target_date,
    latest_data_date,
) -> None:
    target = normalize_date(
        target_date
    )
    latest = normalize_date(
        latest_data_date
    )
    expected = target - pd.Timedelta(
        days=1
    )

    if latest != expected:
        raise RuntimeError(
            "FORWARD_GUARD_REJECTED: latest data date must be exactly "
            "one day before target. "
            f"latest={latest.date()}, expected={expected.date()}, "
            f"target={target.date()}"
        )


def target_actual_paths(
    project_root: Path,
    data_dir: Path,
    target_date,
) -> tuple[Path, Path]:
    target = normalize_date(
        target_date
    )
    ymd = target.strftime(
        "%Y%m%d"
    )
    return (
        data_dir / f"ana_slo_{ymd}.csv",
        project_root / f"ana_slo_{ymd}_source.html",
    )


def validate_target_actual_absent(
    project_root: Path,
    data_dir: Path,
    target_date,
) -> tuple[Path, Path]:
    actual_csv, source_html = target_actual_paths(
        project_root,
        data_dir,
        target_date,
    )

    if actual_csv.exists():
        raise RuntimeError(
            "FORWARD_GUARD_REJECTED: target actual daily CSV already exists: "
            f"{actual_csv}"
        )

    if source_html.exists():
        raise RuntimeError(
            "FORWARD_GUARD_REJECTED: target source HTML already exists: "
            f"{source_html}"
        )

    return (
        actual_csv,
        source_html,
    )


def formal_output_paths(
    output_dir: Path,
    target_date,
) -> tuple[Path, Path, Path]:
    target = normalize_date(
        target_date
    )
    ymd = target.strftime(
        "%Y%m%d"
    )
    return (
        output_dir / f"64_prediction_{ymd}_all514.csv",
        output_dir / f"64_prediction_{ymd}_top10.csv",
        output_dir / f"64_prediction_{ymd}_metadata.csv",
    )


def validate_not_frozen(
    output_dir: Path,
    target_date,
) -> tuple[Path, Path, Path]:
    paths = formal_output_paths(
        output_dir,
        target_date,
    )
    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if len(existing) == len(paths):
        raise RuntimeError(
            "ALREADY_FROZEN: all formal 64 prediction files already exist "
            "and will not be overwritten.\n"
            + "\n".join(
                str(path)
                for path in existing
            )
        )

    if existing:
        missing = [
            path
            for path in paths
            if not path.exists()
        ]
        raise RuntimeError(
            "PARTIAL_FROZEN_OUTPUT: only part of the formal 64 prediction "
            "set exists; automatic repair and overwrite are forbidden.\n"
            "existing:\n"
            + "\n".join(
                str(path)
                for path in existing
            )
            + "\nmissing:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    return paths


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )
    return digest.hexdigest()
