from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import pandas as pd


# ============================================================
# Bic Tsubame Takasaki - One-Click Daily Update V1
#
# Pipeline:
#   1) Ensure Chrome CDP (9222)
#   2) Fetch the expected latest Ana-Slo HTML
#   3) Convert that HTML -> daily CSV
#   4) Freshness Guard
#
# IMPORTANT:
#   - No Forward/ranking stage is connected yet.
#   - No --allow-gap escape hatch is provided.
#   - Known renovation closure 2026-07-21 through 2026-08-07
#     is not treated as missing/freshness failure.
#   - Child-script failures propagate as non-zero.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

MACHINE_DIR = (
    PROJECT_ROOT
    / "machine_number"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "bic_tsubame_takasaki"
    / "machine_number"
)


SCRIPT_FETCH = (
    MACHINE_DIR
    / "ana_slo_bic_tsubame_takasaki_click_fetch_range_v4.py"
)

SCRIPT_CONVERT = (
    MACHINE_DIR
    / "ana_slo_bic_tsubame_takasaki_batch_html_to_daily_csv.py"
)


DAILY_FILE_RE = re.compile(
    r"^ana_slo_bic_tsubame_takasaki_(\d{8})\.csv$",
    re.IGNORECASE,
)


CDP_PORT = 9222

CDP_VERSION_URL = (
    f"http://127.0.0.1:{CDP_PORT}/json/version"
)

REMOTE_PROFILE_DIR = (
    PROJECT_ROOT
    / ".chrome_remote_profile_9222"
)


CHROME_CANDIDATES = [
    (
        Path(
            os.environ.get(
                "PROGRAMFILES",
                r"C:\Program Files",
            )
        )
        / "Google"
        / "Chrome"
        / "Application"
        / "chrome.exe"
    ),
    (
        Path(
            os.environ.get(
                "PROGRAMFILES(X86)",
                r"C:\Program Files (x86)",
            )
        )
        / "Google"
        / "Chrome"
        / "Application"
        / "chrome.exe"
    ),
    (
        Path(
            os.environ.get(
                "LOCALAPPDATA",
                str(
                    Path.home()
                    / "AppData"
                    / "Local"
                ),
            )
        )
        / "Google"
        / "Chrome"
        / "Application"
        / "chrome.exe"
    ),
]


KNOWN_CLOSURES = (
    (
        date(2026, 7, 21),
        date(2026, 8, 7),
        "STORE_RENOVATION",
    ),
)


# ============================================================
# Generic helpers
# ============================================================

def header(title: str) -> None:
    print()
    print("=" * 124)
    print(title)
    print("=" * 124)


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}. Use YYYY-MM-DD."
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-click daily update foundation for "
            "Bic Tsubame Takasaki."
        )
    )

    parser.add_argument(
        "--operation-date",
        type=parse_iso_date,
        default=None,
        help=(
            "Operation date used to determine the expected latest "
            "data date. Default: local system date."
        ),
    )

    parser.add_argument(
        "--min-machines",
        type=int,
        default=200,
        help=(
            "Minimum acceptable machine rows. "
            "Default: 200."
        ),
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip HTML fetch stage.",
    )

    parser.add_argument(
        "--chrome-wait-sec",
        type=int,
        default=15,
        help=(
            "Seconds to wait for auto-started Chrome CDP. "
            "Default: 15."
        ),
    )

    return parser.parse_args()


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found:\n{path}"
        )


def closure_reason(target: date) -> str | None:
    for start_date, end_date, reason in KNOWN_CLOSURES:
        if start_date <= target <= end_date:
            return reason
    return None


def expected_latest_data_date(
    operation_date: date,
) -> tuple[date, list[tuple[date, str]]]:
    """
    Normally the expected latest data date is operation_date - 1 day.

    If that date falls inside a known closure, walk backward until
    the most recent non-closure date. The skipped closure dates are
    returned for transparent logging.
    """

    candidate = operation_date - timedelta(days=1)
    skipped = []

    while True:
        reason = closure_reason(candidate)

        if reason is None:
            return candidate, skipped

        skipped.append(
            (candidate, reason)
        )

        candidate -= timedelta(days=1)


# ============================================================
# CDP helpers
# ============================================================

def try_get_cdp(
    timeout: float = 2.0,
) -> dict | None:
    try:
        with urllib.request.urlopen(
            CDP_VERSION_URL,
            timeout=timeout,
        ) as response:
            raw = response.read().decode(
                "utf-8"
            )

        info = json.loads(raw)

        if not info.get(
            "webSocketDebuggerUrl"
        ):
            return None

        return info

    except Exception:
        return None


def find_chrome_exe() -> Path:
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Google Chrome executable was not found "
        "in the standard Windows locations."
    )


def start_remote_debug_chrome(
    wait_sec: int,
) -> dict:
    chrome = find_chrome_exe()

    REMOTE_PROFILE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    header(
        "AUTO START REMOTE-DEBUGGING CHROME"
    )

    print(
        f"Chrome executable     : {chrome}"
    )
    print(
        f"remote debug port     : {CDP_PORT}"
    )
    print(
        f"remote profile        : {REMOTE_PROFILE_DIR}"
    )

    command = [
        str(chrome),
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={REMOTE_PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank",
    ]

    creationflags = 0

    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )

    subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    deadline = (
        time.time()
        + wait_sec
    )

    while time.time() < deadline:
        info = try_get_cdp(
            timeout=1.0
        )

        if info is not None:
            print(
                f"Chrome                : "
                f"{info.get('Browser')}"
            )
            print(
                "CDP                   : AUTO-START OK"
            )
            return info

        time.sleep(0.5)

    raise RuntimeError(
        "Chrome was started, but CDP did not "
        f"become ready within {wait_sec} seconds."
    )


def ensure_cdp(
    wait_sec: int,
) -> dict:
    info = try_get_cdp()

    if info is not None:
        print(
            f"Chrome                : "
            f"{info.get('Browser')}"
        )
        print(
            "CDP                   : ALREADY RUNNING"
        )
        return info

    print(
        "CDP                   : NOT RUNNING"
    )
    print(
        "action                : auto-start Chrome"
    )

    return start_remote_debug_chrome(
        wait_sec
    )


# ============================================================
# Child-script helpers
# ============================================================

def compile_script(
    path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(path),
        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"py_compile failed: {path.name}"
        )


def run_stage(
    label: str,
    script: Path,
    args: list[str] | None = None,
) -> float:
    args = args or []

    header(label)

    command = [
        sys.executable,
        str(script),
        *args,
    ]

    print(
        "command               : "
        + " ".join(command)
    )

    started = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print()
    print(
        f"{label} return code    : "
        f"{result.returncode}"
    )
    print(
        f"{label} elapsed sec    : "
        f"{elapsed:.2f}"
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with return code "
            f"{result.returncode}."
        )

    return elapsed


# ============================================================
# Daily CSV discovery / Freshness Guard
# ============================================================

def discover_daily_csv_files():
    found = []

    for path in DATA_DIR.glob(
        "ana_slo_bic_tsubame_takasaki_*.csv"
    ):
        match = DAILY_FILE_RE.fullmatch(
            path.name
        )

        if not match:
            continue

        file_date = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="raise",
        ).normalize()

        found.append(
            (
                file_date,
                path,
            )
        )

    if not found:
        raise RuntimeError(
            "No Bic Tsubame Takasaki daily CSV files "
            "were found."
        )

    return sorted(
        found,
        key=lambda item: item[0],
    )


def freshness_guard(
    *,
    operation_date: date,
    min_machines: int,
) -> dict:
    header(
        "FRESHNESS GUARD"
    )

    daily_files = (
        discover_daily_csv_files()
    )

    latest_file_date, latest_path = (
        daily_files[-1]
    )

    raw = pd.read_csv(
        latest_path,
        encoding="utf-8-sig",
    )

    required = {
        "date",
        "machine_name",
        "machine_no",
        "G",
        "diff",
    }

    missing = sorted(
        required - set(raw.columns)
    )

    if missing:
        raise RuntimeError(
            f"{latest_path.name}: missing required "
            f"columns: {missing}"
        )

    raw["date"] = pd.to_datetime(
        raw["date"],
        errors="raise",
    ).dt.normalize()

    raw["machine_no"] = pd.to_numeric(
        raw["machine_no"],
        errors="raise",
    ).astype(int)

    raw["G"] = pd.to_numeric(
        raw["G"],
        errors="raise",
    )

    raw["diff"] = pd.to_numeric(
        raw["diff"],
        errors="raise",
    )

    internal_dates = (
        raw["date"]
        .drop_duplicates()
        .tolist()
    )

    if len(internal_dates) != 1:
        raise RuntimeError(
            f"{latest_path.name}: multiple internal "
            "dates exist."
        )

    internal_date = (
        internal_dates[0]
    )

    if internal_date != latest_file_date:
        raise RuntimeError(
            f"{latest_path.name}: filename date "
            f"{latest_file_date.date()} does not match "
            f"internal date {internal_date.date()}."
        )

    rows = len(raw)

    unique_machines = int(
        raw["machine_no"].nunique()
    )

    duplicates = int(
        raw["machine_no"]
        .duplicated(
            keep=False
        )
        .sum()
    )

    missing_name = int(
        raw["machine_name"]
        .astype(str)
        .str.strip()
        .isin(["", "nan", "None"])
        .sum()
    )

    negative_g = int(
        ((raw["G"] < 0).fillna(False))
        .sum()
    )

    if rows < min_machines:
        raise RuntimeError(
            f"{latest_path.name}: machine rows below "
            f"minimum. rows={rows}, "
            f"minimum={min_machines}"
        )

    if unique_machines != rows:
        raise RuntimeError(
            f"{latest_path.name}: machine_no is not "
            f"unique. rows={rows}, "
            f"unique={unique_machines}"
        )

    if duplicates != 0:
        raise RuntimeError(
            f"{latest_path.name}: duplicate "
            f"machine_no rows exist. "
            f"duplicates={duplicates}"
        )

    if missing_name != 0:
        raise RuntimeError(
            f"{latest_path.name}: missing machine_name "
            f"rows exist. missing={missing_name}"
        )

    if negative_g != 0:
        raise RuntimeError(
            f"{latest_path.name}: negative G rows exist. "
            f"negative={negative_g}"
        )

    expected_latest_date, skipped_closures = (
        expected_latest_data_date(
            operation_date
        )
    )

    expected_latest = pd.Timestamp(
        expected_latest_date
    ).normalize()

    target_date = (
        latest_file_date
        + pd.Timedelta(days=1)
    ).normalize()

    print(
        f"latest daily CSV       : "
        f"{latest_path.name}"
    )
    print(
        f"latest data date       : "
        f"{latest_file_date.date()}"
    )
    print(
        f"internal date          : "
        f"{internal_date.date()}"
    )
    print(
        f"rows                   : "
        f"{rows}"
    )
    print(
        f"unique machines        : "
        f"{unique_machines}"
    )
    print(
        f"minimum machines       : "
        f"{min_machines}"
    )
    print(
        f"operation date         : "
        f"{operation_date}"
    )
    print(
        f"expected latest        : "
        f"{expected_latest.date()}"
    )

    if skipped_closures:
        print(
            "known closure skipped  : "
            f"{len(skipped_closures)} day(s)"
        )

        for closure_date, reason in skipped_closures:
            print(
                f"  {closure_date} : {reason}"
            )
    else:
        print(
            "known closure skipped  : 0"
        )

    if latest_file_date > expected_latest:
        raise RuntimeError(
            "Latest Bic Tsubame daily CSV is dated "
            "in the future relative to the expected "
            "operation date. "
            f"latest={latest_file_date.date()}, "
            f"expected={expected_latest.date()}"
        )

    if latest_file_date != expected_latest:
        raise RuntimeError(
            "Bic Tsubame daily CSV is stale. "
            f"latest={latest_file_date.date()}, "
            f"expected={expected_latest.date()}. "
            "No gap override is permitted."
        )

    print()
    print(
        "FRESHNESS RESULT       : OK"
    )

    return {
        "latest_path": latest_path,
        "latest_data_date": latest_file_date,
        "expected_latest_date": expected_latest,
        "target_date": target_date,
        "rows": rows,
        "unique_machines": unique_machines,
        "skipped_closures": skipped_closures,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if args.min_machines < 1:
        raise ValueError(
            "--min-machines must be >= 1"
        )

    if args.chrome_wait_sec < 1:
        raise ValueError(
            "--chrome-wait-sec must be >= 1"
        )

    operation_date = (
        args.operation_date
        if args.operation_date is not None
        else date.today()
    )

    expected_latest_date, closure_skips = (
        expected_latest_data_date(
            operation_date
        )
    )

    header(
        "Bic Tsubame Takasaki - "
        "One-Click Daily Update V1"
    )

    print(
        f"project root          : "
        f"{PROJECT_ROOT}"
    )
    print(
        f"python                : "
        f"{sys.executable}"
    )
    print(
        f"operation date        : "
        f"{operation_date}"
    )
    print(
        f"expected latest       : "
        f"{expected_latest_date}"
    )
    print(
        f"min machines          : "
        f"{args.min_machines}"
    )
    print(
        f"skip fetch            : "
        f"{args.skip_fetch}"
    )
    print(
        "gap override          : DISABLED"
    )

    if closure_skips:
        print(
            f"known closure adjusted: "
            f"{len(closure_skips)} day(s)"
        )
    else:
        print(
            "known closure adjusted: 0"
        )

    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------

    header(
        "PREFLIGHT"
    )

    required_scripts = (
        SCRIPT_FETCH,
        SCRIPT_CONVERT,
    )

    for path in required_scripts:
        check_file(path)

        print(
            f"script exists         : "
            f"{path.name}"
        )

    # Match existing project behavior:
    # confirm/start CDP even when --skip-fetch is used.
    ensure_cdp(
        args.chrome_wait_sec
    )

    print()
    print(
        "Compiling required scripts..."
    )

    for path in required_scripts:
        compile_script(path)

        print(
            f"py_compile OK         : "
            f"{path.name}"
        )

    total_started = (
        time.perf_counter()
    )

    elapsed_rows = []

    # --------------------------------------------------------
    # 1) Fetch expected latest HTML
    # --------------------------------------------------------

    if not args.skip_fetch:
        elapsed = run_stage(
            "FETCH EXPECTED LATEST HTML",
            SCRIPT_FETCH,
            [
                "--start-date",
                expected_latest_date.isoformat(),
                "--end-date",
                expected_latest_date.isoformat(),
                "--min-machines",
                str(args.min_machines),
            ],
        )

        elapsed_rows.append(
            (
                "FETCH EXPECTED LATEST HTML",
                elapsed,
            )
        )

    else:
        print()
        print(
            "FETCH EXPECTED LATEST HTML : SKIPPED"
        )

    # --------------------------------------------------------
    # 2) HTML -> daily CSV
    # --------------------------------------------------------

    elapsed = run_stage(
        "HTML TO DAILY CSV",
        SCRIPT_CONVERT,
        [
            "--only-date",
            expected_latest_date.isoformat(),
            "--min-machines",
            str(args.min_machines),
        ],
    )

    elapsed_rows.append(
        (
            "HTML TO DAILY CSV",
            elapsed,
        )
    )

    # --------------------------------------------------------
    # 3) Freshness Guard
    # --------------------------------------------------------

    freshness_started = (
        time.perf_counter()
    )

    freshness = freshness_guard(
        operation_date=operation_date,
        min_machines=args.min_machines,
    )

    freshness_elapsed = (
        time.perf_counter()
        - freshness_started
    )

    elapsed_rows.append(
        (
            "FRESHNESS GUARD",
            freshness_elapsed,
        )
    )

    total_elapsed = (
        time.perf_counter()
        - total_started
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    header(
        "BIC TSUBAME DAILY UPDATE SUMMARY"
    )

    for label, elapsed in elapsed_rows:
        print(
            f"{label:<32}: "
            f"OK  ({elapsed:.2f} sec)"
        )

    print()
    print(
        f"latest data date       : "
        f"{freshness['latest_data_date'].date()}"
    )
    print(
        f"expected latest        : "
        f"{freshness['expected_latest_date'].date()}"
    )
    print(
        f"validated machines     : "
        f"{freshness['unique_machines']}"
    )
    print(
        f"total elapsed sec      : "
        f"{total_elapsed:.2f}"
    )

    print()
    print(
        "FINAL RESULT          : OK"
    )
    print()
    print(
        "Bic Tsubame Takasaki daily update "
        "foundation completed."
    )
    print(
        "No Forward/ranking stage is connected yet."
    )
    print(
        "No gap override is permitted."
    )
    print(
        "Known closure dates are explicitly excluded "
        "from freshness-gap interpretation."
    )
    print(
        "No other store files were modified."
    )


if __name__ == "__main__":
    main()
