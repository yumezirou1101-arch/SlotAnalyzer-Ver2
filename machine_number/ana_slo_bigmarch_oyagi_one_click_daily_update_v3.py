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
# 09 V3 - Big March Takasaki Oyagi
# True One-Click Daily Update
# ============================================================
#
# If Chrome remote debugging (9222) is not running:
#   - Detect Chrome automatically
#   - Start a separate Chrome instance with:
#       --remote-debugging-port=9222
#       dedicated user-data-dir under SlotAnalyzer
#   - Wait for CDP to become ready
#
# Pipeline:
#   1) Fetch newest Oyagi Ana-Slo HTML
#   2) Convert HTML -> daily CSV
#   2.5) Freshness guard
#   3) Frozen JUGGLER_RECENT7_WIN Top3 forward
#   4) Frozen NON_JUGGLER_WEEKDAY_AVG Top1 forward
#   5) Juggler future ranking
#   6) Non-Juggler future ranking
#
# IMPORTANT:
#   - Frozen development period through 2026-08-26
#     remains locked.
#   - No automatic model promotion is performed.
#   - A child-script failure propagates to this V3 runner.
#   - Freshness guard prevents a stale daily CSV from being
#     treated as a successful morning update.
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
    / "bigmarch_takasaki_oyagi"
    / "machine_number"
)


SCRIPT_FETCH = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_click_fetch_31days_v3.py"
)

SCRIPT_CONVERT = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_batch_html_to_daily_csv.py"
)

SCRIPT_JUGGLER_FORWARD = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_juggler_recent7_top3_forward.py"
)

SCRIPT_NONJUGGLER_FORWARD = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_nonjuggler_weekday_top1_forward.py"
)

SCRIPT_JUGGLER_FUTURE = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_juggler_recent7_future_ranking.py"
)

SCRIPT_NONJUGGLER_FUTURE = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_nonjuggler_weekday_future_ranking.py"
)


DAILY_FILE_RE = re.compile(
    r"^ana_slo_bigmarch_oyagi_(\d{8})\.csv$",
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


def header(title: str) -> None:
    print()
    print("=" * 124)
    print(title)
    print("=" * 124)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "True one-click daily update for "
            "Big March Takasaki Oyagi."
        )
    )

    parser.add_argument(
        "--fetch-days",
        type=int,
        default=1,
        help=(
            "Newest N date links to inspect. "
            "Default: 1."
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
            "Seconds to wait for auto-started "
            "Chrome CDP. Default: 15."
        ),
    )

    parser.add_argument(
        "--allow-gap",
        action="store_true",
        help=(
            "Allow latest daily CSV to be older than "
            "yesterday. Intended only for controlled "
            "maintenance/testing."
        ),
    )

    return parser.parse_args()


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found:\n{path}"
        )


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

        info = json.loads(
            raw
        )

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

        time.sleep(
            0.5
        )

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

    header(
        label
    )

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


def discover_daily_csv_files():
    found = []

    for path in DATA_DIR.glob(
        "ana_slo_bigmarch_oyagi_*.csv"
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
            "No Big March Oyagi daily CSV files "
            "were found."
        )

    return sorted(
        found,
        key=lambda x: x[0],
    )


def freshness_guard(
    min_machines: int,
    allow_gap: bool,
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

    today = pd.Timestamp(
        date.today()
    ).normalize()

    expected_latest = (
        today
        - pd.Timedelta(days=1)
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
        f"today                  : "
        f"{today.date()}"
    )
    print(
        f"expected latest        : "
        f"{expected_latest.date()}"
    )
    print(
        f"prediction target      : "
        f"{target_date.date()}"
    )
    print(
        f"allow gap              : "
        f"{allow_gap}"
    )

    if latest_file_date > expected_latest:
        raise RuntimeError(
            "Latest Big March daily CSV is dated "
            "in the future relative to expected "
            "morning operation. "
            f"latest={latest_file_date.date()}, "
            f"expected={expected_latest.date()}"
        )

    if (
        not allow_gap
        and latest_file_date != expected_latest
    ):
        raise RuntimeError(
            "Big March daily CSV is stale. "
            f"latest={latest_file_date.date()}, "
            f"expected={expected_latest.date()}. "
            "Use --allow-gap only for controlled "
            "maintenance/testing."
        )

    print()
    print(
        "FRESHNESS RESULT       : OK"
    )

    return {
        "latest_path": latest_path,
        "latest_data_date": latest_file_date,
        "target_date": target_date,
        "rows": rows,
        "unique_machines": unique_machines,
    }


def main() -> None:

    args = parse_args()

    if args.fetch_days < 1:
        raise ValueError(
            "--fetch-days must be >= 1"
        )

    if args.min_machines < 1:
        raise ValueError(
            "--min-machines must be >= 1"
        )

    if args.chrome_wait_sec < 1:
        raise ValueError(
            "--chrome-wait-sec must be >= 1"
        )

    header(
        "09 V3 - Big March Takasaki Oyagi "
        "True One-Click Daily Update"
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
        f"fetch days            : "
        f"{args.fetch_days}"
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
        f"allow gap             : "
        f"{args.allow_gap}"
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
        SCRIPT_JUGGLER_FORWARD,
        SCRIPT_NONJUGGLER_FORWARD,
        SCRIPT_JUGGLER_FUTURE,
        SCRIPT_NONJUGGLER_FUTURE,
    )

    for path in required_scripts:

        check_file(
            path
        )

        print(
            f"script exists         : "
            f"{path.name}"
        )

    # Preserve existing V3 behavior:
    # CDP is confirmed even when --skip-fetch is used.
    ensure_cdp(
        args.chrome_wait_sec
    )

    print()
    print(
        "Compiling required scripts..."
    )

    for path in required_scripts:

        compile_script(
            path
        )

        print(
            f"py_compile OK         : "
            f"{path.name}"
        )

    total_started = (
        time.perf_counter()
    )

    elapsed_rows = []

    # --------------------------------------------------------
    # 1) Fetch newest HTML
    # --------------------------------------------------------
    if not args.skip_fetch:

        elapsed = run_stage(
            "FETCH NEWEST HTML",
            SCRIPT_FETCH,
            [
                "--max-days",
                str(
                    args.fetch_days
                ),
                "--min-machines",
                str(
                    args.min_machines
                ),
            ],
        )

        elapsed_rows.append(
            (
                "FETCH NEWEST HTML",
                elapsed,
            )
        )

    else:
        print()
        print(
            "FETCH NEWEST HTML     : SKIPPED"
        )

    # --------------------------------------------------------
    # 2) HTML -> daily CSV
    # --------------------------------------------------------
    elapsed = run_stage(
        "BATCH HTML TO DAILY CSV",
        SCRIPT_CONVERT,
        [
            "--min-machines",
            str(
                args.min_machines
            ),
        ],
    )

    elapsed_rows.append(
        (
            "BATCH HTML TO DAILY CSV",
            elapsed,
        )
    )

    # --------------------------------------------------------
    # 2.5) Freshness guard
    # --------------------------------------------------------
    freshness_started = (
        time.perf_counter()
    )

    freshness = freshness_guard(
        min_machines=args.min_machines,
        allow_gap=args.allow_gap,
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

    # --------------------------------------------------------
    # 3) Juggler frozen forward
    # --------------------------------------------------------
    elapsed = run_stage(
        "JUGGLER FROZEN FORWARD",
        SCRIPT_JUGGLER_FORWARD,
    )

    elapsed_rows.append(
        (
            "JUGGLER FROZEN FORWARD",
            elapsed,
        )
    )

    # --------------------------------------------------------
    # 4) Non-Juggler frozen forward
    # --------------------------------------------------------
    elapsed = run_stage(
        "NONJUGGLER FROZEN FORWARD",
        SCRIPT_NONJUGGLER_FORWARD,
    )

    elapsed_rows.append(
        (
            "NONJUGGLER FROZEN FORWARD",
            elapsed,
        )
    )

    # --------------------------------------------------------
    # 5) Juggler future ranking
    # --------------------------------------------------------
    elapsed = run_stage(
        "JUGGLER FUTURE RANKING",
        SCRIPT_JUGGLER_FUTURE,
    )

    elapsed_rows.append(
        (
            "JUGGLER FUTURE RANKING",
            elapsed,
        )
    )

    # --------------------------------------------------------
    # 6) Non-Juggler future ranking
    # --------------------------------------------------------
    elapsed = run_stage(
        "NONJUGGLER FUTURE RANKING",
        SCRIPT_NONJUGGLER_FUTURE,
    )

    elapsed_rows.append(
        (
            "NONJUGGLER FUTURE RANKING",
            elapsed,
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
        "09 V3 PIPELINE SUMMARY"
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
        f"prediction target      : "
        f"{freshness['target_date'].date()}"
    )
    print(
        f"validated machines     : "
        f"{freshness['unique_machines']}"
    )

    print()
    print(
        f"total elapsed sec      : "
        f"{total_elapsed:.2f}"
    )

    print()
    print(
        "09 V3 daily update complete."
    )
    print(
        "If CDP was not running, Chrome was "
        "started automatically."
    )
    print(
        "If the Oyagi store tab was absent, "
        "fetch V3 opened it automatically."
    )
    print(
        "Freshness validation passed before "
        "forward evaluation and future ranking."
    )
    print(
        "The development period through "
        "2026-08-26 remains locked."
    )
    print(
        "Juggler and Non-Juggler frozen "
        "forward evaluations were updated."
    )
    print(
        "Juggler and Non-Juggler future "
        "rankings were generated."
    )
    print(
        "No automatic model promotion "
        "is performed."
    )
    print(
        "No Maruhan Maebashi files were "
        "modified."
    )


if __name__ == "__main__":
    main()