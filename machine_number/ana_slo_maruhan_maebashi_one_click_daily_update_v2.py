from __future__ import annotations

from pathlib import Path
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
# Maruhan Mega City Maebashi Inter
# True One-Click Daily Update V2
# ============================================================
#
# Flow
# ----
# 0) Preflight
#    - verify required scripts
#    - ensure Chrome remote debugging on 127.0.0.1:9222
#      (auto-start dedicated Chrome if needed)
#
# 1) Fetch newest Ana-Slo page
#    - ana_slo_maruhan_maebashi_click_fetch_v3.py
#
# 2) Convert newest source HTML -> validated daily CSV
#    - ana_slo_source_html_to_daily_csv_auto.py
#    - strict 514-machine validation is performed by this converter
#
# 3) Evaluate already-saved live predictions against newly available actual data
#    - ana_slo_prediction_v4_2_live_prediction_backtest.py
#
# 4) Update Champion / Challenger forward tracking
#    - ana_slo_prediction_v4_2_forward_champion_challenger.py
#
# 5) Build live predictions
#    - ana_slo_prediction_v4_2_one_click_live_pipeline.py
#      -> 64 NORMAL
#      -> 74 A-TYPE
#      -> 75 JUGGLER
#      -> 77 INTEGRATED
#
# Safety
# ------
# - Existing source HTML is skipped unless child fetch rules say otherwise
# - Existing prediction model weights are not recalculated here
# - 78 lottery filtering is NOT run here
# - 76 actual-result evaluation is NOT run here
# - Big March files are not modified
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
    / "maruhan_maebashi"
    / "machine_number"
)

SCRIPT_FETCH = (
    MACHINE_DIR
    / "ana_slo_maruhan_maebashi_click_fetch_v3.py"
)

SCRIPT_CONVERT = (
    PROJECT_ROOT
    / "ana_slo_source_html_to_daily_csv_auto.py"
)

SCRIPT_69 = (
    MACHINE_DIR
    / "ana_slo_prediction_v4_2_live_prediction_backtest.py"
)

SCRIPT_FORWARD = (
    MACHINE_DIR
    / "ana_slo_prediction_v4_2_forward_champion_challenger.py"
)

SCRIPT_LIVE = (
    MACHINE_DIR
    / "ana_slo_prediction_v4_2_one_click_live_pipeline.py"
)

BACKTEST_69_DIR = (
    DATA_DIR
    / "analysis_31days_deep"
    / "69_Ver4_2_live_prediction_backtest"
)

CDP_PORT = 9222

CDP_VERSION_URL = (
    f"http://127.0.0.1:{CDP_PORT}/json/version"
)

REMOTE_PROFILE_DIR = (
    PROJECT_ROOT
    / ".chrome_remote_profile_9222"
)

SOURCE_RE = re.compile(
    r"^ana_slo_(\d{8})_source\.html$",
    re.IGNORECASE,
)

CHROME_CANDIDATES = [
    Path(
        os.environ.get(
            "PROGRAMFILES",
            r"C:\Program Files",
        )
    )
    / "Google"
    / "Chrome"
    / "Application"
    / "chrome.exe",

    Path(
        os.environ.get(
            "PROGRAMFILES(X86)",
            r"C:\Program Files (x86)",
        )
    )
    / "Google"
    / "Chrome"
    / "Application"
    / "chrome.exe",

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
    / "chrome.exe",
]


def header(
    title: str,
) -> None:
    print()
    print(
        "=" * 126
    )
    print(
        title
    )
    print(
        "=" * 126
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "True one-click daily update for "
            "Maruhan Mega City Maebashi Inter."
        )
    )

    parser.add_argument(
        "--fetch-days",
        type=int,
        default=1,
        help=(
            "Newest N Ana-Slo date links to inspect. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help=(
            "Skip HTML fetch and use the latest existing "
            "ana_slo_YYYYMMDD_source.html."
        ),
    )

    parser.add_argument(
        "--target-date",
        default=None,
        help=(
            "Optional prediction target date YYYY-MM-DD "
            "passed to the 79 live pipeline."
        ),
    )

    parser.add_argument(
        "--allow-gap",
        action="store_true",
        help=(
            "Pass --allow-gap to the 79 live pipeline "
            "for intentionally non-consecutive prediction dates."
        ),
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


def require_file(
    path: Path,
) -> None:
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
            raw = (
                response
                .read()
                .decode(
                    "utf-8"
                )
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
        str(
            chrome
        ),
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

    while (
        time.time()
        < deadline
    ):
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
        "Chrome was started, but CDP did not become ready "
        f"within {wait_sec} seconds."
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
            str(
                path
            ),
        ],
        cwd=PROJECT_ROOT,
    )

    if (
        result.returncode
        != 0
    ):
        raise RuntimeError(
            f"py_compile failed: {path.name}"
        )


def run_stage(
    label: str,
    script: Path,
    args: list[str] | None = None,
) -> float:
    args = (
        args
        or []
    )

    header(
        label
    )

    command = [
        sys.executable,
        str(
            script
        ),
        *args,
    ]

    print(
        "command               : "
        + " ".join(
            f'"{value}"'
            if " " in value
            else value
            for value in command
        )
    )

    started = (
        time.perf_counter()
    )

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

    if (
        result.returncode
        != 0
    ):
        raise RuntimeError(
            f"{label} failed with return code "
            f"{result.returncode}."
        )

    return elapsed


def discover_latest_source() -> tuple[
    Path,
    pd.Timestamp,
]:
    candidates = []

    for path in PROJECT_ROOT.glob(
        "ana_slo_????????_source.html"
    ):
        match = SOURCE_RE.fullmatch(
            path.name
        )

        if not match:
            continue

        dt = pd.to_datetime(
            match.group(
                1
            ),
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(
            dt
        ):
            continue

        candidates.append(
            (
                pd.Timestamp(
                    dt
                ).normalize(),
                path,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No Maruhan-style ana_slo_YYYYMMDD_source.html "
            "was found in the project root."
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    return (
        candidates[-1][1],
        candidates[-1][0],
    )


def expected_daily_csv(
    source_date: pd.Timestamp,
) -> Path:
    return (
        DATA_DIR
        / f"ana_slo_{source_date:%Y%m%d}.csv"
    )


def require_nonempty_file(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} missing:\n{path}"
        )

    if (
        path.stat().st_size
        <= 0
    ):
        raise RuntimeError(
            f"{label} is empty:\n{path}"
        )



def discover_latest_69_output() -> Path:
    if not BACKTEST_69_DIR.exists():
        raise FileNotFoundError(
            f"69 backtest output directory missing:\n{BACKTEST_69_DIR}"
        )

    csv_files = [
        path
        for path in BACKTEST_69_DIR.glob("*.csv")
        if path.is_file() and path.stat().st_size > 0
    ]

    if not csv_files:
        raise RuntimeError(
            f"No non-empty 69 backtest CSV was produced in:\n{BACKTEST_69_DIR}"
        )

    status_candidates = [
        path for path in csv_files
        if "status" in path.name.lower()
    ]

    candidates = status_candidates or csv_files
    candidates.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0]


def summarize_69_output(path: Path) -> str:
    df = None
    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "cp932"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except Exception as exc:
            last_error = exc

    if df is None:
        return f"CSV readable check unavailable: {last_error}"

    status_col = next(
        (
            col for col in df.columns
            if str(col).strip().lower()
            in ("status", "evaluation_status", "result_status")
        ),
        None,
    )

    if status_col is None:
        return f"rows={len(df)}"

    counts = (
        df[status_col]
        .astype("string")
        .fillna("NA")
        .value_counts(dropna=False)
    )

    return ", ".join(
        f"{key}={int(value)}"
        for key, value in counts.items()
    )


def main() -> None:
    args = parse_args()

    if (
        args.fetch_days
        < 1
    ):
        raise ValueError(
            "--fetch-days must be >= 1"
        )

    if (
        args.chrome_wait_sec
        < 1
    ):
        raise ValueError(
            "--chrome-wait-sec must be >= 1"
        )

    if args.target_date:
        pd.to_datetime(
            args.target_date,
            format="%Y-%m-%d",
            errors="raise",
        )

    header(
        "Maruhan Mega City Maebashi Inter "
        "- True One-Click Daily Update V2"
    )

    print(
        f"project root          : {PROJECT_ROOT}"
    )
    print(
        f"python                : {sys.executable}"
    )
    print(
        f"fetch days            : {args.fetch_days}"
    )
    print(
        f"skip fetch            : {args.skip_fetch}"
    )
    print(
        f"requested target      : "
        f"{args.target_date if args.target_date else '(auto)'}"
    )
    print(
        f"allow gap             : {args.allow_gap}"
    )

    # --------------------------------------------------------
    # PRE-FLIGHT
    # --------------------------------------------------------
    header(
        "PREFLIGHT"
    )

    for path in (
        SCRIPT_FETCH,
        SCRIPT_CONVERT,
        SCRIPT_69,
        SCRIPT_FORWARD,
        SCRIPT_LIVE,
    ):
        require_file(
            path
        )

        print(
            f"script exists         : "
            f"{path.name}"
        )

    ensure_cdp(
        args.chrome_wait_sec
    )

    print()
    print(
        "Compiling required scripts..."
    )

    for path in (
        SCRIPT_FETCH,
        SCRIPT_CONVERT,
        SCRIPT_69,
        SCRIPT_FORWARD,
        SCRIPT_LIVE,
    ):
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

    stage_rows = []

    # --------------------------------------------------------
    # 1) FETCH
    # --------------------------------------------------------
    if not args.skip_fetch:
        elapsed = run_stage(
            "STEP 1 / 5 - FETCH NEWEST HTML",
            SCRIPT_FETCH,
            [
                "--max-days",
                str(
                    args.fetch_days
                ),
            ],
        )

        stage_rows.append(
            (
                "FETCH NEWEST HTML",
                elapsed,
            )
        )
    else:
        print()
        print(
            "STEP 1 / 5 - FETCH NEWEST HTML : SKIPPED"
        )

    # --------------------------------------------------------
    # Resolve the source explicitly.
    # --------------------------------------------------------
    source_html, source_date = (
        discover_latest_source()
    )

    header(
        "RESOLVED SOURCE"
    )

    print(
        f"source HTML           : {source_html}"
    )
    print(
        f"source date           : {source_date.date()}"
    )

    # --------------------------------------------------------
    # 2) CONVERT
    # --------------------------------------------------------
    elapsed = run_stage(
        "STEP 2 / 5 - HTML TO VALIDATED DAILY CSV",
        SCRIPT_CONVERT,
        [
            str(
                source_html
            ),
        ],
    )

    stage_rows.append(
        (
            "HTML TO DAILY CSV",
            elapsed,
        )
    )

    daily_csv = expected_daily_csv(
        source_date
    )

    require_nonempty_file(
        daily_csv,
        "validated daily CSV",
    )

    print()
    print(
        f"validated daily CSV   : {daily_csv}"
    )

    # --------------------------------------------------------
    # 3) 69 LIVE PREDICTION BACKTEST
    # --------------------------------------------------------
    elapsed = run_stage(
        "STEP 3 / 5 - LIVE PREDICTION BACKTEST 69",
        SCRIPT_69,
    )

    stage_rows.append(
        (
            "69 LIVE BACKTEST",
            elapsed,
        )
    )

    backtest_69_output = discover_latest_69_output()
    require_nonempty_file(
        backtest_69_output,
        "69 live prediction backtest output",
    )
    backtest_69_summary = summarize_69_output(
        backtest_69_output
    )

    print()
    print(
        f"69 backtest output    : {backtest_69_output}"
    )
    print(
        f"69 evaluation status : {backtest_69_summary}"
    )

    # --------------------------------------------------------
    # 4) 63 FORWARD TRACKING
    # --------------------------------------------------------
    elapsed = run_stage(
        "STEP 4 / 5 - CHAMPION / CHALLENGER FORWARD",
        SCRIPT_FORWARD,
    )

    stage_rows.append(
        (
            "63 FORWARD",
            elapsed,
        )
    )

    # --------------------------------------------------------
    # 5) 79 LIVE PIPELINE
    # --------------------------------------------------------
    live_args = []

    if args.target_date:
        live_args += [
            "--target-date",
            args.target_date,
        ]

    if args.allow_gap:
        live_args.append(
            "--allow-gap"
        )

    elapsed = run_stage(
        "STEP 5 / 5 - LIVE PREDICTION PIPELINE 79",
        SCRIPT_LIVE,
        live_args,
    )

    stage_rows.append(
        (
            "79 LIVE PIPELINE",
            elapsed,
        )
    )

    total_elapsed = (
        time.perf_counter()
        - total_started
    )

    prediction_target = (
        pd.Timestamp(
            pd.to_datetime(
                args.target_date,
                format="%Y-%m-%d",
                errors="raise",
            )
        ).normalize()
        if args.target_date
        else (
            source_date
            + pd.Timedelta(
                days=1
            )
        )
    )

    integrated_path = (
        DATA_DIR
        / "analysis_31days_deep"
        / "77_live_integrated_prediction_report"
        / (
            f"77_integrated_prediction_"
            f"{prediction_target:%Y%m%d}.csv"
        )
    )

    require_nonempty_file(
        integrated_path,
        "77 integrated prediction",
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    header(
        "ONE-CLICK PIPELINE SUMMARY"
    )

    for label, elapsed in stage_rows:
        print(
            f"{label:<28}: "
            f"OK  ({elapsed:.2f} sec)"
        )

    print()
    print(
        f"source date           : "
        f"{source_date.date()}"
    )
    print(
        f"prediction target     : "
        f"{prediction_target.date()}"
    )
    print(
        f"validated daily CSV   : "
        f"{daily_csv}"
    )
    print(
        f"69 backtest output    : "
        f"{backtest_69_output}"
    )
    print(
        f"69 evaluation status : "
        f"{backtest_69_summary}"
    )
    print(
        f"77 integrated report : "
        f"{integrated_path}"
    )
    print(
        f"total elapsed sec     : "
        f"{total_elapsed:.2f}"
    )

    print()
    print(
        "Maruhan Maebashi one-click daily update complete."
    )
    print(
        "79 generated 64 NORMAL / 74 A-TYPE / "
        "75 JUGGLER / 77 INTEGRATED."
    )
    print(
        "78 lottery filtering was not run."
    )
    print(
        "69 evaluated frozen live predictions against available actual data."
    )
    print(
        "76 actual-result evaluation was not run."
    )
    print(
        "No Big March files were modified."
    )


if __name__ == "__main__":
    main()
