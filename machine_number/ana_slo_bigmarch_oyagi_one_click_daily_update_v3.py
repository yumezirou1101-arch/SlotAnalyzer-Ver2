from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request


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
# Then:
#   1) Fetch newest Oyagi Ana-Slo HTML
#      (fetch V3 auto-opens store list tab if needed)
#   2) Convert HTML -> daily CSV
#   3) Run frozen JUGGLER_RECENT7_WIN Top3 forward test
#
# This dedicated Chrome profile is separate from the user's
# normal Chrome profile, reducing profile-lock conflicts.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

MACHINE_DIR = PROJECT_ROOT / "machine_number"

SCRIPT_FETCH = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_click_fetch_31days_v3.py"
)

SCRIPT_CONVERT = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_batch_html_to_daily_csv.py"
)

SCRIPT_FORWARD = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_juggler_recent7_top3_forward.py"
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
            str(Path.home() / "AppData" / "Local"),
        )
    )
    / "Google"
    / "Chrome"
    / "Application"
    / "chrome.exe",
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

    # Keep the browser independent of the Python process.
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
                f"Chrome                : {info.get('Browser')}"
            )
            print(
                "CDP                   : AUTO-START OK"
            )
            return info

        time.sleep(
            0.5
        )

    raise RuntimeError(
        f"Chrome was started, but CDP did not become ready "
        f"within {wait_sec} seconds."
    )


def ensure_cdp(
    wait_sec: int,
) -> dict:

    info = try_get_cdp()

    if info is not None:
        print(
            f"Chrome                : {info.get('Browser')}"
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
        f"project root          : {PROJECT_ROOT}"
    )
    print(
        f"python                : {sys.executable}"
    )
    print(
        f"fetch days            : {args.fetch_days}"
    )
    print(
        f"min machines          : {args.min_machines}"
    )
    print(
        f"skip fetch            : {args.skip_fetch}"
    )

    # --------------------------------------------------------
    # Preflight
    # --------------------------------------------------------
    header(
        "PREFLIGHT"
    )

    for path in (
        SCRIPT_FETCH,
        SCRIPT_CONVERT,
        SCRIPT_FORWARD,
    ):
        check_file(
            path
        )

        print(
            f"script exists         : {path.name}"
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
        SCRIPT_FORWARD,
    ):
        compile_script(
            path
        )

        print(
            f"py_compile OK         : {path.name}"
        )

    total_started = time.perf_counter()
    elapsed_rows = []

    # --------------------------------------------------------
    # Fetch newest HTML
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
    # HTML -> daily CSV
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
    # Frozen forward test
    # --------------------------------------------------------
    elapsed = run_stage(
        "FROZEN TOP3 FORWARD TEST",
        SCRIPT_FORWARD,
    )

    elapsed_rows.append(
        (
            "FROZEN TOP3 FORWARD TEST",
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
            f"{label:<28}: "
            f"OK  ({elapsed:.2f} sec)"
        )

    print()
    print(
        f"total elapsed sec     : "
        f"{total_elapsed:.2f}"
    )

    print()
    print(
        "09 V3 daily update complete."
    )
    print(
        "If CDP was not running, Chrome was started automatically."
    )
    print(
        "If the Oyagi store tab was absent, fetch V3 opened it automatically."
    )
    print(
        "The development period through 2026-08-26 remains locked."
    )
    print(
        "No automatic model promotion is performed."
    )
    print(
        "No Maruhan Maebashi files were modified."
    )


if __name__ == "__main__":
    main()
