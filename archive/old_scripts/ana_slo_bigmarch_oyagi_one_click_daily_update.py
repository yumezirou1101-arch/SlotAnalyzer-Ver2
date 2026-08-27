from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import urllib.request


# ============================================================
# 09 - Big March Takasaki Oyagi
# One-Click Daily Update Pipeline
# ============================================================
#
# Daily flow
# ----------
# 1) Preflight:
#    - confirm Chrome remote debugging (127.0.0.1:9222)
#    - confirm required Python scripts exist
# 2) Fetch newest Ana-Slo daily page by normal date-link click
# 3) Batch-convert saved HTML -> daily CSV
# 4) Run frozen JUGGLER_RECENT7_WIN Top3 forward test
#
# Existing HTML / CSV files are skipped by the child scripts.
# Development data through 2026-08-26 remains locked.
# No Maruhan Maebashi files are modified.
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\user\Desktop\Documents\SlotAnalyzer"
)

MACHINE_DIR = PROJECT_ROOT / "machine_number"

SCRIPT_FETCH = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_click_fetch_31days_v2.py"
)

SCRIPT_CONVERT = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_batch_html_to_daily_csv.py"
)

SCRIPT_FORWARD = (
    MACHINE_DIR
    / "ana_slo_bigmarch_oyagi_juggler_recent7_top3_forward.py"
)

CDP_VERSION_URL = "http://127.0.0.1:9222/json/version"


def header(title: str) -> None:
    print()
    print("=" * 122)
    print(title)
    print("=" * 122)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "One-click daily update for Big March Takasaki Oyagi."
        )
    )

    p.add_argument(
        "--fetch-days",
        type=int,
        default=1,
        help=(
            "Newest N date links to inspect. "
            "Default: 1. Existing HTML is skipped."
        ),
    )

    p.add_argument(
        "--min-machines",
        type=int,
        default=200,
        help=(
            "Minimum acceptable machines for historical/current pages. "
            "Default: 200."
        ),
    )

    p.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip Ana-Slo HTML fetch stage.",
    )

    return p.parse_args()


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required script not found:\n{path}"
        )


def check_cdp() -> dict:
    try:
        with urllib.request.urlopen(
            CDP_VERSION_URL,
            timeout=3,
        ) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            "Chrome remote debugging is not available at "
            "127.0.0.1:9222.\n"
            "Start/use the remote-debugging Chrome first."
        ) from exc

    try:
        info = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(
            "Could not parse Chrome CDP version response."
        ) from exc

    if not info.get("webSocketDebuggerUrl"):
        raise RuntimeError(
            "CDP response has no webSocketDebuggerUrl."
        )

    return info


def compile_script(path: Path) -> None:
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

    elapsed = time.perf_counter() - started

    print()
    print(
        f"{label} return code    : {result.returncode}"
    )
    print(
        f"{label} elapsed sec    : {elapsed:.2f}"
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

    header(
        "09 - Big March Takasaki Oyagi One-Click Daily Update"
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
        check_file(path)
        print(
            f"script exists         : {path.name}"
        )

    cdp = check_cdp()

    print(
        f"Chrome                : {cdp.get('Browser')}"
    )
    print(
        "CDP                   : OK"
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
        compile_script(path)
        print(
            f"py_compile OK         : {path.name}"
        )

    total_started = time.perf_counter()
    elapsed_rows = []

    # --------------------------------------------------------
    # Stage 1: fetch
    # --------------------------------------------------------
    if not args.skip_fetch:
        elapsed = run_stage(
            "FETCH NEWEST HTML",
            SCRIPT_FETCH,
            [
                "--max-days",
                str(args.fetch_days),
                "--min-machines",
                str(args.min_machines),
            ],
        )
        elapsed_rows.append(
            ("FETCH NEWEST HTML", elapsed)
        )
    else:
        print()
        print(
            "FETCH NEWEST HTML     : SKIPPED"
        )

    # --------------------------------------------------------
    # Stage 2: HTML -> CSV
    # --------------------------------------------------------
    elapsed = run_stage(
        "BATCH HTML TO DAILY CSV",
        SCRIPT_CONVERT,
        [
            "--min-machines",
            str(args.min_machines),
        ],
    )

    elapsed_rows.append(
        ("BATCH HTML TO DAILY CSV", elapsed)
    )

    # --------------------------------------------------------
    # Stage 3: frozen forward test
    # --------------------------------------------------------
    elapsed = run_stage(
        "FROZEN TOP3 FORWARD TEST",
        SCRIPT_FORWARD,
    )

    elapsed_rows.append(
        ("FROZEN TOP3 FORWARD TEST", elapsed)
    )

    total_elapsed = (
        time.perf_counter()
        - total_started
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    header(
        "09 PIPELINE SUMMARY"
    )

    for label, elapsed in elapsed_rows:
        print(
            f"{label:<28}: OK  ({elapsed:.2f} sec)"
        )

    print()
    print(
        f"total elapsed sec     : {total_elapsed:.2f}"
    )

    print()
    print(
        "09 daily update complete."
    )
    print(
        "Existing HTML/CSV files were preserved unless child-script "
        "rules explicitly refreshed them."
    )
    print(
        "The JUGGLER_RECENT7_WIN Top3 development period "
        "through 2026-08-26 remains locked."
    )
    print(
        "No automatic model promotion is performed."
    )
    print(
        "No Maruhan Maebashi files were modified."
    )


if __name__ == "__main__":
    main()
